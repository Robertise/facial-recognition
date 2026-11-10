"""
tests/innovation/test_purifier.py - Unit tests for `innovation/defense/` (plan_purifier.md).

Tests:
1. `band_in_crop` on synthetic frame: mask covers 100% of painted H09 PhotoFrame fill-lens pixels.
2. Output compositing: pixels outside `band` are bit-for-bit unchanged.
3. Strict invariance: two crops differing only inside `band` produce bit-for-bit identical purifier outputs.
4. `prepare`: excludes gallery LFW identities (raises SystemExit on unmapped ID) and splits train/val disjointly by identity.
5. `train_purifier` and `eval_purifier`: run end-to-end on a small fixture with a random PurifierUNet.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from innovation.attacks.adv_glasses import faces, glasses, printability
from innovation.attacks.adv_glasses.evaluate import paint
from innovation.defense.band import band_in_crop, build_band_template
from innovation.defense.eval_purifier import parse_attack_spec, run_eval_purifier
from innovation.defense.prepare import (
    prepare_lfw_dataset,
    resolve_gallery_lfw_names,
    split_lfw_identities,
)
from innovation.defense.purifier import (
    PurifierUNet,
    apply_gray,
    apply_purifier,
    count_parameters,
    purifier_loss,
)
from innovation.defense.train_purifier import train_purifier


def test_band_in_crop_covers_h09_photoframe_paint(tmp_path):
    """Verify that `band_in_crop` covers every pixel painted by H09 (PhotoFrame fill-lens) on a 112x112 crop."""
    band_tmpl, meta = build_band_template(
        out_template=tmp_path / "band_template.png",
        out_meta=tmp_path / "band_meta.json",
    )
    assert band_tmpl.shape == (glasses.TEMPLATE_H, glasses.TEMPLATE_W)
    assert meta["band_area_px"] > 0

    # H09 fill-lens PhotoFrame (grow_px=8, clearance_mm=9)
    h09_shape = glasses.PhotoFrame(
        mask_path="data/innovation/frame/owner_frame_mask.png",
        lens_path="data/innovation/frame/owner_frame_lens.png",
        grow_px=8.0,
        clearance_mm=9.0,
        ipd_mm=63.0,
    )
    h09_mask_np = h09_shape.mask()
    # Every active pixel of H09 in template space must be inside band_template
    assert np.all(band_tmpl[h09_mask_np > 0] == 1.0)

    # Synthetic 480x640 frame, bbox, and eye landmarks
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    bbox = [200, 120, 240, 240]
    eyes = np.array([[275.0, 215.0], [365.0, 215.0]], dtype=np.float32)

    A = glasses.similarity(eyes)
    h09_mask_t = torch.from_numpy(h09_mask_np)[None]
    tex_white = torch.ones((3, glasses.TEMPLATE_H, glasses.TEMPLATE_W), dtype=torch.float32)
    img_painted = paint(img, tex_white, h09_mask_t, A)

    crop_clean = cv2.resize(faces.crop(img, bbox), (112, 112), interpolation=cv2.INTER_LINEAR)
    crop_painted = cv2.resize(faces.crop(img_painted, bbox), (112, 112), interpolation=cv2.INTER_LINEAR)

    mask112 = band_in_crop(img, bbox, eyes, band_template=band_tmpl, size=112)
    assert mask112 is not None
    assert mask112.shape == (112, 112)
    assert mask112.dtype == np.uint8
    assert set(np.unique(mask112)).issubset({0, 1})

    # Every pixel modified by H09 paint in the 112x112 crop must lie inside mask112 == 1
    changed_pixels = np.any(crop_painted != crop_clean, axis=2)
    assert changed_pixels.sum() > 100
    outside_band_changes = np.logical_and(changed_pixels, mask112 == 0)
    assert int(outside_band_changes.sum()) == 0, (
        f"Found {outside_band_changes.sum()} painted H09 pixels outside band_in_crop mask"
    )

    # Missing eyes returns None
    assert band_in_crop(img, bbox, None, band_template=band_tmpl) is None


def test_purifier_compositing_preserves_outside_band():
    """Verify that pixels outside `band` remain bit-for-bit unchanged after PurifierUNet and gray baseline."""
    torch.manual_seed(0)
    model = PurifierUNet().eval()
    n_params = count_parameters(model)
    assert 2_000_000 <= n_params <= 5_000_000

    rng = np.random.default_rng(42)
    crop_u8 = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    band = np.zeros((112, 112), dtype=np.uint8)
    band[30:65, 15:95] = 1

    # 1. Float forward pass
    x = torch.from_numpy(crop_u8.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    m = torch.from_numpy(band.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(x, m)

    outside = (m.expand_as(x) == 0)
    assert torch.equal(out[outside], x[outside])

    # 2. uint8 helper apply_purifier
    out_u8 = apply_purifier(model, crop_u8, band)
    assert np.array_equal(out_u8[band == 0], crop_u8[band == 0])

    # 3. uint8 helper apply_gray
    gray_u8 = apply_gray(crop_u8, band)
    assert np.array_equal(gray_u8[band == 0], crop_u8[band == 0])
    assert np.all(gray_u8[band == 1] == 128)


def test_purifier_strict_invariance_to_inside_band():
    """Two crops that differ ONLY inside `band` must produce bit-for-bit identical purifier outputs."""
    torch.manual_seed(7)
    model = PurifierUNet().eval()

    rng = np.random.default_rng(7)
    crop1 = rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
    crop2 = crop1.copy()

    band = np.zeros((112, 112), dtype=np.uint8)
    band[25:70, 10:102] = 1

    # Replace inside of band in crop2 with arbitrary adversarial pattern
    crop2[band == 1] = rng.integers(0, 256, size=(int(band.sum()), 3), dtype=np.uint8)
    assert not np.array_equal(crop1, crop2)
    assert np.array_equal(crop1[band == 0], crop2[band == 0])

    # Tensor invariance
    x1 = torch.from_numpy(crop1.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x2 = torch.from_numpy(crop2.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    m = torch.from_numpy(band.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        y1 = model(x1, m)
        y2 = model(x2, m)
    assert torch.equal(y1, y2)

    # uint8 invariance
    u1 = apply_purifier(model, crop1, band)
    u2 = apply_purifier(model, crop2, band)
    assert np.array_equal(u1, u2)


def test_prepare_excludes_gallery_and_splits_disjoint_identities(tmp_path):
    """Verify `prepare` excludes gallery LFW identities, fails on unmapped IDs, and splits train/val disjointly."""
    lfw_dir = tmp_path / "lfw"
    lfw_dir.mkdir()
    rng = np.random.default_rng(0)

    all_person_names = [f"Person_{i:02d}" for i in range(1, 13)]
    for p_name in all_person_names:
        p_dir = lfw_dir / p_name
        p_dir.mkdir()
        for j in range(2):
            img = rng.integers(40, 200, size=(120, 120, 3), dtype=np.uint8)
            cv2.imwrite(str(p_dir / f"{p_name}_{j:04d}.jpg"), img)

    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()
    for folder in ("LFW_0001", "LFW_0002", "LFW_victim", "owner"):
        (gallery_dir / folder).mkdir()

    build_report = {
        "id_to_lfw_name": {
            "LFW_0001": "Person_01",
            "LFW_0002": "Person_02",
            "LFW_victim": "Person_03",
        }
    }
    (gallery_dir / "_build_report.json").write_text(json.dumps(build_report), encoding="utf-8")

    split_info = split_lfw_identities(lfw_dir=lfw_dir, gallery_dir=gallery_dir, seed=0)
    excluded = set(split_info["excluded_gallery_identities"])
    train_ids = set(split_info["train_ids"])
    val_ids = set(split_info["val_ids"])

    assert excluded == {"Person_01", "Person_02", "Person_03"}
    assert train_ids.isdisjoint(excluded)
    assert val_ids.isdisjoint(excluded)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(all_person_names) - excluded

    # Mock detector and eyes finder to run prepare_lfw_dataset quickly
    class FakeDetector:
        def detect(self, img):
            return [10, 10, 100, 100]

    class FakeEyes:
        def eyes(self, img):
            return np.array([[40.0, 50.0], [80.0, 50.0]], dtype=np.float32)

    band_tmpl = np.zeros((glasses.TEMPLATE_H, glasses.TEMPLATE_W), dtype=np.float32)
    band_tmpl[40:110, 40:360] = 1.0

    out_dir = tmp_path / "lfw_crops"
    rep = prepare_lfw_dataset(
        lfw_dir=lfw_dir,
        gallery_dir=gallery_dir,
        out_dir=out_dir,
        seed=0,
        shard_size=10,
        detector=FakeDetector(),
        eyes_finder=FakeEyes(),
        band_template=band_tmpl,
    )
    assert rep["used_train"] == len(train_ids) * 2
    assert rep["used_val"] == len(val_ids) * 2

    tr_shard = np.load(out_dir / "train_000.npz", allow_pickle=True)
    va_shard = np.load(out_dir / "val_000.npz", allow_pickle=True)
    assert tr_shard["crops"].shape[1:] == (112, 112, 3)
    assert tr_shard["bands"].shape[1:] == (112, 112)
    assert set(tr_shard["identities"]).isdisjoint(excluded)
    assert set(va_shard["identities"]).isdisjoint(excluded)
    assert set(tr_shard["identities"]).isdisjoint(set(va_shard["identities"]))

    # Unmapped gallery folder must trigger SystemExit
    (gallery_dir / "LFW_9999").mkdir()
    with pytest.raises(SystemExit, match="Could not map gallery identity"):
        resolve_gallery_lfw_names(gallery_dir, lfw_dir=lfw_dir)


def test_train_and_eval_purifier_on_small_fixture(tmp_path, monkeypatch):
    """Verify `train_purifier` and `eval_purifier` execute end-to-end on a small synthetic fixture."""
    import backend.face_recognizer as fr_mod
    from innovation.defense import eval_purifier as eval_pur_mod

    # 1. Create tiny shard dataset and run 1 epoch of train_purifier
    data_dir = tmp_path / "lfw_crops"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    band_small = np.zeros((4, 112, 112), dtype=np.uint8)
    band_small[:, 35:65, 20:92] = 1
    np.savez_compressed(
        data_dir / "train_000.npz",
        crops=rng.integers(50, 200, size=(4, 112, 112, 3), dtype=np.uint8),
        bands=band_small,
        identities=np.array(["id1", "id1", "id2", "id2"]),
        paths=np.array(["a.jpg", "b.jpg", "c.jpg", "d.jpg"]),
    )
    np.savez_compressed(
        data_dir / "val_000.npz",
        crops=rng.integers(50, 200, size=(2, 112, 112, 3), dtype=np.uint8),
        bands=band_small[:2],
        identities=np.array(["id3", "id3"]),
        paths=np.array(["e.jpg", "f.jpg"]),
    )

    class DummyEmbedder(torch.nn.Module):
        def forward(self, x):
            pooled = x.mean(dim=(2, 3))  # (B, 3)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    train_out = tmp_path / "train_out"
    train_purifier(
        data_dir=data_dir,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        seed=0,
        out_dir=train_out,
        embedder=DummyEmbedder(),
    )
    best_ckpt = train_out / "best.pt"
    assert best_ckpt.exists()
    assert (train_out / "train_log.csv").exists()
    assert (train_out / "summary.json").exists()

    # 2. Setup fixture for eval_purifier
    owner_s6 = tmp_path / "owner_s6"
    owner_s6.mkdir()
    p01_root = tmp_path / "p01"
    p01_s2 = p01_root / "s2"
    p01_s2.mkdir(parents=True)

    for idx in range(3):
        cv2.imwrite(str(owner_s6 / f"{idx:03d}.jpg"), np.full((120, 120, 3), 100 + idx * 10, dtype=np.uint8))
        cv2.imwrite(str(p01_s2 / f"{idx:03d}.jpg"), np.full((120, 120, 3), 150 + idx * 10, dtype=np.uint8))

    main_gal = tmp_path / "gallery"
    for gid in ("owner", "LFW_victim"):
        g_folder = main_gal / gid
        g_folder.mkdir(parents=True)
        cv2.imwrite(str(g_folder / "000.jpg"), np.full((120, 120, 3), 90, dtype=np.uint8))

    extra_gal = tmp_path / "gallery_people"
    (extra_gal / "p01").mkdir(parents=True)
    cv2.imwrite(str(extra_gal / "p01" / "000.jpg"), np.full((120, 120, 3), 140, dtype=np.uint8))

    # Create mock H09 and R2b attack runs
    for run_name in ("h09_run", "r2b_run"):
        r_dir = tmp_path / run_name
        r_dir.mkdir()
        (r_dir / "split.json").write_text(json.dumps({"train": [], "dev": [], "eval": []}), encoding="utf-8")
        printability.ColorMap.identity().save(r_dir / "colormap.json")
        torch.save({"logits": torch.zeros(3, 15, 40), "tex_res": [15, 40]}, r_dir / "texture.pt")
        (r_dir / "attack.json").write_text(json.dumps({
            "mode": "impersonate",
            "victim": "LFW_victim",
            "attacker_id": "owner",
            "gallery": str(main_gal),
            "checkpoint": "checkpoints/method3_best.pth",
            "frame_shape": {"thickness": 25.0, "rim_scale": 1.1},
        }), encoding="utf-8")

    class FakeDetector:
        def detect(self, img):
            return [10, 10, 100, 100]

    class FakeEyes:
        def eyes(self, img):
            return np.array([[40.0, 50.0], [80.0, 50.0]], dtype=np.float32)

    class FakeRecognizer:
        def __init__(self, *a, **kw):
            self.threshold = 0.5
            self.ckpt_path = "fake.pth"
            self.load_report = {"loaded": True}

        def extract_embedding(self, face):
            v = np.zeros(16, dtype=np.float32)
            v[0] = float(face.mean()) / 255.0
            v[1] = 1.0
            return v / np.linalg.norm(v)

    monkeypatch.setattr(faces, "FaceDetector", FakeDetector)
    monkeypatch.setattr(faces, "EyeLandmarks", FakeEyes)
    monkeypatch.setattr(faces, "enhance", lambda crop: (crop, {"applied": False}))
    monkeypatch.setattr(eval_pur_mod, "FaceRecognizer", FakeRecognizer)
    monkeypatch.setattr(fr_mod, "FaceRecognizer", FakeRecognizer)

    band_tmpl = np.zeros((glasses.TEMPLATE_H, glasses.TEMPLATE_W), dtype=np.float32)
    band_tmpl[40:110, 40:360] = 1.0

    eval_out = tmp_path / "eval_out"
    run_eval_purifier(
        modes=["none", "gray", "purifier"],
        purifier_ckpt=best_ckpt,
        h09_run=tmp_path / "h09_run",
        r2b_run=tmp_path / "r2b_run",
        owner_s6_dir=owner_s6,
        p01_root=p01_root,
        gallery_dir=main_gal,
        extra_gallery=[str(extra_gal)],
        liveness=False,
        device="cpu",
        output_dir=eval_out,
        band_template=band_tmpl,
    )

    assert (eval_out / "frames.csv").exists()
    assert (eval_out / "summary.json").exists()

    with open(eval_out / "frames.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 6 sets (H09, R2b, owner_s6, owner_control_random, owner_control_plain, p01_s2) * 3 frames * 3 modes = 54 rows
    assert len(rows) == 54
    assert {r["mode"] for r in rows} == {"none", "gray", "purifier"}
    assert {r["set"] for r in rows} == {
        "H09", "R2b", "owner_s6", "owner_control_random", "owner_control_plain", "p01_s2"
    }

    summary = json.loads((eval_out / "summary.json").read_text(encoding="utf-8"))
    assert "purifier/H09" in summary["results"]
    assert "purifier/owner_s6" in summary["results"]
    assert "purifier/p01_s2" in summary["results"]
    assert "purifier" in summary["consistency"]
    assert "gray" in summary["consistency"]
    assert "rule_check" in summary


def test_parse_attack_spec_windows_and_posix():
    """Verify `parse_attack_spec` handles both POSIX paths and Windows drive-letter paths."""
    name, run_p, sess_p = parse_attack_spec("P01=results/run1:data/frames/p01/a3")
    assert name == "P01"
    assert run_p == Path("results/run1")
    assert sess_p == Path("data/frames/p01/a3")

    name_w, run_w, sess_w = parse_attack_spec(r"P01=C:\runs\r1:D:\frames\p01\a3")
    assert name_w == "P01"
    assert run_w == Path(r"C:\runs\r1")
    assert sess_w == Path(r"D:\frames\p01\a3")


def test_eval_purifier_fail_closed_when_no_band_and_missing_p01(tmp_path, monkeypatch):
    """Verify §7: frames without eye landmarks (`band_ok == 0`) are fail-closed in `gray`/`purifier`
    (`success = 0`, `match_id = ""`, `top1_after_id = ""`, `top1_drop = ""`, `rejected_no_band = 1`),
    unchanged in `none`, treated as `+inf` in consistency detector, and missing `p01_s2` gives `purifier_pass = null`.
    """
    import backend.face_recognizer as fr_mod
    from innovation.defense import eval_purifier as eval_pur_mod

    owner_s6 = tmp_path / "owner_s6"
    owner_s6.mkdir()
    # Frame 000 has eyes (pixel value < 180); frame 001 has no eyes on rendered frame (pixel value >= 180)
    cv2.imwrite(str(owner_s6 / "000.jpg"), np.full((120, 120, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(owner_s6 / "001.jpg"), np.full((120, 120, 3), 200, dtype=np.uint8))

    main_gal = tmp_path / "gallery"
    (main_gal / "owner").mkdir(parents=True)
    cv2.imwrite(str(main_gal / "owner" / "000.jpg"), np.full((120, 120, 3), 120, dtype=np.uint8))
    (main_gal / "LFW_victim").mkdir(parents=True)
    cv2.imwrite(str(main_gal / "LFW_victim" / "000.jpg"), np.full((120, 120, 3), 30, dtype=np.uint8))

    for run_name in ("h09_run", "r2b_run"):
        r_dir = tmp_path / run_name
        r_dir.mkdir()
        (r_dir / "split.json").write_text(json.dumps({"train": [], "dev": [], "eval": []}), encoding="utf-8")
        printability.ColorMap.identity().save(r_dir / "colormap.json")
        torch.save({"logits": torch.zeros(3, 15, 40), "tex_res": [15, 40]}, r_dir / "texture.pt")
        (r_dir / "attack.json").write_text(json.dumps({
            "mode": "impersonate",
            "victim": "LFW_victim",
            "attacker_id": "owner",
            "gallery": str(main_gal),
            "checkpoint": "checkpoints/method3_best.pth",
            "frame_shape": {"thickness": 25.0, "rim_scale": 1.1},
        }), encoding="utf-8")

    class FakeDetector:
        def detect(self, img):
            return [10, 10, 100, 100]

    class FakeEyes:
        def eyes(self, img):
            # Return None when image brightness >= 180 to simulate failed landmark detection
            if float(img.mean()) >= 180.0:
                return None
            return np.array([[40.0, 50.0], [80.0, 50.0]], dtype=np.float32)

    class FakeRecognizer:
        def __init__(self, *a, **kw):
            self.threshold = 0.5
            self.ckpt_path = "fake.pth"
            self.load_report = {"loaded": True}

        def extract_embedding(self, face):
            # Distinguish owner (brightness >= 80) from LFW_victim (brightness < 80)
            v = np.zeros(16, dtype=np.float32)
            if float(face.mean()) >= 80.0:
                v[0] = 1.0
            else:
                v[1] = 1.0
            return v

    monkeypatch.setattr(faces, "FaceDetector", FakeDetector)
    monkeypatch.setattr(faces, "EyeLandmarks", FakeEyes)
    monkeypatch.setattr(faces, "enhance", lambda crop: (crop, {"applied": False}))
    monkeypatch.setattr(eval_pur_mod, "FaceRecognizer", FakeRecognizer)
    monkeypatch.setattr(fr_mod, "FaceRecognizer", FakeRecognizer)

    band_tmpl = np.zeros((glasses.TEMPLATE_H, glasses.TEMPLATE_W), dtype=np.float32)
    band_tmpl[40:110, 40:360] = 1.0

    eval_out = tmp_path / "eval_no_band"
    run_eval_purifier(
        modes=["none", "gray", "purifier"],
        purifier_model=PurifierUNet().eval(),
        h09_run=tmp_path / "h09_run",
        r2b_run=tmp_path / "r2b_run",
        owner_s6_dir=owner_s6,
        p01_root=tmp_path / "missing_p01",
        gallery_dir=main_gal,
        liveness=False,
        device="cpu",
        output_dir=eval_out,
        band_template=band_tmpl,
    )

    with open(eval_out / "frames.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    no_eye_rows = {
        r["mode"]: r
        for r in rows
        if r["set"] == "owner_s6" and Path(r["frame"]).name == "001.jpg"
    }
    assert set(no_eye_rows.keys()) == {"none", "gray", "purifier"}

    # Mode 'none': unchanged, matches owner and succeeds
    r_none = no_eye_rows["none"]
    assert r_none["band_ok"] == "0"
    assert r_none["rejected_no_band"] == "0"
    assert r_none["match_id"] != ""
    assert r_none["top1_after_id"] != ""
    assert r_none["success"] == "1"

    # Modes 'purifier' and 'gray': fail-closed when band_ok == 0
    for def_mode in ("purifier", "gray"):
        r_def = no_eye_rows[def_mode]
        assert r_def["band_ok"] == "0"
        assert r_def["rejected_no_band"] == "1"
        assert r_def["match_id"] == ""
        assert r_def["top1_after_id"] == ""
        assert r_def["top1_drop"] == ""
        assert r_def["success"] == "0"

    summary = json.loads((eval_out / "summary.json").read_text(encoding="utf-8"))
    assert summary["results"]["none/owner_s6"]["rejected_no_band_rate"] == 0.0
    assert summary["results"]["purifier/owner_s6"]["rejected_no_band_rate"] == 0.5
    assert summary["results"]["gray/owner_s6"]["rejected_no_band_rate"] == 0.5

    # Missing p01_s2 -> purifier_pass and rule2_preserve_genuine must be null (None)
    assert summary["rule_check"]["rule2_preserve_genuine"] is None
    assert summary["rule_check"]["purifier_pass"] is None

