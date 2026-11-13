"""
tests/innovation/test_adaptive.py - Unit tests for adaptive attack against the purifier (plan_adaptive.md).

Tests:
1. `render_padded` with P = 0 and mask 400x150 matches `glasses.render` bit-for-bit.
2. `ring(r)` is disjoint from `band_padded`; area and coverage strictly increase with r.
3. Gradient of purifier branch loss with respect to pixels/texture inside the band is strictly zero.
4. End-to-end evaluation runs on a small synthetic fixture and outputs all keys required by plan_adaptive §3.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from innovation.attacks.adv_glasses import faces, glasses, pipeline
from innovation.defense.adaptive import (
    H_PRIME,
    PAD_PX,
    W_PRIME,
    coverage_padded,
    make_band_padded,
    make_ring,
    render_padded,
    run_eval_adaptive,
    similarity_padded,
    upsample_logits_padded,
)
from innovation.defense.purifier import PurifierUNet, apply_purifier
from innovation.eval.runs import write_json


def test_render_padded_matches_glasses_render_when_pad_zero():
    """Verify that render_padded with w_prime=400, h_prime=150 is identical to glasses.render."""
    rng = np.random.default_rng(42)
    region = torch.from_numpy(rng.uniform(0, 255, size=(3, 140, 140)).astype(np.float32))
    tex = torch.from_numpy(rng.uniform(0, 1, size=(3, 150, 400)).astype(np.float32))
    mask = torch.from_numpy((rng.uniform(0, 1, size=(1, 150, 400)) > 0.4).astype(np.float32))

    eyes = np.array([[45.0, 65.0], [105.0, 65.0]], dtype=np.float32)
    win = (15, 15, 155, 155)

    A_orig = glasses.similarity(eyes, src_eyes=glasses.EYES)
    out_orig = glasses.render(region, tex, mask, A_orig, win)

    A_pad = similarity_padded(eyes, src_eyes=glasses.EYES)
    out_pad = render_padded(region, tex, mask, A_pad, win, w_prime=400, h_prime=150)

    torch.testing.assert_close(out_pad, out_orig, atol=1e-5, rtol=1e-5)


def test_ring_disjoint_from_band_and_coverage_increases_with_r():
    """Verify that ring(r) is strictly disjoint from band_padded and area/coverage increases with r."""
    band_orig = np.zeros((150, 400), dtype=np.float32)
    band_orig[40:110, 80:320] = 1.0  # Synthetic rectangle band

    band_padded = make_band_padded(band_orig, pad=PAD_PX)
    assert band_padded.shape == (H_PRIME, W_PRIME)
    assert np.all((band_padded == 0) | (band_padded == 255))

    ring8 = make_ring(band_padded, 8)
    ring16 = make_ring(band_padded, 16)
    ring32 = make_ring(band_padded, 32)
    ring48 = make_ring(band_padded, 48)

    # 1. Strictly disjoint from band (no active pixels overlap)
    assert np.all((ring8 > 0) & (band_padded > 0) == 0)
    assert np.all((ring16 > 0) & (band_padded > 0) == 0)
    assert np.all((ring32 > 0) & (band_padded > 0) == 0)
    assert np.all((ring48 > 0) & (band_padded > 0) == 0)

    # 2. Area strictly increases with r
    a8 = int(np.sum(ring8 > 0))
    a16 = int(np.sum(ring16 > 0))
    a32 = int(np.sum(ring32 > 0))
    a48 = int(np.sum(ring48 > 0))
    assert 0 < a8 < a16 < a32 < a48

    # 3. Measured coverage strictly increases with r
    eyes = np.array([[120.0, 140.0], [220.0, 140.0]], dtype=np.float32)
    A_pad = similarity_padded(eyes)
    win = (40, 40, 300, 300)

    cov8 = coverage_padded(torch.from_numpy((ring8 > 0).astype(np.float32))[None], A_pad, win)
    cov16 = coverage_padded(torch.from_numpy((ring16 > 0).astype(np.float32))[None], A_pad, win)
    cov32 = coverage_padded(torch.from_numpy((ring32 > 0).astype(np.float32))[None], A_pad, win)
    cov48 = coverage_padded(torch.from_numpy((ring48 > 0).astype(np.float32))[None], A_pad, win)
    assert 0.0 < cov8 < cov16 < cov32 < cov48


def test_purifier_loss_gradient_zero_inside_band():
    """Verify that modifications inside the band produce identically ZERO gradient
    through the purifier branch, establishing strict mathematical invariance."""
    purifier = PurifierUNet(base_channels=(8, 16, 32, 64))
    purifier.eval()

    # Input face crop (1, 3, 112, 112) in [0, 1]
    crop = torch.full((1, 3, 112, 112), 0.5, requires_grad=True)

    # Band mask covering a box in the center
    band_mask = torch.zeros((1, 1, 112, 112), dtype=torch.float32)
    band_mask[0, 0, 30:70, 20:90] = 1.0

    # Purified output
    out = purifier(crop, band_mask)

    # Compute a gradient of the purified output
    loss = (out ** 2).sum()
    loss.backward()

    # All gradients inside the band must be strictly 0.0
    band_indices = (band_mask[0, 0] > 0.5)
    grads_in_band = crop.grad[0, :, band_indices]
    assert torch.all(grads_in_band == 0.0)


def test_adaptive_eval_fixture_end_to_end(tmp_path: Path, monkeypatch):
    """Run run_eval_adaptive on a small synthetic fixture and check that summary.json has all keys from §3."""
    class FakeDetector:
        def detect(self, img):
            return [10, 10, 100, 100]

    class FakeEyes:
        def eyes(self, img):
            return np.array([[40.0, 50.0], [80.0, 50.0]], dtype=np.float32)

    class FakeRecognizer:
        def __init__(self, *a, **kw):
            self.threshold = 0.5
            self.ckpt_path = "checkpoints/method3_best.pth"
            self.load_report = {"loaded": True}
            self.preprocess = type("Pre", (), {"size": 112, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]})()

        def extract_embedding(self, face):
            v = np.zeros(16, dtype=np.float32)
            if float(face.mean()) >= 80.0:
                v[0] = 1.0
            else:
                v[1] = 1.0
            return v

    from innovation.defense import adaptive as adaptive_mod
    monkeypatch.setattr(faces, "FaceDetector", FakeDetector)
    monkeypatch.setattr(faces, "EyeLandmarks", FakeEyes)
    monkeypatch.setattr(faces, "enhance", lambda crop: (crop, {"applied": False}))
    monkeypatch.setattr(adaptive_mod, "FaceRecognizer", FakeRecognizer)

    # 1. Create synthetic band template (150 x 400)
    band_tmpl = np.zeros((150, 400), dtype=np.uint8)
    band_tmpl[40:110, 80:320] = 255
    band_tmpl_path = tmp_path / "band_template.png"
    cv2.imwrite(str(band_tmpl_path), band_tmpl)

    # 2. Attacker frames: owner/s6_vary with 2 synthetic 640x480 face images
    session_dir = tmp_path / "frames" / "owner" / "s6_vary"
    session_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Center square to ensure face-like region
        img[150:350, 230:430] = 160
        cv2.imwrite(str(session_dir / f"{idx:04d}.jpg"), img)
    with open(session_dir / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "client_ms", "bytes"])
        w.writerow([0, 0.0, 1000])
        w.writerow([1, 200.0, 1000])

    # 3. Create synthetic gallery: owner and victim
    gal_dir = tmp_path / "gallery"
    for person in ["owner", "LFW_victim"]:
        p_dir = gal_dir / person
        p_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            img = np.full((120, 120, 3), 130 if person == "owner" else 90, dtype=np.uint8)
            cv2.imwrite(str(p_dir / f"{idx}.jpg"), img)

    # 4. Small purifier model with default channels
    purifier = PurifierUNet()
    purifier_ckpt_path = tmp_path / "purifier_best.pt"
    torch.save(
        {
            "model_state_dict": purifier.state_dict(),
            "best_epoch": 1,
            "best_val_cos": 0.85,
        },
        purifier_ckpt_path,
    )

    # 5. Adaptive optimize run folder
    opt_run = tmp_path / "adaptive_opt_run"
    opt_run.mkdir(parents=True, exist_ok=True)

    tex_res = [10, 20]
    logits = torch.randn(3, tex_res[0], tex_res[1], dtype=torch.float32)
    torch.save(
        {"logits": logits, "tex_res": tex_res, "ring_px": 8, "canvas_pad": PAD_PX},
        opt_run / "texture.pt",
    )

    # Mask: 262 x 512
    ring_mask_img = np.zeros((H_PRIME, W_PRIME), dtype=np.uint8)
    ring_mask_img[30:230, 40:470] = 255
    cv2.imwrite(str(opt_run / "mask.png"), ring_mask_img)

    write_json(
        opt_run / "split.json",
        {
            "train": [str(tmp_path / "frames" / "owner" / "s1" / "0000.jpg")],
            "eval": [str(session_dir / "0000.jpg")],
        },
    )

    write_json(
        opt_run / "attack.json",
        {
            "adaptive": True,
            "ring_px": 8,
            "canvas_pad": PAD_PX,
            "victim": "LFW_victim",
            "attacker_id": "owner",
            "purifier_ckpt": str(purifier_ckpt_path),
            "tex_res": tex_res,
        },
    )

    # 6. Run evaluation (using checkpoint checkpoints/method3_best.pth)
    eval_out = tmp_path / "eval_out"
    eval_run_dir = run_eval_adaptive(
        run_dir=opt_run,
        eval_session_name="s6_vary",
        attacker_frames_dir=tmp_path / "frames" / "owner",
        gallery_dir=gal_dir,
        checkpoint_path=Path("checkpoints/method3_best.pth"),
        purifier_ckpt_path=purifier_ckpt_path,
        consistency_threshold=0.4253,
        band_template_path=band_tmpl_path,
        liveness_enabled=False,
        device="cpu",
        output_dir=eval_out,
    )

    assert eval_run_dir == eval_out
    assert eval_run_dir.exists()
    assert (eval_run_dir / "frames.csv").exists()
    assert (eval_run_dir / "summary.json").exists()

    summary = json.loads((eval_run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ring_px"] == 8
    assert "ring_coverage_s6" in summary
    assert "mean" in summary["ring_coverage_s6"]
    assert "consistency_threshold" in summary
    assert "results" in summary

    results = summary["results"]
    for tier in ("D", "control_random"):
        for mode in ("none", "gray", "purifier"):
            key = f"{tier}/{mode}"
            assert key in results, f"Missing key {key} in results"
            assert "rate" in results[key]
            assert "wilson95" in results[key]
            if mode == "purifier":
                assert "success_full_rate" in results[key]
                assert "flag_rate" in results[key]


def test_adaptive_optimize_purifier_branch_gallery(tmp_path: Path, monkeypatch):
    """Test A (plan_adaptive §7): cmd_optimize builds purifier-mode gallery, saves attack.json with
    'purifier_branch_gallery': 'purifier' and 'gallery_pur_report' containing victim, in tmp_path."""
    import argparse
    from innovation.defense import adaptive as adaptive_mod

    class FakeDetector:
        def detect(self, img):
            return [10, 10, 100, 100]

    class FakeEyes:
        def eyes(self, img):
            return np.array([[40.0, 50.0], [80.0, 50.0]], dtype=np.float32)

    class FakeRecognizer:
        def __init__(self, *a, **kw):
            self.threshold = 0.5
            self.ckpt_path = Path("checkpoints/method3_best.pth")
            self.load_report = {"loaded": True}
            self.preprocess = type("Pre", (), {"size": 112, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]})()
            self._model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 112 * 112, 16))
            self._model.get_embedding = lambda x: F.normalize(self._model(x), p=2, dim=1)

        def extract_embedding(self, face):
            tensor = torch.from_numpy(face.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            if tensor.shape[-2:] != (112, 112):
                tensor = F.interpolate(tensor, size=(112, 112), mode="bilinear", align_corners=False)
            with torch.no_grad():
                return self._model.get_embedding(tensor).cpu().numpy().squeeze()

    monkeypatch.setattr(faces, "FaceDetector", FakeDetector)
    monkeypatch.setattr(faces, "EyeLandmarks", FakeEyes)
    monkeypatch.setattr(faces, "enhance", lambda crop: (crop, {"applied": False}))
    monkeypatch.setattr(adaptive_mod, "FaceRecognizer", FakeRecognizer)

    # 1. Band template
    band_tmpl = np.zeros((150, 400), dtype=np.uint8)
    band_tmpl[40:110, 80:320] = 255
    band_tmpl_path = tmp_path / "band_template.png"
    cv2.imwrite(str(band_tmpl_path), band_tmpl)

    # 2. Attacker frames (need >= 4 usable frames)
    frames_dir = tmp_path / "frames" / "owner" / "s1_day1"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(5):
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        img[150:350, 230:430] = 160 + idx * 5
        cv2.imwrite(str(frames_dir / f"{idx:04d}.jpg"), img)
    with open(frames_dir / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "client_ms", "bytes"])
        for idx in range(5):
            w.writerow([idx, idx * 200.0, 1000])

    # 3. Gallery (owner and LFW_victim)
    gal_dir = tmp_path / "gallery"
    for person in ["owner", "LFW_victim"]:
        p_dir = gal_dir / person
        p_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            img = np.full((120, 120, 3), 130 if person == "owner" else 90, dtype=np.uint8)
            cv2.imwrite(str(p_dir / f"{idx}.jpg"), img)

    # 4. Small purifier model
    purifier = PurifierUNet()
    purifier_ckpt_path = tmp_path / "purifier_best.pt"
    torch.save(
        {
            "model_state_dict": purifier.state_dict(),
            "best_epoch": 1,
            "best_val_cos": 0.85,
        },
        purifier_ckpt_path,
    )

    # 5. Output directory inside tmp_path (NOT in results/innovation/)
    opt_run_dir = tmp_path / "adaptive_opt_run"
    opt_run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(adaptive_mod, "new_run", lambda *a, **kw: opt_run_dir)

    args = argparse.Namespace(
        ring_px=8,
        purifier_ckpt=str(purifier_ckpt_path),
        attacker_frames=str(tmp_path / "frames" / "owner"),
        gallery=str(gal_dir),
        attacker_id="owner",
        victim="LFW_victim",
        checkpoint="checkpoints/method3_best.pth",
        band_template=str(band_tmpl_path),
        chart=None,
        steps=2,
        lr=0.03,
        batch=2,
        tau=0.05,
        margin=0.05,
        lambda_tv=2.5,
        lambda_nps=0.5,
        tex_res=[10, 20],
        eval_sessions=[],
        dev_frac=0.0,
        dev_mode="random",
        no_eot=True,
        max_gallery=faces.GALLERY_PER_ID,
        device="cpu",
        seed=0,
        log_every=1,
    )

    adaptive_mod.cmd_optimize(args)

    attack_json_path = opt_run_dir / "attack.json"
    assert attack_json_path.exists()
    attack_data = json.loads(attack_json_path.read_text(encoding="utf-8"))
    assert attack_data.get("purifier_branch_gallery") == "purifier"
    assert "gallery_pur_report" in attack_data
    assert "LFW_victim" in attack_data["gallery_pur_report"]
    assert "owner" in attack_data["gallery_pur_report"]


def test_adaptive_optimize_gallery_purifier_matches_eval(tmp_path: Path):
    """Test B (plan_adaptive §7): Target embedding in E_pur matches build_gallery_for_mode(mode='purifier')
    identically (atol 1e-5) and differs from clean gallery embedding."""
    from innovation.defense import adaptive as adaptive_mod
    from innovation.defense.eval_purifier import build_gallery_for_mode

    class FakeDetector:
        def detect(self, img):
            return [10, 10, 100, 100]

    class FakeEyes:
        def eyes(self, img):
            return np.array([[40.0, 50.0], [80.0, 50.0]], dtype=np.float32)

    class FakeRecognizer:
        def __init__(self, *a, **kw):
            self.threshold = 0.5
            self.ckpt_path = Path("checkpoints/method3_best.pth")
            self.load_report = {"loaded": True}
            self.preprocess = type("Pre", (), {"size": 112, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]})()
            torch.manual_seed(123)
            self._weight = torch.randn(16, 3 * 112 * 112)

        def extract_embedding(self, face):
            if face.shape[:2] != (112, 112):
                face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(face.transpose(2, 0, 1)).float().reshape(-1) / 255.0
            emb = self._weight @ tensor
            norm = float(torch.norm(emb))
            return (emb / max(norm, 1e-8)).numpy()

    detector = FakeDetector()
    eyes_finder = FakeEyes()
    recognizer = FakeRecognizer()

    # 1. Band template
    band_tmpl = np.zeros((150, 400), dtype=np.uint8)
    band_tmpl[40:110, 80:320] = 255
    band_tmpl_float = band_tmpl.astype(np.float32) / 255.0

    # 2. Gallery (owner and LFW_victim)
    gal_dir = tmp_path / "gallery"
    for person in ["owner", "LFW_victim"]:
        p_dir = gal_dir / person
        p_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            img = np.full((120, 120, 3), 130 if person == "owner" else 90, dtype=np.uint8)
            cv2.imwrite(str(p_dir / f"{idx}.jpg"), img)

    # 3. Purifier model with random weights
    purifier = PurifierUNet()
    purifier.eval()

    # Call build_adaptive_optimize_galleries
    E, E_pur, g_report, g_pur_report = adaptive_mod.build_adaptive_optimize_galleries(
        gallery_dir=gal_dir,
        recognizer=recognizer,
        detector=detector,
        eyes_finder=eyes_finder,
        band_template=band_tmpl_float,
        purifier_model=purifier,
        purifier_tag="test_tag",
        device="cpu",
        max_gallery=5,
        cache_dir=tmp_path / "_cache_opt",
        attacker_id="owner",
        victim="LFW_victim",
    )

    # Call build_gallery_for_mode with mode='purifier' identically to run_eval_adaptive
    gal_eval_pur, g_eval_report = build_gallery_for_mode(
        root=gal_dir,
        recognizer=recognizer,
        detector=detector,
        eyes_finder=eyes_finder,
        mode="purifier",
        band_template=band_tmpl_float,
        purifier_model=purifier,
        purifier_tag="test_tag",
        device="cpu",
        max_per_id=5,
        cache_dir=tmp_path / "_cache_eval",
    )

    target_emb_pur = E_pur["LFW_victim"]
    target_emb_eval = torch.from_numpy(gal_eval_pur.embeddings["LFW_victim"]).float()
    target_emb_clean = E["LFW_victim"]

    # Must match eval purifier gallery embedding
    assert torch.allclose(target_emb_pur, target_emb_eval, atol=1e-5)
    # Must differ from clean gallery embedding
    assert not torch.allclose(target_emb_pur, target_emb_clean, atol=1e-3)

