"""
test_certified.py - Unit and regression tests for certified multi-mask face defense.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from backend.face_recognizer import FaceRecognizer
from innovation.eval.runs import write_json
from innovation.attacks.adv_glasses import faces, glasses
from innovation.defense.band import band_in_crop, load_band_template
from innovation.defense.eval_certified import (
    compute_sets_summary,
    parse_sets_spec,
    transform_branch_crop,
)
from innovation.defense.maskset import (
    BROW_TOP,
    CANVAS_EYES,
    CANVAS_H,
    CANVAS_OFF,
    CANVAS_W,
    DELTA,
    EYE_ZONE_RECTS,
    FACE_OVAL,
    L_TOP_LANDMARK,
    MASK_NAMES,
    build_maskset,
    mask_in_crop,
    mask_in_window,
    similarity_canvas,
)
from innovation.defense.purifier import PurifierUNet
from innovation.defense.train_inpainter import LFWMultiMaskDataset


def test_mask_in_window_matches_band_in_crop():
    """mask_in_window with canvas containing only band (DELTA=0) matches band_in_crop (IoU >= 0.99)."""
    tmpl = load_band_template()
    canvas_band = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    ox, oy = CANVAS_OFF
    canvas_band[oy : oy + glasses.TEMPLATE_H, ox : ox + glasses.TEMPLATE_W] = (tmpl > 0.5).astype(np.uint8) * 255

    # Simulated frontal face
    eyes = np.array([[120.0, 160.0], [200.0, 160.0]], dtype=np.float32)
    bbox = [100, 120, 120, 140]
    window = faces.crop_window(bbox, (480, 640))
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    m112 = mask_in_window(window, eyes, canvas_band, size=112)
    b112 = band_in_crop(dummy_img, bbox, eyes, band_template=tmpl, size=112)

    assert m112 is not None and b112 is not None
    inter = int(np.sum((m112 > 0) & (b112 > 0)))
    union = int(np.sum((m112 > 0) | (b112 > 0)))
    iou = float(inter / union) if union > 0 else 1.0
    assert iou >= 0.99, f"Expected IoU >= 0.99, got {iou:.4f}"


def test_end_to_end_invariance_across_masks():
    """End-to-end invariance: two crops differing only inside mask m yield identical inpaint output."""
    torch.manual_seed(42)
    np.random.seed(42)

    inpainter = PurifierUNet().eval()

    # Create synthetic canvas mask with an active region
    canvas_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    ox, oy = CANVAS_OFF
    canvas_mask[oy + 20 : oy + 120, ox + 50 : ox + 350] = 255

    eyes = np.array([[120.0, 160.0], [200.0, 160.0]], dtype=np.float32)
    bbox = [100, 120, 120, 140]
    img_shape = (480, 640, 3)

    # 1. Normal light
    base_crop = np.random.randint(50, 200, size=(112, 112, 3), dtype=np.uint8)
    mask112 = mask_in_window(faces.crop_window(bbox, img_shape), eyes, canvas_mask, size=112)
    assert mask112 is not None and np.sum(mask112 > 0) > 100

    cropA = base_crop.copy()
    cropB = base_crop.copy()
    # Randomize only inside mask
    cropA[mask112 > 0] = np.random.randint(0, 50, size=np.sum(mask112 > 0) * 3, dtype=np.uint8).reshape(-1, 3)
    cropB[mask112 > 0] = np.random.randint(200, 255, size=np.sum(mask112 > 0) * 3, dtype=np.uint8).reshape(-1, 3)

    outA, m112_A = transform_branch_crop(
        face_raw=cropA,
        img_shape=img_shape,
        bbox=bbox,
        eyes=eyes,
        mask_canvas=canvas_mask,
        fill="inpaint",
        inpainter=inpainter,
        device="cpu",
    )
    outB, m112_B = transform_branch_crop(
        face_raw=cropB,
        img_shape=img_shape,
        bbox=bbox,
        eyes=eyes,
        mask_canvas=canvas_mask,
        fill="inpaint",
        inpainter=inpainter,
        device="cpu",
    )

    diff = np.max(np.abs(outA.astype(np.float32) - outB.astype(np.float32)))
    assert diff < 1e-4, f"Normal light: expected diff < 1e-4, got max diff {diff}"

    # 2. Dark crop (where enhancer would fire)
    dark_crop = np.random.randint(5, 25, size=(112, 112, 3), dtype=np.uint8)
    darkA = dark_crop.copy()
    darkB = dark_crop.copy()
    darkA[mask112 > 0] = 0
    darkB[mask112 > 0] = 255

    outA_dark, _ = transform_branch_crop(
        face_raw=darkA,
        img_shape=img_shape,
        bbox=bbox,
        eyes=eyes,
        mask_canvas=canvas_mask,
        fill="inpaint",
        inpainter=inpainter,
        device="cpu",
    )
    outB_dark, _ = transform_branch_crop(
        face_raw=darkB,
        img_shape=img_shape,
        bbox=bbox,
        eyes=eyes,
        mask_canvas=canvas_mask,
        fill="inpaint",
        inpainter=inpainter,
        device="cpu",
    )
    diff_dark = np.max(np.abs(outA_dark.astype(np.float32) - outB_dark.astype(np.float32)))
    assert diff_dark < 1e-4, f"Dark light: expected diff < 1e-4, got max diff {diff_dark}"


def test_end_to_end_invariance_at_native_resolution():
    """Verify bit-for-bit invariance when accessories differ at native resolution within region_m."""
    torch.manual_seed(42)
    np.random.seed(42)

    inpainter = PurifierUNet().eval()

    img_shape = (720, 1280, 3)
    bbox = [200, 150, 240, 260]  # x, y, w, h -> crop shape is (260, 240, 3)
    win = faces.crop_window(bbox, img_shape)
    eyes = np.array([[260.0, 240.0], [380.0, 240.0]], dtype=np.float32)

    native_h, native_w = 260, 240
    base_native = np.random.randint(50, 200, size=(native_h, native_w, 3), dtype=np.uint8)

    band_tmpl = load_band_template()
    band_canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    ox, oy = CANVAS_OFF
    band_canvas[oy : oy + glasses.TEMPLATE_H, ox : ox + glasses.TEMPLATE_W] = (band_tmpl > 0.5).astype(np.uint8) * 255

    test_masks = {}
    for m in ("G0", "G48", "L", "F"):
        if m == "G0":
            reg = band_canvas.copy()
            k_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DELTA + 1, 2 * DELTA + 1))
            mask = cv2.dilate(band_canvas, k_mask)
        elif m == "G48":
            k_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 48 + 1, 2 * 48 + 1))
            reg = cv2.dilate(band_canvas, k_r)
            k_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (48 + DELTA) + 1, 2 * (48 + DELTA) + 1))
            mask = cv2.dilate(band_canvas, k_mask)
        elif m == "L":
            reg = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
            reg[500:750, 250:550] = 255
            k_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DELTA + 1, 2 * DELTA + 1))
            mask = cv2.dilate(reg, k_mask)
        else:  # F
            reg = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
            reg[200:320, 260:540] = 255
            k_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DELTA + 1, 2 * DELTA + 1))
            mask = cv2.dilate(reg, k_mask)
        test_masks[m] = (reg, mask)

    for m_name in ("G0", "G48", "L", "F"):
        reg_canvas, mask_canvas = test_masks[m_name]

        reg_native = mask_in_window(win, eyes, reg_canvas, size=max(native_h, native_w))
        assert reg_native is not None and np.sum(reg_native > 0) > 50
        reg_native_crop = cv2.resize(reg_native, (native_w, native_h), interpolation=cv2.INTER_NEAREST)

        # 1. Normal light test
        cropA = base_native.copy()
        cropB = base_native.copy()
        cropA[reg_native_crop > 0] = np.random.randint(0, 40, size=np.sum(reg_native_crop > 0) * 3, dtype=np.uint8).reshape(-1, 3)
        cropB[reg_native_crop > 0] = np.random.randint(210, 255, size=np.sum(reg_native_crop > 0) * 3, dtype=np.uint8).reshape(-1, 3)

        outA, _ = transform_branch_crop(
            face_raw=cropA,
            img_shape=img_shape,
            bbox=bbox,
            eyes=eyes,
            mask_canvas=mask_canvas,
            fill="inpaint",
            inpainter=inpainter,
            device="cpu",
        )
        outB, _ = transform_branch_crop(
            face_raw=cropB,
            img_shape=img_shape,
            bbox=bbox,
            eyes=eyes,
            mask_canvas=mask_canvas,
            fill="inpaint",
            inpainter=inpainter,
            device="cpu",
        )
        diff = np.max(np.abs(outA.astype(np.float32) - outB.astype(np.float32)))
        assert diff < 1e-4, f"Native normal {m_name}: diff must be < 1e-4, got {diff}"

        # 2. Dark crop test: verify enhancer actually fires (applied is True) and outputs remain identical
        dark_base = np.random.randint(5, 20, size=(native_h, native_w, 3), dtype=np.uint8)
        darkA = dark_base.copy()
        darkB = dark_base.copy()
        darkA[reg_native_crop > 0] = 0
        darkB[reg_native_crop > 0] = 255

        c112_darkA = cv2.resize(darkA, (112, 112), interpolation=cv2.INTER_LINEAR)
        mask112 = mask_in_crop(img_shape, bbox, eyes, mask_canvas, size=112)
        c112_darkA[mask112 > 0] = 128
        _, meta_enh = faces.enhance(c112_darkA)
        assert meta_enh and meta_enh.get("applied"), f"Dark crop for {m_name} must trigger enhancer applied=True"

        outA_dark, _ = transform_branch_crop(
            face_raw=darkA,
            img_shape=img_shape,
            bbox=bbox,
            eyes=eyes,
            mask_canvas=mask_canvas,
            fill="inpaint",
            inpainter=inpainter,
            device="cpu",
        )
        outB_dark, _ = transform_branch_crop(
            face_raw=darkB,
            img_shape=img_shape,
            bbox=bbox,
            eyes=eyes,
            mask_canvas=mask_canvas,
            fill="inpaint",
            inpainter=inpainter,
            device="cpu",
        )
        diff_dark = np.max(np.abs(outA_dark.astype(np.float32) - outB_dark.astype(np.float32)))
        assert diff_dark < 1e-4, f"Native dark {m_name}: diff must be < 1e-4, got {diff_dark}"


def test_consensus_and_certification_logic_hand_calculated():
    """Verify consensus, GAR, and certification metrics against hand-calculated table."""
    # Build 4 frames for set G48+L+F with fill inpaint
    # Frame 1: Genuine owner -> G48, L, F all match 'owner' -> Accepted, Certified
    # Frame 2: Genuine owner -> G48, F match 'owner', L matches '' -> Not accepted, but Certified
    # Frame 3: Attack on owner -> G48 matches 'victim', L matches '', F matches 'victim' -> Blocked
    # Frame 4: Attack on owner -> G48, L, F all match 'victim' -> Attack succeeds
    rows = [
        # Frame 1
        {"set": "owner_set", "frame": "f1", "kind": "genuine", "target_id": "owner", "branch": "G48/inpaint", "eyes_ok": 1, "match_id": "owner", "is_real": 1, "light": "normal"},
        {"set": "owner_set", "frame": "f1", "kind": "genuine", "target_id": "owner", "branch": "L/inpaint", "eyes_ok": 1, "match_id": "owner", "is_real": 1, "light": "normal"},
        {"set": "owner_set", "frame": "f1", "kind": "genuine", "target_id": "owner", "branch": "F/inpaint", "eyes_ok": 1, "match_id": "owner", "is_real": 1, "light": "normal"},
        # Frame 2
        {"set": "owner_set", "frame": "f2", "kind": "genuine", "target_id": "owner", "branch": "G48/inpaint", "eyes_ok": 1, "match_id": "owner", "is_real": 1, "light": "normal"},
        {"set": "owner_set", "frame": "f2", "kind": "genuine", "target_id": "owner", "branch": "L/inpaint", "eyes_ok": 1, "match_id": "", "is_real": 1, "light": "normal"},
        {"set": "owner_set", "frame": "f2", "kind": "genuine", "target_id": "owner", "branch": "F/inpaint", "eyes_ok": 1, "match_id": "owner", "is_real": 1, "light": "normal"},
        # Frame 3
        {"set": "att_set", "frame": "f3", "kind": "attack", "target_id": "victim", "branch": "G48/inpaint", "eyes_ok": 1, "match_id": "victim", "is_real": 1, "light": "normal"},
        {"set": "att_set", "frame": "f3", "kind": "attack", "target_id": "victim", "branch": "L/inpaint", "eyes_ok": 1, "match_id": "", "is_real": 1, "light": "normal"},
        {"set": "att_set", "frame": "f3", "kind": "attack", "target_id": "victim", "branch": "F/inpaint", "eyes_ok": 1, "match_id": "victim", "is_real": 1, "light": "normal"},
        # Frame 4
        {"set": "att_set", "frame": "f4", "kind": "attack", "target_id": "victim", "branch": "G48/inpaint", "eyes_ok": 1, "match_id": "victim", "is_real": 1, "light": "normal"},
        {"set": "att_set", "frame": "f4", "kind": "attack", "target_id": "victim", "branch": "L/inpaint", "eyes_ok": 1, "match_id": "victim", "is_real": 1, "light": "normal"},
        {"set": "att_set", "frame": "f4", "kind": "attack", "target_id": "victim", "branch": "F/inpaint", "eyes_ok": 1, "match_id": "victim", "is_real": 1, "light": "normal"},
    ]

    specs = parse_sets_spec(["G48+L+F"])
    summary = compute_sets_summary(rows, specs)

    inpaint_stats = summary["G48+L+F"]["inpaint"]
    assert inpaint_stats["genuine_total"] == 2
    assert inpaint_stats["gar"] == 0.5  # 1 out of 2 accepted
    assert inpaint_stats["certified_rate"] == 1.0  # both f1 and f2 are certified
    assert inpaint_stats["genuine_sets"]["owner_set"]["gar"] == 0.5
    assert inpaint_stats["genuine_sets"]["owner_set"]["certified_rate"] == 1.0
    assert inpaint_stats["attacks"]["att_set"]["total"] == 2
    assert inpaint_stats["attacks"]["att_set"]["success"] == 1  # only f4 succeeded
    assert inpaint_stats["attacks"]["att_set"]["rate"] == 0.5


def test_select_priority_order_on_mock_runs(tmp_path):
    """Test priority order in select logic under 4 scenarios."""
    from innovation.defense.eval_certified import cmd_summarize
    from types import SimpleNamespace

    # Helper to create mock run
    def make_mock_run(run_dir: Path, set_gars: dict[str, float], none_gar: float):
        run_dir.mkdir(parents=True, exist_ok=True)
        # write frames.csv with none branch
        rows = [
            {"set": "dev", "frame": f"f{i}", "kind": "genuine", "target_id": "target",
             "branch": "none", "eyes_ok": 1, "match_id": "target" if i < int(none_gar * 100) else "",
             "is_real": 1, "light": "normal"}
            for i in range(100)
        ]
        with open(run_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        summary_sets = {}
        for s, g in set_gars.items():
            summary_sets[s] = {"inpaint": {"gar": g, "genuine_total": 100}}
        write_json(run_dir / "summary.json", {"sets": summary_sets})

    # Scenario 1: Full triplet G48+L+F passes -> selects G48+L+F
    run_owner = tmp_path / "scen1_owner"
    run_lfw = tmp_path / "scen1_lfw"
    make_mock_run(run_owner, {"G48+L+F": 0.95, "G48+L": 0.96}, none_gar=1.0)
    make_mock_run(run_lfw, {"G48+L+F": 0.92, "G48+L": 0.94}, none_gar=1.0)
    args = SimpleNamespace(select=True, dev_runs=[str(run_owner), str(run_lfw)], runs=[], sets=None)
    cmd_summarize(args)
    res = json.loads((run_owner / "selection.json").read_text(encoding="utf-8"))
    assert res["selected_set"] == "G48+L+F"

    # Scenario 2: Triplet fails, doublet G48+L passes -> selects G48+L
    run_owner2 = tmp_path / "scen2_owner"
    run_lfw2 = tmp_path / "scen2_lfw"
    make_mock_run(run_owner2, {"G48+L+F": 0.80, "G48+L": 0.95, "G48+F": 0.82}, none_gar=1.0)
    make_mock_run(run_lfw2, {"G48+L+F": 0.70, "G48+L": 0.92, "G48+F": 0.81}, none_gar=1.0)
    args2 = SimpleNamespace(select=True, dev_runs=[str(run_owner2), str(run_lfw2)], runs=[], sets=None)
    cmd_summarize(args2)
    res2 = json.loads((run_owner2 / "selection.json").read_text(encoding="utf-8"))
    assert res2["selected_set"] == "G48+L"

    # Scenario 3: Doublets G16+L and G16+F both pass at r=16 -> selects the one with higher min GAR
    run_owner3 = tmp_path / "scen3_owner"
    run_lfw3 = tmp_path / "scen3_lfw"
    # G16+L min GAR is min(0.92, 0.94) = 0.92
    # G16+F min GAR is min(0.95, 0.90) = 0.90 -> G16+L wins
    make_mock_run(run_owner3, {"G16+L": 0.92, "G16+F": 0.95}, none_gar=1.0)
    make_mock_run(run_lfw3, {"G16+L": 0.94, "G16+F": 0.90}, none_gar=1.0)
    args3 = SimpleNamespace(select=True, dev_runs=[str(run_owner3), str(run_lfw3)], runs=[], sets=None)
    cmd_summarize(args3)
    res3 = json.loads((run_owner3 / "selection.json").read_text(encoding="utf-8"))
    assert res3["selected_set"] == "G16+L"

    # Scenario 4: No candidate set passes -> selects None
    run_owner4 = tmp_path / "scen4_owner"
    run_lfw4 = tmp_path / "scen4_lfw"
    make_mock_run(run_owner4, {"G48+L+F": 0.50, "G48": 0.50}, none_gar=1.0)
    make_mock_run(run_lfw4, {"G48+L+F": 0.50, "G48": 0.50}, none_gar=1.0)
    args4 = SimpleNamespace(select=True, dev_runs=[str(run_owner4), str(run_lfw4)], runs=[], sets=None)
    cmd_summarize(args4)
    res4 = json.loads((run_owner4 / "selection.json").read_text(encoding="utf-8"))
    assert res4["selected_set"] is None


def test_dataset_train_aug_and_shift(tmp_path):
    """Test LFWMultiMaskDataset item shapes, augmentations, and shift."""
    crops_dir = tmp_path / "crops"
    landmarks_dir = tmp_path / "landmarks"
    maskset_dir = tmp_path / "maskset"
    crops_dir.mkdir()
    landmarks_dir.mkdir()
    maskset_dir.mkdir()

    # Create dummy masks
    for m in MASK_NAMES:
        dummy_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        dummy_mask[350:450, 250:550] = 255
        cv2.imwrite(str(maskset_dir / f"{m}.png"), dummy_mask)

    # Create dummy crop & landmark shards
    dummy_crops = np.random.randint(0, 255, size=(4, 112, 112, 3), dtype=np.uint8)
    dummy_eyes = np.array([[[120.0, 160.0], [200.0, 160.0]]] * 4, dtype=np.float32)
    dummy_win = np.array([[100, 120, 220, 260]] * 4, dtype=np.int32)
    dummy_ok = np.array([1, 1, 0, 1], dtype=np.uint8)  # 3 valid

    np.savez_compressed(crops_dir / "train_000.npz", crops=dummy_crops, paths=["p1", "p2", "p3", "p4"])
    np.savez_compressed(landmarks_dir / "train_000.npz", eyes=dummy_eyes, window=dummy_win, ok=dummy_ok, paths=["p1", "p2", "p3", "p4"])

    ds = LFWMultiMaskDataset(
        crops_dir=crops_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        split="train",
        augment=True,
        seed=123,
    )
    assert len(ds) == 3

    crop_t, mask_t = ds[0]
    assert crop_t.shape == (3, 112, 112)
    assert mask_t.shape == (1, 112, 112)
    assert crop_t.dtype == torch.float32
    assert mask_t.dtype == torch.float32
    assert float(crop_t.min()) >= 0.0 and float(crop_t.max()) <= 1.0


def test_eval_certified_run_smoke_fixture(tmp_path):
    """eval_certified run smoke test on minimal synthetic fixture with --output-dir in tmp_path."""
    from innovation.defense.eval_certified import cmd_run
    from types import SimpleNamespace

    # Create dummy maskset
    maskset_dir = tmp_path / "maskset"
    maskset_dir.mkdir()
    for m in MASK_NAMES:
        dummy_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        dummy_mask[350:450, 250:550] = 255
        cv2.imwrite(str(maskset_dir / f"{m}.png"), dummy_mask)

    # Create dummy inpainter checkpoint
    ckpt_path = tmp_path / "mock_inpainter.pt"
    dummy_model = PurifierUNet()
    torch.save({"model_state_dict": dummy_model.state_dict(), "model_class": "PurifierUNet"}, ckpt_path)

    # Create dummy gallery
    gallery_dir = tmp_path / "gallery"
    owner_gal = gallery_dir / "owner"
    owner_gal.mkdir(parents=True)
    # 2 small images for owner
    for i in range(2):
        img = np.full((240, 240, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(owner_gal / f"{i}.jpg"), img)

    out_dir = tmp_path / "run_out"
    args = SimpleNamespace(
        split="dev",
        gallery=str(gallery_dir),
        extra_gallery=[],
        checkpoint="checkpoints/method3_best.pth",
        maskset_dir=str(maskset_dir),
        inpainter_ckpt=str(ckpt_path),
        liveness=False,
        device="cpu",
        limit_frames=1,
        output_dir=str(out_dir),
    )
    cmd_run(args)
    assert (out_dir / "frames.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary_data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "branches" in summary_data
    assert "sets_evaluated" in summary_data


def test_maskset_eye_zone_check_and_landmark_195(tmp_path):
    """Test build_maskset eye zone check: landmark 195 below eyes passes, landmark 195 at eye level raises SystemExit."""
    landmarks_dir = tmp_path / "landmarks"
    landmarks_dir.mkdir()
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    out_dir_pass = tmp_path / "maskset_pass"
    out_dir_fail = tmp_path / "maskset_fail"

    def make_shard(path: Path, lm195_y: float):
        eyes = np.array([[[310.0, 380.0], [490.0, 380.0]]], dtype=np.float32)
        mesh = np.zeros((1, 468, 2), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
        for idx, ang in zip(FACE_OVAL, angles):
            mesh[0, idx] = [400.0 + 180.0 * np.cos(ang), 480.0 + 200.0 * np.sin(ang)]
        for idx in BROW_TOP:
            mesh[0, idx] = [400.0, 280.0]
        mesh[0, L_TOP_LANDMARK] = [400.0, lm195_y]
        ok = np.array([1], dtype=np.uint8)
        win = np.array([[100, 100, 700, 800]], dtype=np.int32)
        np.savez_compressed(path, eyes=eyes, mesh=mesh, window=win, ok=ok, paths=["dummy.jpg"])

    # 1. Landmark 195 below eyes (y = 430) -> passes eye zone check with 0 overlap
    shard_pass = landmarks_dir / "train_000.npz"
    make_shard(shard_pass, lm195_y=430.0)
    meta = build_maskset(
        landmarks_dir=landmarks_dir,
        crops_dir=crops_dir,
        owner_s1_dir=tmp_path / "nonexistent_owner",
        out_dir=out_dir_pass,
    )
    assert "eye_zone_check" in meta
    assert meta["eye_zone_check"]["pass"] is True
    assert meta["eye_zone_check"]["overlap_L_px"] == 0
    assert meta["eye_zone_check"]["overlap_F_px"] == 0
    assert meta["landmarks"]["nose_bridge"] == 195
    assert meta["landmarks"]["l_top_landmark"] == 195
    assert (out_dir_pass / "L.png").exists()
    assert (out_dir_pass / "maskset_preview.png").exists()
    prev_img = cv2.imread(str(out_dir_pass / "maskset_preview.png"))
    assert prev_img.shape == (112, 784, 3)

    # 2. Landmark 195 at eye level (y = 375) -> overlaps eye zone -> raises SystemExit
    shard_fail = landmarks_dir / "train_000.npz"
    make_shard(shard_fail, lm195_y=375.0)
    with pytest.raises(SystemExit) as excinfo:
        build_maskset(
            landmarks_dir=landmarks_dir,
            crops_dir=crops_dir,
            owner_s1_dir=tmp_path / "nonexistent_owner",
            out_dir=out_dir_fail,
        )
    assert "overlaps eye zone" in str(excinfo.value)


def test_eval_certified_lfwval_smoke_fixture(tmp_path):
    """eval_certified lfwval smoke test on minimal synthetic fixture with --output-dir in tmp_path."""
    from innovation.defense.eval_certified import cmd_lfwval
    from types import SimpleNamespace

    # Create dummy maskset
    maskset_dir = tmp_path / "maskset"
    maskset_dir.mkdir()
    for m in MASK_NAMES:
        dummy_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        dummy_mask[350:450, 250:550] = 255
        cv2.imwrite(str(maskset_dir / f"{m}.png"), dummy_mask)

    # Create dummy inpainter checkpoint
    ckpt_path = tmp_path / "mock_inpainter.pt"
    dummy_model = PurifierUNet()
    torch.save({"model_state_dict": dummy_model.state_dict(), "model_class": "PurifierUNet"}, ckpt_path)

    # Create dummy crops and landmarks with at least one identity having >= 2 valid images
    crops_dir = tmp_path / "crops"
    landmarks_dir = tmp_path / "landmarks"
    crops_dir.mkdir()
    landmarks_dir.mkdir()

    dummy_crops = np.random.randint(0, 255, size=(3, 112, 112, 3), dtype=np.uint8)
    dummy_eyes = np.array([[[120.0, 160.0], [200.0, 160.0]]] * 3, dtype=np.float32)
    dummy_win = np.array([[100, 120, 220, 260]] * 3, dtype=np.int32)
    dummy_ok = np.array([1, 1, 1], dtype=np.uint8)
    dummy_paths = ["p1", "p2", "p3"]
    dummy_ids = ["person_A", "person_A", "person_B"]

    np.savez_compressed(crops_dir / "val_000.npz", crops=dummy_crops, paths=dummy_paths, identities=dummy_ids)
    np.savez_compressed(landmarks_dir / "val_000.npz", eyes=dummy_eyes, window=dummy_win, ok=dummy_ok, paths=dummy_paths)

    out_dir = tmp_path / "lfwval_out"
    args = SimpleNamespace(
        crops_dir=crops_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        inpainter_ckpt=str(ckpt_path),
        checkpoint="checkpoints/method3_best.pth",
        device="cpu",
        limit=None,
        output_dir=str(out_dir),
    )
    cmd_lfwval(args)
    assert (out_dir / "frames.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary_data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "branches" in summary_data
    assert "sets" in summary_data



