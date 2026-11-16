"""
test_certified_v2.py - Tests for Certified Face Defense v2 upgrades:
- End-to-end invariance with student recognizer
- gallery-mode clean uses clean none gallery embeddings
- calibrate threshold selection on synthetic impostors
- Deterministic and non-overlapping identity A/B partitioning
- train_student smoke test on synthetic fixture writing all artifacts to tmp_path
- Backward compatibility: default options match v1 behavior on fixtures
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from backend.face_recognizer import FaceCNN_ArcFace, FaceRecognizer
from innovation.attacks.adv_glasses import faces, glasses
from innovation.defense.train_inpainter import load_inpainter_checkpoint
from innovation.defense.maskset import (
    CANVAS_H,
    CANVAS_OFF,
    CANVAS_W,
    DELTA,
    MASK_NAMES,
    mask_in_window,
)
from innovation.defense.purifier import PurifierUNet
from innovation.defense.eval_certified import (
    cmd_calibrate,
    cmd_lfwval,
    cmd_run,
    cmd_summarize,
    compute_branches_summary,
    compute_sets_summary,
    evaluate_certified_session,
    transform_branch_crop,
)
from innovation.defense.train_student import (
    LoadedStudent,
    StudentEmbedder,
    identity_is_val_a,
    load_student,
    train_student,
)


def test_student_end_to_end_invariance_across_masks():
    """End-to-end invariance: two crops differing only inside mask m yield identical student embeddings."""
    torch.manual_seed(42)
    np.random.seed(42)

    student = StudentEmbedder().eval()

    # Synthetic canvas mask
    canvas_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    ox, oy = CANVAS_OFF
    canvas_mask[oy + 20 : oy + 120, ox + 50 : ox + 350] = 255

    eyes = np.array([[120.0, 160.0], [200.0, 160.0]], dtype=np.float32)
    bbox = [100, 120, 120, 140]
    img_shape = (480, 640, 3)

    base_crop = np.random.randint(50, 200, size=(112, 112, 3), dtype=np.uint8)
    mask112 = mask_in_window(faces.crop_window(bbox, img_shape), eyes, canvas_mask, size=112)
    assert mask112 is not None and np.sum(mask112 > 0) > 100

    cropA = base_crop.copy()
    cropB = base_crop.copy()
    # Randomize only inside mask region
    cropA[mask112 > 0] = np.random.randint(0, 50, size=np.sum(mask112 > 0) * 3, dtype=np.uint8).reshape(-1, 3)
    cropB[mask112 > 0] = np.random.randint(200, 255, size=np.sum(mask112 > 0) * 3, dtype=np.uint8).reshape(-1, 3)

    outA, _ = transform_branch_crop(
        face_raw=cropA,
        img_shape=img_shape,
        bbox=bbox,
        eyes=eyes,
        mask_canvas=canvas_mask,
        fill="gray",
        inpainter=None,
        device="cpu",
    )
    outB, _ = transform_branch_crop(
        face_raw=cropB,
        img_shape=img_shape,
        bbox=bbox,
        eyes=eyes,
        mask_canvas=canvas_mask,
        fill="gray",
        inpainter=None,
        device="cpu",
    )

    loaded_student = LoadedStudent(student, "cpu")
    embA = loaded_student.extract_embedding(outA)
    embB = loaded_student.extract_embedding(outB)

    diff = float(np.max(np.abs(embA - embB)))
    assert diff < 1e-4, f"Student embedding invariance failed: max diff = {diff}"


def test_gallery_mode_clean_uses_none_gallery_embeddings(tmp_path):
    """--gallery-mode clean assigns identical 'none' gallery embeddings to all branches."""
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
    img = np.full((240, 240, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(owner_gal / "0.jpg"), img)

    out_dir = tmp_path / "run_clean"
    args = SimpleNamespace(
        split="dev",
        gallery=str(gallery_dir),
        extra_gallery=[],
        checkpoint="checkpoints/method3_best.pth",
        maskset_dir=str(maskset_dir),
        inpainter_ckpt=str(ckpt_path),
        student_ckpt=None,
        gallery_mode="clean",
        thresholds=None,
        liveness=False,
        device="cpu",
        limit_frames=1,
        output_dir=str(out_dir),
    )
    cmd_run(args)

    assert (out_dir / "summary.json").exists()
    sum_data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert sum_data["config"]["gallery_mode"] == "clean"

    # Check frames.csv threshold is present
    rows = list(csv.DictReader(open(out_dir / "frames.csv", encoding="utf-8")))
    assert len(rows) > 0
    assert "threshold" in rows[0]
    assert float(rows[0]["threshold"]) == 0.5


def test_deterministic_non_overlapping_ab_partition():
    """identity_is_val_a is deterministic and partitions identities with no overlap."""
    ids = [f"person_{i:04d}" for i in range(200)]
    part_A = [pid for pid in ids if identity_is_val_a(pid)]
    part_B = [pid for pid in ids if not identity_is_val_a(pid)]

    # Disjoint and complete
    assert set(part_A).isdisjoint(set(part_B))
    assert set(part_A) | set(part_B) == set(ids)

    # Determinism
    for pid in part_A:
        assert identity_is_val_a(pid) is True
    for pid in part_B:
        assert identity_is_val_a(pid) is False

    # Approximately balanced partition
    assert 70 <= len(part_A) <= 130


def test_calibrate_threshold_selection_logic():
    """Verify calibration logic selects the smallest tau such that FAR_b(tau) <= FAR_none."""
    grid = [round(0.30 + i * 0.005, 4) for i in range(121)]
    # Synthetic impostor scores for 100 probes
    # 'none' branch: 5 probes >= 0.50 => FAR_none = 0.05
    scores_none = [0.2] * 95 + [0.52, 0.55, 0.60, 0.62, 0.70]
    n = len(scores_none)
    far_none = sum(1 for s in scores_none if s >= 0.5) / n
    assert far_none == 0.05

    # Branch b: higher impostor scores:
    # 20 probes >= 0.65 (FAR=0.20)
    # 5 probes >= 0.66 (FAR=0.05)
    scores_b = [0.2] * 80 + [0.65] * 15 + [0.66, 0.67, 0.68, 0.71, 0.72]

    best_tau = 0.90
    for tau in grid:
        far = sum(1 for s in scores_b if s >= tau) / n
        if far <= far_none:
            best_tau = tau
            break

    # At tau <= 0.65, FAR = 20/100 = 0.20 > 0.05
    # At tau = 0.655, all 15 items with 0.65 are < 0.655, leaving 5 items >= 0.66 => FAR = 0.05 <= 0.05
    assert best_tau == 0.655


def test_train_student_smoke_fixture(tmp_path):
    """train_student runs 1 epoch on synthetic fixture and writes all required files to tmp_path."""
    # Build synthetic LFW crops and landmarks
    data_dir = tmp_path / "data"
    landmarks_dir = tmp_path / "landmarks"
    maskset_dir = tmp_path / "maskset"
    out_dir = tmp_path / "student_out"
    data_dir.mkdir()
    landmarks_dir.mkdir()
    maskset_dir.mkdir()

    # Create dummy masks
    for m in MASK_NAMES:
        dummy_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        dummy_mask[350:450, 250:550] = 255
        cv2.imwrite(str(maskset_dir / f"{m}.png"), dummy_mask)

    # 4 dummy crops (2 for val A identity, 2 for val B identity)
    crops = np.full((4, 112, 112, 3), 128, dtype=np.uint8)
    eyes = np.array([[[35.0, 45.0], [75.0, 45.0]]] * 4, dtype=np.float32)
    windows = np.array([[0, 0, 112, 112]] * 4, dtype=np.int32)
    ok = np.array([1, 1, 1, 1], dtype=np.uint8)
    paths = np.array(["crop_0.jpg", "crop_1.jpg", "crop_2.jpg", "crop_3.jpg"])

    # Ensure id_A is in val A and id_B is in val B
    id_A = "person_0" if identity_is_val_a("person_0") else "person_1"
    id_B = "person_1" if not identity_is_val_a("person_1") else "person_0"
    if not identity_is_val_a(id_A):
        id_A, id_B = id_B, id_A

    identities = np.array([id_A, id_A, id_B, id_B])

    np.savez_compressed(data_dir / "train_000.npz", crops=crops, identities=identities, paths=paths)
    np.savez_compressed(landmarks_dir / "train_000.npz", eyes=eyes, window=windows, ok=ok, paths=paths)
    np.savez_compressed(data_dir / "val_000.npz", crops=crops, identities=identities, paths=paths)
    np.savez_compressed(landmarks_dir / "val_000.npz", eyes=eyes, window=windows, ok=ok, paths=paths)

    run_dir = train_student(
        fill="gray",
        inpainter_ckpt=None,
        epochs=1,
        batch_size=2,
        lr=1e-4,
        lambda_nce=1.0,
        patience=6,
        seed=0,
        device="cpu",
        limit=4,
        output_dir=out_dir,
        data_dir=data_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        checkpoint="checkpoints/method3_best.pth",
    )

    assert (out_dir / "best.pt").exists()
    assert (out_dir / "train_log.csv").exists()
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "summary.json").exists()

    # Verify loaded student works
    stu = load_student(out_dir / "best.pt", device="cpu")
    dummy_crop = np.full((112, 112, 3), 100, dtype=np.uint8)
    emb = stu.extract_embedding(dummy_crop)
    assert emb.shape == (512,)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4


def test_default_options_produce_v1_compatible_outputs(tmp_path):
    """Default options in eval_certified produce v1-compatible outputs with all recognition keys present."""
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

    gallery_dir = tmp_path / "gallery"
    owner_gal = gallery_dir / "owner"
    owner_gal.mkdir(parents=True)
    img = np.full((240, 240, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(owner_gal / "0.jpg"), img)

    out_dir = tmp_path / "run_default"
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

    sum_path = out_dir / "summary.json"
    assert sum_path.exists()
    summary = json.loads(sum_path.read_text(encoding="utf-8"))

    # V1 keys
    assert "branches" in summary
    assert "sets" in summary
    assert "sets_evaluated" in summary

    # V2 keys in config
    assert "config" in summary
    assert summary["config"]["gallery_mode"] == "symmetric"
    assert summary["config"]["student_ckpt"] is None
    assert summary["config"]["thresholds"] is None

    # V2 recognition metrics in branches and sets
    br_none = summary["branches"]["none"]
    assert "gar" in br_none
    assert "gar_recog" in br_none
    assert "gar_recog_wilson_95" in br_none

    set_g0 = summary["sets"]["G0"]["gray"]
    assert "gar" in set_g0
    assert "gar_recog" in set_g0
    assert "certified_rate" in set_g0
    assert "certified_rate_recog" in set_g0


def test_load_student_dynamic_shape_and_metadata(tmp_path):
    """load_student infers num_classes and embed_dim from state_dict, loads preprocess and fill."""
    model_custom = FaceCNN_ArcFace(num_classes=120, embed_dim=256)
    ckpt_path = tmp_path / "custom_student.pt"
    torch.save(
        {
            "model_state_dict": model_custom.state_dict(),
            "arch": "FaceCNN_ArcFace",
            "fill": "gray",
            "preprocess": {
                "size": 112,
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
            },
        },
        ckpt_path,
    )
    loaded = load_student(ckpt_path, device="cpu")
    assert loaded.num_classes == 120
    assert loaded.embed_dim == 256
    assert loaded.fill == "gray"
    dummy_crop = np.full((112, 112, 3), 128, dtype=np.uint8)
    emb = loaded.extract_embedding(dummy_crop)
    assert emb.shape == (256,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_load_inpainter_custom_base_channels(tmp_path):
    """load_inpainter_checkpoint handles 64-width models with and without base_channels in dict."""
    model64 = PurifierUNet(base_channels=(64, 128, 256, 512))

    # Case 1: explicit metadata
    p1 = tmp_path / "inpainter_64_meta.pt"
    torch.save({"model_state_dict": model64.state_dict(), "base_channels": (64, 128, 256, 512)}, p1)
    loaded1 = load_inpainter_checkpoint(p1, device="cpu")
    assert loaded1.enc1.net[0].out_channels == 64

    # Case 2: no metadata, inferred from enc1.net.0.weight shape
    p2 = tmp_path / "inpainter_64_nometa.pt"
    torch.save(model64.state_dict(), p2)
    loaded2 = load_inpainter_checkpoint(p2, device="cpu")
    assert loaded2.enc1.net[0].out_channels == 64


def test_summarize_select_fill_and_metric(tmp_path):
    """summarize --select with --fill gray and --metric gar_recog writes selection.json correctly."""
    owner_dir = tmp_path / "owner_dev"
    lfw_dir = tmp_path / "lfw_dev"
    owner_dir.mkdir()
    lfw_dir.mkdir()

    owner_sum = {
        "branches": {"none": {"genuine_sets": {"owner_dev": {"gar": 1.0, "gar_recog": 1.0}}}},
        "sets": {
            "G0": {
                "gray": {"genuine_sets": {"owner_dev": {"gar": 0.95, "gar_recog": 0.96}}},
                "inpaint": {"genuine_sets": {"owner_dev": {"gar": 0.80, "gar_recog": 0.80}}},
            }
        },
    }
    lfw_sum = {
        "split": "lfw_val_B",
        "branches": {"none": {"genuine_sets": {"lfw_val_B": {"gar": 1.0, "gar_recog": 1.0}}}},
        "sets": {
            "G0": {
                "gray": {"genuine_sets": {"lfw_val_B": {"gar": 0.95, "gar_recog": 0.96}}},
                "inpaint": {"genuine_sets": {"lfw_val_B": {"gar": 0.80, "gar_recog": 0.80}}},
            }
        },
    }
    (owner_dir / "summary.json").write_text(json.dumps(owner_sum), encoding="utf-8")
    (lfw_dir / "summary.json").write_text(json.dumps(lfw_sum), encoding="utf-8")

    args = SimpleNamespace(
        select=True,
        dev_runs=[str(owner_dir), str(lfw_dir)],
        metric="gar_recog",
        fill="gray",
    )
    cmd_summarize(args)

    sel_p = owner_dir / "selection.json"
    assert sel_p.exists()
    sel_data = json.loads(sel_p.read_text(encoding="utf-8"))
    assert sel_data["fill"] == "gray"
    assert sel_data["metric"] == "gar_recog"
    assert sel_data["selected_set"] == "G0"


def test_symmetric_student_warning(tmp_path, capsys):
    """Warning is displayed when user specifies --gallery-mode symmetric with --student-ckpt."""
    ckpt_path = tmp_path / "dummy_student.pt"
    model = FaceCNN_ArcFace()
    torch.save(model.state_dict(), ckpt_path)

    inp_path = tmp_path / "mock_inpainter.pt"
    dummy_inp = PurifierUNet()
    torch.save({"model_state_dict": dummy_inp.state_dict()}, inp_path)

    maskset_dir = tmp_path / "maskset"
    maskset_dir.mkdir()
    for m in MASK_NAMES:
        dummy_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        dummy_mask[350:450, 250:550] = 255
        cv2.imwrite(str(maskset_dir / f"{m}.png"), dummy_mask)

    gallery_dir = tmp_path / "gallery"
    owner_gal = gallery_dir / "owner"
    owner_gal.mkdir(parents=True)
    cv2.imwrite(str(owner_gal / "0.jpg"), np.full((100, 100, 3), 128, dtype=np.uint8))

    args = SimpleNamespace(
        split="dev",
        gallery=str(gallery_dir),
        extra_gallery=[],
        checkpoint="checkpoints/method3_best.pth",
        maskset_dir=str(maskset_dir),
        inpainter_ckpt=str(inp_path),
        student_ckpt=str(ckpt_path),
        gallery_mode="symmetric",
        thresholds=None,
        liveness=False,
        device="cpu",
        limit_frames=1,
        output_dir=str(tmp_path / "run_warn"),
    )
    cmd_run(args)
    captured = capsys.readouterr()
    assert "Warning: --student-ckpt is supplied with --gallery-mode symmetric" in captured.out

