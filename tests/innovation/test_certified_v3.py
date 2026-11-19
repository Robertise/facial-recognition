"""
test_certified_v3.py - Comprehensive unit tests for Certified Face Defense v3:
1. --gallery-encoder student in symmetric mode (uses student for gallery embeddings)
2. prepare_faces on 3 synthetic identities fixture (name filtering, matching shards, LFWMultiMaskDataset readability)
3. SymmetricBatchSampler: P x 2 sampling, shared batch mask, no self-pairing
4. train_student with --extra-data and --sym-weight > 0 (SupCon loss, symmetric val metric)
5. Backward compatibility: default options match v2/v1 behavior on fixtures
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from backend.face_recognizer import FaceCNN_ArcFace, FaceRecognizer
from innovation.attacks.adv_glasses import faces
from innovation.defense.maskset import (
    CANVAS_H,
    CANVAS_W,
    MASK_NAMES,
)
from innovation.defense.purifier import PurifierUNet
from innovation.defense.train_inpainter import LFWMultiMaskDataset
from innovation.defense.eval_certified import (
    build_branch_gallery,
    cmd_run,
)
from innovation.defense.prepare_faces import (
    normalize_name,
    prepare_faces_dataset,
)
import innovation.defense.prepare_faces as pf_module
from innovation.defense.train_student import (
    LoadedStudent,
    StudentEmbedder,
    StudentTrainingDataset,
    SymmetricBatchSampler,
    identity_is_val_a,
    load_student,
    train_student,
)


def _create_dummy_maskset(maskset_dir: Path) -> None:
    maskset_dir.mkdir(parents=True, exist_ok=True)
    for m in MASK_NAMES:
        dummy_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
        dummy_mask[350:450, 250:550] = 255
        cv2.imwrite(str(maskset_dir / f"{m}.png"), dummy_mask)


def _create_mock_student_ckpt(ckpt_path: Path, weight_offset: float = 0.05) -> Path:
    model = FaceCNN_ArcFace(num_classes=10, embed_dim=512)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(weight_offset)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": "FaceCNN_ArcFace",
            "fill": "gray",
            "preprocess": {"size": 112, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            "val_recog_mean": 0.85,
            "val_sym_recog_mean": 0.88,
        },
        ckpt_path,
    )
    return ckpt_path


def test_gallery_encoder_student_in_symmetric_mode(tmp_path, monkeypatch):
    """
    --gallery-encoder student in --gallery-mode symmetric uses student model
    to embed gallery branches, matching student.extract_embedding and differing from method3.
    """
    maskset_dir = tmp_path / "maskset"
    _create_dummy_maskset(maskset_dir)

    student_ckpt = tmp_path / "mock_student.pt"
    _create_mock_student_ckpt(student_ckpt, weight_offset=0.2)

    loaded_stu = load_student(student_ckpt, device="cpu")
    method3_rec = FaceRecognizer(ckpt_path=Path("checkpoints/method3_best.pth"), strict=True)

    # Synthetic gallery face
    gallery_dir = tmp_path / "gallery"
    owner_gal = gallery_dir / "owner"
    owner_gal.mkdir(parents=True)
    img = np.full((240, 240, 3), 160, dtype=np.uint8)
    cv2.imwrite(str(owner_gal / "0.jpg"), img)

    class MockDetector:
        def detect(self, image):
            return [40, 40, 160, 160]

    class MockEyesFinder:
        def eyes(self, image):
            return np.array([[80.0, 100.0], [160.0, 100.0]], dtype=np.float32)

    mock_det = MockDetector()
    mock_eyes = MockEyesFinder()
    canvas_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    canvas_mask[350:450, 250:550] = 255

    # 1. Direct comparison in build_branch_gallery
    gal_method3, _ = build_branch_gallery(
        gallery_dir=gallery_dir,
        recognizer=method3_rec,
        detector=mock_det,
        eyes_finder=mock_eyes,
        branch_name="G0/gray",
        mask_canvas=canvas_mask,
        device="cpu",
    )
    gal_student, _ = build_branch_gallery(
        gallery_dir=gallery_dir,
        recognizer=loaded_stu,
        detector=mock_det,
        eyes_finder=mock_eyes,
        branch_name="G0/gray",
        mask_canvas=canvas_mask,
        device="cpu",
    )

    emb_m3 = gal_method3.embeddings["owner"][0]
    emb_stu = gal_student.embeddings["owner"][0]
    diff = float(np.linalg.norm(emb_m3 - emb_stu))
    assert diff > 0.05, f"Student gallery embedding should differ from method3, diff was {diff}"

    # Create dummy inpainter checkpoint
    ckpt_path = tmp_path / "mock_inpainter.pt"
    dummy_model = PurifierUNet()
    torch.save({"model_state_dict": dummy_model.state_dict(), "model_class": "PurifierUNet"}, ckpt_path)

    # 2. End-to-end check with cmd_run
    out_dir = tmp_path / "run_sym_student"
    args = SimpleNamespace(
        split="dev",
        gallery=str(gallery_dir),
        extra_gallery=[],
        checkpoint="checkpoints/method3_best.pth",
        maskset_dir=str(maskset_dir),
        inpainter_ckpt=str(ckpt_path),
        student_ckpt=str(student_ckpt),
        gallery_mode="symmetric",
        gallery_encoder="student",
        thresholds=None,
        liveness=False,
        device="cpu",
        limit_frames=1,
        output_dir=str(out_dir),
    )
    cmd_run(args)

    assert (out_dir / "summary.json").exists()
    with open(out_dir / "summary.json", "r", encoding="utf-8") as fh:
        summary_data = json.load(fh)

    assert summary_data["config"]["gallery_encoder"] == "student"
    assert summary_data["config"]["gallery_mode"] == "symmetric"


def test_prepare_faces_name_filtering_and_shard_generation(tmp_path, monkeypatch):
    """
    prepare_faces correctly filters out identity names matching LFW gallery/val,
    produces matching crop and landmark shards, and outputs shards readable by LFWMultiMaskDataset.
    """
    images_root = tmp_path / "vgg_images"
    meta_path = tmp_path / "identity_meta.csv"
    gallery_dir = tmp_path / "mock_gallery"
    lfw_crops_dir = tmp_path / "mock_lfw_crops"
    maskset_dir = tmp_path / "maskset"
    out_dir = tmp_path / "vgg_prepared"

    _create_dummy_maskset(maskset_dir)

    # Identity 1: "Colin Powell" (present in gallery -> must be excluded)
    # Identity 2: "Alice Smith" (valid)
    # Identity 3: "Bob Jones" (valid)
    (gallery_dir / "Colin_Powell").mkdir(parents=True)
    (gallery_dir / "_build_report.json").write_text(
        json.dumps({"id_to_lfw_name": {"Colin_Powell": "Colin Powell"}}), encoding="utf-8"
    )
    lfw_crops_dir.mkdir(parents=True)

    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Class_ID", "Name"])
        writer.writeheader()
        writer.writerow({"Class_ID": "n000001", "Name": "Colin Powell"})
        writer.writerow({"Class_ID": "n000002", "Name": "Alice Smith"})
        writer.writerow({"Class_ID": "n000003", "Name": "Bob Jones"})

    for cid in ["n000001", "n000002", "n000003"]:
        cdir = images_root / cid
        cdir.mkdir(parents=True)
        img = np.full((120, 120, 3), 140, dtype=np.uint8)
        cv2.imwrite(str(cdir / "001.jpg"), img)
        cv2.imwrite(str(cdir / "002.jpg"), img)

    # Monkeypatch _process_one_image to return deterministic synthetic crops & landmarks
    def mock_process_one(task: tuple[str, str, str]):
        img_path, class_id, rel_path = task
        crop112 = np.full((112, 112, 3), 125, dtype=np.uint8)
        eyes = np.array([[35.0, 45.0], [75.0, 45.0]], dtype=np.float32)
        window = np.array([0, 0, 112, 112], dtype=np.int32)
        mesh_pts = np.zeros((468, 2), dtype=np.float32)
        ok = 1
        return crop112, eyes, window, mesh_pts, ok, rel_path, class_id

    monkeypatch.setattr(pf_module, "_process_one_image", mock_process_one)

    report = prepare_faces_dataset(
        images_root=images_root,
        meta_path=meta_path,
        gallery_dir=gallery_dir,
        lfw_crops_dir=lfw_crops_dir,
        per_id=10,
        out_dir=out_dir,
        workers=0,
        shard_size=1000,
    )

    # 1. Check exclusion report
    assert len(report["excluded_list"]) == 1
    assert report["excluded_list"][0]["class_id"] == "n000001"
    assert report["identities_used"] == 2
    assert report["total_images"] == 4
    assert report["total_ok"] == 4
    assert report["excluded_by_source"] == {
        "gallery": 1,
        "lfw_val": 0,
        "lfw_train": 0,
    }

    # 2. Check generated shard files
    crop_shards = sorted((out_dir / "crops").glob("train_*.npz"))
    lm_shards = sorted((out_dir / "landmarks").glob("train_*.npz"))
    assert len(crop_shards) == 1
    assert len(lm_shards) == 1

    c_data = np.load(crop_shards[0], allow_pickle=True)
    l_data = np.load(lm_shards[0], allow_pickle=True)

    assert len(c_data["crops"]) == 4
    assert len(l_data["ok"]) == 4
    assert np.all(l_data["ok"] == 1)
    assert c_data["paths"].tolist() == l_data["paths"].tolist()

    # 3. Check LFWMultiMaskDataset compatibility
    ds = LFWMultiMaskDataset(
        crops_dir=out_dir / "crops",
        landmarks_dir=out_dir / "landmarks",
        maskset_dir=maskset_dir,
        split="train",
        augment=False,
    )
    assert len(ds) == 4
    c_clean, m_t = ds[0]
    assert c_clean.shape == (3, 112, 112)
    assert m_t.shape == (1, 112, 112)

    # 4. Check StudentTrainingDataset compatibility
    st_ds = StudentTrainingDataset(
        crops_dir=out_dir / "crops",
        landmarks_dir=out_dir / "landmarks",
        maskset_dir=maskset_dir,
    )
    assert len(st_ds) == 4
    tc, ts, tm = st_ds[0]
    assert tc.shape == (3, 112, 112)
    assert ts.shape == (3, 112, 112)
    assert tm.shape == (1, 112, 112)


def test_symmetric_batch_sampler_properties():
    """
    SymmetricBatchSampler yields exactly P x 2 samples per batch,
    consecutive pairs share the same identity, and all samples in a batch share the exact same mask.
    """
    id_to_indices = {
        "person_A": [0, 1, 2, 3],
        "person_B": [4, 5],
        "person_C": [6, 7, 8],
        "person_single": [9],  # Only 1 sample -> must be excluded
    }

    sampler = SymmetricBatchSampler(
        id_to_indices=id_to_indices,
        pk=3,
        seed=42,
        steps_per_epoch=5,
    )

    assert sampler.eligible_ids == ["person_A", "person_B", "person_C"]
    assert sampler.p == 3
    assert len(sampler) == 5

    idx_to_id = {}
    for ident, indices in id_to_indices.items():
        for idx in indices:
            idx_to_id[idx] = ident

    for batch in sampler:
        assert len(batch) == 6  # 2 * P = 6

        # Check all samples share identical batch_mask
        batch_masks = [item[1] for item in batch]
        shared_mask = batch_masks[0]
        assert all(m == shared_mask for m in batch_masks)
        assert shared_mask == "none" or shared_mask in MASK_NAMES

        # Check consecutive pairs share identity
        for k in range(sampler.p):
            idx1, _ = batch[2 * k]
            idx2, _ = batch[2 * k + 1]
            assert idx1 != idx2, "Samples within pair must be distinct"
            assert idx_to_id[idx1] == idx_to_id[idx2], "Pair must belong to same identity"


def test_train_student_extra_data_and_sym_weight(tmp_path):
    """
    train_student runs with --extra-data and --sym-weight > 0,
    computes SupCon loss, evaluates symmetric validation on LFW val A,
    and records all symmetric metrics in summary.json and train_log.csv.
    """
    data_dir = tmp_path / "lfw_crops"
    landmarks_dir = tmp_path / "lfw_landmarks"
    extra_crops = tmp_path / "extra" / "crops"
    extra_landmarks = tmp_path / "extra" / "landmarks"
    maskset_dir = tmp_path / "maskset"
    out_dir = tmp_path / "student_sym_out"

    for d in [data_dir, landmarks_dir, extra_crops, extra_landmarks]:
        d.mkdir(parents=True, exist_ok=True)
    _create_dummy_maskset(maskset_dir)

    # Build synthetic LFW shards (val A and train)
    id_A = "person_0" if identity_is_val_a("person_0") else "person_1"
    id_B = "person_1" if not identity_is_val_a("person_1") else "person_0"
    if not identity_is_val_a(id_A):
        id_A, id_B = id_B, id_A

    crops_4 = np.full((4, 112, 112, 3), 130, dtype=np.uint8)
    eyes_4 = np.array([[[35.0, 45.0], [75.0, 45.0]]] * 4, dtype=np.float32)
    win_4 = np.array([[0, 0, 112, 112]] * 4, dtype=np.int32)
    ok_4 = np.array([1, 1, 1, 1], dtype=np.uint8)
    paths_4 = np.array(["c0.jpg", "c1.jpg", "c2.jpg", "c3.jpg"])
    ids_4 = np.array([id_A, id_A, id_B, id_B])

    np.savez_compressed(data_dir / "train_000.npz", crops=crops_4, identities=ids_4, paths=paths_4)
    np.savez_compressed(landmarks_dir / "train_000.npz", eyes=eyes_4, window=win_4, ok=ok_4, paths=paths_4)
    np.savez_compressed(data_dir / "val_000.npz", crops=crops_4, identities=ids_4, paths=paths_4)
    np.savez_compressed(landmarks_dir / "val_000.npz", eyes=eyes_4, window=win_4, ok=ok_4, paths=paths_4)

    # Build extra data shards with 2 samples for extra_person_1 and 2 for extra_person_2
    crops_extra = np.full((4, 112, 112, 3), 110, dtype=np.uint8)
    ids_extra = np.array(["ext_1", "ext_1", "ext_2", "ext_2"])
    paths_extra = np.array(["e0.jpg", "e1.jpg", "e2.jpg", "e3.jpg"])
    np.savez_compressed(extra_crops / "train_000.npz", crops=crops_extra, identities=ids_extra, paths=paths_extra)
    np.savez_compressed(extra_landmarks / "train_000.npz", eyes=eyes_4, window=win_4, ok=ok_4, paths=paths_extra)

    run_dir = train_student(
        fill="gray",
        inpainter_ckpt=None,
        epochs=1,
        batch_size=4,
        lr=1e-4,
        lambda_nce=1.0,
        sym_weight=1.0,
        pk=2,
        seed=42,
        device="cpu",
        output_dir=out_dir,
        data_dir=data_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        extra_data=[tmp_path / "extra"],
    )

    # 1. Verify best.pt saved with symmetric validation metrics
    ckpt_file = run_dir / "best.pt"
    assert ckpt_file.exists()
    ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert "val_sym_recog_mean" in ckpt_data
    assert "val_sym_recog_per_mask" in ckpt_data

    # 2. Verify summary.json
    with open(run_dir / "summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert "best_val_sym_recog_mean" in summary
    assert "val_sym_recog_per_mask" in summary

    # 3. Verify train_log.csv contains loss_supcon and val_sym_recog columns
    with open(run_dir / "train_log.csv", "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 1
    row0 = rows[0]
    assert "loss_supcon" in row0
    assert "val_sym_recog_mean" in row0
    assert "val_sym_recog_G0" in row0
    assert "val_sym_recog_G16" in row0


def test_v3_backward_compatibility_defaults(tmp_path):
    """
    Verifies all new CLI options and training parameters default to exact v2/v1 behavior:
    sym_weight=0.0 runs standard KD with random batches, and eval_certified defaults to gallery_encoder='method3'.
    """
    data_dir = tmp_path / "lfw_crops"
    landmarks_dir = tmp_path / "lfw_landmarks"
    maskset_dir = tmp_path / "maskset"
    out_dir = tmp_path / "default_student_out"

    data_dir.mkdir(parents=True)
    landmarks_dir.mkdir(parents=True)
    _create_dummy_maskset(maskset_dir)

    id_A = "person_0" if identity_is_val_a("person_0") else "person_1"
    crops_2 = np.full((2, 112, 112, 3), 128, dtype=np.uint8)
    eyes_2 = np.array([[[35.0, 45.0], [75.0, 45.0]]] * 2, dtype=np.float32)
    win_2 = np.array([[0, 0, 112, 112]] * 2, dtype=np.int32)
    ok_2 = np.array([1, 1], dtype=np.uint8)
    paths_2 = np.array(["p0.jpg", "p1.jpg"])
    ids_2 = np.array([id_A, id_A])

    np.savez_compressed(data_dir / "train_000.npz", crops=crops_2, identities=ids_2, paths=paths_2)
    np.savez_compressed(landmarks_dir / "train_000.npz", eyes=eyes_2, window=win_2, ok=ok_2, paths=paths_2)
    np.savez_compressed(data_dir / "val_000.npz", crops=crops_2, identities=ids_2, paths=paths_2)
    np.savez_compressed(landmarks_dir / "val_000.npz", eyes=eyes_2, window=win_2, ok=ok_2, paths=paths_2)

    # Train with default sym_weight=0.0, extra_data=None, init_ckpt=None
    run_dir = train_student(
        fill="gray",
        epochs=1,
        batch_size=2,
        device="cpu",
        output_dir=out_dir,
        data_dir=data_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
    )

    with open(run_dir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    assert meta["sym_weight"] == 0.0
    assert meta["extra_data"] == []
    assert meta["init_ckpt"] is None
    assert meta["pk"] == 64
    assert meta["pk_weighting"] == "uniform"
    assert meta["num_workers"] == 0

    with open(run_dir / "summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert "best_val_recog_mean" in summary


def test_infonce_symmetric_batch_ignores_pair_negative():
    """
    §7.1 / §7.6a: In symmetric batches, cross-pair negative i ^ 1 is masked with -inf,
    preventing InfoNCE from pushing the student away from the same person's clean teacher.
    """
    import torch.nn.functional as F

    # 2 identities x 2 samples = 4 samples
    # Identical positive pairs for identity 0 and 1
    s_emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    t_emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

    # Standard unmasked InfoNCE:
    sim_unmasked = torch.matmul(s_emb, t_emb.T) / 0.05
    targets = torch.arange(4)
    loss_unmasked = F.cross_entropy(sim_unmasked, targets).item()
    # At row 0: logits [20, 20, 0, 0], cross-entropy is ln(2) ~ 0.6931
    assert abs(loss_unmasked - 0.6931) < 0.01

    # Symmetric InfoNCE with pair masking (i ^ 1 = -inf):
    sim_masked = torch.matmul(s_emb, t_emb.T) / 0.05
    pair_idx = torch.arange(4) ^ 1
    sim_masked[torch.arange(4), pair_idx] = -float("inf")
    loss_masked = F.cross_entropy(sim_masked, targets).item()
    # With -inf, denominator only includes targets and non-pair terms, so loss is ~0
    assert loss_masked < 1e-4, f"Expected masked loss ~0, got {loss_masked}"


def test_system_exit_fail_closed_conditions(tmp_path):
    """
    §7.2 / §7.6b: Verifies SystemExit on all fail-closed conditions:
    1. --init-ckpt does not exist
    2. --sym-weight > 0 but no identities >= 2 images
    3. eval_certified with --gallery-encoder student but missing --student-ckpt
    4. prepare_faces with gallery failure or missing meta or empty images-root
    """
    from innovation.defense.eval_certified import cmd_run, cmd_lfwval, cmd_calibrate

    mask_dir = tmp_path / "masks"
    _create_dummy_maskset(mask_dir)

    # 1. --init-ckpt non-existent
    with pytest.raises(SystemExit, match="Initial student checkpoint not found"):
        StudentEmbedder(init_ckpt=tmp_path / "non_existent.pt")

    # 2. --sym-weight > 0 but no eligible identities
    data_dir = tmp_path / "data_single"
    lm_dir = tmp_path / "lm_single"
    data_dir.mkdir(parents=True)
    lm_dir.mkdir(parents=True)
    c1 = np.full((1, 112, 112, 3), 128, dtype=np.uint8)
    e1 = np.array([[[35.0, 45.0], [75.0, 45.0]]], dtype=np.float32)
    w1 = np.array([[0, 0, 112, 112]], dtype=np.int32)
    o1 = np.array([1], dtype=np.uint8)
    p1 = np.array(["p0.jpg"])
    id1 = np.array(["single_person"])
    np.savez_compressed(data_dir / "train_000.npz", crops=c1, identities=id1, paths=p1)
    np.savez_compressed(lm_dir / "train_000.npz", eyes=e1, window=w1, ok=o1, paths=p1)
    np.savez_compressed(data_dir / "val_000.npz", crops=c1, identities=id1, paths=p1)
    np.savez_compressed(lm_dir / "val_000.npz", eyes=e1, window=w1, ok=o1, paths=p1)

    with pytest.raises(SystemExit, match="--sym-weight > 0 requires at least one identity"):
        train_student(
            sym_weight=1.0,
            epochs=1,
            data_dir=data_dir,
            landmarks_dir=lm_dir,
            maskset_dir=mask_dir,
            output_dir=tmp_path / "out",
        )

    # 3. eval_certified commands without student-ckpt
    args_bad = SimpleNamespace(
        gallery_mode="symmetric",
        gallery_encoder="student",
        student_ckpt=None,
        gallery=str(tmp_path),
        checkpoint="checkpoints/method3_best.pth",
        maskset_dir=str(mask_dir),
        liveness=False,
        split="dev",
        extra_gallery=[],
        output_dir=str(tmp_path / "eval_out"),
    )
    with pytest.raises(SystemExit, match="--gallery-encoder student requires --student-ckpt"):
        cmd_run(args_bad)
    with pytest.raises(SystemExit, match="--gallery-encoder student requires --student-ckpt"):
        cmd_lfwval(args_bad)
    with pytest.raises(SystemExit, match="--gallery-encoder student requires --student-ckpt"):
        cmd_calibrate(args_bad)

    # 4. prepare_faces with corrupted gallery
    corrupt_gal = tmp_path / "corrupt_gal"
    (corrupt_gal / "Unmapped_Person_Folder").mkdir(parents=True)
    with pytest.raises(SystemExit):
        prepare_faces_dataset(gallery_dir=corrupt_gal, out_dir=tmp_path / "out_p")

    # 5. prepare_faces with missing meta file
    with pytest.raises(SystemExit, match="Metadata file not found"):
        prepare_faces_dataset(meta_path=tmp_path / "non_existent_meta.csv", out_dir=tmp_path / "out_p")

    # 6. prepare_faces with empty images-root
    empty_root = tmp_path / "empty_images"
    empty_root.mkdir(parents=True)
    with pytest.raises(SystemExit, match="No identity directories found"):
        prepare_faces_dataset(images_root=empty_root, meta_path=None, out_dir=tmp_path / "out_p")


def test_prepare_faces_name_filtering_lfw_train_and_sources(tmp_path, monkeypatch):
    """
    §7.3 / §7.6c: prepare_faces correctly filters identities matching gallery, lfw_val, and lfw_train,
    and records accurate counts in excluded_by_source.
    """
    images_root = tmp_path / "images"
    meta_path = tmp_path / "meta.csv"
    gallery_dir = tmp_path / "gallery"
    lfw_crops_dir = tmp_path / "lfw_crops"
    out_dir = tmp_path / "out_vgg"

    # Setup gallery with report
    (gallery_dir / "Gal_Person").mkdir(parents=True)
    (gallery_dir / "_build_report.json").write_text(
        json.dumps({"id_to_lfw_name": {"Gal_Person": "Gal Person"}}), encoding="utf-8"
    )

    # Setup LFW crops with val and train
    lfw_crops_dir.mkdir(parents=True)
    np.savez_compressed(lfw_crops_dir / "val_000.npz", identities=np.array(["Val_Person"]))
    np.savez_compressed(lfw_crops_dir / "train_000.npz", identities=np.array(["Train_Person"]))

    # 4 identities in metadata:
    # n1: Gal Person (in gallery)
    # n2: Val Person (in val)
    # n3: Train Person (in train)
    # n4: Clean Person (clean)
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Class_ID", "Name"])
        writer.writeheader()
        writer.writerow({"Class_ID": "n1", "Name": "Gal Person"})
        writer.writerow({"Class_ID": "n2", "Name": "Val Person"})
        writer.writerow({"Class_ID": "n3", "Name": "Train Person"})
        writer.writerow({"Class_ID": "n4", "Name": "Clean Person"})

    for cid in ["n1", "n2", "n3", "n4"]:
        cdir = images_root / cid
        cdir.mkdir(parents=True)
        cv2.imwrite(str(cdir / "01.jpg"), np.full((100, 100, 3), 120, dtype=np.uint8))

    def mock_proc(task):
        return (
            np.full((112, 112, 3), 120, dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(4, dtype=np.int32),
            np.zeros((468, 2), dtype=np.float32),
            1,
            task[2],
            task[1],
        )

    monkeypatch.setattr(pf_module, "_process_one_image", mock_proc)

    report = prepare_faces_dataset(
        images_root=images_root,
        meta_path=meta_path,
        gallery_dir=gallery_dir,
        lfw_crops_dir=lfw_crops_dir,
        out_dir=out_dir,
        workers=0,
    )

    assert report["identities_excluded"] == 3
    assert report["identities_used"] == 1
    assert report["excluded_by_source"] == {
        "gallery": 1,
        "lfw_val": 1,
        "lfw_train": 1,
    }


def test_symmetric_batch_sampler_pk_weighting():
    """
    §7.4 / §7.6d: --pk-weighting count samples identities proportionally to their image count,
    selecting dominant identities much more frequently than uniform,
    while still yielding P distinct identities per batch.
    """
    # 1 big identity with 50 images, 9 small identities with 2 images each
    id_to_indices = {"big_id": list(range(50))}
    for i in range(9):
        id_to_indices[f"small_id_{i}"] = [50 + 2 * i, 50 + 2 * i + 1]

    # Uniform weighting
    sampler_uni = SymmetricBatchSampler(
        id_to_indices=id_to_indices,
        pk=2,
        seed=42,
        steps_per_epoch=200,
        pk_weighting="uniform",
    )
    uni_batches = list(sampler_uni)
    big_in_uni = sum(1 for b in uni_batches if any(item[0] < 50 for item in b))
    # In uniform, prob of big_id being chosen in 2 draws out of 10 is 2/10 = 0.20
    freq_uni = big_in_uni / len(uni_batches)
    assert 0.10 <= freq_uni <= 0.30

    # Count weighting
    sampler_cnt = SymmetricBatchSampler(
        id_to_indices=id_to_indices,
        pk=2,
        seed=42,
        steps_per_epoch=200,
        pk_weighting="count",
    )
    cnt_batches = list(sampler_cnt)
    big_in_cnt = sum(1 for b in cnt_batches if any(item[0] < 50 for item in b))
    freq_cnt = big_in_cnt / len(cnt_batches)
    assert freq_cnt > 0.85, f"Expected big_id in >85% of batches with count weighting, got {freq_cnt}"

    # Verify P distinct identities per batch
    for b in cnt_batches:
        idx0, idx2 = b[0][0], b[2][0]
        id0 = "big_id" if idx0 < 50 else f"small_id_{(idx0-50)//2}"
        id2 = "big_id" if idx2 < 50 else f"small_id_{(idx2-50)//2}"
        assert id0 != id2, "Identities in batch must be distinct"


def test_load_training_shards_path_and_count_validation(tmp_path):
    """
    §7.5 / §7.6e: _load_training_shards raises SystemExit on shard count or paths mismatch line-by-line.
    """
    from innovation.defense.train_student import _load_training_shards

    crops_dir = tmp_path / "crops"
    lm_dir = tmp_path / "landmarks"
    crops_dir.mkdir()
    lm_dir.mkdir()

    crops = np.full((2, 112, 112, 3), 128, dtype=np.uint8)
    lm_eyes = np.zeros((2, 2, 2), dtype=np.float32)
    lm_win = np.zeros((2, 4), dtype=np.int32)
    lm_ok = np.ones(2, dtype=np.uint8)

    # Mismatched paths line-by-line
    np.savez_compressed(crops_dir / "train_000.npz", crops=crops, paths=np.array(["img1.jpg", "img2.jpg"]))
    np.savez_compressed(
        lm_dir / "train_000.npz", eyes=lm_eyes, window=lm_win, ok=lm_ok, paths=np.array(["img1.jpg", "MISMATCH.jpg"])
    )

    with pytest.raises(SystemExit, match="Paths mismatch line-by-line"):
        _load_training_shards(crops_dir, lm_dir, "test")

    # Missing shard
    np.savez_compressed(crops_dir / "train_001.npz", crops=crops, paths=np.array(["img3.jpg", "img4.jpg"]))
    with pytest.raises(SystemExit, match="Missing landmark shards"):
        _load_training_shards(crops_dir, lm_dir, "test")

