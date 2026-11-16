"""
train_student.py - Train student recognizer via knowledge distillation on masked face crops.

Teacher: method3 (frozen, FrozenMethod3Embedder from checkpoints/method3_best.pth).
Student: initialized from method3, trained with AdamW on LFW train crops + 6 masks.
Loss: (1 - cos(s_i, t_i)) + lambda * InfoNCE(s_i; {t_j}, tau=0.05).
Validation: evaluated deterministically on LFW val part A (sha1(id) % 2 == 0).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.face_recognizer import FaceCNN_ArcFace, FaceRecognizer
from innovation.defense.maskset import MASK_NAMES, load_maskset, mask_in_window
from innovation.defense.purifier import (
    FrozenMethod3Embedder,
    PurifierUNet,
    apply_purifier,
)
from innovation.defense.train_inpainter import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    DEFAULT_LANDMARKS_DIR,
    DEFAULT_MASKSET_DIR,
    LFWMultiMaskDataset,
    load_inpainter_checkpoint,
)
from innovation.eval.runs import new_run, write_json


def identity_is_val_a(identity: str) -> bool:
    """Deterministically partition identities into A (True) or B (False)."""
    h = int(hashlib.sha1(identity.encode("utf-8")).hexdigest(), 16)
    return (h % 2 == 0)


def apply_webcam_aug(
    crop_bgr: np.ndarray,
    mask112: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    webcam-aug (applied to student crop before masking / fill):
    - Brightness / contrast +-0.1
    - Gaussian noise sigma <= 0.02
    - downscale factor in [0.5, 1.0] and upscale
    - Gaussian blur sigma in [0, 1.2]
    - JPEG q in [50, 95]
    Each transform applied with probability 0.5.
    (Horizontal flip is handled jointly for student and teacher outside this function).
    crop_bgr: float32 in [0, 1].
    mask112: uint8 in {0, 1}.
    """
    c = crop_bgr.copy()
    m = mask112.copy()

    # 1. Brightness & contrast +-0.1
    if rng.random() < 0.5:
        b_gain = float(rng.uniform(-0.10, 0.10))
        c_gain = float(rng.uniform(-0.10, 0.10))
        mean_val = float(c.mean())
        c = (c - mean_val) * (1.0 + c_gain) + mean_val + b_gain
        c = np.clip(c, 0.0, 1.0)

    # 2. Gaussian noise (sigma <= 0.02)
    if rng.random() < 0.5:
        sigma_noise = float(rng.uniform(0.0, 0.02))
        if sigma_noise > 0:
            noise = rng.normal(0.0, sigma_noise, c.shape).astype(np.float32)
            c = np.clip(c + noise, 0.0, 1.0)

    # 3. Downscale and upscale (factor in [0.5, 1.0])
    if rng.random() < 0.5:
        factor = float(rng.uniform(0.5, 1.0))
        if factor < 0.98:
            h, w = c.shape[:2]
            h_small = max(8, int(round(h * factor)))
            w_small = max(8, int(round(w * factor)))
            small = cv2.resize(c, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
            c = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4. Gaussian blur (sigma in [0, 1.2])
    if rng.random() < 0.5:
        sigma_blur = float(rng.uniform(0.0, 1.2))
        if sigma_blur > 0.05:
            c = cv2.GaussianBlur(c, (0, 0), sigmaX=sigma_blur, sigmaY=sigma_blur)

    # 5. JPEG compression (q in [50, 95])
    if rng.random() < 0.5:
        q = int(rng.integers(50, 96))
        c_u8 = (c * 255.0).clip(0, 255).astype(np.uint8)
        ret, enc = cv2.imencode(".jpg", c_u8, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ret:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                c = dec.astype(np.float32) / 255.0

    return c, m


class StudentTrainingDataset(Dataset):
    """Dataset for student training with clean crops, webcam aug, and mask selection."""

    def __init__(
        self,
        crops_dir: Path,
        landmarks_dir: Path,
        maskset_dir: Path,
        fill: str = "gray",
        seed: int = 0,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.base_ds = LFWMultiMaskDataset(
            crops_dir=crops_dir,
            landmarks_dir=landmarks_dir,
            maskset_dir=maskset_dir,
            split="train",
            augment=False,
            seed=seed,
            limit=limit,
        )
        self.fill = fill
        self.rng = np.random.default_rng(seed)
        self.masks = self.base_ds.masks

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crop_u8 = self.base_ds.crops[idx]
        eyes = self.base_ds.eyes[idx]
        win = tuple(self.base_ds.windows[idx])

        c_clean = crop_u8.astype(np.float32) / 255.0

        # Choose a mask or no-mask (15% probability unmasked)
        if self.rng.random() < 0.15:
            m112 = np.zeros((112, 112), dtype=np.uint8)
        else:
            m_name = self.rng.choice(MASK_NAMES)
            canvas_mask = self.masks[m_name]
            m112 = mask_in_window(win, eyes, canvas_mask, size=112)
            if m112 is None:
                m112 = np.zeros((112, 112), dtype=np.uint8)

        # Joint horizontal flip for BOTH teacher and student (and mask)
        if self.rng.random() < 0.5:
            c_clean = np.ascontiguousarray(c_clean[:, ::-1, :])
            m112 = np.ascontiguousarray(m112[:, ::-1])

        # Apply remaining webcam augs only to student input
        c_student, m112 = apply_webcam_aug(c_clean, m112, self.rng)

        # Apply gray fill directly if fill == 'gray'
        if self.fill == "gray":
            c_student[m112 > 0] = 0.5

        t_clean = torch.from_numpy(np.ascontiguousarray(c_clean.transpose(2, 0, 1))).float()
        t_student = torch.from_numpy(np.ascontiguousarray(c_student.transpose(2, 0, 1))).float()
        t_mask = torch.from_numpy(np.ascontiguousarray(m112.astype(np.float32))).unsqueeze(0)
        return t_clean, t_student, t_mask


class StudentEmbedder(nn.Module):
    """Trainable student recognizer initialized from method3."""

    def __init__(self, recognizer: Optional[FaceRecognizer] = None, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        if recognizer is None:
            recognizer = FaceRecognizer(ckpt_path=DEFAULT_CHECKPOINT)
        self.model = copy.deepcopy(recognizer._model).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(True)
        pre = recognizer.preprocess
        self.size = int(pre.size)
        self.register_buffer("mean", torch.tensor(pre.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(pre.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1))

    def forward(self, x_bgr_01: torch.Tensor) -> torch.Tensor:
        x = x_bgr_01
        if x.shape[-2:] != (self.size, self.size):
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        rgb = x.flip(1)
        norm = (rgb - self.mean) / self.std
        emb = self.model.get_embedding(norm)
        return F.normalize(emb, p=2, dim=1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_embedding(x)


class LoadedStudent:
    """Inference wrapper for trained student recognizer matching FaceRecognizer.extract_embedding."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        size: int = 112,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.size = int(size)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std
        transposed = np.transpose(norm, (2, 0, 1))
        t = torch.from_numpy(np.ascontiguousarray(transposed)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.get_embedding(t)
            emb = F.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().squeeze()


def load_student(ckpt_path: str | Path, device: str | torch.device = "cpu") -> LoadedStudent:
    """Load trained student model from checkpoint file."""
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {p}")
    data = torch.load(str(p), map_location=device, weights_only=False)
    fill = None
    if isinstance(data, dict) and "model_state_dict" in data:
        state_dict = data["model_state_dict"]
        pre = data.get("preprocess", {})
        size = pre.get("size", 112)
        mean = pre.get("mean", (0.5, 0.5, 0.5))
        std = pre.get("std", (0.5, 0.5, 0.5))
        fill = data.get("fill", None)
    else:
        state_dict = data
        size = 112
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    # Infer num_classes and embed_dim from state_dict
    num_classes = 4000
    embed_dim = 512
    if "head.weight" in state_dict:
        num_classes = int(state_dict["head.weight"].shape[0])
        embed_dim = int(state_dict["head.weight"].shape[1])
    elif "classifier.weight" in state_dict:
        num_classes = int(state_dict["classifier.weight"].shape[0])
        embed_dim = int(state_dict["classifier.weight"].shape[1])

    model = FaceCNN_ArcFace(num_classes=num_classes, embed_dim=embed_dim)
    model.load_state_dict(state_dict, strict=True)
    loaded = LoadedStudent(model=model, device=device, size=size, mean=mean, std=std)
    loaded.fill = fill
    loaded.num_classes = num_classes
    loaded.embed_dim = embed_dim
    return loaded


def load_val_a_probes_and_gallery(
    crops_dir: Path,
    landmarks_dir: Path,
    maskset_dir: Path,
    limit: Optional[int] = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Load LFW val A identities (>= 2 images, sha1 % 2 == 0) for validation."""
    masks = load_maskset(maskset_dir)
    val_c_map = {p.name: p for p in Path(crops_dir).glob("val_*.npz")}
    val_l_map = {p.name: p for p in Path(landmarks_dir).glob("val_*.npz")}

    by_identity: dict[str, list[dict[str, Any]]] = {}
    for sname in sorted(val_l_map.keys()):
        if sname not in val_c_map:
            continue
        c_data = np.load(val_c_map[sname], allow_pickle=True)
        l_data = np.load(val_l_map[sname], allow_pickle=True)

        c_crops = c_data["crops"]
        c_paths = c_data["paths"]
        identities = c_data["identities"]
        l_eyes = l_data["eyes"]
        l_win = l_data["window"]
        l_ok = l_data["ok"]

        for i in range(len(l_ok)):
            if l_ok[i] != 1:
                continue
            person = str(identities[i])
            if not identity_is_val_a(person):
                continue
            if person not in by_identity:
                by_identity[person] = []
            by_identity[person].append({
                "crop": c_crops[i],
                "path": str(c_paths[i]),
                "eyes": l_eyes[i],
                "window": tuple(l_win[i]),
            })

    eligible_ids = sorted(k for k, v in by_identity.items() if len(v) >= 2)
    if limit is not None:
        eligible_ids = eligible_ids[:limit]

    gallery_raw: dict[str, np.ndarray] = {}
    probes: list[dict[str, Any]] = []

    for pid in eligible_ids:
        items = sorted(by_identity[pid], key=lambda x: x["path"])
        gallery_raw[pid] = items[0]["crop"]
        for p_it in items[1:]:
            probes.append({"person": pid, **p_it})

    return gallery_raw, probes, masks


def evaluate_student_val(
    student: StudentEmbedder,
    teacher: FrozenMethod3Embedder,
    gallery_raw: dict[str, np.ndarray],
    probes: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    fill: str,
    inpainter: Optional[PurifierUNet],
    device: str | torch.device,
    batch_size: int = 128,
) -> tuple[float, float, dict[str, float]]:
    """
    Deterministic evaluation on LFW val A:
    - Gallery: clean teacher embedding of first image.
    - Probe: evaluated per mask with student. Top-1 correct and sim >= 0.5.
    Returns: (val_recog_mean, val_cos_st_mean, val_recog_per_mask).
    """
    student.eval()

    # 1. Compute clean teacher embeddings for gallery
    gal_keys = list(gallery_raw.keys())
    if not gal_keys or not probes:
        return 0.0, 0.0, {m: 0.0 for m in MASK_NAMES}

    gal_crops = [gallery_raw[k].astype(np.float32) / 255.0 for k in gal_keys]
    with torch.no_grad():
        x_gal = torch.from_numpy(np.stack([c.transpose(2, 0, 1) for c in gal_crops])).float().to(device)
        gal_embs = teacher(x_gal)  # (N_gal, 512), normalized

    recog_per_mask: dict[str, float] = {}
    cos_st_scores = []

    # 2. Evaluate probes across each mask
    for m_name in MASK_NAMES:
        canvas_mask = masks[m_name]
        correct_cnt = 0
        tot_cnt = len(probes)

        for start_idx in range(0, tot_cnt, batch_size):
            end_idx = min(tot_cnt, start_idx + batch_size)
            batch_probes = probes[start_idx:end_idx]

            batch_crops_clean = [p["crop"].astype(np.float32) / 255.0 for p in batch_probes]
            batch_windows = [p["window"] for p in batch_probes]
            batch_eyes = [p["eyes"] for p in batch_probes]

            x_clean_t = torch.from_numpy(np.stack([c.transpose(2, 0, 1) for c in batch_crops_clean])).float().to(device)

            batch_crops_masked = [c.copy() for c in batch_crops_clean]
            batch_masks112 = []
            for i, (win, eyes) in enumerate(zip(batch_windows, batch_eyes)):
                m112 = mask_in_window(win, eyes, canvas_mask, size=112)
                if m112 is None:
                    m112 = np.zeros((112, 112), dtype=np.uint8)
                batch_masks112.append(m112)
                if fill == "gray":
                    batch_crops_masked[i][m112 > 0] = 0.5

            x_student_t = torch.from_numpy(np.stack([c.transpose(2, 0, 1) for c in batch_crops_masked])).float().to(device)
            m_t = torch.from_numpy(np.stack(batch_masks112).astype(np.float32)).unsqueeze(1).to(device)

            with torch.no_grad():
                if fill == "inpaint":
                    if inpainter is None:
                        raise ValueError("Inpainter required for fill == 'inpaint'")
                    x_student_t = inpainter(x_student_t, m_t)

                t_emb = teacher(x_clean_t)
                s_emb = student(x_student_t)

                cos_st = (s_emb * t_emb).sum(dim=1).cpu().tolist()
                cos_st_scores.extend(cos_st)

                # 1:N match against gallery
                sims = torch.matmul(s_emb, gal_embs.T)  # (B_probe, N_gal)
                top_sims, top_indices = sims.max(dim=1)

                top_indices = top_indices.cpu().numpy()
                top_sims = top_sims.cpu().numpy()

                for j, p_it in enumerate(batch_probes):
                    matched_id = gal_keys[top_indices[j]]
                    if matched_id == p_it["person"] and top_sims[j] >= 0.5:
                        correct_cnt += 1

        recog_per_mask[m_name] = round(float(correct_cnt / tot_cnt), 4) if tot_cnt > 0 else 0.0

    val_recog_mean = float(np.mean(list(recog_per_mask.values())))
    val_cos_st_mean = float(np.mean(cos_st_scores)) if cos_st_scores else 0.0
    return val_recog_mean, val_cos_st_mean, recog_per_mask


def train_student(
    fill: str = "gray",
    inpainter_ckpt: Optional[Path] = None,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-4,
    lambda_nce: float = 1.0,
    patience: int = 6,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    landmarks_dir: Path = DEFAULT_LANDMARKS_DIR,
    maskset_dir: Path = DEFAULT_MASKSET_DIR,
    checkpoint: Path = DEFAULT_CHECKPOINT,
) -> Path:
    """Train student recognizer using knowledge distillation on masked face crops."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta_args = {
        "fill": str(fill),
        "inpainter_ckpt": str(inpainter_ckpt) if inpainter_ckpt else None,
        "epochs": int(epochs),
        "batch": int(batch_size),
        "lr": float(lr),
        "lambda_nce": float(lambda_nce),
        "patience": int(patience),
        "seed": int(seed),
        "device": str(device),
        "limit": limit,
    }

    if output_dir is not None:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "meta.json", meta_args)
    else:
        run_dir = new_run("student_train", meta_args)

    # Load frozen teacher recognizer
    rec = FaceRecognizer(ckpt_path=checkpoint, strict=True)
    teacher = FrozenMethod3Embedder(rec, device=device)

    # Initialize trainable student from method3
    student = StudentEmbedder(rec, device=device)

    # Inpainter if fill == 'inpaint'
    inpainter = None
    if fill == "inpaint":
        if not inpainter_ckpt or not Path(inpainter_ckpt).exists():
            raise FileNotFoundError(f"Inpainter checkpoint required for fill 'inpaint': {inpainter_ckpt}")
        inpainter = load_inpainter_checkpoint(inpainter_ckpt, device=device)
        inpainter.eval()

    train_ds = StudentTrainingDataset(
        crops_dir=data_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        fill=fill,
        seed=seed,
        limit=limit,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # Load LFW val A for deterministic evaluation
    val_gal_raw, val_probes, val_masks = load_val_a_probes_and_gallery(
        crops_dir=data_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        limit=limit,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup 1 epoch then cosine annealing
    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < 1:
            return 1.0
        progress = float(epoch_idx - 1) / float(max(1, epochs - 1))
        return float(0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    fieldnames = [
        "epoch", "loss_total", "loss_cos", "loss_nce", "lr",
        "val_recog_mean", "val_cos_st_mean",
    ] + [f"val_recog_{m}" for m in MASK_NAMES]

    log_rows: list[dict[str, Any]] = []
    best_val_recog = -1.0
    best_epoch = -1
    best_recog_per_mask: dict[str, float] = {}
    best_cos_st = 0.0
    patience_cnt = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        student.train()

        ep_loss, ep_cos, ep_nce = [], [], []

        for c_clean, c_student, m_t in train_loader:
            c_clean = c_clean.to(device)
            c_student = c_student.to(device)
            m_t = m_t.to(device)

            with torch.no_grad():
                if fill == "inpaint" and inpainter is not None:
                    c_student = inpainter(c_student, m_t)
                t_emb = teacher(c_clean)

            s_emb = student(c_student)

            # Cosine distance loss: 1 - cos(s_i, t_i)
            cos_sim = (s_emb * t_emb).sum(dim=1)
            l_cos = (1.0 - cos_sim).mean()

            # InfoNCE loss with tau = 0.05
            sim_mat = torch.matmul(s_emb, t_emb.T) / 0.05
            targets = torch.arange(s_emb.size(0), device=device)
            l_nce = F.cross_entropy(sim_mat, targets)

            loss = l_cos + lambda_nce * l_nce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss.append(float(loss.item()))
            ep_cos.append(float(l_cos.item()))
            ep_nce.append(float(l_nce.item()))

        scheduler.step()
        curr_lr = float(optimizer.param_groups[0]["lr"])

        # Deterministic val A evaluation
        val_recog_mean, val_cos_st_mean, recog_per_mask = evaluate_student_val(
            student=student,
            teacher=teacher,
            gallery_raw=val_gal_raw,
            probes=val_probes,
            masks=val_masks,
            fill=fill,
            inpainter=inpainter,
            device=device,
            batch_size=batch_size,
        )

        row = {
            "epoch": ep,
            "loss_total": round(float(np.mean(ep_loss)), 6),
            "loss_cos": round(float(np.mean(ep_cos)), 6),
            "loss_nce": round(float(np.mean(ep_nce)), 6),
            "lr": round(curr_lr, 8),
            "val_recog_mean": round(val_recog_mean, 6),
            "val_cos_st_mean": round(val_cos_st_mean, 6),
        }
        for m in MASK_NAMES:
            row[f"val_recog_{m}"] = round(recog_per_mask[m], 6)
        log_rows.append(row)

        elapsed_ep = time.time() - t_ep
        masks_str = " ".join(f"{m}:{recog_per_mask[m]:.3f}" for m in MASK_NAMES)
        print(
            f"Epoch {ep:02d}/{epochs:02d} | loss={row['loss_total']:.4f} "
            f"(cos={row['loss_cos']:.4f}, nce={row['loss_nce']:.4f}) | "
            f"val_recog={val_recog_mean:.4f} ({masks_str}) cos_st={val_cos_st_mean:.4f} | "
            f"time={elapsed_ep:.1f}s",
            flush=True,
        )

        with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

        if val_recog_mean > best_val_recog:
            best_val_recog = val_recog_mean
            best_epoch = ep
            best_recog_per_mask = recog_per_mask
            best_cos_st = val_cos_st_mean
            patience_cnt = 0
            pre = rec.preprocess
            torch.save(
                {
                    "model_state_dict": student.model.state_dict(),
                    "arch": "FaceCNN_ArcFace",
                    "fill": fill,
                    "preprocess": {
                        "size": int(pre.size),
                        "mean": tuple(float(x) for x in pre.mean),
                        "std": tuple(float(x) for x in pre.std),
                    },
                    "epoch": ep,
                    "val_recog_mean": val_recog_mean,
                    "val_recog_per_mask": recog_per_mask,
                    "val_cos_st": val_cos_st_mean,
                },
                run_dir / "best.pt",
            )
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping triggered at epoch {ep} (patience {patience})")
                break

    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    summary = {
        "best_epoch": best_epoch,
        "best_val_recog_mean": round(best_val_recog, 6),
        "val_recog_per_mask": {k: round(v, 6) for k, v in best_recog_per_mask.items()},
        "val_cos_st": round(best_cos_st, 6),
        "total_time_seconds": round(time.time() - t0, 2),
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Train student recognizer with knowledge distillation.")
    ap.add_argument("--fill", choices=("gray", "inpaint"), default="gray")
    ap.add_argument("--inpainter-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-nce", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--landmarks", default=str(DEFAULT_LANDMARKS_DIR))
    ap.add_argument("--maskset-dir", default=str(DEFAULT_MASKSET_DIR))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = ap.parse_args()

    run = train_student(
        fill=args.fill,
        inpainter_ckpt=Path(args.inpainter_ckpt) if args.inpainter_ckpt else None,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        lambda_nce=args.lambda_nce,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        limit=args.limit,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        data_dir=Path(args.data),
        landmarks_dir=Path(args.landmarks),
        maskset_dir=Path(args.maskset_dir),
        checkpoint=Path(args.checkpoint),
    )
    print(f"Student training completed: {run}")


if __name__ == "__main__":
    main()
