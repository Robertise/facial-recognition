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


def _load_training_shards(
    crops_dir: Path,
    landmarks_dir: Path,
    source_prefix: str,
    limit: Optional[int] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    crops_dir = Path(crops_dir)
    landmarks_dir = Path(landmarks_dir)
    crop_shards_map = {p.name: p for p in crops_dir.glob("train_*.npz")}
    lm_shards_map = {p.name: p for p in landmarks_dir.glob("train_*.npz")}

    if not lm_shards_map:
        raise SystemExit(f"No landmark shards found for train in {landmarks_dir}")
    if not crop_shards_map:
        raise SystemExit(f"No crop shards found for train in {crops_dir}")

    if limit is None:
        missing_lm = set(crop_shards_map.keys()) - set(lm_shards_map.keys())
        if missing_lm:
            raise SystemExit(f"Missing landmark shards in {landmarks_dir}: {sorted(missing_lm)}")
        missing_crops = set(lm_shards_map.keys()) - set(crop_shards_map.keys())
        if missing_crops:
            raise SystemExit(f"Missing crop shards in {crops_dir}: {sorted(missing_crops)}")

    out_crops: list[np.ndarray] = []
    out_eyes: list[np.ndarray] = []
    out_windows: list[np.ndarray] = []
    out_ids: list[str] = []

    for sname in sorted(lm_shards_map.keys()):
        if sname not in crop_shards_map:
            raise SystemExit(f"Missing crop shard for landmark shard {sname}")
        c_data = np.load(crop_shards_map[sname], allow_pickle=True)
        l_data = np.load(lm_shards_map[sname], allow_pickle=True)

        c_crops = c_data["crops"]
        l_eyes = l_data["eyes"]
        l_win = l_data["window"]
        l_ok = l_data["ok"]
        c_ids = c_data["identities"] if "identities" in c_data else np.array([f"{i}" for i in range(len(l_ok))])

        if len(c_crops) != len(l_ok):
            raise SystemExit(
                f"Row count mismatch for shard {sname}: crop {len(c_crops)} vs landmark {len(l_ok)}"
            )

        if "paths" in c_data and "paths" in l_data:
            c_paths = c_data["paths"]
            l_paths = l_data["paths"]
            if len(c_paths) != len(l_paths):
                raise SystemExit(
                    f"Paths count mismatch for shard {sname}: crop {len(c_paths)} vs landmark {len(l_paths)}"
                )
            if not np.array_equal(c_paths, l_paths):
                raise SystemExit(f"Paths mismatch line-by-line for shard {sname}")

        for i in range(len(l_ok)):
            if l_ok[i] == 1:
                out_crops.append(c_crops[i])
                out_eyes.append(l_eyes[i])
                out_windows.append(l_win[i])
                out_ids.append(f"{source_prefix}:{c_ids[i]}")
                if limit is not None and len(out_crops) >= limit:
                    return out_crops, out_eyes, out_windows, out_ids

    return out_crops, out_eyes, out_windows, out_ids


class StudentTrainingDataset(Dataset):
    """Dataset for student training with clean crops, webcam aug, and mask selection."""

    def __init__(
        self,
        crops_dir: Path,
        landmarks_dir: Path,
        maskset_dir: Path,
        extra_data: Optional[list[Path | str]] = None,
        fill: str = "gray",
        seed: int = 0,
        limit: Optional[int] = None,
    ):
        super().__init__()
        crops, eyes, windows, identities = _load_training_shards(
            crops_dir, landmarks_dir, "lfw", limit=limit
        )

        if extra_data:
            for ed in extra_data:
                ed_path = Path(ed)
                c_dir = ed_path / "crops" if (ed_path / "crops").exists() else ed_path
                l_dir = ed_path / "landmarks" if (ed_path / "landmarks").exists() else ed_path
                rem_limit = limit - len(crops) if limit is not None else None
                if rem_limit is not None and rem_limit <= 0:
                    break
                e_crops, e_eyes, e_win, e_ids = _load_training_shards(
                    c_dir, l_dir, ed_path.name, limit=rem_limit
                )
                crops.extend(e_crops)
                eyes.extend(e_eyes)
                windows.extend(e_win)
                identities.extend(e_ids)

        if not crops:
            raise SystemExit(f"No valid images (ok == 1) found in training shards")

        self.crops = crops
        self.eyes = eyes
        self.windows = windows
        self.identities = identities
        self.fill = fill
        self.rng = np.random.default_rng(seed)
        self.masks = load_maskset(maskset_dir)

        # Build ID to indices mapping for eligible identities (>= 2 images)
        self.id_to_indices: dict[str, list[int]] = {}
        for idx, ident in enumerate(self.identities):
            self.id_to_indices.setdefault(ident, []).append(idx)
        self.eligible_ids = sorted(k for k, v in self.id_to_indices.items() if len(v) >= 2)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, item: int | tuple[int, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(item, (tuple, list)):
            idx, mask_choice = item
        else:
            idx = item
            mask_choice = None

        crop_u8 = self.crops[idx]
        eyes = self.eyes[idx]
        win = tuple(self.windows[idx])

        c_clean = crop_u8.astype(np.float32) / 255.0

        if mask_choice is not None:
            if mask_choice == "none":
                m112 = np.zeros((112, 112), dtype=np.uint8)
            else:
                canvas_mask = self.masks[mask_choice]
                m112 = mask_in_window(win, eyes, canvas_mask, size=112)
                if m112 is None:
                    m112 = np.zeros((112, 112), dtype=np.uint8)
        else:
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


class SymmetricBatchSampler:
    """Batch sampler that yields P identities x 2 samples per batch with a shared mask."""

    def __init__(
        self,
        id_to_indices: dict[str, list[int]],
        pk: int = 64,
        seed: int = 0,
        steps_per_epoch: Optional[int] = None,
        pk_weighting: str = "uniform",
    ):
        self.id_to_indices = {k: v for k, v in id_to_indices.items() if len(v) >= 2}
        self.eligible_ids = sorted(self.id_to_indices.keys())
        self.pk = max(1, pk)
        self.rng = np.random.default_rng(seed)
        self.pk_weighting = pk_weighting
        if not self.eligible_ids:
            self.p = 0
            self.steps = 0
            self.prob_weights = None
        else:
            self.p = min(self.pk, len(self.eligible_ids))
            total_samples = sum(len(self.id_to_indices[k]) for k in self.eligible_ids)
            self.steps = steps_per_epoch or max(1, total_samples // (2 * self.p))
            if self.pk_weighting == "count":
                counts = np.array([len(self.id_to_indices[k]) for k in self.eligible_ids], dtype=np.float64)
                self.prob_weights = counts / counts.sum()
            else:
                self.prob_weights = None

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        if not self.eligible_ids:
            return
        for _ in range(self.steps):
            if self.pk_weighting == "count" and self.prob_weights is not None:
                chosen_ids = self.rng.choice(self.eligible_ids, size=self.p, replace=False, p=self.prob_weights)
            else:
                chosen_ids = self.rng.choice(self.eligible_ids, size=self.p, replace=False)

            if self.rng.random() < 0.15:
                batch_mask = "none"
            else:
                batch_mask = str(self.rng.choice(MASK_NAMES))

            batch: list[tuple[int, str]] = []
            for cid in chosen_ids:
                indices = self.id_to_indices[cid]
                pair = self.rng.choice(indices, size=2, replace=False)
                batch.append((int(pair[0]), batch_mask))
                batch.append((int(pair[1]), batch_mask))
            yield batch


def _worker_init(worker_id: int) -> None:
    """Seed per-worker RNG for DataLoader multiprocessing."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.rng = np.random.default_rng(worker_info.seed % (2**32))


class StudentEmbedder(nn.Module):
    """Trainable student recognizer initialized from method3 or init_ckpt."""

    def __init__(
        self,
        recognizer: Optional[FaceRecognizer] = None,
        device: str | torch.device = "cpu",
        init_ckpt: Optional[Path | str] = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        if recognizer is None:
            recognizer = FaceRecognizer(ckpt_path=DEFAULT_CHECKPOINT)
        self.model = copy.deepcopy(recognizer._model).to(self.device)
        if init_ckpt is not None:
            p_init = Path(init_ckpt)
            if not p_init.exists():
                raise SystemExit(f"Initial student checkpoint not found: {p_init}")
            data = torch.load(str(p_init), map_location=self.device, weights_only=False)
            state_dict = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    clean_state[k[len("module."):]] = v
                else:
                    clean_state[k] = v
            self.model.load_state_dict(clean_state, strict=True)
            print(f"[StudentEmbedder] Initialized student weights from: {p_init}", flush=True)

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
    loaded.ckpt_path = p.resolve()
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

    gallery_raw: dict[str, Any] = {}
    probes: list[dict[str, Any]] = []

    for pid in eligible_ids:
        items = sorted(by_identity[pid], key=lambda x: x["path"])
        gallery_raw[pid] = {
            "crop": items[0]["crop"],
            "path": items[0]["path"],
            "eyes": items[0]["eyes"],
            "window": items[0]["window"],
        }
        for p_it in items[1:]:
            probes.append({"person": pid, **p_it})

    return gallery_raw, probes, masks


def evaluate_student_val(
    student: StudentEmbedder,
    teacher: FrozenMethod3Embedder,
    gallery_raw: dict[str, Any],
    probes: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    fill: str,
    inpainter: Optional[PurifierUNet],
    device: str | torch.device,
    batch_size: int = 128,
) -> tuple[float, float, dict[str, float], float, dict[str, float]]:
    """
    Deterministic evaluation on LFW val A:
    - Clean: Gallery clean teacher embedding of first image vs student probe under mask m.
    - Symmetric: Gallery masked with m, filled, student embedding vs student probe under mask m.
    Returns: (val_recog_mean, val_cos_st_mean, recog_per_mask, val_sym_recog_mean, sym_recog_per_mask).
    """
    student.eval()

    gal_keys = list(gallery_raw.keys())
    if not gal_keys or not probes:
        empty_masks = {m: 0.0 for m in MASK_NAMES}
        return 0.0, 0.0, empty_masks, 0.0, dict(empty_masks)

    # 1. Clean gallery crops for teacher
    gal_crops_clean = []
    for k in gal_keys:
        val_item = gallery_raw[k]
        c = val_item["crop"] if isinstance(val_item, dict) else val_item
        gal_crops_clean.append(c.astype(np.float32) / 255.0)

    with torch.no_grad():
        x_gal_clean = torch.from_numpy(np.stack([c.transpose(2, 0, 1) for c in gal_crops_clean])).float().to(device)
        gal_embs_clean = teacher(x_gal_clean)  # (N_gal, 512), normalized

    recog_per_mask: dict[str, float] = {}
    sym_recog_per_mask: dict[str, float] = {}
    cos_st_scores = []

    # 2. Evaluate probes across each mask
    for m_name in MASK_NAMES:
        canvas_mask = masks[m_name]

        # Compute symmetric gallery embeddings for this mask
        gal_crops_masked = [c.copy() for c in gal_crops_clean]
        gal_masks112 = []
        for i, k in enumerate(gal_keys):
            val_item = gallery_raw[k]
            win = val_item.get("window") if isinstance(val_item, dict) else None
            eyes = val_item.get("eyes") if isinstance(val_item, dict) else None
            if win is not None and eyes is not None:
                m112 = mask_in_window(win, eyes, canvas_mask, size=112)
                if m112 is None:
                    m112 = np.zeros((112, 112), dtype=np.uint8)
            else:
                m112 = np.zeros((112, 112), dtype=np.uint8)
            gal_masks112.append(m112)
            if fill == "gray":
                gal_crops_masked[i][m112 > 0] = 0.5

        with torch.no_grad():
            x_gal_sym = torch.from_numpy(np.stack([c.transpose(2, 0, 1) for c in gal_crops_masked])).float().to(device)
            if fill == "inpaint":
                if inpainter is None:
                    raise ValueError("Inpainter required for fill == 'inpaint'")
                m_gal_t = torch.from_numpy(np.stack(gal_masks112).astype(np.float32)).unsqueeze(1).to(device)
                x_gal_sym = inpainter(x_gal_sym, m_gal_t)
            gal_embs_sym = student(x_gal_sym)  # (N_gal, 512), normalized

        correct_cnt_clean = 0
        correct_cnt_sym = 0
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

                # Clean match (probe student vs gallery clean teacher)
                sims_clean = torch.matmul(s_emb, gal_embs_clean.T)
                top_sims_c, top_idx_c = sims_clean.max(dim=1)
                top_idx_c = top_idx_c.cpu().numpy()
                top_sims_c = top_sims_c.cpu().numpy()

                # Symmetric match (probe student vs gallery masked student)
                sims_sym = torch.matmul(s_emb, gal_embs_sym.T)
                top_sims_s, top_idx_s = sims_sym.max(dim=1)
                top_idx_s = top_idx_s.cpu().numpy()
                top_sims_s = top_sims_s.cpu().numpy()

                for j, p_it in enumerate(batch_probes):
                    matched_id_c = gal_keys[top_idx_c[j]]
                    if matched_id_c == p_it["person"] and top_sims_c[j] >= 0.5:
                        correct_cnt_clean += 1

                    matched_id_s = gal_keys[top_idx_s[j]]
                    if matched_id_s == p_it["person"] and top_sims_s[j] >= 0.5:
                        correct_cnt_sym += 1

        recog_per_mask[m_name] = round(float(correct_cnt_clean / tot_cnt), 4) if tot_cnt > 0 else 0.0
        sym_recog_per_mask[m_name] = round(float(correct_cnt_sym / tot_cnt), 4) if tot_cnt > 0 else 0.0

    val_recog_mean = float(np.mean(list(recog_per_mask.values())))
    val_sym_recog_mean = float(np.mean(list(sym_recog_per_mask.values())))
    val_cos_st_mean = float(np.mean(cos_st_scores)) if cos_st_scores else 0.0
    return val_recog_mean, val_cos_st_mean, recog_per_mask, val_sym_recog_mean, sym_recog_per_mask


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
    extra_data: Optional[list[Path | str]] = None,
    init_ckpt: Optional[Path | str] = None,
    sym_weight: float = 0.0,
    pk: int = 64,
    pk_weighting: str = "uniform",
    num_workers: int = 0,
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
        "extra_data": [str(p) for p in extra_data] if extra_data else [],
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "sym_weight": float(sym_weight),
        "pk": int(pk),
        "pk_weighting": str(pk_weighting),
        "num_workers": int(num_workers),
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

    # Initialize trainable student from method3 or init_ckpt
    student = StudentEmbedder(rec, device=device, init_ckpt=init_ckpt)

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
        extra_data=extra_data,
        fill=fill,
        seed=seed,
        limit=limit,
    )

    if sym_weight > 0:
        if not train_ds.eligible_ids:
            raise SystemExit("--sym-weight > 0 requires at least one identity with >= 2 images, but none found")
        batch_sampler = SymmetricBatchSampler(
            id_to_indices=train_ds.id_to_indices,
            pk=pk,
            seed=seed,
            pk_weighting=pk_weighting,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=_worker_init if num_workers > 0 else None,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=min(batch_size, len(train_ds)),
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=_worker_init if num_workers > 0 else None,
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
        "epoch", "loss_total", "loss_cos", "loss_nce", "loss_supcon", "lr",
        "val_recog_mean", "val_cos_st_mean", "val_sym_recog_mean",
    ] + [f"val_recog_{m}" for m in MASK_NAMES] + [f"val_sym_recog_{m}" for m in MASK_NAMES]

    log_rows: list[dict[str, Any]] = []
    best_eval_metric = -1.0
    best_val_recog = -1.0
    best_val_sym_recog = -1.0
    best_epoch = -1
    best_recog_per_mask: dict[str, float] = {}
    best_sym_recog_per_mask: dict[str, float] = {}
    best_cos_st = 0.0
    patience_cnt = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        student.train()

        ep_loss, ep_cos, ep_nce, ep_supcon = [], [], [], []

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
            if sym_weight > 0 and s_emb.size(0) >= 2 and s_emb.size(0) % 2 == 0:
                N_b = s_emb.size(0)
                pair_idx = torch.arange(N_b, device=device) ^ 1
                sim_mat[torch.arange(N_b, device=device), pair_idx] = -float("inf")
            targets = torch.arange(s_emb.size(0), device=device)
            l_nce = F.cross_entropy(sim_mat, targets)

            # Supervised symmetric contrastive loss (SupCon) with tau = 0.05
            if sym_weight > 0 and s_emb.size(0) >= 2 and s_emb.size(0) % 2 == 0:
                N_s = s_emb.size(0)
                sim_s = torch.matmul(s_emb, s_emb.T) / 0.05
                sim_s.fill_diagonal_(-1e9)
                targets_supcon = torch.empty(N_s, dtype=torch.long, device=device)
                targets_supcon[0::2] = torch.arange(1, N_s, 2, device=device)
                targets_supcon[1::2] = torch.arange(0, N_s, 2, device=device)
                l_supcon = F.cross_entropy(sim_s, targets_supcon)
            else:
                l_supcon = torch.tensor(0.0, device=device)

            loss = l_cos + lambda_nce * l_nce + sym_weight * l_supcon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss.append(float(loss.item()))
            ep_cos.append(float(l_cos.item()))
            ep_nce.append(float(l_nce.item()))
            ep_supcon.append(float(l_supcon.item()))

        scheduler.step()
        curr_lr = float(optimizer.param_groups[0]["lr"])

        # Deterministic val A evaluation (both clean teacher gallery and symmetric student gallery)
        val_recog_mean, val_cos_st_mean, recog_per_mask, val_sym_recog_mean, sym_recog_per_mask = evaluate_student_val(
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
            "loss_supcon": round(float(np.mean(ep_supcon)), 6),
            "lr": round(curr_lr, 8),
            "val_recog_mean": round(val_recog_mean, 6),
            "val_cos_st_mean": round(val_cos_st_mean, 6),
            "val_sym_recog_mean": round(val_sym_recog_mean, 6),
        }
        for m in MASK_NAMES:
            row[f"val_recog_{m}"] = round(recog_per_mask[m], 6)
        for m in MASK_NAMES:
            row[f"val_sym_recog_{m}"] = round(sym_recog_per_mask[m], 6)
        log_rows.append(row)

        elapsed_ep = time.time() - t_ep
        masks_str = " ".join(f"{m}:{recog_per_mask[m]:.3f}" for m in MASK_NAMES)
        sym_str = f" sym_recog={val_sym_recog_mean:.4f}" if sym_weight > 0 else ""
        supcon_str = f", supcon={row['loss_supcon']:.4f}" if sym_weight > 0 else ""
        print(
            f"Epoch {ep:02d}/{epochs:02d} | loss={row['loss_total']:.4f} "
            f"(cos={row['loss_cos']:.4f}, nce={row['loss_nce']:.4f}{supcon_str}) | "
            f"val_recog={val_recog_mean:.4f}{sym_str} ({masks_str}) cos_st={val_cos_st_mean:.4f} | "
            f"time={elapsed_ep:.1f}s",
            flush=True,
        )

        with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

        eval_metric = val_sym_recog_mean if sym_weight > 0 else val_recog_mean
        if eval_metric > best_eval_metric:
            best_eval_metric = eval_metric
            best_val_recog = val_recog_mean
            best_val_sym_recog = val_sym_recog_mean
            best_epoch = ep
            best_recog_per_mask = recog_per_mask
            best_sym_recog_per_mask = sym_recog_per_mask
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
                    "val_sym_recog_mean": val_sym_recog_mean,
                    "val_sym_recog_per_mask": sym_recog_per_mask,
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
        "best_val_sym_recog_mean": round(best_val_sym_recog, 6),
        "val_sym_recog_per_mask": {k: round(v, 6) for k, v in best_sym_recog_per_mask.items()},
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
    ap.add_argument(
        "--extra-data",
        action="append",
        default=None,
        help="Extra dataset directory with crops/ and landmarks/ (repeatable)",
    )
    ap.add_argument(
        "--init-ckpt",
        default=None,
        help="Path to pre-trained student checkpoint to initialize weights",
    )
    ap.add_argument(
        "--sym-weight",
        type=float,
        default=0.0,
        help="Supervised symmetric contrastive loss weight (default 0.0)",
    )
    ap.add_argument(
        "--pk",
        type=int,
        default=64,
        help="Number of identities P per batch for symmetric sampling (batch size = 2P, default 64)",
    )
    ap.add_argument(
        "--pk-weighting",
        choices=("uniform", "count"),
        default="uniform",
        help="Sampling probability weighting for identities in symmetric batches ('uniform' or 'count', default 'uniform')",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default 0)",
    )
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
        extra_data=args.extra_data,
        init_ckpt=args.init_ckpt,
        sym_weight=args.sym_weight,
        pk=args.pk,
        pk_weighting=args.pk_weighting,
        num_workers=args.num_workers,
    )
    print(f"Student training completed: {run}")


if __name__ == "__main__":
    main()
