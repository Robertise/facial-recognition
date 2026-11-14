"""
train_inpainter.py - Train PurifierUNet inpainter on multi-mask set with frozen ArcFace identity loss.

Dataset: LFW crops + landmarks (where ok == 1).
Randomly samples one mask from {G0, G16, G32, G48, L, F} per training sample.
Evaluates validation on all 6 masks individually. Best checkpoint chosen by average validation
cosine similarity across all 6 masks.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.face_recognizer import FaceRecognizer
from innovation.defense.maskset import MASK_NAMES, load_maskset, mask_in_window
from innovation.defense.purifier import (
    FrozenMethod3Embedder,
    PurifierUNet,
    gray_baseline_tensor,
    purifier_loss,
)
from innovation.defense.train_purifier import shift_band_mask
from innovation.eval.runs import new_run, write_json

DEFAULT_DATA_DIR = Path("data/innovation/defense/lfw_crops")
DEFAULT_LANDMARKS_DIR = Path("data/innovation/defense/lfw_landmarks")
DEFAULT_MASKSET_DIR = Path("data/innovation/defense/maskset")
DEFAULT_CHECKPOINT = Path("checkpoints/method3_best.pth")


class LFWMultiMaskDataset(Dataset):
    """Dataset pairing LFW crops with dynamically rendered masks from the 6-mask set."""

    def __init__(
        self,
        crops_dir: Path,
        landmarks_dir: Path,
        maskset_dir: Path,
        split: str = "train",
        augment: bool = False,
        seed: int = 0,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.augment = augment
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.masks = load_maskset(maskset_dir)

        crop_shards_map = {p.name: p for p in Path(crops_dir).glob(f"{split}_*.npz")}
        lm_shards_map = {p.name: p for p in Path(landmarks_dir).glob(f"{split}_*.npz")}

        if not lm_shards_map:
            raise SystemExit(f"No landmark shards found for {split} in {landmarks_dir}")
        if not crop_shards_map:
            raise SystemExit(f"No crop shards found for {split} in {crops_dir}")

        if limit is None:
            missing_lm = set(crop_shards_map.keys()) - set(lm_shards_map.keys())
            if missing_lm:
                raise SystemExit(f"Missing landmark shards: {sorted(missing_lm)}")

        crops_list = []
        eyes_list = []
        windows_list = []

        total_loaded = 0
        for sname in sorted(lm_shards_map.keys()):
            if sname not in crop_shards_map:
                raise SystemExit(f"Missing crop shard for landmark shard {sname}")
            cs = crop_shards_map[sname]
            ls = lm_shards_map[sname]

            c_data = np.load(cs, allow_pickle=True)
            l_data = np.load(ls, allow_pickle=True)

            c_paths = c_data["paths"]
            c_crops = c_data["crops"]
            if "paths" in l_data:
                l_paths = l_data["paths"]
                if len(c_paths) != len(l_paths):
                    if limit is not None and len(l_paths) < len(c_paths) and np.array_equal(c_paths[:len(l_paths)], l_paths):
                        c_crops = c_crops[:len(l_paths)]
                        c_paths = c_paths[:len(l_paths)]
                    else:
                        raise SystemExit(
                            f"Row count mismatch for shard {cs.name}: crop {len(c_paths)} vs landmark {len(l_paths)}"
                        )
                if not np.array_equal(c_paths, l_paths):
                    raise SystemExit(f"Paths mismatch line-by-line for shard {cs.name}")
            else:
                if len(c_paths) != len(l_data["ok"]):
                    raise SystemExit(
                        f"Row count mismatch for shard {cs.name}: crop {len(c_paths)} vs landmark {len(l_data['ok'])}"
                    )
            l_eyes = l_data["eyes"]
            l_win = l_data["window"]
            l_ok = l_data["ok"]

            n = len(l_ok)
            for i in range(n):
                if l_ok[i] == 1:
                    crops_list.append(c_crops[i])
                    eyes_list.append(l_eyes[i])
                    windows_list.append(l_win[i])
                    total_loaded += 1
                    if limit is not None and total_loaded >= limit:
                        break
            if limit is not None and total_loaded >= limit:
                break

        if not crops_list:
            raise SystemExit(f"No valid images (ok == 1) found in {split} shards")

        self.crops = crops_list
        self.eyes = eyes_list
        self.windows = windows_list

        # Optional speedup: precompute 6 masks for val split
        self.precomputed_masks: Optional[list[dict[str, np.ndarray]]] = None
        if split == "val":
            self.precomputed_masks = []
            for i in range(len(self.crops)):
                win = tuple(self.windows[i])
                eyes = self.eyes[i]
                sample_masks = {}
                for m_name in MASK_NAMES:
                    m112 = mask_in_window(win, eyes, self.masks[m_name], size=112)
                    if m112 is None:
                        m112 = np.zeros((112, 112), dtype=np.uint8)
                    sample_masks[m_name] = m112
                self.precomputed_masks.append(sample_masks)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        crop_u8 = self.crops[idx]
        eyes = self.eyes[idx]
        win = tuple(self.windows[idx])

        crop = crop_u8.astype(np.float32) / 255.0
        m_name = self.rng.choice(MASK_NAMES)
        canvas_mask = self.masks[m_name]

        mask112 = mask_in_window(win, eyes, canvas_mask, size=112)
        if mask112 is None:
            mask112 = np.zeros((112, 112), dtype=np.uint8)

        if self.augment:
            # 1. Horizontal flip
            if self.rng.random() < 0.5:
                crop = np.ascontiguousarray(crop[:, ::-1, :])
                mask112 = np.ascontiguousarray(mask112[:, ::-1])

            # 2. Brightness and contrast jitter
            b_gain = float(self.rng.uniform(-0.10, 0.10))
            c_gain = float(self.rng.uniform(-0.10, 0.10))
            mean_val = float(crop.mean())
            crop = (crop - mean_val) * (1.0 + c_gain) + mean_val + b_gain
            crop = np.clip(crop, 0.0, 1.0)

            # 3. Shift mask by +-3 px
            dx = int(self.rng.integers(-3, 4))
            dy = int(self.rng.integers(-3, 4))
            if dx != 0 or dy != 0:
                mask112 = shift_band_mask(mask112, dx, dy)

        crop_t = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask112.astype(np.float32))).unsqueeze(0)
        return crop_t, mask_t


def evaluate_inpainter_val(
    model: PurifierUNet,
    val_dataset: LFWMultiMaskDataset,
    embedder: FrozenMethod3Embedder,
    device: str | torch.device,
    batch_size: int = 64,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Evaluate inpainter on validation set across all 6 masks without jitter or flips."""
    model.eval()
    cos_scores_pred: dict[str, list[float]] = {m: [] for m in MASK_NAMES}
    cos_scores_gray: dict[str, list[float]] = {m: [] for m in MASK_NAMES}

    masks = val_dataset.masks
    n_val = len(val_dataset)

    # Process in batches
    with torch.no_grad():
        for start_idx in range(0, n_val, batch_size):
            end_idx = min(n_val, start_idx + batch_size)
            batch_crops = [val_dataset.crops[i].astype(np.float32) / 255.0 for i in range(start_idx, end_idx)]
            batch_windows = [tuple(val_dataset.windows[i]) for i in range(start_idx, end_idx)]
            batch_eyes = [val_dataset.eyes[i] for i in range(start_idx, end_idx)]

            x_clean = torch.from_numpy(np.stack([c.transpose(2, 0, 1) for c in batch_crops])).float().to(device)
            emb_clean = embedder(x_clean)

            for m_name in MASK_NAMES:
                if val_dataset.precomputed_masks is not None:
                    batch_masks = [val_dataset.precomputed_masks[i][m_name] for i in range(start_idx, end_idx)]
                else:
                    canvas_mask = masks[m_name]
                    batch_masks = []
                    for win, eyes in zip(batch_windows, batch_eyes):
                        m112 = mask_in_window(win, eyes, canvas_mask, size=112)
                        if m112 is None:
                            m112 = np.zeros((112, 112), dtype=np.uint8)
                        batch_masks.append(m112)

                m_t = torch.from_numpy(np.stack(batch_masks).astype(np.float32)).unsqueeze(1).to(device)
                pred = model(x_clean, m_t)
                gray = gray_baseline_tensor(x_clean, m_t)

                emb_pred = embedder(pred)
                emb_gray = embedder(gray)

                sim_pred = F.cosine_similarity(emb_pred, emb_clean, dim=1).cpu().tolist()
                sim_gray = F.cosine_similarity(emb_gray, emb_clean, dim=1).cpu().tolist()

                cos_scores_pred[m_name].extend(sim_pred)
                cos_scores_gray[m_name].extend(sim_gray)

    val_cos_per_mask = {m: float(np.mean(cos_scores_pred[m])) for m in MASK_NAMES}
    val_gray_per_mask = {m: float(np.mean(cos_scores_gray[m])) for m in MASK_NAMES}
    val_cos_mean = float(np.mean(list(val_cos_per_mask.values())))
    return val_cos_mean, val_cos_per_mask, val_gray_per_mask


def train_inpainter(
    data_dir: Path = DEFAULT_DATA_DIR,
    landmarks_dir: Path = DEFAULT_LANDMARKS_DIR,
    maskset_dir: Path = DEFAULT_MASKSET_DIR,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta_args = {
        "data": str(data_dir),
        "landmarks": str(landmarks_dir),
        "maskset_dir": str(maskset_dir),
        "checkpoint": str(checkpoint),
        "epochs": int(epochs),
        "batch": int(batch_size),
        "lr": float(lr),
        "device": str(device),
        "seed": int(seed),
        "limit": limit,
    }

    if output_dir is not None:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "meta.json", meta_args)
    else:
        run_dir = new_run("inpainter_train", meta_args)

    train_ds = LFWMultiMaskDataset(
        crops_dir=data_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        split="train",
        augment=True,
        seed=seed,
        limit=limit,
    )
    val_ds = LFWMultiMaskDataset(
        crops_dir=data_dir,
        landmarks_dir=landmarks_dir,
        maskset_dir=maskset_dir,
        split="val",
        augment=False,
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

    model = PurifierUNet().to(device)
    rec = FaceRecognizer(ckpt_path=checkpoint, strict=True)
    embedder = FrozenMethod3Embedder(rec, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fieldnames = [
        "epoch", "loss_total", "loss_band", "loss_full", "loss_id", "val_cos_mean",
    ] + [f"val_cos_{m}" for m in MASK_NAMES] + [f"val_gray_{m}" for m in MASK_NAMES]

    log_rows = []
    best_val_cos = -1.0
    best_epoch = -1
    best_val_cos_per_mask: dict[str, float] = {}
    best_val_gray_per_mask: dict[str, float] = {}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        ep_loss, ep_band, ep_full, ep_id = [], [], [], []

        for x_clean, mask in train_loader:
            x_clean = x_clean.to(device)
            mask = mask.to(device)

            pred = model(x_clean, mask)
            loss, loss_dict = purifier_loss(pred, x_clean, mask, embedder)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss.append(float(loss.item()))
            ep_band.append(loss_dict["l1_band"])
            ep_full.append(loss_dict["l1_full"])
            ep_id.append(loss_dict["id_loss"])

        val_cos_mean, val_cos_per_mask, val_gray_per_mask = evaluate_inpainter_val(
            model=model,
            val_dataset=val_ds,
            embedder=embedder,
            device=device,
            batch_size=batch_size,
        )

        row = {
            "epoch": ep,
            "loss_total": round(float(np.mean(ep_loss)), 6),
            "loss_band": round(float(np.mean(ep_band)), 6),
            "loss_full": round(float(np.mean(ep_full)), 6),
            "loss_id": round(float(np.mean(ep_id)), 6),
            "val_cos_mean": round(val_cos_mean, 6),
        }
        for m in MASK_NAMES:
            row[f"val_cos_{m}"] = round(val_cos_per_mask[m], 6)
            row[f"val_gray_{m}"] = round(val_gray_per_mask[m], 6)
        log_rows.append(row)

        elapsed_ep = time.time() - t_ep
        masks_cos_str = " ".join(f"{m}:{val_cos_per_mask[m]:.4f}" for m in MASK_NAMES)
        print(
            f"Epoch {ep:02d}/{epochs:02d} | loss={row['loss_total']:.4f} | "
            f"val_cos={val_cos_mean:.4f} ({masks_cos_str}) | time={elapsed_ep:.1f}s",
            flush=True,
        )

        # Write train_log.csv after each epoch for real-time monitoring
        with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

        if val_cos_mean > best_val_cos:
            best_val_cos = val_cos_mean
            best_epoch = ep
            best_val_cos_per_mask = val_cos_per_mask
            best_val_gray_per_mask = val_gray_per_mask
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": ep,
                    "val_cos_mean": val_cos_mean,
                    "val_cos_per_mask": val_cos_per_mask,
                    "val_gray_per_mask": val_gray_per_mask,
                    "model_class": "PurifierUNet",
                },
                run_dir / "best.pt",
            )

    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    summary = {
        "best_epoch": best_epoch,
        "best_val_cos_mean": round(best_val_cos, 6),
        "val_cos_per_mask": {k: round(v, 6) for k, v in best_val_cos_per_mask.items()},
        "val_gray_per_mask": {k: round(v, 6) for k, v in best_val_gray_per_mask.items()},
        "total_time_seconds": round(time.time() - t0, 2),
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Train multi-mask face inpainter.")
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--landmarks", default=str(DEFAULT_LANDMARKS_DIR))
    ap.add_argument("--maskset-dir", default=str(DEFAULT_MASKSET_DIR))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    run = train_inpainter(
        data_dir=Path(args.data),
        landmarks_dir=Path(args.landmarks),
        maskset_dir=Path(args.maskset_dir),
        checkpoint=Path(args.checkpoint),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        limit=args.limit,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Inpainter training completed: {run}")


if __name__ == "__main__":
    main()
