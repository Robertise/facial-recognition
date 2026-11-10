"""
train_purifier.py - Train the PurifierUNet on LFW crops + band masks with frozen method3 identity loss.

CLI:
  python -m innovation.defense.train_purifier --data data/innovation/defense/lfw_crops --device cuda
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from backend.face_recognizer import FaceRecognizer
from innovation.defense.purifier import (
    FrozenMethod3Embedder,
    PurifierUNet,
    count_parameters,
    gray_baseline_tensor,
    purifier_loss,
)
from innovation.eval.runs import new_run, write_json

DEFAULT_DATA_DIR = Path("data/innovation/defense/lfw_crops")
DEFAULT_CHECKPOINT = Path("checkpoints/method3_best.pth")


def shift_band_mask(band: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a 2D binary mask by (dx, dy) pixels with zero padding at boundaries."""
    h, w = band.shape[:2]
    out = np.zeros_like(band)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = band[src_y0:src_y1, src_x0:src_x1]
    return out


class LFWCropBandDataset(Dataset):
    """Dataset loading `.npz` shards of (crops uint8 Nx112x112x3, bands uint8 Nx112x112)."""

    def __init__(self, data_dir: Path, split: str = "train", augment: bool = False, seed: int = 0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = bool(augment)
        self.seed = int(seed)

        shard_files = sorted(self.data_dir.glob(f"{split}_*.npz"))
        if not shard_files:
            raise FileNotFoundError(f"No '{split}_*.npz' shards found in {self.data_dir}")

        crops_list = []
        bands_list = []
        for sf in shard_files:
            d = np.load(sf, allow_pickle=True)
            crops_list.append(d["crops"].astype(np.uint8))
            bands_list.append((d["bands"] > 0).astype(np.uint8))

        self.crops = np.concatenate(crops_list, axis=0)
        self.bands = np.concatenate(bands_list, axis=0)

    def __len__(self) -> int:
        return int(len(self.crops))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        crop_u8 = self.crops[idx]
        band_u8 = self.bands[idx]

        crop = crop_u8.astype(np.float32) / 255.0
        band = band_u8.copy()

        if self.augment:
            rng = np.random.default_rng()
            # 1. Horizontal flip (both crop and band)
            if rng.random() < 0.5:
                crop = np.ascontiguousarray(crop[:, ::-1, :])
                band = np.ascontiguousarray(band[:, ::-1])

            # 2. Slight brightness and contrast jitter
            b_gain = float(rng.uniform(-0.10, 0.10))
            c_gain = float(rng.uniform(-0.10, 0.10))
            mean_val = float(crop.mean())
            crop = (crop - mean_val) * (1.0 + c_gain) + mean_val + b_gain
            crop = np.clip(crop, 0.0, 1.0)

            # 3. Shift band by +-3 px
            dx = int(rng.integers(-3, 4))
            dy = int(rng.integers(-3, 4))
            if dx != 0 or dy != 0:
                band = shift_band_mask(band, dx, dy)

        crop_t = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()
        band_t = torch.from_numpy(np.ascontiguousarray(band.astype(np.float32))).unsqueeze(0)
        return crop_t, band_t


def train_purifier(
    data_dir: Path = DEFAULT_DATA_DIR,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    out_dir: Optional[Path] = None,
    embedder: Optional[Any] = None,
) -> Path:
    """Train PurifierUNet and save best checkpoint by validation identity cosine similarity."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta_args = {
        "data": str(data_dir),
        "checkpoint": str(checkpoint),
        "epochs": int(epochs),
        "batch": int(batch_size),
        "lr": float(lr),
        "device": str(device),
        "seed": int(seed),
    }
    if out_dir is not None:
        run_dir = Path(out_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "meta.json", meta_args)
    else:
        run_dir = new_run("purifier_train", meta_args)

    dev = torch.device(device)
    train_ds = LFWCropBandDataset(data_dir, split="train", augment=True, seed=seed)
    val_ds = LFWCropBandDataset(data_dir, split="val", augment=False, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = PurifierUNet().to(dev)
    n_params = count_parameters(model)

    if embedder is None:
        recognizer = FaceRecognizer(ckpt_path=checkpoint, strict=True)
        embedder = FrozenMethod3Embedder(recognizer, device=dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Evaluate gray baseline cosine similarity on val once
    val_gray_cos_sum = 0.0
    val_total = 0
    with torch.no_grad():
        for x_val, m_val in val_loader:
            x_val = x_val.to(dev)
            m_val = m_val.to(dev)
            x_gray = gray_baseline_tensor(x_val, m_val)
            emb_g = embedder(x_gray)
            emb_c = embedder(x_val)
            cos_g = (emb_g * emb_c).sum(dim=1)
            val_gray_cos_sum += float(cos_g.sum().item())
            val_total += int(x_val.size(0))
    val_gray_cos = val_gray_cos_sum / max(1, val_total)

    log_path = run_dir / "train_log.csv"
    log_fields = [
        "epoch",
        "train_loss", "train_l1_band", "train_l1_full", "train_id_loss", "train_cos",
        "val_loss", "val_l1_band", "val_l1_full", "val_id_loss", "val_cos",
        "seconds",
    ]

    best_val_cos = -float("inf")
    best_epoch = 0
    t_start = time.time()

    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            tr_sums = {"loss": 0.0, "l1_band": 0.0, "l1_full": 0.0, "id_loss": 0.0, "cos": 0.0}
            tr_n = 0

            for x_batch, m_batch in train_loader:
                x_batch = x_batch.to(dev)
                m_batch = m_batch.to(dev)
                bs = int(x_batch.size(0))

                optimizer.zero_grad()
                pred = model(x_batch, m_batch)
                loss, m_dict = purifier_loss(pred, x_batch, m_batch, embedder=embedder)
                loss.backward()
                optimizer.step()

                for k in tr_sums:
                    tr_sums[k] += m_dict[k] * bs
                tr_n += bs

            tr_avg = {k: tr_sums[k] / max(1, tr_n) for k in tr_sums}

            model.eval()
            va_sums = {"loss": 0.0, "l1_band": 0.0, "l1_full": 0.0, "id_loss": 0.0, "cos": 0.0}
            va_n = 0
            with torch.no_grad():
                for x_batch, m_batch in val_loader:
                    x_batch = x_batch.to(dev)
                    m_batch = m_batch.to(dev)
                    bs = int(x_batch.size(0))
                    pred = model(x_batch, m_batch)
                    _, m_dict = purifier_loss(pred, x_batch, m_batch, embedder=embedder)
                    for k in va_sums:
                        va_sums[k] += m_dict[k] * bs
                    va_n += bs

            va_avg = {k: va_sums[k] / max(1, va_n) for k in va_sums}
            elapsed = time.time() - t_start

            row = {
                "epoch": epoch,
                "train_loss": round(tr_avg["loss"], 6),
                "train_l1_band": round(tr_avg["l1_band"], 6),
                "train_l1_full": round(tr_avg["l1_full"], 6),
                "train_id_loss": round(tr_avg["id_loss"], 6),
                "train_cos": round(tr_avg["cos"], 6),
                "val_loss": round(va_avg["loss"], 6),
                "val_l1_band": round(va_avg["l1_band"], 6),
                "val_l1_full": round(va_avg["l1_full"], 6),
                "val_id_loss": round(va_avg["id_loss"], 6),
                "val_cos": round(va_avg["cos"], 6),
                "seconds": round(elapsed, 2),
            }
            writer.writerow(row)
            fh.flush()

            if va_avg["cos"] > best_val_cos:
                best_val_cos = va_avg["cos"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "val_cos": best_val_cos,
                        "val_loss": va_avg["loss"],
                        "model_state_dict": model.state_dict(),
                    },
                    run_dir / "best.pt",
                )

            torch.save(
                {
                    "epoch": epoch,
                    "val_cos": va_avg["cos"],
                    "val_loss": va_avg["loss"],
                    "model_state_dict": model.state_dict(),
                },
                run_dir / "last.pt",
            )
            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"train_loss={tr_avg['loss']:.4f} train_cos={tr_avg['cos']:.4f} | "
                f"val_loss={va_avg['loss']:.4f} val_cos={va_avg['cos']:.4f} "
                f"(best={best_val_cos:.4f} @ ep {best_epoch}, gray={val_gray_cos:.4f})"
            )

    summary = {
        **meta_args,
        "run_dir": str(run_dir),
        "n_parameters": int(n_params),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "val_gray_cos": round(float(val_gray_cos), 6),
        "best_epoch": int(best_epoch),
        "best_val_cos": round(float(best_val_cos), 6),
        "best_checkpoint": str(run_dir / "best.pt"),
    }
    write_json(run_dir / "summary.json", summary)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Purifier U-Net on LFW crops.")
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="Directory with LFW crop shards")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Frozen method3 checkpoint")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Optional explicit output directory")
    args = ap.parse_args()

    train_purifier(
        data_dir=Path(args.data),
        checkpoint=Path(args.checkpoint),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        out_dir=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
