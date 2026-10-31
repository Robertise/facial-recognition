"""Train or fine-tune the optional Zero-DCE++ reconstruction model on face crops.

Zero-DCE++ is trained with zero-reference losses, so no paired "well-lit"
ground truth is needed: any folder of face crops works. With ``--synthesize``
normally exposed crops are randomly degraded into under/over-exposed
variants (gamma + gain + shot/read noise), so an ordinary face dataset becomes
an extreme-lighting training set.

Typical use (run from the repository root)::

    # 1) Use the official pretrained weights as-is (no training). Place the file
    #    locally yourself; nothing is downloaded by this project.
    LIGHTING_ENHANCER_ENABLE_RECONSTRUCTION=true \\
    LIGHTING_ENHANCER_MODEL_ARCH=zero_dce_pp \\
    LIGHTING_ENHANCER_CHECKPOINT=checkpoints/zero_dce_pp/Epoch99.pth  uvicorn backend.main:app

    # 2) Optional: fine-tune on your own face crops, starting from those weights
    python -m extreme_cond_enhancer.train_reconstruction --data dataset/faces \\
        --init checkpoints/zero_dce_pp/Epoch99.pth --synthesize --epochs 20 \\
        --out checkpoints/zero_dce_pp_faces --export-torchscript

    # 3) Only convert a state dict to TorchScript
    python -m extreme_cond_enhancer.train_reconstruction --init Epoch99.pth \\
        --export-only --out checkpoints/zero_dce_pp
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .zero_dce import ZeroDCEPP, ZeroReferenceLoss, export_torchscript, load_zero_dce_pp

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def synthesize_exposure(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """Degrade a float RGB ``[0, 1]`` crop into an extreme exposure.

    Under-exposure: ``x' = g * x ** gamma`` with ``gamma in [1.2, 2.5]`` and
    ``g in [0.08, 0.45]`` plus Poisson-Gaussian noise. Over-exposure:
    ``x' = clip(g * x ** gamma)`` with ``gamma in [0.4, 0.8]``, ``g in [1.2, 2.2]``.
    """
    if rng.random() < 0.7:
        out = rng.uniform(0.08, 0.45) * np.power(rgb, rng.uniform(1.2, 2.5))
        photons = rng.uniform(80.0, 400.0)
        out = np.random.poisson(out * photons) / photons + np.random.normal(0, rng.uniform(0.0, 0.01), out.shape)
    else:
        out = rng.uniform(1.2, 2.2) * np.power(rgb, rng.uniform(0.4, 0.8))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class FaceCropDataset(Dataset):
    def __init__(self, root: Path, size: int, synthesize: bool, seed: int = 0) -> None:
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")
        self.size, self.synthesize = size, synthesize
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        bgr = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((self.size, self.size, 3), np.uint8)
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.rng.random() < 0.5:
            rgb = rgb[:, ::-1]
        if self.synthesize:
            rgb = synthesize_exposure(rgb, self.rng)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_epoch(net, loader, criterion, device, optimizer=None, grad_clip: float = 0.1) -> dict[str, float]:
    training = optimizer is not None
    net.train(training)
    totals: dict[str, float] = {}
    batches = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            enhanced, curve = net(batch)
            loss, parts = criterion(batch, enhanced, curve)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()
            for key, value in {"loss": float(loss.detach()), **parts,
                               "out_mean": float(enhanced.detach().mean())}.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    return {k: v / max(batches, 1) for k, v in totals.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="folder of face crops (searched recursively)")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/zero_dce_pp_faces"))
    parser.add_argument("--init", type=Path, help="state dict to start from, e.g. official Epoch99.pth")
    parser.add_argument("--synthesize", action="store_true", help="degrade normal crops into extreme exposures")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=128, help="training crop side (face crops are ~112px)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--exposure-target", type=float, default=0.6)
    parser.add_argument("--scale-factor", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-torchscript", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="convert --init to TorchScript and exit")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    net = (load_zero_dce_pp(args.init, args.scale_factor) if args.init else ZeroDCEPP(args.scale_factor))
    if args.export_only:
        if not args.init:
            parser.error("--export-only requires --init")
        print(f"exported {export_torchscript(net, args.out / 'zero_dce_pp.torchscript.pt')}")
        return
    if args.data is None:
        parser.error("--data is required for training")

    device = pick_device(args.device)
    dataset = FaceCropDataset(args.data, args.size, args.synthesize, args.seed)
    val_len = int(len(dataset) * args.val_split) if len(dataset) > 10 else 0
    train_set, val_set = random_split(dataset, [len(dataset) - val_len, val_len],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.workers) if val_len else None

    net = net.to(device)
    criterion = ZeroReferenceLoss(args.exposure_target).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"device={device} train={len(train_set)} val={val_len} init={args.init}")

    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_stats = run_epoch(net, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(net, val_loader, criterion, device) if val_loader else train_stats
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        torch.save(net.state_dict(), args.out / "last.pth")
        if val_stats["loss"] < best:
            best = val_stats["loss"]
            torch.save(net.state_dict(), args.out / "best.pth")
        print(f"epoch {epoch:3d}  train {train_stats['loss']:.4f}  val {val_stats['loss']:.4f}  "
              f"val_out_mean {val_stats['out_mean']:.3f}  {time.perf_counter() - started:.1f}s")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    if args.export_torchscript:
        best_net = load_zero_dce_pp(args.out / "best.pth", args.scale_factor)
        print(f"exported {export_torchscript(best_net, args.out / 'zero_dce_pp.torchscript.pt')}")
    print(f"best val loss {best:.4f}; weights in {args.out}")


if __name__ == "__main__":
    main()
