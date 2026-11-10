"""
purifier.py - 4-level U-Net inpainting model for erasing and reconstructing the eyeglass band region.

Architecture:
  - 4 encoder/decoder levels with channels (32, 64, 128, 256) and a 256-channel bottleneck (~3.7M params).
  - Input (4 channels): 3-channel crop (in [0, 1] BGR) with the band region replaced by 0.5 gray,
    concatenated with the 1-channel binary band mask (in {0.0, 1.0}).
  - Output (3 channels): composited so pixels outside the band are bit-for-bit identical to the input crop,
    and pixels inside the band come from the U-Net prediction.
  - Strictly invariant to any pixel values inside the band region.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PurifierUNet(nn.Module):
    """4-level U-Net (channels 32-64-128-256, ~3.7M parameters) for fixed-band face inpainting."""

    def __init__(self, base_channels: tuple[int, int, int, int] = (32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = base_channels

        self.enc1 = _DoubleConv(4, c1)         # 112 x 112
        self.pool1 = nn.MaxPool2d(2)           # 56 x 56
        self.enc2 = _DoubleConv(c1, c2)        # 56 x 56
        self.pool2 = nn.MaxPool2d(2)           # 28 x 28
        self.enc3 = _DoubleConv(c2, c3)        # 28 x 28
        self.pool3 = nn.MaxPool2d(2)           # 14 x 14
        self.enc4 = _DoubleConv(c3, c4)        # 14 x 14
        self.pool4 = nn.MaxPool2d(2)           # 7 x 7

        self.bottleneck = _DoubleConv(c4, c4)  # 7 x 7

        self.up4 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(c4 + c4, c3)   # 14 x 14

        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(c3 + c3, c2)   # 28 x 28

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c2 + c2, c1)   # 56 x 56

        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c1 + c1, c1)   # 112 x 112

        self.out_conv = nn.Conv2d(c1, 3, kernel_size=1)

    @staticmethod
    def _normalize_mask(mask: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        cond = (mask > 0.5).to(device=ref.device)
        m = cond.to(dtype=ref.dtype)
        return cond, m

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 112, 112) float tensor in [0, 1] (BGR).
        mask: (B, 1, 112, 112) or (B, 112, 112) binary band mask (> 0.5 = inside band).
        Returns composited (B, 3, 112, 112) float tensor in [0, 1].
        """
        cond, m = self._normalize_mask(mask, x)
        cond_3 = cond.expand_as(x)

        # Erase band pixels to constant 0.5 gray before entering U-Net (strict invariance)
        gray = torch.full_like(x, 0.5)
        x_masked = torch.where(cond_3, gray, x)
        inp = torch.cat([x_masked, m], dim=1)

        e1 = self.enc1(inp)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        raw_out = torch.sigmoid(self.out_conv(d1))

        # Composite: outside band keeps original pixels identically; inside band uses U-Net output
        return torch.where(cond_3, raw_out, x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gray_baseline_tensor(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Non-learned baseline: replace band pixels with 0.5 gray, keep outside unchanged."""
    cond, _ = PurifierUNet._normalize_mask(mask, x)
    cond_3 = cond.expand_as(x)
    return torch.where(cond_3, torch.full_like(x, 0.5), x)


def apply_gray(crop_bgr: np.ndarray, band: Optional[np.ndarray], gray_val: int = 128) -> np.ndarray:
    """Apply non-learned gray baseline on a BGR uint8 crop (112x112x3)."""
    if crop_bgr.shape[:2] != (112, 112):
        crop_bgr = cv2.resize(crop_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    out = crop_bgr.copy()
    if band is None:
        return out
    if band.shape[:2] != (112, 112):
        band = cv2.resize(band.astype(np.uint8), (112, 112), interpolation=cv2.INTER_NEAREST)
    out[band > 0] = gray_val
    return out


@torch.no_grad()
def apply_purifier(
    model: PurifierUNet,
    crop_bgr: np.ndarray,
    band: Optional[np.ndarray],
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Apply trained PurifierUNet on a BGR uint8 crop (112x112x3), preserving outside-band pixels."""
    if crop_bgr.shape[:2] != (112, 112):
        crop_bgr = cv2.resize(crop_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    if band is None:
        return crop_bgr.copy()
    if band.shape[:2] != (112, 112):
        band = cv2.resize(band.astype(np.uint8), (112, 112), interpolation=cv2.INTER_NEAREST)

    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(crop_bgr.transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255.0
    m = torch.from_numpy(np.ascontiguousarray((band > 0).astype(np.float32))).unsqueeze(0).unsqueeze(0).to(device)
    pred = model(x, m)[0]
    out_u8 = (pred * 255.0).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # Guarantee bit-exact preservation of uint8 pixels outside the band
    mask_bool = (band > 0)
    res = crop_bgr.copy()
    res[mask_bool] = out_u8[mask_bool]
    return res


class FrozenMethod3Embedder(nn.Module):
    """Differentiable wrapper around frozen FaceRecognizer (method3) matching `extract_embedding` preprocessing."""

    def __init__(self, recognizer: Any, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = recognizer._model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        pre = recognizer.preprocess
        self.size = int(pre.size)
        self.register_buffer("mean", torch.tensor(pre.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(pre.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1))

    def forward(self, x_bgr_01: torch.Tensor) -> torch.Tensor:
        """
        x_bgr_01: (B, 3, H, W) BGR float tensor in [0, 1].
        Returns (B, 512) L2-normalized embeddings.
        """
        self.model.eval()
        x = x_bgr_01
        if x.shape[-2:] != (self.size, self.size):
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        rgb = x.flip(1)
        norm = (rgb - self.mean) / self.std
        return self.model.get_embedding(norm)


def purifier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    embedder: Optional[Any] = None,
    w_band: float = 1.0,
    w_full: float = 0.1,
    w_id: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute composite training loss:
      loss = 1.0 * L1_band + 0.1 * L1_full + 0.5 * (1 - cos(emb(pred), emb(target)))
    """
    _, m = PurifierUNet._normalize_mask(mask, pred)
    diff = (pred - target).abs()

    denom = (m.sum() * 3.0).clamp(min=1.0)
    l1_band = (diff * m).sum() / denom
    l1_full = diff.mean()

    if embedder is not None:
        emb_pred = embedder(pred)
        with torch.no_grad():
            emb_clean = embedder(target)
        cos_sim = (emb_pred * emb_clean).sum(dim=1)
        id_loss = (1.0 - cos_sim).mean()
        mean_cos = float(cos_sim.detach().mean().item())
    else:
        id_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        mean_cos = 0.0

    total = w_band * l1_band + w_full * l1_full + w_id * id_loss
    metrics = {
        "loss": float(total.detach().item()),
        "l1_band": float(l1_band.detach().item()),
        "l1_full": float(l1_full.detach().item()),
        "id_loss": float(id_loss.detach().item()),
        "cos": mean_cos,
    }
    return total, metrics


def load_purifier_checkpoint(ckpt_path: str | Path, device: str | torch.device = "cpu") -> PurifierUNet:
    """Load a PurifierUNet from a saved checkpoint file."""
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Purifier checkpoint not found: {p}")
    state = torch.load(str(p), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    model = PurifierUNet().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
