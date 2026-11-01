"""
eot.py – Expectation over Transformation: the random changes a printed frame
goes through before the recogniser sees it, applied to the face crop.

`TRAIN` is sampled while optimising; `UNSEEN` is a wider, held-out set for
the D+EOT tier (stronger rotation, light, blur, compression and distance).
JPEG has no gradient, so it is applied straight-through (BPDA): the forward
pass sees the real codec output, the backward pass treats it as identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EOTConfig:
    shift_px: float = 3.0          # placement jitter (template -> image, in image pixels)
    rot_deg: float = 5.0
    scale: float = 0.05
    brightness: float = 0.20       # multiplicative, ±
    contrast: float = 0.20         # around the crop mean, ±
    channel_gain: float = 0.05     # per-channel colour cast, ±
    blur_sigma: float = 1.0        # Gaussian, uniform in [0, max]
    jpeg_q: tuple[int, int] = (60, 95)
    distance_px: tuple[int, int] = (80, 200)   # crop side before the pipeline's resize to 112


TRAIN = EOTConfig()
UNSEEN = EOTConfig(shift_px=5.0, rot_deg=10.0, scale=0.10, brightness=0.35, contrast=0.35,
                   channel_gain=0.10, blur_sigma=1.6, jpeg_q=(40, 75), distance_px=(60, 120))


def jpeg_bpda(x: torch.Tensor, quality: int) -> torch.Tensor:
    """(3, h, w) BGR 0..255 through a real JPEG encode/decode, identity gradient."""
    img = x.detach().clamp(0, 255).round().byte().permute(1, 2, 0).cpu().numpy()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    y = torch.from_numpy(dec).permute(2, 0, 1).float().to(x.device)
    return x + (y - x).detach()


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.3:
        return x
    k = _gauss_kernel(sigma, x.device)
    pad = len(k) // 2
    y = F.conv2d(F.pad(x[:, None], (pad, pad, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, pad, pad), mode="replicate"), k.view(1, 1, -1, 1))
    return y[:, 0]


def photometric(crop: torch.Tensor, cfg: EOTConfig, rng: np.random.Generator) -> torch.Tensor:
    """Light, colour cast, blur, distance and JPEG on a (3, h, w) BGR 0..255 crop."""
    x = crop * float(1 + rng.uniform(-cfg.brightness, cfg.brightness))
    mean = x.mean()
    x = (x - mean) * float(1 + rng.uniform(-cfg.contrast, cfg.contrast)) + mean
    gains = torch.tensor(1 + rng.uniform(-cfg.channel_gain, cfg.channel_gain, size=3),
                         dtype=x.dtype, device=x.device).view(3, 1, 1)
    x = (x * gains).clamp(0, 255)
    x = blur(x, float(rng.uniform(0, cfg.blur_sigma)))
    side = int(rng.integers(cfg.distance_px[0], cfg.distance_px[1] + 1))
    h, w = x.shape[-2:]
    if side < max(h, w):
        k = side / max(h, w)
        x = F.interpolate(x[None], size=(max(8, int(h * k)), max(8, int(w * k))), mode="bilinear",
                          align_corners=False, antialias=True)[0]
    return jpeg_bpda(x, int(rng.integers(cfg.jpeg_q[0], cfg.jpeg_q[1] + 1)))
