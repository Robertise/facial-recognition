"""
glasses.py – the eyeglass-frame template, where it sits on a face, and a
differentiable renderer that paints a texture onto it.

Template space is a 400 x 150 px canvas: the eye centres sit at (110, 80) and
(290, 80); the frame is two elliptical rims, a bridge and short temple arms.
The rim thickness sets how much of the face the frame covers. A frame is
placed by the similarity transform that maps the template eye centres onto
the eye centres FaceMesh finds (scale, rotation, shift).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

TEMPLATE_W, TEMPLATE_H = 400, 150
EYES = np.array([[110.0, 80.0], [290.0, 80.0]], np.float32)
RIM_RX, RIM_RY = 80.0, 58.0


@dataclass(frozen=True)
class FrameShape:
    """Frame geometry in template pixels.

    `thickness` (rim band width) and `rim_scale` (rim size relative to an
    ordinary frame) set how much of the face the frame covers; `coverage`
    measures it on a real placement. The lens area inside the rims stays clear.
    """
    thickness: float = 14.0
    rim_scale: float = 1.0

    def mask(self, supersample: int = 4) -> np.ndarray:
        """(H, W) float32 in [0, 1], anti-aliased."""
        s = supersample
        m = np.zeros((TEMPLATE_H * s, TEMPLATE_W * s), np.uint8)
        t = self.thickness
        rx, ry = RIM_RX * self.rim_scale, RIM_RY * self.rim_scale
        for cx, cy in EYES:
            c = (int(cx * s), int(cy * s))
            cv2.ellipse(m, c, (int(rx * s), int(ry * s)), 0, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(m, c, (int(max(rx - t, 1) * s), int(max(ry - t, 1) * s)), 0, 0, 360, 0, -1, cv2.LINE_AA)
        # bridge across the top of the nose, and temple arms out to the canvas edges
        bx0, bx1 = EYES[0, 0] + rx - t, EYES[1, 0] - rx + t
        by = EYES[0, 1] - ry * 0.55
        cv2.rectangle(m, (int(bx0 * s), int(by * s)), (int(bx1 * s), int((by + t * 0.9) * s)), 255, -1)
        ay = EYES[0, 1] - ry * 0.6
        for x0, x1 in ((0, EYES[0, 0] - rx + t), (EYES[1, 0] + rx - t, TEMPLATE_W)):
            cv2.rectangle(m, (int(x0 * s), int(ay * s)), (int(x1 * s), int((ay + t) * s)), 255, -1)
        m = cv2.resize(m, (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA)
        return (m.astype(np.float32) / 255.0)


def similarity(dst_eyes: np.ndarray, src_eyes: np.ndarray = EYES) -> np.ndarray:
    """2x3 matrix mapping template pixels to image pixels so the eye centres line up."""
    s0, s1 = src_eyes.astype(np.float64)
    d0, d1 = dst_eyes.astype(np.float64)
    vs, vd = s1 - s0, d1 - d0
    scale = np.linalg.norm(vd) / np.linalg.norm(vs)
    ang = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])
    c, s = scale * np.cos(ang), scale * np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    t = d0 - R @ s0
    return np.hstack([R, t[:, None]]).astype(np.float32)


def jitter(A: np.ndarray, shift_px: float, rot_deg: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb a placement about the eye midpoint: shift, rotate and scale."""
    mid = A[:, :2] @ EYES.mean(axis=0) + A[:, 2]
    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    k = 1 + rng.uniform(-scale, scale)
    J = k * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    R = J @ A[:, :2]
    t = J @ (A[:, 2] - mid) + mid + rng.uniform(-shift_px, shift_px, size=2)
    return np.hstack([R, t[:, None]]).astype(np.float32)


def _grid(A: np.ndarray, window: tuple[int, int, int, int], device) -> torch.Tensor:
    """Sampling grid (1, h, w, 2) taking each window pixel back to template coordinates."""
    x0, y0, x1, y1 = window
    A3 = np.vstack([A, [0, 0, 1]]).astype(np.float64)
    inv = np.linalg.inv(A3)[:2]
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    tx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    ty = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    gx = (2 * tx + 1) / TEMPLATE_W - 1      # align_corners=False convention
    gy = (2 * ty + 1) / TEMPLATE_H - 1
    return torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32)).to(device)


def render(region_bgr: torch.Tensor, texture_rgb: torch.Tensor, mask: torch.Tensor,
           A: np.ndarray, window: tuple[int, int, int, int]) -> torch.Tensor:
    """Paint the frame onto an image region.

    region_bgr: (3, h, w) floats in 0..255, the pixels of `window` (x0, y0, x1, y1) of the frame.
    texture_rgb: (3, Ht, Wt) in 0..1, the colours the camera sees (any size; resampled to the template).
    mask: (1, TEMPLATE_H, TEMPLATE_W) in 0..1.
    Returns the composited region, differentiable in `texture_rgb`.
    """
    grid = _grid(A, window, region_bgr.device)
    tex = texture_rgb
    if tex.shape[-2:] != (TEMPLATE_H, TEMPLATE_W):
        tex = F.interpolate(tex[None], size=(TEMPLATE_H, TEMPLATE_W), mode="bilinear", align_corners=False)[0]
    tex_bgr = tex.flip(0) * 255.0
    t = F.grid_sample(tex_bgr[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    m = F.grid_sample(mask[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    return region_bgr * (1 - m) + t * m


def coverage(mask: torch.Tensor, A: np.ndarray, window: tuple[int, int, int, int], size: int = 112) -> float:
    """Share of the 112 x 112 recognition crop the frame covers."""
    x0, y0, x1, y1 = window
    grid = _grid(A, window, mask.device)
    m = F.grid_sample(mask[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
    return float(m.mean())
