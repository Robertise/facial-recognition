"""Deterministic, explainable exposure correction for face crops.

All curves operate on luma ``Y`` (BT.601 grey, ``[0, 255]``) and are applied to
the image as a per-pixel gain, so chromaticity (the B:G:R ratio) is preserved:

1. Edge-preserving smoothing: ``Y_b = bilateral(Y)`` so the gain map does not
   follow sensor noise.
2. Robust stretch with percentiles ``P2``/``P98`` of ``Y_b``:
   ``Y_s = clip(255 * (Y_b - P2) / max(P98 - P2, eps), 0, 255)``,
   skipped when ``P98 - P2 < 8`` (flat/degenerate crop).
3. Adaptive gamma that maps the median ``m = median(Y_s) / 255`` onto a target
   level ``t``: ``gamma = clip(ln t / ln m, min_gamma, max_gamma)`` with
   ``gamma <= 1`` for dark and ``gamma >= 1`` for bright crops, and
   ``T = 255 * (Y_s / 255) ** gamma``.
4. Gain map ``G = clip(T / max(Y_b, 1), 0, max_gain)``; output
   ``I_out[c] = clip(I[c] * G, 0, 255)`` for every colour channel ``c``.
5. Dark crops only: light non-local-means denoising, since brightening
   amplifies sensor noise.
6. CLAHE on LAB ``L`` only (``a``/``b`` untouched) to restore local contrast.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import LightingEnhancerConfig

_MIN_DYNAMIC_RANGE = 8.0
_EPS = 1e-3


def robust_stretch(luma: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """``Y_s = clip(255 (Y - P2) / (P98 - P2))``; identity when the range is degenerate."""
    if upper - lower < _MIN_DYNAMIC_RANGE:
        return luma.astype(np.float32, copy=True)
    return np.clip(255.0 * (luma - lower) / max(upper - lower, _EPS), 0.0, 255.0).astype(np.float32)


def adaptive_gamma(stretched: np.ndarray, target: float, lo: float, hi: float) -> float:
    """``gamma = ln(t) / ln(m)`` so that ``m ** gamma == t``, clipped to ``[lo, hi]``."""
    median = float(np.clip(np.median(stretched) / 255.0, _EPS, 1.0 - _EPS))
    target = float(np.clip(target, _EPS, 1.0 - _EPS))
    return float(np.clip(np.log(target) / np.log(median), lo, hi))


def enhance_color_scaling(image_bgr: np.ndarray, condition: str,
                          config: LightingEnhancerConfig) -> tuple[np.ndarray, dict[str, float]]:
    """Return the exposure-corrected crop and the parameters that produced it."""
    luma = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.bilateralFilter(luma, 5, 12.0, 3.0)
    low, high = (float(v) for v in np.percentile(smooth, (2, 98)))
    info = {"p02": round(low, 3), "p98": round(high, 3), "gamma": 1.0, "mean_gain": 1.0}

    if condition == "too_dark":
        gamma_range = (config.min_gamma, 1.0)
        target = config.dark_target_level
    elif condition == "too_bright":
        gamma_range = (1.0, config.max_gamma)
        target = config.bright_target_level
    else:
        return image_bgr.copy(), info

    stretched = robust_stretch(smooth, low, high)
    gamma = adaptive_gamma(stretched, target, *gamma_range)
    mapped = 255.0 * np.power(stretched / 255.0, gamma)
    gain = np.clip(mapped / np.maximum(smooth, 1.0), 0.0, config.max_gain)
    output = np.clip(image_bgr.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)

    if condition == "too_dark" and config.denoise_strength > 0:
        h = float(config.denoise_strength)
        output = cv2.fastNlMeansDenoisingColored(output, None, h, h, 5, 15)

    if config.clahe_clip_limit > 0:
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        grid = max(1, config.clahe_grid_size)
        clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(grid, grid))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    info.update(gamma=round(gamma, 4), mean_gain=round(float(gain.mean()), 4))
    return output, info
