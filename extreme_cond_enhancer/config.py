"""Configuration for extreme lighting enhancement.

Every value may be overridden with an ``LIGHTING_ENHANCER_*`` environment
variable.  Defaults deliberately target only materially bad exposures.

Detection (all on LAB ``L`` in ``[0, 255]``)
    LIGHTING_ENHANCER_DARK_BRIGHTNESS      55.0   mean L below this -> too_dark
    LIGHTING_ENHANCER_BRIGHT_BRIGHTNESS    205.0  mean L above this -> too_bright
    LIGHTING_ENHANCER_DARK_CLIP_RATIO      0.22   fraction of L <= NEAR_BLACK -> too_dark
    LIGHTING_ENHANCER_BRIGHT_CLIP_RATIO    0.18   fraction of L >= NEAR_WHITE -> too_bright
    LIGHTING_ENHANCER_NEAR_BLACK           12
    LIGHTING_ENHANCER_NEAR_WHITE           243
    LIGHTING_ENHANCER_MIN_FACE_SIDE        8      smaller crops are returned unchanged

Deterministic transformation (see ``color_scaler.py`` for the math)
    LIGHTING_ENHANCER_DARK_TARGET          0.42   target median luma (0-1) for dark crops
    LIGHTING_ENHANCER_BRIGHT_TARGET        0.60   target median luma (0-1) for bright crops
    LIGHTING_ENHANCER_MIN_GAMMA            0.40   strongest brightening gamma allowed
    LIGHTING_ENHANCER_MAX_GAMMA            1.80   strongest darkening gamma allowed
    LIGHTING_ENHANCER_MAX_GAIN             6.0    per-pixel luminance gain ceiling
    LIGHTING_ENHANCER_DENOISE_STRENGTH     4.0    NL-means strength after brightening; 0 disables
    LIGHTING_ENHANCER_CLAHE_CLIP_LIMIT     1.4
    LIGHTING_ENHANCER_CLAHE_GRID_SIZE      4

Optional reconstruction model (never downloaded; local file only)
    LIGHTING_ENHANCER_ENABLE_RECONSTRUCTION  false
    LIGHTING_ENHANCER_CHECKPOINT             ""        path to the local weights
    LIGHTING_ENHANCER_MODEL_ARCH             torchscript | zero_dce_pp
    LIGHTING_ENHANCER_MODEL_SCALE            1         Zero-DCE++ curve-map downscale factor
    LIGHTING_ENHANCER_DEVICE                 auto      auto | cpu | cuda | mps

Quality gate
    LIGHTING_ENHANCER_MAX_COLOR_SHIFT      24.0   mean |delta a,b| in LAB above which a candidate is rejected
    LIGHTING_ENHANCER_MIN_QUALITY_GAIN     -0.02  required score gain over the original
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LightingEnhancerConfig:
    # Detection
    dark_brightness_threshold: float = _env_float("LIGHTING_ENHANCER_DARK_BRIGHTNESS", 55.0)
    bright_brightness_threshold: float = _env_float("LIGHTING_ENHANCER_BRIGHT_BRIGHTNESS", 205.0)
    dark_clip_ratio_threshold: float = _env_float("LIGHTING_ENHANCER_DARK_CLIP_RATIO", 0.22)
    bright_clip_ratio_threshold: float = _env_float("LIGHTING_ENHANCER_BRIGHT_CLIP_RATIO", 0.18)
    near_black: int = int(_env_float("LIGHTING_ENHANCER_NEAR_BLACK", 12))
    near_white: int = int(_env_float("LIGHTING_ENHANCER_NEAR_WHITE", 243))
    min_face_side: int = int(_env_float("LIGHTING_ENHANCER_MIN_FACE_SIDE", 8))
    # Deterministic transformation
    dark_target_level: float = _env_float("LIGHTING_ENHANCER_DARK_TARGET", 0.42)
    bright_target_level: float = _env_float("LIGHTING_ENHANCER_BRIGHT_TARGET", 0.60)
    min_gamma: float = _env_float("LIGHTING_ENHANCER_MIN_GAMMA", 0.40)
    max_gamma: float = _env_float("LIGHTING_ENHANCER_MAX_GAMMA", 1.80)
    max_gain: float = _env_float("LIGHTING_ENHANCER_MAX_GAIN", 6.0)
    denoise_strength: float = _env_float("LIGHTING_ENHANCER_DENOISE_STRENGTH", 4.0)
    clahe_clip_limit: float = _env_float("LIGHTING_ENHANCER_CLAHE_CLIP_LIMIT", 1.4)
    clahe_grid_size: int = int(_env_float("LIGHTING_ENHANCER_CLAHE_GRID_SIZE", 4))
    # Optional reconstruction model
    enable_reconstruction: bool = _env_bool("LIGHTING_ENHANCER_ENABLE_RECONSTRUCTION", False)
    checkpoint_path: str = os.getenv("LIGHTING_ENHANCER_CHECKPOINT", "")
    model_arch: str = os.getenv("LIGHTING_ENHANCER_MODEL_ARCH", "torchscript").strip().lower()
    model_scale_factor: int = max(1, int(_env_float("LIGHTING_ENHANCER_MODEL_SCALE", 1)))
    device: str = os.getenv("LIGHTING_ENHANCER_DEVICE", "auto")
    # Quality gate
    max_color_shift: float = _env_float("LIGHTING_ENHANCER_MAX_COLOR_SHIFT", 24.0)
    min_quality_gain: float = _env_float("LIGHTING_ENHANCER_MIN_QUALITY_GAIN", -0.02)

    @property
    def checkpoint(self) -> Path | None:
        return Path(self.checkpoint_path).expanduser() if self.checkpoint_path else None
