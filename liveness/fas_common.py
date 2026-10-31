# -*- coding: utf-8 -*-
"""
fas_common.py
=============
Single source of truth for Face Anti-Spoofing (FAS) preprocessing,
crop calculations, Fourier target generation, and configuration loading.

Used by both training and inference pipelines.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

INPUT_SIZE = 80
FT_SIZE = 10
LIVE_INDEX = 0        # class 0 = live, 1 = spoof
CLASS_NAMES = ["live", "spoof"]


def resize_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize arbitrary size uint8 BGR face image to 80x80 using cv2.INTER_AREA."""
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Invalid or empty face image provided to resize_face")
    if face_bgr.shape[0] == INPUT_SIZE and face_bgr.shape[1] == INPUT_SIZE:
        return face_bgr.copy()
    return cv2.resize(face_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def to_tensor(face80_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert 80x80 uint8 BGR face to normalized float32 tensor (3, 80, 80).
    Color: BGR -> RGB. Range: [0, 255] -> [-1.0, 1.0] via (x / 255.0 - 0.5) / 0.5.
    Does NOT resize.
    """
    if face80_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        raise ValueError(
            f"to_tensor expects ({INPUT_SIZE}, {INPUT_SIZE}) image, got {face80_bgr.shape[:2]}"
        )
    rgb = cv2.cvtColor(face80_bgr, cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    chw = np.transpose(norm, (2, 0, 1))
    return torch.from_numpy(chw).float()


def prepare_face(face_bgr: np.ndarray) -> torch.Tensor:
    """Convenience pipeline: resize_face followed by to_tensor."""
    return to_tensor(resize_face(face_bgr))


def generate_ft_target(face80_bgr: np.ndarray) -> torch.Tensor:
    """
    Generate Fourier Transform auxiliary supervision target (1, 10, 10).
    Formula: gray -> fft2 -> fftshift -> log(|.| + 1) -> min-max (x - min + 1) / (max - min + 1) -> resize(10, 10).
    Fully vectorized with numpy (no python loop).
    """
    if face80_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        face80_bgr = resize_face(face80_bgr)

    gray = cv2.cvtColor(face80_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1.0)
    minn = float(np.min(fimg))
    maxx = float(np.max(fimg))
    fimg = (fimg - minn + 1.0) / (maxx - minn + 1.0)

    fimg_resized = cv2.resize(fimg.astype(np.float32), (FT_SIZE, FT_SIZE))
    return torch.from_numpy(fimg_resized).unsqueeze(0).float()


def crop_face(img_bgr: np.ndarray, bbox: list | tuple, crop_cfg: dict[str, float]) -> np.ndarray | None:
    """
    Crop face based on calibrated parameters relative to MediaPipe bbox.
    bbox = [x, y, w, h] (MediaPipe).
    crop_cfg = {"scale_w": ..., "scale_h": ..., "shift_x": ..., "shift_y": ...}.

    cx = x + w/2 + shift_x*w ; cy = y + h/2 + shift_y*h
    new_w = w*scale_w ; new_h = h*scale_h
    box = [cx - new_w/2, cy - new_h/2, cx + new_w/2, cy + new_h/2]
    Intersects with image boundaries (no padding).
    Returns uint8 BGR crop, or None if intersection is empty.
    """
    if img_bgr is None or img_bgr.size == 0 or bbox is None:
        return None

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    scale_w = float(crop_cfg["scale_w"])
    scale_h = float(crop_cfg["scale_h"])
    shift_x = float(crop_cfg["shift_x"])
    shift_y = float(crop_cfg["shift_y"])

    cx = x + w / 2.0 + shift_x * w
    cy = y + h / 2.0 + shift_y * h
    new_w = w * scale_w
    new_h = h * scale_h

    x1 = int(round(cx - new_w / 2.0))
    y1 = int(round(cy - new_h / 2.0))
    x2 = int(round(cx + new_w / 2.0))
    y2 = int(round(cy + new_h / 2.0))

    H, W = img_bgr.shape[:2]
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(W, x2)
    iy2 = min(H, y2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    crop = img_bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    return crop


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load and validate liveness_config.json.
    Raises FileNotFoundError if file is missing.
    Raises ValueError if required keys are missing.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Liveness configuration file not found at: {config_path.resolve()}\n"
            f"Please ensure liveness_config.json exists."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["input_size", "ft_size", "class_names", "live_index", "crop", "threshold", "model"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(
            f"Invalid liveness config in {config_path}: missing top-level keys: {missing_keys}"
        )

    crop_required = ["scale_w", "scale_h", "shift_x", "shift_y"]
    missing_crop = [k for k in crop_required if k not in config.get("crop", {})]
    if missing_crop:
        raise ValueError(
            f"Invalid liveness config in {config_path}: missing crop keys: {missing_crop}"
        )

    return config
