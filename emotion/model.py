"""
model.py – Task 3: Emotion Detection
=====================================
Shared model architecture + preprocessing for the *own-trained* emotion model.
The architecture here MUST match the one trained in
`training/training_emotion.ipynb` (the notebook also exports a TorchScript copy,
which is preferred at inference time so this file is only a fallback).

Public API:
    preprocess_face(face_bgr, size=48, channels=1, mean=0.5, std=0.5) -> Tensor[1,C,S,S]
    EmotionCNN(num_classes=7, width=32, dropout=0.3)                  -> logits[B,num_classes]
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


# ── Preprocessing (must match the notebook's *validation* transform) ───────────
def preprocess_face(face_bgr: np.ndarray,
                    size: int = 48,
                    channels: int = 1,
                    mean: float | list = 0.5,
                    std: float | list = 0.5) -> torch.Tensor:
    """
    Convert a BGR face crop (any size) into a normalized (1, C, size, size) tensor.

    Training used: cv2/PIL grayscale -> resize(size) -> /255 -> Normalize(mean, std).
    """
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        raise ValueError("preprocess_face received an empty image")

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    x = gray.astype(np.float32) / 255.0
    if channels == 3:
        x = np.stack([x, x, x], axis=-1)          # H,W,3
        m = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        s = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    else:
        x = x[..., None]                          # H,W,1
        m = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1)
        s = np.asarray(std, dtype=np.float32).reshape(1, 1, 1)

    x = (x - m) / s
    x = np.transpose(x, (2, 0, 1))                # C,H,W
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).float()


# ── Architecture ──────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> optional MaxPool2d(2)."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    Small VGG-style CNN for 48x48 (1-channel) facial-expression classification.

    Input : (B, 1, 48, 48)
    Output: (B, num_classes) logits
    Spatial: 48 -> 24 -> 12 -> 6 -> 3 -> GAP -> 1
    """

    def __init__(self, num_classes: int = 7, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, width),            # 48 -> 24
            ConvBlock(width, width * 2),    # 24 -> 12
            ConvBlock(width * 2, width * 4),  # 12 -> 6
            ConvBlock(width * 4, width * 4),  # 6 -> 3
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def embed(self, x):
        """Feature vector before the classifier head (used for Grad-CAM / SSL)."""
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))
