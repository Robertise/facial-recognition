# -*- coding: utf-8 -*-
"""
liveness_detector.py – Task 2: Anti-Spoofing (Liveness Detection)
================================================================
Wraps MultiFTNet (MiniFASNetV2SE backbone) trained from scratch
with calibrated face cropping, Fourier supervision features,
and full MediaPipe face detection integration.

Usage:
    from liveness.liveness_detector import LivenessDetector

    detector = LivenessDetector()
    is_real, confidence, label, bbox = detector.check(img_bgr)       # full pipeline (detect + check)
    is_real, confidence, label = detector.check_face(face_bgr)       # already-cropped face
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from liveness.fas_model import MultiFTNet, build_model
from liveness.fas_common import crop_face, prepare_face, load_config, LIVE_INDEX


class LivenessDetector:
    """
    Trained MultiFTNet liveness detector for anti-spoofing verification.

    Args:
        device: 'cpu', 'cuda', or 'mps'. Auto-detected if None.
        threshold: Probability threshold to classify as Real. If None,
                   reads from LIVENESS_THRESHOLD env var, else from liveness_config.json.
    """

    def __init__(self, device: str | torch.device | None = None, threshold: float | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Resolve checkpoint and config paths
        ckpt_env = os.environ.get("LIVENESS_CHECKPOINT")
        if ckpt_env:
            self.ckpt_path = Path(ckpt_env)
        else:
            self.ckpt_path = PROJECT_ROOT / "checkpoints" / "liveness_best.pth"

        config_env = os.environ.get("LIVENESS_CONFIG")
        if config_env:
            self.config_path = Path(config_env)
        else:
            self.config_path = self.ckpt_path.parent / "liveness_config.json"

        # Verify existence
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(
                f"[LivenessDetector] Checkpoint file not found: {self.ckpt_path.resolve()}\n"
                f"Please ensure 'liveness_best.pth' is placed in checkpoints/ or set LIVENESS_CHECKPOINT."
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"[LivenessDetector] Configuration file not found: {self.config_path.resolve()}\n"
                f"Please ensure 'liveness_config.json' is placed alongside the checkpoint or set LIVENESS_CONFIG."
            )

        # Load config
        self.config = load_config(self.config_path)
        self.crop_cfg = self.config["crop"]
        self.live_index = int(self.config.get("live_index", LIVE_INDEX))

        # Threshold resolution: argument > env var > config
        if threshold is not None:
            self.threshold = float(threshold)
            thresh_src = "argument"
        elif os.environ.get("LIVENESS_THRESHOLD") is not None:
            self.threshold = float(os.environ["LIVENESS_THRESHOLD"])
            thresh_src = "environment variable"
        else:
            self.threshold = float(self.config["threshold"])
            thresh_src = "config file"

        # Initialize model
        model_cfg = self.config["model"]
        self.model = build_model(model_cfg, pretrained=False).to(self.device)

        # Load weights
        state = torch.load(str(self.ckpt_path), map_location=self.device, weights_only=False)
        if next(iter(state)).startswith("module."):
            state = OrderedDict((k[7:], v) for k, v in state.items())
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # Face detector – MediaPipe Face Detection (short-range <2m)
        import mediapipe as mp
        self._mp_fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

        print(
            f"[LivenessDetector] Loaded checkpoint: {self.ckpt_path.name} on {self.device} "
            f"(threshold={self.threshold:.3f} from {thresh_src})"
        )

    # ── Face Detection (MediaPipe) ────────────────────────────────────────────
    def detect_face(self, img_bgr: np.ndarray) -> list[int] | None:
        """
        Returns bbox [x, y, w, h] of the most prominent face, or None.
        Uses MediaPipe Face Detection.
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_fd.process(img_rgb)

        if not result.detections:
            return None

        # Pick detection with highest score
        best = max(result.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box
        x = max(0, int(bb.xmin * w))
        y = max(0, int(bb.ymin * h))
        bw = int(bb.width * w)
        bh = int(bb.height * h)
        return [x, y, bw, bh]

    # ── Liveness Check on cropped face ────────────────────────────────────────
    @torch.no_grad()
    def check_face(self, face_bgr: np.ndarray) -> tuple[bool, float, str]:
        """
        Run liveness check on an already-cropped face image (BGR numpy, any size).

        Returns:
            is_real    (bool)
            confidence (float)  probability of being real/live [0, 1]
            label      (str)    'Real' or 'Fake'
        """
        inp = prepare_face(face_bgr).unsqueeze(0).to(self.device)
        logits = self.model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        confidence = float(probs[self.live_index])
        is_real = confidence >= self.threshold
        label = "Real" if is_real else "Fake"
        return is_real, confidence, label

    # ── Full pipeline: detect + check ─────────────────────────────────────────
    def check(self, img_bgr: np.ndarray) -> tuple[bool, float, str, list[int] | None]:
        """
        Full pipeline: detect face from full image, crop with calibrated config, then check liveness.

        Returns:
            is_real    (bool)
            confidence (float)
            label      (str)   'Real' | 'Fake' | 'No Face'
            bbox       (list)  [x, y, w, h] or None (original MediaPipe bbox)
        """
        bbox = self.detect_face(img_bgr)
        if bbox is None:
            return False, 0.0, "No Face", None

        face_patch = crop_face(img_bgr, bbox, self.crop_cfg)
        if face_patch is None or face_patch.size == 0:
            return False, 0.0, "No Face", None

        is_real, confidence, label = self.check_face(face_patch)
        return is_real, confidence, label, bbox
