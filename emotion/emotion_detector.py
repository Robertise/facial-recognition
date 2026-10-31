"""
emotion_detector.py  – Task 3: Emotion Detection
=================================================
Inference wrapper for the **own-trained** emotion classifier produced by
`training/training_emotion.ipynb`.

The public API is unchanged from the previous (HSEmotion-based) version so that
`backend/processing.py`, the FastAPI app, the frontend and the test scripts keep
working without modification::

    from emotion.emotion_detector import EmotionDetector

    detector = EmotionDetector()
    label, emoji, confidence = detector.predict(face_bgr)   # e.g. ('Happy', '😊', 0.93)

Model resolution order:
    1. <root>/checkpoints/emotion/emotion_best.pt   (TorchScript – preferred, arch-free)
    2. <root>/checkpoints/emotion/emotion_best.pth  (state_dict for EmotionCNN)
    3. legacy <root>/checkpoints/emotion_best.{pt,pth}  (pre-move location)
    4. HSEmotion fallback (development only – NOT the submitted model)

`EMOTION_CHECKPOINT` overrides the search entirely.

All preprocessing metadata (input size, channels, normalization, class order,
emoji map) is read from `emotion/labels.json`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from emotion.model import EmotionCNN, preprocess_face  # noqa: E402


# ── Defaults (used if labels.json is missing) ─────────────────────────────────
DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOJI_MAP = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
    # legacy HSEmotion-style names (fallback path)
    "Happiness": "😊", "Sadness": "😢", "Anger": "😠",
    "Contempt": "😒", "Disgust": "🤢", "Fear": "😨",
    "Surprise": "😲", "Neutral": "😐",
    "Happy": "😊", "Sad": "😢", "Angry": "😠",
}

DEFAULT_EMOJI = "❓"


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmotionDetector:
    """
    Emotion classifier backed by the project's own trained model.

    Args:
        model_path: Explicit path to the TorchScript (.pt) or state_dict (.pth).
        labels_path: Path to labels.json (class order / emoji / preprocessing).
        device: 'cpu', 'cuda', 'mps' or a torch.device. Auto-detected if None.
        allow_fallback: If the trained checkpoint is missing, fall back to
            HSEmotion when installed instead of raising (development convenience).
    """

    def __init__(self,
                 model_path: str | None = None,
                 labels_path: str | None = None,
                 device=None,
                 allow_fallback: bool = True):
        self.device = torch.device(device) if device is not None else _auto_device()
        self.allow_fallback = allow_fallback

        self.meta = self._load_meta(labels_path)
        self.class_names = list(self.meta.get("class_names", DEFAULT_LABELS))
        self.emoji_map = {**EMOJI_MAP, **self.meta.get("emoji", {})}
        self.input_size = int(self.meta.get("input_size", 48))
        self.channels = int(self.meta.get("channels", 1))
        self.mean = self.meta.get("mean", 0.5)
        self.std = self.meta.get("std", 0.5)
        self.logit_offsets = self._load_offsets()

        self._scripted = None
        self._model = None
        self._hs = None
        self.backend = "none"

        if model_path:
            given = Path(model_path)
            if given.suffix == ".pth":
                candidates = [(given.with_suffix(".pt"), given)]
            else:
                candidates = [(given, given.with_suffix(".pth"))]
        else:
            env = os.environ.get("EMOTION_CHECKPOINT")
            if env:
                given = Path(env)
                candidates = ([(given.with_suffix(".pt"), given)] if given.suffix == ".pth"
                              else [(given, given.with_suffix(".pth"))])
            else:
                # Preferred home (checkpoints/emotion/), then the legacy pre-move location.
                ck_dir = _PROJECT_ROOT / "checkpoints" / "emotion"
                candidates = [
                    (ck_dir / "emotion_best.pt", ck_dir / "emotion_best.pth"),
                    (_PROJECT_ROOT / "checkpoints" / "emotion_best.pt",
                     _PROJECT_ROOT / "checkpoints" / "emotion_best.pth"),
                ]

        for pt_path, pth_path in candidates:
            if pt_path.exists():
                self._scripted = torch.jit.load(str(pt_path), map_location=self.device)
                self._scripted.eval()
                self.backend = "torchscript"
                print(f"[EmotionDetector] Loaded TorchScript model: {pt_path}")
                break
            if pth_path.exists():
                state = torch.load(str(pth_path), map_location=self.device, weights_only=False)
                if isinstance(state, dict) and "model_state" in state:
                    state = state["model_state"]
                model = EmotionCNN(num_classes=len(self.class_names))
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    print(f"[EmotionDetector] WARNING: missing keys: {missing[:5]}...")
                model.to(self.device).eval()
                self._model = model
                self.backend = "state_dict"
                print(f"[EmotionDetector] Loaded state_dict model: {pth_path}")
                break
        else:
            searched = ", ".join(str(pt) for pt, _ in candidates)
            msg = (f"[EmotionDetector] No trained checkpoint found (searched: {searched}). "
                   f"Train one with training/training_emotion.ipynb.")
            print(msg)
            if allow_fallback:
                self._init_hsemotion()
            else:
                raise FileNotFoundError(msg)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load_meta(labels_path: str | None) -> dict:
        path = Path(labels_path) if labels_path else (_HERE / "labels.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_offsets(self):
        """Per-class logit offsets from labels.json, validated against the class order.

        Returns None when the notebook did not fit calibration, so an older checkpoint
        behaves exactly as before.
        """
        raw = self.meta.get("logit_offsets")
        if not raw:
            return None
        off = np.asarray(raw, dtype=np.float32).reshape(-1)
        if off.shape[0] != len(self.class_names):
            print(f"[EmotionDetector] Ignoring logit_offsets: expected {len(self.class_names)} "
                  f"values, got {off.shape[0]}")
            return None
        if not np.all(np.isfinite(off)):
            print("[EmotionDetector] Ignoring logit_offsets: non-finite values")
            return None
        if not np.any(off):
            return None                      # all-zero == notebook shipped without calibration
        print("[EmotionDetector] Applying fitted class-prior calibration")
        return off

    def _init_hsemotion(self):
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self._hs = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
            self.backend = "hsemotion"
            print("[EmotionDetector] Fallback: HSEmotion (dev only – not the trained model).")
        except Exception as exc:  # pragma: no cover
            print(f"[EmotionDetector] HSEmotion fallback unavailable: {exc}")
            self._hs = None

    def _emoji_for(self, label: str) -> str:
        return self.emoji_map.get(label, self.emoji_map.get(label.lower(), DEFAULT_EMOJI))

    def _forward_logits(self, face_bgr: np.ndarray) -> np.ndarray:
        x = preprocess_face(face_bgr, self.input_size, self.channels,
                            self.mean, self.std).to(self.device)
        with torch.no_grad():
            out = self._scripted(x) if self._scripted is not None else self._model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.reshape(out.shape[0], -1)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        # Per-class prior correction fitted on validation by the training notebook.
        # It matches the predicted class distribution to the validation prior, which
        # mainly fixes the neutral<->sad trade-off; it is off by default because it did
        # not measurably lift fear/disgust (see section 15 of the notebook).
        if self.logit_offsets is not None:
            z = np.log(np.clip(probs, 1e-9, 1.0)) + self.logit_offsets
            z -= z.max()
            e = np.exp(z)
            probs = e / e.sum()
        return probs

    # ── public API ────────────────────────────────────────────────────────────
    def predict(self, face_bgr: np.ndarray):
        """
        Predict emotion from a cropped face (BGR numpy).

        Returns:
            label      (str)   e.g. 'Happy'
            emoji      (str)   e.g. '😊'
            confidence (float) in [0, 1]
        """
        if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
            return "Neutral", self._emoji_for("neutral"), 0.0

        if self.backend in ("torchscript", "state_dict"):
            probs = self._forward_logits(face_bgr)
            idx = int(np.argmax(probs))
            cls = self.class_names[idx]
            label = cls.capitalize()
            return label, self._emoji_for(cls), float(probs[idx])

        if self.backend == "hsemotion" and self._hs is not None:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            label, scores = self._hs.predict_emotions(face_rgb, logits=False)
            return label, self._emoji_for(label), float(np.max(scores))

        # No usable model at all – fail soft so the UI keeps working.
        return "Neutral", self._emoji_for("neutral"), 0.0

    def predict_batch(self, faces_bgr: list):
        """Predict for a list of face crops. Returns list of (label, emoji, confidence)."""
        return [self.predict(f) for f in faces_bgr]
