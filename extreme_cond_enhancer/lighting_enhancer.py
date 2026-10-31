"""Facade with detector, quality gate, and safe no-throw fallback behavior."""
from __future__ import annotations

from typing import Any
import time
import cv2
import numpy as np

from .color_scaler import enhance_color_scaling
from .config import LightingEnhancerConfig
from .reconstruction_model import ReconstructionModel, TorchScriptReconstructionModel


class LightingEnhancer:
    """Enhance only extreme exposures; normal images are returned byte-for-byte."""
    def __init__(self, config: LightingEnhancerConfig | None = None,
                 reconstruction_model: ReconstructionModel | None = None) -> None:
        self.config = config or LightingEnhancerConfig()
        self.reconstruction_model = reconstruction_model or TorchScriptReconstructionModel(self.config)

    def _stats(self, image: np.ndarray) -> dict[str, float]:
        lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {"brightness": float(lab_l.mean()), "contrast": float(lab_l.std()),
                "black_clip": float(np.mean(lab_l <= self.config.near_black)),
                "white_clip": float(np.mean(lab_l >= self.config.near_white)),
                "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var())}

    def _condition(self, stats: dict[str, float]) -> str:
        if stats["brightness"] < self.config.dark_brightness_threshold or stats["black_clip"] > self.config.dark_clip_ratio_threshold:
            return "too_dark"
        if stats["brightness"] > self.config.bright_brightness_threshold or stats["white_clip"] > self.config.bright_clip_ratio_threshold:
            return "too_bright"
        return "normal"

    def _quality(self, candidate: np.ndarray, original: np.ndarray) -> float:
        if candidate.shape != original.shape or candidate.dtype != np.uint8 or not np.isfinite(candidate).all():
            return -1e9
        s = self._stats(candidate)
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        cand_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB)
        color_shift = float(np.mean(np.abs(cand_lab[:, :, 1:].astype(np.float32) - orig_lab[:, :, 1:].astype(np.float32))))
        if color_shift > self.config.max_color_shift:
            return -1e9
        # Favor balanced exposure and clipping reduction, penalize blur/color artifacts.
        exposure = 1.0 - min(abs(s["brightness"] - 128.0) / 128.0, 1.0)
        clipping = 1.0 - min(s["black_clip"] + s["white_clip"], 1.0)
        contrast = min(s["contrast"] / 64.0, 1.0)
        blur = min(s["blur"] / 80.0, 1.0)
        return 0.43 * exposure + 0.27 * clipping + 0.20 * contrast + 0.10 * blur - color_shift / 500.0

    @staticmethod
    def _metadata(applied: bool, condition: str, before: dict[str, float], after: dict[str, float],
                  quality: float, reason: str, color: bool = False, deep: bool = False, fallback: bool = False,
                  started: float | None = None, transform: dict[str, float] | None = None) -> dict[str, Any]:
        latency = 0.0 if started is None else (time.perf_counter() - started) * 1000.0
        return {"applied": applied, "condition": condition,
                "brightness_before": round(before["brightness"], 3), "brightness_after": round(after["brightness"], 3),
                "contrast_before": round(before["contrast"], 3), "contrast_after": round(after["contrast"], 3),
                "color_scale_applied": color, "deep_reconstruction_applied": deep,
                "quality_score": round(float(quality), 4), "fallback_used": fallback, "reason": reason,
                "transform": transform or {}, "latency_ms": round(latency, 3)}

    def enhance(self, face_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Return an image of the same BGR shape plus diagnostics; never raises."""
        started = time.perf_counter()
        fallback = np.asarray(face_bgr).copy() if isinstance(face_bgr, np.ndarray) else np.empty((0, 0, 3), np.uint8)
        zero = {"brightness": 0.0, "contrast": 0.0, "black_clip": 0.0, "white_clip": 0.0, "blur": 0.0}
        try:
            if fallback.ndim != 3 or fallback.shape[2] != 3 or fallback.size == 0:
                return fallback, self._metadata(False, "normal", zero, zero, 0.0, "invalid BGR image", fallback=True)
            if fallback.dtype != np.uint8:
                fallback = np.clip(fallback, 0, 255).astype(np.uint8)
            if min(fallback.shape[:2]) < self.config.min_face_side:
                s = self._stats(fallback)
                return fallback, self._metadata(False, "normal", s, s, 0.0, "image is too small", fallback=True)
            before = self._stats(fallback)
            condition = self._condition(before)
            original_quality = self._quality(fallback, fallback)
            if condition == "normal":
                return fallback, self._metadata(False, condition, before, before, original_quality, "normal lighting; bypassed",
                                                started=started)
            scaled, transform = enhance_color_scaling(fallback, condition, self.config)
            candidates: list[tuple[np.ndarray, bool, bool, str]] = [(fallback, False, False, "quality gate retained original"),
                                                                     (scaled, True, False, "deterministic color scaling")]
            try:
                if self.reconstruction_model.is_available():
                    # The curve model is trained on raw exposures, so feed it the original crop.
                    candidates.append((self.reconstruction_model.enhance(fallback), False, True, "local reconstruction model"))
            except Exception:
                pass
            chosen, color, deep, reason = max(candidates, key=lambda item: self._quality(item[0], fallback))
            quality = self._quality(chosen, fallback)
            if quality < original_quality + self.config.min_quality_gain:
                chosen, color, deep, reason, quality = fallback, False, False, "quality gate rejected enhancement", original_quality
            after = self._stats(chosen)
            return chosen, self._metadata(color or deep, condition, before, after, quality, reason, color, deep,
                                          chosen is fallback, started, transform if color else None)
        except Exception:
            return fallback, self._metadata(False, "normal", zero, zero, 0.0, "enhancement failure; original retained",
                                            fallback=True, started=started)
