# -*- coding: utf-8 -*-
"""
liveness/temporal.py
====================
Temporal smoother for Face Anti-Spoofing predictions with hysteresis,
timestamp-based sliding window, face disappearance timeout, and spatial jump detection.

Pure python implementation without torch or cv2 dependencies.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple, List


class LivenessSmoother:
    """
    Temporal prediction smoother with dual-threshold hysteresis.

    Args:
        window_s: Sliding window duration in seconds (default: 1.0).
        t_on: Threshold to switch from Checking/Fake -> Real (default: 0.30).
        t_off: Threshold to switch from Checking/Real -> Fake (default: 0.10).
        min_frames: Minimum valid frames required in window to exit Checking (default: 3).
        reset_s: Max allowed duration of face absence before state resets (default: 1.0).
        jump_ratio: Center displacement relative to previous face width triggering a reset (default: 0.5).
    """

    def __init__(
        self,
        window_s: float = 1.0,
        t_on: float = 0.30,
        t_off: float = 0.10,
        min_frames: int = 3,
        reset_s: float = 1.0,
        jump_ratio: float = 0.5,
    ):
        assert t_off <= t_on, f"t_off ({t_off}) must be <= t_on ({t_on})"
        self.window_s = float(window_s)
        self.t_on = float(t_on)
        self.t_off = float(t_off)
        self.min_frames = int(min_frames)
        self.reset_s = float(reset_s)
        self.jump_ratio = float(jump_ratio)

        self.history: deque[Tuple[float, float]] = deque()
        self.state: str = "Checking"
        self.last_face_t: Optional[float] = None
        self.last_bbox: Optional[List[int]] = None

    def reset(self) -> None:
        """Clear temporal window and reset state to 'Checking'."""
        self.history.clear()
        self.state = "Checking"
        self.last_face_t = None
        self.last_bbox = None

    def _prune(self, current_t: float) -> None:
        cutoff = current_t - self.window_s
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def update(
        self,
        t: float,
        p_live: Optional[float],
        bbox: Optional[List[int]] = None,
    ) -> Tuple[str, Optional[float]]:
        """
        Update smoother with current frame information.

        Args:
            t: Monotonic timestamp of the frame in seconds.
            p_live: Liveness confidence in [0, 1] or None if no face detected.
            bbox: Bounding box [x, y, w, h] or None.

        Returns:
            (state, mean_p) where state is 'Checking', 'Real', or 'Fake',
            and mean_p is average confidence in window (or None if empty).
        """
        # Case 1: No face in this frame
        if p_live is None:
            if self.last_face_t is not None and (t - self.last_face_t) > self.reset_s:
                self.reset()
            else:
                self._prune(t)
                if len(self.history) < self.min_frames:
                    self.state = "Checking"

            mean_p = (
                sum(p for _, p in self.history) / len(self.history)
                if self.history
                else None
            )
            return self.state, mean_p

        # Case 2: Face is detected
        # Check face absence timeout
        if self.last_face_t is not None and (t - self.last_face_t) > self.reset_s:
            self.reset()

        # Check spatial jump if previous bbox is known
        if bbox is not None and len(bbox) >= 4 and self.last_bbox is not None:
            curr_cx = bbox[0] + bbox[2] / 2.0
            curr_cy = bbox[1] + bbox[3] / 2.0
            prev_cx = self.last_bbox[0] + self.last_bbox[2] / 2.0
            prev_cy = self.last_bbox[1] + self.last_bbox[3] / 2.0
            dist = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
            prev_w = float(self.last_bbox[2])
            if dist > self.jump_ratio * prev_w:
                self.reset()

        self.last_face_t = t
        if bbox is not None and len(bbox) >= 4:
            self.last_bbox = [int(v) for v in bbox[:4]]

        self.history.append((t, float(p_live)))
        self._prune(t)

        if len(self.history) < self.min_frames:
            self.state = "Checking"
            mean_p = sum(p for _, p in self.history) / len(self.history)
            return self.state, mean_p

        mean_p = sum(p for _, p in self.history) / len(self.history)

        if self.state == "Checking":
            if mean_p >= self.t_on:
                self.state = "Real"
            else:
                self.state = "Fake"
        elif self.state == "Real":
            if mean_p < self.t_off:
                self.state = "Fake"
        elif self.state == "Fake":
            if mean_p >= self.t_on:
                self.state = "Real"

        return self.state, mean_p


def make_smoother(config: Optional[dict]) -> Optional[LivenessSmoother]:
    """
    Factory function to create a LivenessSmoother from a configuration dict.
    Returns None if smoothing is not enabled or configuration is missing.
    """
    if not config or not isinstance(config, dict):
        return None
    smoothing_cfg = config.get("smoothing")
    if not smoothing_cfg or not isinstance(smoothing_cfg, dict):
        return None
    if not smoothing_cfg.get("enabled"):
        return None
    kwargs = {k: v for k, v in smoothing_cfg.items() if k != "enabled"}
    return LivenessSmoother(**kwargs)

