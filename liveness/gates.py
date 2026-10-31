# -*- coding: utf-8 -*-
"""
liveness/gates.py
=================
Face bounding box size ratio gate for Face Anti-Spoofing.
"""

from __future__ import annotations
from typing import Optional, Sequence


def face_size_status(
    bbox: Optional[Sequence[int | float]],
    frame_w: int | float,
    min_ratio: float,
    max_ratio: float,
) -> str:
    """
    Check if detected face bounding box size is within valid operational distance.

    Args:
        bbox: [x, y, w, h] or (x, y, w, h). bbox[2] is face width.
        frame_w: Full frame width in pixels.
        min_ratio: Minimum acceptable ratio of face_w / frame_w.
        max_ratio: Maximum acceptable ratio of face_w / frame_w.

    Returns:
        'too_far'   if ratio < min_ratio (or bbox missing/invalid)
        'too_close' if ratio > max_ratio
        'ok'        if min_ratio <= ratio <= max_ratio
    """
    if bbox is None or len(bbox) < 3 or frame_w <= 0:
        return "too_far"

    face_w = float(bbox[2])
    if face_w <= 0:
        return "too_far"

    ratio = face_w / float(frame_w)
    if ratio < min_ratio:
        return "too_far"
    if ratio > max_ratio:
        return "too_close"
    return "ok"
