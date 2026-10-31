# -*- coding: utf-8 -*-
"""
liveness/phone_detector.py
==========================
YOLOv8-based cell phone detector and face-in-phone bounding box overlap test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, List, Tuple


CELL_PHONE_CLASS_ID = 67


def face_in_phone(
    face_bbox: Optional[Sequence[int | float]],
    phone_boxes: Sequence[Sequence[int | float]],
    min_overlap: float = 0.8,
) -> bool:
    """
    Determine if any detected phone bounding box contains >= min_overlap fraction
    of the face bounding box area. Pure function without external dependencies.

    Args:
        face_bbox: [x, y, w, h] of face bounding box.
        phone_boxes: List of (x, y, w, h, ...) tuples/lists for detected phones.
        min_overlap: Minimum ratio of intersection area to face bounding box area (default: 0.8).

    Returns:
        True if any phone box covers >= min_overlap of the face box, else False.
    """
    if face_bbox is None or len(face_bbox) < 4 or not phone_boxes:
        return False

    fx, fy, fw, fh = float(face_bbox[0]), float(face_bbox[1]), float(face_bbox[2]), float(face_bbox[3])
    face_area = fw * fh
    if face_area <= 0:
        return False

    fx2 = fx + fw
    fy2 = fy + fh

    for pbox in phone_boxes:
        if len(pbox) < 4:
            continue
        px, py, pw, ph = float(pbox[0]), float(pbox[1]), float(pbox[2]), float(pbox[3])
        if pw <= 0 or ph <= 0:
            continue

        px2 = px + pw
        py2 = py + ph

        ix1 = max(fx, px)
        iy1 = max(fy, py)
        ix2 = min(fx2, px2)
        iy2 = min(fy2, py2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        overlap = inter_area / face_area
        if overlap >= min_overlap:
            return True

    return False


class PhoneDetector:
    """
    Cell phone detector using YOLOv8 (COCO class 67).

    Args:
        weights: Path to YOLO weights (default: 'checkpoints/yolov8n.pt').
        imgsz: Inference input size (default: 416).
        conf: Confidence threshold for cell phone detection (default: 0.25).
        device: Device to run inference on, e.g. 'cpu', 'cuda' (default: 'cpu').
    """

    def __init__(
        self,
        weights: str = "checkpoints/yolov8n.pt",
        imgsz: int = 416,
        conf: float = 0.25,
        device: str = "cpu",
    ):
        # Lazy import ultralytics so files importing this module do not crash if not installed
        from ultralytics import YOLO

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = device

        weights_path = Path(weights)
        if not weights_path.is_file():
            # If path points to checkpoints/yolov8n.pt and doesn't exist, download to checkpoints/
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            model_tmp = YOLO("yolov8n.pt")
            downloaded = Path("yolov8n.pt")
            if downloaded.exists() and not weights_path.exists():
                downloaded.rename(weights_path)
            self.model = YOLO(str(weights_path))
        else:
            self.model = YOLO(str(weights_path))

    def detect(self, frame_bgr) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect cell phones in a BGR image frame.

        Args:
            frame_bgr: BGR image numpy array.

        Returns:
            List of (x, y, w, h, conf) tuples for detected cell phones.
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return []

        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[CELL_PHONE_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: List[Tuple[int, int, int, int, float]] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box, conf in zip(xyxy, confs):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            detections.append((
                int(round(float(x1))),
                int(round(float(y1))),
                int(round(float(w))),
                int(round(float(h))),
                float(conf),
            ))

        return detections
