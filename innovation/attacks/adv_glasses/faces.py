"""
faces.py – the backend's face path, reused as-is: detection (the
LivenessDetector's MediaPipe call), the bbox crop, the lighting enhancer and
the recogniser. Plus eye landmarks for placing the glasses, and galleries of
enrolled identities built from images.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from backend.processing import _enhance_inference_crop, crop_face_from_bbox
from liveness.liveness_detector import LivenessDetector

CROP_PAD = 0.15                  # backend.processing.crop_face_from_bbox default
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# FaceMesh indices of the eye corners. "Left" means the left of the image
# (the subject's right eye).
LEFT_EYE = (33, 133)
RIGHT_EYE = (362, 263)


class FaceDetector:
    """MediaPipe face detection with the backend's settings and selection rule."""

    def __init__(self, model_selection: int = 0):
        import mediapipe as mp
        self._fd = mp.solutions.face_detection.FaceDetection(model_selection=model_selection,
                                                             min_detection_confidence=0.5)
        # LivenessDetector.detect_face only reads self._mp_fd: call it unbound so
        # the bbox comes from the backend's own code, without loading the liveness model.
        self._self = SimpleNamespace(_mp_fd=self._fd)

    def detect(self, img_bgr: np.ndarray) -> Optional[list[int]]:
        return LivenessDetector.detect_face(self._self, img_bgr)


def crop_window(bbox: list[int], shape: tuple[int, ...], pad: float = CROP_PAD) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the region crop_face_from_bbox returns."""
    x, y, w, h = bbox
    H, W = shape[:2]
    dx, dy = int(w * pad), int(h * pad)
    return max(0, x - dx), max(0, y - dy), min(W, x + w + dx), min(H, y + h + dy)


def crop(img_bgr: np.ndarray, bbox: list[int]) -> np.ndarray:
    return crop_face_from_bbox(img_bgr, bbox)


def enhance(face_bgr: np.ndarray) -> tuple[np.ndarray, Optional[dict]]:
    """The lighting enhancer exactly as check-in applies it (bypassed in normal light)."""
    return _enhance_inference_crop(face_bgr)


class EyeLandmarks:
    """Eye centres from MediaPipe FaceMesh (still images)."""

    def __init__(self):
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                                     refine_landmarks=False, min_detection_confidence=0.5)

    def eyes(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """[[x_left, y_left], [x_right, y_right]] in pixels, or None when no face mesh is found."""
        h, w = img_bgr.shape[:2]
        res = self._mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pt = lambda ids: np.mean([[lm[i].x * w, lm[i].y * h] for i in ids], axis=0)
        return np.array([pt(LEFT_EYE), pt(RIGHT_EYE)], np.float32)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


@dataclass
class Gallery:
    """Enrolled embeddings per identity, as the DB would hold them (one per enrolled image)."""
    embeddings: dict[str, np.ndarray]          # id -> (n, 512), L2-normalised

    def as_db(self) -> list[dict]:
        """The list FaceRecognizer.find_match expects."""
        return [{"employee_id": k, "name": k, "angle": str(i), "embedding": e}
                for k, embs in self.embeddings.items() for i, e in enumerate(embs)]

    def ids(self) -> list[str]:
        return sorted(self.embeddings)


def build_gallery(root: Path, recognizer, detector: FaceDetector, max_per_id: int = 10,
                  cache_dir: Optional[Path] = None) -> tuple[Gallery, dict]:
    """Embed every identity folder under `root` (root/<id>/*.jpg) through the backend path.

    Returns the gallery and a report: images used and skipped (no face) per id.
    Cached per checkpoint file so transfer runs do not recompute.
    """
    root = Path(root)
    ckpt = getattr(recognizer, "ckpt_path", None)
    key = hashlib.sha1(f"{root.resolve()}|{ckpt}|{max_per_id}".encode()).hexdigest()[:12]
    cache = (cache_dir / f"gallery_{key}.npz") if cache_dir else None
    if cache and cache.exists() and Path(str(ckpt)).exists():
        data = np.load(cache, allow_pickle=True)
        return Gallery(dict(data["embeddings"].item())), dict(data["report"].item())

    embeddings, report = {}, {}
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        embs, skipped = [], 0
        for f in list_images(person)[:max_per_id * 3]:
            if len(embs) >= max_per_id:
                break
            img = cv2.imread(str(f))
            bbox = detector.detect(img) if img is not None else None
            if bbox is None:
                skipped += 1
                continue
            face, _ = enhance(crop(img, bbox))
            if face.size == 0:
                skipped += 1
                continue
            embs.append(recognizer.extract_embedding(face))
        if embs:
            embeddings[person.name] = np.stack(embs)
        report[person.name] = {"used": len(embs), "skipped_no_face": skipped}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embeddings=np.array(embeddings, dtype=object), report=np.array(report, dtype=object))
    return Gallery(embeddings), report
