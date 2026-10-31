"""
processing.py – Full AI pipeline for attendance and registration frames
"""
import base64, sys, os, time
from typing import Optional

# Optional per-machine CPU thread cap (unset = torch default). On a hybrid Intel
# P/E-core Windows laptop, torch's default 12 threads was ~30% slower per frame
# than TORCH_THREADS=1. Must be applied before torch is imported.
_TORCH_THREADS = os.environ.get("TORCH_THREADS")
if _TORCH_THREADS:
    os.environ.setdefault("OMP_NUM_THREADS", _TORCH_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", _TORCH_THREADS)

import numpy as np
import cv2
import torch
from pathlib import Path

if _TORCH_THREADS:
    torch.set_num_threads(int(_TORCH_THREADS))

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from liveness.liveness_detector import LivenessDetector
from liveness.gates             import face_size_status
from liveness.temporal          import LivenessSmoother
from emotion.emotion_detector   import EmotionDetector
from backend.head_pose          import HeadPoseEstimator
from backend.face_recognizer    import FaceRecognizer
from backend.database           import get_all_embeddings

THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.5"))

# ── Singletons (lazy-loaded) ──────────────────────────────────────────────────
_liveness   = None
_emotion    = None
_head_pose  = None
_recognizer = None
_lighting_enhancer = None
_lighting_unavailable = False

def _get_liveness():
    global _liveness
    if _liveness is None:
        _liveness = LivenessDetector()
    return _liveness

def _get_emotion():
    global _emotion
    if _emotion is None:
        _emotion = EmotionDetector()
    return _emotion

def _get_head_pose():
    global _head_pose
    if _head_pose is None:
        _head_pose = HeadPoseEstimator()
    return _head_pose

def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer(threshold=THRESHOLD)
    return _recognizer

def _get_lighting_enhancer():
    """Lazy optional dependency; inference must work when it cannot load."""
    global _lighting_enhancer, _lighting_unavailable
    if _lighting_enhancer is None and not _lighting_unavailable:
        try:
            from extreme_cond_enhancer import LightingEnhancer
            _lighting_enhancer = LightingEnhancer()
        except Exception:
            _lighting_unavailable = True
    return _lighting_enhancer


def _enhance_inference_crop(face_bgr: np.ndarray) -> tuple[np.ndarray, dict | None]:
    enhancer = _get_lighting_enhancer()
    if enhancer is None:
        return face_bgr, {"applied": False, "reason": "enhancer unavailable", "fallback_used": True}
    try:
        return enhancer.enhance(face_bgr)
    except Exception:
        return face_bgr, {"applied": False, "reason": "enhancement failure; original retained", "fallback_used": True}


def decode_frame(b64_data: str) -> np.ndarray | None:
    """Decode base64 JPEG/PNG string to BGR numpy array."""
    try:
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def crop_face_from_bbox(frame: np.ndarray, bbox: list, pad: float = 0.15) -> np.ndarray:
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    dx, dy = int(w*pad), int(h*pad)
    return frame[max(0,y-dy):min(H,y+h+dy), max(0,x-dx):min(W,x+w+dx)]


def _liveness_decision(
    live: LivenessDetector,
    frame_bgr: np.ndarray,
    smoother: LivenessSmoother,
    t: Optional[float] = None,
) -> tuple[bool, bool, Optional[dict], Optional[list]]:
    """
    Helper for smoothed liveness decision and distance gating.
    Returns: (face_detected, is_real, liveness_dict, bbox)
    """
    if t is None:
        t = time.monotonic()

    is_real_raw, p, label_raw, bbox = live.check(frame_bgr)

    if label_raw == "No Face":
        smoother.update(t, None, None)
        return False, False, None, None

    # Gate check (only if gate is enabled in config)
    gate_cfg = getattr(live, "config", None)
    gate = gate_cfg.get("gate") if isinstance(gate_cfg, dict) else None
    if gate and isinstance(gate, dict) and gate.get("enabled"):
        frame_w = frame_bgr.shape[1] if hasattr(frame_bgr, "shape") and len(frame_bgr.shape) >= 2 else 0
        min_ratio = float(gate.get("min_ratio", 0.22))
        max_ratio = float(gate.get("max_ratio", 0.70))
        status = face_size_status(bbox, frame_w, min_ratio, max_ratio)
        if status == "too_far":
            liveness_dict = {"label": "Move closer", "score": round(p, 3), "raw_score": round(p, 3)}
            return True, False, liveness_dict, bbox
        elif status == "too_close":
            liveness_dict = {"label": "Move back", "score": round(p, 3), "raw_score": round(p, 3)}
            return True, False, liveness_dict, bbox

    # Gate passed or disabled -> update temporal smoother
    state, mean = smoother.update(t, p, bbox)
    score = mean if mean is not None else p
    is_real = (state == "Real")
    liveness_dict = {"label": state, "score": round(score, 3), "raw_score": round(p, 3)}
    return True, is_real, liveness_dict, bbox


# ── Attendance pipeline ────────────────────────────────────────────────────────

def process_attendance_frame(
    frame_bgr: np.ndarray,
    smoother: Optional[LivenessSmoother] = None,
    t: Optional[float] = None,
) -> dict:
    """
    Full pipeline for attendance mode.
    Returns structured result dict.
    """
    result = {
        "face_detected": False,
        "bbox": None,
        "liveness": None,
        "emotion": None,
        "recognition": None,
        "head_pose": None,
        "lighting_enhancement": None,
    }

    live = _get_liveness()

    if smoother is None:
        is_real, live_score, live_label, bbox = live.check(frame_bgr)
        if live_label == "No Face":
            return result
        result["face_detected"] = True
        result["bbox"] = bbox
        result["liveness"] = {"label": live_label, "score": round(live_score, 3)}
    else:
        face_detected, is_real, liveness_dict, bbox = _liveness_decision(live, frame_bgr, smoother, t)
        if not face_detected:
            return result
        result["face_detected"] = True
        result["bbox"] = bbox
        result["liveness"] = liveness_dict

    # Head pose (always)
    pose = _get_head_pose().estimate(frame_bgr)
    if pose:
        result["head_pose"] = pose

    if not is_real:
        return result

    # Emotion
    if bbox:
        face = crop_face_from_bbox(frame_bgr, bbox)
        if face.size > 0:
            inference_face, lighting_metadata = _enhance_inference_crop(face)
            result["lighting_enhancement"] = lighting_metadata
            emo_label, emo_emoji, emo_conf = _get_emotion().predict(inference_face)
            result["emotion"] = {
                "label": emo_label,
                "emoji": emo_emoji,
                "score": round(emo_conf, 3),
            }

            # Recognition uses the selected inference crop, not the audit crop.
            emb = _get_recognizer().extract_embedding(inference_face)
            db_embs = get_all_embeddings()
            match = _get_recognizer().find_match(emb, db_embs)
            result["recognition"] = {
                "matched":     match["matched"],
                "employee_id": match["employee_id"],
                "name":        match["name"],
                "similarity":  round(match["similarity"], 3),
            }

    return result


# ── Registration pipeline ──────────────────────────────────────────────────────

def process_register_frame(
    frame_bgr: np.ndarray,
    smoother: Optional[LivenessSmoother] = None,
    t: Optional[float] = None,
) -> dict:
    """
    Pipeline for registration mode.
    Returns head_pose angle + liveness + face crop (base64) for preview + duplicate face check.
    """
    result = {
        "face_detected": False,
        "bbox": None,
        "liveness": None,
        "head_pose": None,
        "face_preview": None,  # base64 JPEG of face crop
        "ready_to_capture": False,
        "already_registered": False,
        "recognition": None,
        "lighting_enhancement": None,
    }

    live = _get_liveness()

    if smoother is None:
        is_real, live_score, live_label, bbox = live.check(frame_bgr)
        if live_label == "No Face":
            return result
        result["face_detected"] = True
        result["bbox"] = bbox
        result["liveness"] = {"label": live_label, "score": round(live_score, 3)}
    else:
        face_detected, is_real, liveness_dict, bbox = _liveness_decision(live, frame_bgr, smoother, t)
        if not face_detected:
            return result
        result["face_detected"] = True
        result["bbox"] = bbox
        result["liveness"] = liveness_dict

    pose = _get_head_pose().estimate(frame_bgr)
    if pose:
        result["head_pose"] = pose

    if is_real and bbox:
        face = crop_face_from_bbox(frame_bgr, bbox)
        if face.size > 0:
            inference_face, lighting_metadata = _enhance_inference_crop(face)
            result["lighting_enhancement"] = lighting_metadata
            # Check if this face already exists in DB
            db_embs = get_all_embeddings()
            if db_embs:
                emb = _get_recognizer().extract_embedding(inference_face)
                match = _get_recognizer().find_match(emb, db_embs)
                result["recognition"] = {
                    "matched": match["matched"],
                    "employee_id": match["employee_id"],
                    "name": match["name"],
                    "similarity": round(match["similarity"], 3),
                }
                if match["matched"]:
                    result["already_registered"] = True
                    result["ready_to_capture"] = False
                else:
                    result["already_registered"] = False
                    result["ready_to_capture"] = True
            else:
                result["already_registered"] = False
                result["ready_to_capture"] = True

            # Encode face preview as base64 JPEG
            _, buf = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 80])
            result["face_preview"] = base64.b64encode(buf).decode()

    return result



def extract_embedding_from_b64(b64_face: str) -> np.ndarray | None:
    """Extract embedding from a base64 face image."""
    if "," in b64_face:
        b64_face = b64_face.split(",", 1)[1]
    raw  = base64.b64decode(b64_face)
    arr  = np.frombuffer(raw, dtype=np.uint8)
    face = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if face is None:
        return None
    return _get_recognizer().extract_embedding(face)
