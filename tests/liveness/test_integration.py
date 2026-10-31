# -*- coding: utf-8 -*-
"""
tests/liveness/test_integration.py
==================================
Integration unit tests for FAS temporal smoothing and distance gating
in backend pipelines (process_attendance_frame, process_register_frame).

All tests use mocks for AI models (LivenessDetector, EmotionDetector,
FaceRecognizer, HeadPoseEstimator) to ensure fast, deterministic execution
without loading real weights or hardware dependencies.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from liveness.temporal import LivenessSmoother, make_smoother
import backend.processing as processing


# ── 1. make_smoother factory tests ───────────────────────────────────────────

def test_make_smoother_none_and_empty():
    assert make_smoother(None) is None
    assert make_smoother({}) is None
    assert make_smoother({"other_key": 123}) is None


def test_make_smoother_disabled():
    cfg = {
        "smoothing": {
            "enabled": False,
            "window_s": 1.5,
            "t_on": 0.20,
            "t_off": 0.05,
        }
    }
    assert make_smoother(cfg) is None


def test_make_smoother_enabled_with_params():
    cfg = {
        "smoothing": {
            "enabled": True,
            "window_s": 1.5,
            "t_on": 0.20,
            "t_off": 0.05,
            "min_frames": 3,
            "reset_s": 1.0,
            "jump_ratio": 0.5,
        }
    }
    smoother = make_smoother(cfg)
    assert smoother is not None
    assert isinstance(smoother, LivenessSmoother)
    assert smoother.window_s == 1.5
    assert smoother.t_on == 0.20
    assert smoother.t_off == 0.05
    assert smoother.min_frames == 3
    assert smoother.reset_s == 1.0
    assert smoother.jump_ratio == 0.5


# ── 2. Backward compatibility (smoother=None) ────────────────────────────────

def test_smoother_none_backward_compatible_face_detected(monkeypatch):
    """When smoother=None, output dict must strictly match original behavior (no raw_score key)."""
    mock_live = MagicMock()
    mock_live.check.return_value = (True, 0.95, "Real", [50, 50, 150, 150])
    mock_live.config = {}

    mock_pose = MagicMock()
    mock_pose.estimate.return_value = {"yaw": 1.2, "pitch": -0.5, "roll": 0.1}

    mock_emo = MagicMock()
    mock_emo.predict.return_value = ("Happy", "😊", 0.98)

    mock_rec = MagicMock()
    mock_rec.extract_embedding.return_value = np.zeros(128)
    mock_rec.find_match.return_value = {
        "matched": True,
        "employee_id": "EMP01",
        "name": "Alice",
        "similarity": 0.88,
    }

    monkeypatch.setattr(processing, "_get_liveness", lambda: mock_live)
    monkeypatch.setattr(processing, "_get_head_pose", lambda: mock_pose)
    monkeypatch.setattr(processing, "_get_emotion", lambda: mock_emo)
    monkeypatch.setattr(processing, "_get_recognizer", lambda: mock_rec)
    monkeypatch.setattr(processing, "get_all_embeddings", lambda: {"EMP01": [np.zeros(128)]})
    monkeypatch.setattr(processing, "_enhance_inference_crop", lambda face: (face, {"applied": False}))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = processing.process_attendance_frame(frame, smoother=None)

    assert res["face_detected"] is True
    assert res["bbox"] == [50, 50, 150, 150]
    # Backward compatibility: exact dict with only "label" and "score"
    assert res["liveness"] == {"label": "Real", "score": 0.95}
    assert "raw_score" not in res["liveness"]
    assert res["head_pose"] == {"yaw": 1.2, "pitch": -0.5, "roll": 0.1}
    assert res["emotion"] == {"label": "Happy", "emoji": "😊", "score": 0.98}
    assert res["recognition"] == {
        "matched": True,
        "employee_id": "EMP01",
        "name": "Alice",
        "similarity": 0.88,
    }


def test_smoother_none_backward_compatible_no_face(monkeypatch):
    """When smoother=None and detector returns 'No Face', result has face_detected=False."""
    mock_live = MagicMock()
    mock_live.check.return_value = (False, 0.0, "No Face", None)
    mock_live.config = {}

    monkeypatch.setattr(processing, "_get_liveness", lambda: mock_live)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = processing.process_attendance_frame(frame, smoother=None)

    assert res["face_detected"] is False
    assert res["bbox"] is None
    assert res["liveness"] is None
    assert res["emotion"] is None
    assert res["recognition"] is None


# ── 3. Temporal smoothing stream, hysteresis, downstream gating ──────────────

def test_stream_temporal_hysteresis_and_downstream_calls(monkeypatch):
    """
    Synthetic frame stream:
    - High p continuously: Checking -> Checking -> Real.
    - Emotion/recognition ONLY called when Real.
    - Single p=0 dip during Real: hysteresis retains Real state, downstream called.
    - Sustained low p: flips to Fake, downstream NOT called.
    """
    mock_live = MagicMock()
    mock_live.config = {
        "gate": {"enabled": True, "min_ratio": 0.22, "max_ratio": 0.70}
    }

    mock_pose = MagicMock()
    mock_pose.estimate.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    mock_emo = MagicMock()
    mock_emo.predict.return_value = ("Neutral", "😐", 0.95)

    mock_rec = MagicMock()
    mock_rec.extract_embedding.return_value = np.zeros(128)
    mock_rec.find_match.return_value = {
        "matched": True,
        "employee_id": "EMP01",
        "name": "Alice",
        "similarity": 0.85,
    }

    monkeypatch.setattr(processing, "_get_liveness", lambda: mock_live)
    monkeypatch.setattr(processing, "_get_head_pose", lambda: mock_pose)
    monkeypatch.setattr(processing, "_get_emotion", lambda: mock_emo)
    monkeypatch.setattr(processing, "_get_recognizer", lambda: mock_rec)
    monkeypatch.setattr(processing, "get_all_embeddings", lambda: {"EMP01": [np.zeros(128)]})
    monkeypatch.setattr(processing, "_enhance_inference_crop", lambda face: (face, {"applied": False}))

    # Frame 640w, bbox width 200 -> ratio = 200/640 = 0.3125 (within gate [0.22, 0.70])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    valid_bbox = [100, 100, 200, 200]

    smoother = LivenessSmoother(
        window_s=1.5,
        t_on=0.10,
        t_off=0.05,
        min_frames=3,
        reset_s=1.0,
        jump_ratio=0.5,
    )

    # Frame 1 (t=0.0): p=0.9 -> 1 frame (< min_frames=3) -> Checking
    mock_live.check.return_value = (True, 0.90, "Real", valid_bbox)
    res1 = processing.process_attendance_frame(frame, smoother, t=0.0)
    assert res1["liveness"]["label"] == "Checking"
    assert res1["liveness"]["raw_score"] == 0.90
    assert mock_emo.predict.call_count == 0
    assert mock_rec.find_match.call_count == 0

    # Frame 2 (t=0.1): p=0.9 -> 2 frames (< min_frames=3) -> Checking
    res2 = processing.process_attendance_frame(frame, smoother, t=0.1)
    assert res2["liveness"]["label"] == "Checking"
    assert mock_emo.predict.call_count == 0
    assert mock_rec.find_match.call_count == 0

    # Frame 3 (t=0.2): p=0.9 -> 3 frames (>= 3, mean=0.9 >= t_on=0.10) -> Real!
    res3 = processing.process_attendance_frame(frame, smoother, t=0.2)
    assert res3["liveness"]["label"] == "Real"
    assert res3["liveness"]["score"] == 0.90
    assert res3["liveness"]["raw_score"] == 0.90
    assert mock_emo.predict.call_count == 1
    assert mock_rec.find_match.call_count == 1

    # Frame 4 (t=0.3): Single isolated p=0.0 dip -> mean=(0.9+0.9+0.9+0.0)/4 = 0.675 >= t_off(0.05) -> Real!
    mock_live.check.return_value = (False, 0.0, "Fake", valid_bbox)
    res4 = processing.process_attendance_frame(frame, smoother, t=0.3)
    assert res4["liveness"]["label"] == "Real"
    assert res4["liveness"]["raw_score"] == 0.0
    assert res4["liveness"]["score"] == 0.675
    assert mock_emo.predict.call_count == 2
    assert mock_rec.find_match.call_count == 2

    # Sustained low confidence frames past window_s (1.5s): older 0.9 frames slide out
    for t_step in [0.5, 0.8, 1.1, 1.4, 1.7, 1.9]:
        mock_live.check.return_value = (False, 0.0, "Fake", valid_bbox)
        processing.process_attendance_frame(frame, smoother, t=t_step)

    # Frame at t=2.0: all frames in window are 0.0 -> mean=0.0 < t_off=0.05 -> Fake!
    call_count_before_fake = mock_emo.predict.call_count
    res_fake = processing.process_attendance_frame(frame, smoother, t=2.0)
    assert res_fake["liveness"]["label"] == "Fake"
    assert res_fake["liveness"]["score"] == 0.0
    # Downstream was NOT called on this fake frame
    assert mock_emo.predict.call_count == call_count_before_fake
    assert mock_rec.find_match.call_count == call_count_before_fake


# ── 4. Distance gate tests ───────────────────────────────────────────────────

def test_distance_gate_too_far_and_too_close(monkeypatch):
    """
    When distance gate is enabled:
    - Bbox ratio < 0.22 -> label "Move closer", is_real=False, smoother NOT updated.
    - Bbox ratio > 0.70 -> label "Move back", is_real=False, smoother NOT updated.
    - Downstream emotion/recognition NOT called.
    """
    mock_live = MagicMock()
    mock_live.config = {
        "gate": {"enabled": True, "min_ratio": 0.22, "max_ratio": 0.70}
    }

    mock_pose = MagicMock()
    mock_pose.estimate.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    mock_emo = MagicMock()
    mock_rec = MagicMock()

    monkeypatch.setattr(processing, "_get_liveness", lambda: mock_live)
    monkeypatch.setattr(processing, "_get_head_pose", lambda: mock_pose)
    monkeypatch.setattr(processing, "_get_emotion", lambda: mock_emo)
    monkeypatch.setattr(processing, "_get_recognizer", lambda: mock_rec)

    # Frame width 1000
    frame = np.zeros((480, 1000, 3), dtype=np.uint8)

    smoother = LivenessSmoother(window_s=1.5, t_on=0.10, t_off=0.05, min_frames=3)

    # Case A: Too far (bbox width 150 -> 150/1000 = 0.15 < 0.22)
    mock_live.check.return_value = (True, 0.92, "Real", [100, 100, 150, 150])
    res_far = processing.process_attendance_frame(frame, smoother, t=0.0)

    assert res_far["face_detected"] is True
    assert res_far["liveness"]["label"] == "Move closer"
    assert res_far["liveness"]["score"] == 0.92
    assert res_far["liveness"]["raw_score"] == 0.92
    assert res_far["head_pose"] is not None
    assert mock_emo.predict.call_count == 0
    assert len(smoother.history) == 0  # Smoother must NOT receive the frame

    # Case B: Too close (bbox width 750 -> 750/1000 = 0.75 > 0.70)
    mock_live.check.return_value = (True, 0.95, "Real", [100, 100, 750, 750])
    res_close = processing.process_attendance_frame(frame, smoother, t=0.1)

    assert res_close["face_detected"] is True
    assert res_close["liveness"]["label"] == "Move back"
    assert res_close["liveness"]["score"] == 0.95
    assert res_close["liveness"]["raw_score"] == 0.95
    assert res_close["head_pose"] is not None
    assert mock_emo.predict.call_count == 0
    assert len(smoother.history) == 0  # Smoother must NOT receive the frame


# ── 5. No Face timeout reset ─────────────────────────────────────────────────

def test_no_face_timeout_reset(monkeypatch):
    """
    When face disappears ('No Face') for > reset_s (1.0s),
    the smoother resets so that the next detected face starts from 'Checking'.
    """
    mock_live = MagicMock()
    mock_live.config = {"gate": {"enabled": True, "min_ratio": 0.22, "max_ratio": 0.70}}

    mock_pose = MagicMock()
    mock_pose.estimate.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    mock_emo = MagicMock()
    mock_emo.predict.return_value = ("Neutral", "😐", 0.95)
    mock_rec = MagicMock()
    mock_rec.extract_embedding.return_value = np.zeros(128)
    mock_rec.find_match.return_value = {"matched": False, "employee_id": None, "name": None, "similarity": 0.0}

    monkeypatch.setattr(processing, "_get_liveness", lambda: mock_live)
    monkeypatch.setattr(processing, "_get_head_pose", lambda: mock_pose)
    monkeypatch.setattr(processing, "_get_emotion", lambda: mock_emo)
    monkeypatch.setattr(processing, "_get_recognizer", lambda: mock_rec)
    monkeypatch.setattr(processing, "get_all_embeddings", lambda: {})
    monkeypatch.setattr(processing, "_enhance_inference_crop", lambda face: (face, {"applied": False}))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    valid_bbox = [100, 100, 200, 200]

    smoother = LivenessSmoother(window_s=1.5, t_on=0.10, t_off=0.05, min_frames=3, reset_s=1.0)

    # 1. Establish Real state (3 frames at t=0.0, 0.1, 0.2)
    mock_live.check.return_value = (True, 0.9, "Real", valid_bbox)
    processing.process_attendance_frame(frame, smoother, t=0.0)
    processing.process_attendance_frame(frame, smoother, t=0.1)
    res_real = processing.process_attendance_frame(frame, smoother, t=0.2)
    assert res_real["liveness"]["label"] == "Real"

    # 2. Face absent: "No Face" arrives at t=1.5 (> 1.0s timeout since last face at 0.2)
    mock_live.check.return_value = (False, 0.0, "No Face", None)
    res_noface = processing.process_attendance_frame(frame, smoother, t=1.5)
    assert res_noface["face_detected"] is False
    assert smoother.state == "Checking"
    assert len(smoother.history) == 0

    # 3. Next detected face at t=1.6 must start from Checking (not Real)
    mock_live.check.return_value = (True, 0.9, "Real", valid_bbox)
    res_reappear = processing.process_attendance_frame(frame, smoother, t=1.6)
    assert res_reappear["liveness"]["label"] == "Checking"


# ── 6. Registration mode: ready_to_capture requires Real state ───────────────

def test_process_register_frame_ready_to_capture(monkeypatch):
    """
    In registration pipeline with smoother:
    - Checking -> ready_to_capture=False, face_preview=None.
    - Real -> ready_to_capture=True, face_preview is valid base64 string.
    - Gate distance violation -> ready_to_capture=False.
    - Fake -> ready_to_capture=False.
    """
    mock_live = MagicMock()
    mock_live.config = {"gate": {"enabled": True, "min_ratio": 0.22, "max_ratio": 0.70}}

    mock_pose = MagicMock()
    mock_pose.estimate.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    mock_rec = MagicMock()
    # Unregistered face
    mock_rec.extract_embedding.return_value = np.zeros(128)
    mock_rec.find_match.return_value = {
        "matched": False,
        "employee_id": None,
        "name": None,
        "similarity": 0.1,
    }

    monkeypatch.setattr(processing, "_get_liveness", lambda: mock_live)
    monkeypatch.setattr(processing, "_get_head_pose", lambda: mock_pose)
    monkeypatch.setattr(processing, "_get_recognizer", lambda: mock_rec)
    monkeypatch.setattr(processing, "get_all_embeddings", lambda: {})
    monkeypatch.setattr(processing, "_enhance_inference_crop", lambda face: (face, {"applied": False}))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    valid_bbox = [100, 100, 200, 200]

    smoother = LivenessSmoother(window_s=1.5, t_on=0.10, t_off=0.05, min_frames=3)

    # Frame 1: Checking -> ready_to_capture=False
    mock_live.check.return_value = (True, 0.90, "Real", valid_bbox)
    res1 = processing.process_register_frame(frame, smoother, t=0.0)
    assert res1["liveness"]["label"] == "Checking"
    assert res1["ready_to_capture"] is False
    assert res1["face_preview"] is None

    # Frame 2: Checking -> ready_to_capture=False
    res2 = processing.process_register_frame(frame, smoother, t=0.1)
    assert res2["liveness"]["label"] == "Checking"
    assert res2["ready_to_capture"] is False

    # Frame 3: Real -> ready_to_capture=True, face_preview generated
    res3 = processing.process_register_frame(frame, smoother, t=0.2)
    assert res3["liveness"]["label"] == "Real"
    assert res3["ready_to_capture"] is True
    assert res3["already_registered"] is False
    assert isinstance(res3["face_preview"], str) and len(res3["face_preview"]) > 0

    # Frame with gate too_far: ready_to_capture=False
    mock_live.check.return_value = (True, 0.90, "Real", [100, 100, 50, 50])
    res_gate = processing.process_register_frame(frame, smoother, t=0.3)
    assert res_gate["liveness"]["label"] == "Move closer"
    assert res_gate["ready_to_capture"] is False
