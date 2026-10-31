# -*- coding: utf-8 -*-
"""
tests/test_liveness_stability.py
================================
Unit tests for FAS stability modules:
1. LivenessSmoother (hysteresis, drop-below t_off, timeout reset, jump reset, fps invariance, asserts)
2. face_size_status (boundary checks)
3. face_in_phone (100% enclosed, 50% partial overlap, disjoint, invalid inputs)
4. Decision rules arbitration (PASS, 1-person fail, INSUFFICIENT, priority sorting 1->4)
"""

import pytest
from liveness.temporal import LivenessSmoother
from liveness.gates import face_size_status
from liveness.phone_detector import face_in_phone
from utils.liveness_offline_eval import evaluate_decision_rules


# ── 1. Smoother Tests ─────────────────────────────────────────────────────────

def test_smoother_assertion_invalid_thresholds():
    """Assert t_off <= t_on must raise AssertionError if violated."""
    with pytest.raises(AssertionError):
        LivenessSmoother(t_on=0.10, t_off=0.30)


def test_smoother_hysteresis():
    """Oscillating confidence between t_off and t_on must NOT flip state."""
    smoother = LivenessSmoother(window_s=1.0, t_on=0.30, t_off=0.10, min_frames=3)

    # 1. Start with high confidence -> transitions to 'Real'
    st, _ = smoother.update(0.0, 0.9)
    assert st == "Checking"
    st, _ = smoother.update(0.2, 0.9)
    assert st == "Checking"
    st, _ = smoother.update(0.4, 0.9)
    assert st == "Real"

    # 2. Confidence hovers around 0.20 (between t_off=0.10 and t_on=0.30)
    # Mean stays around 0.20. State should remain 'Real' (hysteresis!)
    st, m = smoother.update(0.6, 0.20)
    assert st == "Real"
    st, m = smoother.update(0.8, 0.18)
    assert st == "Real"
    st, m = smoother.update(1.0, 0.22)
    assert st == "Real"
    st, m = smoother.update(1.2, 0.15)
    assert st == "Real"

    # 3. Drops below t_off (0.10) after older values slide out of window_s (1.0s)
    smoother.update(1.5, 0.05)
    smoother.update(1.8, 0.05)
    st, m = smoother.update(2.1, 0.05)
    assert st == "Fake"

    # 4. Confidence rises back to 0.22 (between t_off=0.10 and t_on=0.30)
    # State should remain 'Fake' until mean >= 0.30
    smoother.update(2.4, 0.22)
    smoother.update(2.7, 0.22)
    st, m = smoother.update(3.0, 0.22)
    assert st == "Fake"


def test_smoother_face_absence_timeout_reset():
    """When face is absent (None) for > reset_s, smoother must reset to Checking."""
    smoother = LivenessSmoother(window_s=1.0, t_on=0.30, t_off=0.10, reset_s=1.0, min_frames=3)

    # Establish Real state
    smoother.update(0.0, 0.9)
    smoother.update(0.2, 0.9)
    st, _ = smoother.update(0.4, 0.9)
    assert st == "Real"

    # Face absent for 0.5s (<= reset_s)
    st, _ = smoother.update(0.9, None)
    # Still within window or in grace period
    assert st in ("Real", "Checking")

    # Face absent for > 1.0s (t=2.0, gap = 2.0 - 0.4 = 1.6s > reset_s)
    st, m = smoother.update(2.0, None)
    assert st == "Checking"
    assert m is None

    # Next detected face should start from Checking
    st, _ = smoother.update(2.1, 0.9)
    assert st == "Checking"


def test_smoother_bbox_spatial_jump_reset():
    """When face center jumps > jump_ratio * face_w, smoother must reset."""
    smoother = LivenessSmoother(window_s=1.0, t_on=0.30, t_off=0.10, min_frames=3, jump_ratio=0.5)

    # Initial stable face
    smoother.update(0.0, 0.9, bbox=[100, 100, 60, 60])
    smoother.update(0.2, 0.9, bbox=[102, 100, 60, 60])
    st, _ = smoother.update(0.4, 0.9, bbox=[101, 101, 60, 60])
    assert st == "Real"

    # Large sudden jump: center moves by 200px (jump > 0.5 * 60 = 30px)
    st, _ = smoother.update(0.6, 0.9, bbox=[300, 100, 60, 60])
    # Jump triggered reset, so with only 1 frame in history it must be Checking
    assert st == "Checking"


def test_smoother_fps_invariance():
    """5 fps and 10 fps streams observing the same signal reach matching states."""
    s_5fps = LivenessSmoother(window_s=1.0, t_on=0.30, t_off=0.10, min_frames=3)
    s_10fps = LivenessSmoother(window_s=1.0, t_on=0.30, t_off=0.10, min_frames=3)

    # Phase 1: High live signal for 1.2s
    for i in range(7):  # 0.0, 0.2, ..., 1.2
        t = i * 0.2
        s_5fps.update(t, 0.85)

    for i in range(13):  # 0.0, 0.1, ..., 1.2
        t = i * 0.1
        s_10fps.update(t, 0.85)

    # Both should be Real
    assert s_5fps.state == "Real"
    assert s_10fps.state == "Real"

    # Phase 2: Low signal for 1.2s
    for i in range(7, 14):
        t = i * 0.2
        s_5fps.update(t, 0.02)

    for i in range(13, 26):
        t = i * 0.1
        s_10fps.update(t, 0.02)

    # Both should be Fake
    assert s_5fps.state == "Fake"
    assert s_10fps.state == "Fake"


# ── 2. Gate Tests ─────────────────────────────────────────────────────────────

def test_face_size_status_boundaries():
    frame_w = 640
    min_r = 0.20  # min_w = 128
    max_r = 0.70  # max_w = 448

    # too far (< min_ratio)
    assert face_size_status([100, 100, 120, 120], frame_w, min_r, max_r) == "too_far"

    # exact boundary min_ratio
    assert face_size_status([100, 100, 128, 128], frame_w, min_r, max_r) == "ok"

    # comfortably in range
    assert face_size_status([100, 100, 250, 250], frame_w, min_r, max_r) == "ok"

    # exact boundary max_ratio
    assert face_size_status([100, 100, 448, 448], frame_w, min_r, max_r) == "ok"

    # too close (> max_ratio)
    assert face_size_status([100, 100, 480, 480], frame_w, min_r, max_r) == "too_close"

    # edge cases
    assert face_size_status(None, frame_w, min_r, max_r) == "too_far"
    assert face_size_status([0, 0, 0, 0], frame_w, min_r, max_r) == "too_far"


# ── 3. Phone Detector Helper Tests ────────────────────────────────────────────

def test_face_in_phone():
    face = [100, 100, 50, 50]  # area = 2500

    # 1. Phone completely contains face (100% overlap >= 80%)
    phone_enclosing = [(80, 80, 100, 100, 0.95)]
    assert face_in_phone(face, phone_enclosing, min_overlap=0.8) is True

    # 2. Phone covers 50% of face horizontally (x: 125..150, y: 100..150 -> 25*50 = 1250 / 2500 = 50% < 80%)
    phone_partial = [(125, 100, 50, 50, 0.90)]
    assert face_in_phone(face, phone_partial, min_overlap=0.8) is False

    # 3. Phone disjoint (0% overlap)
    phone_disjoint = [(300, 300, 80, 80, 0.90)]
    assert face_in_phone(face, phone_disjoint, min_overlap=0.8) is False

    # 4. Invalid inputs
    assert face_in_phone(None, phone_enclosing) is False
    assert face_in_phone(face, []) is False


# ── 4. Decision Rule Arbitration Tests ────────────────────────────────────────

def _make_dummy_records(model, policy, param_str, params, scores_map, n_windows=15):
    """
    scores_map: {person: {scenario: score}}
    """
    records = []
    sc_labels = {
        "real_still": "real",
        "real_motion": "real",
        "spoof_photo_close": "spoof",
        "spoof_photo_far": "spoof",
        "spoof_video": "spoof",
    }
    for person, s_dict in scores_map.items():
        for sc, score in s_dict.items():
            records.append({
                "model": model,
                "policy": policy,
                "param_str": param_str,
                "params": params,
                "person": person,
                "scenario": sc,
                "label": sc_labels[sc],
                "score_name": "real_accept" if sc_labels[sc] == "real" else "spoof_reject",
                "score": score,
                "n_windows": n_windows,
                "flips_per_min": 0.5,
            })
    return records


def test_decision_rules_pass_and_priorities():
    perfect_scores = {
        "user1": {
            "real_still": 1.0,
            "real_motion": 0.98,
            "spoof_photo_close": 0.95,
            "spoof_photo_far": 0.95,
            "spoof_video": 0.95,
        },
        "user2": {
            "real_still": 0.96,
            "real_motion": 0.95,
            "spoof_photo_close": 0.92,
            "spoof_photo_far": 0.93,
            "spoof_video": 0.91,
        },
    }

    # Config A: P1 on run2 (window_s=1.0)
    data_a = _make_dummy_records("run2", "P1", "w=1.0", {"window_s": 1.0}, perfect_scores)
    # Config B: P0 on run2
    data_b = _make_dummy_records("run2", "P0", "none", {}, perfect_scores)
    # Config C: P1 on run3
    data_c = _make_dummy_records("run3", "P1", "w=1.0", {"window_s": 1.0}, perfect_scores)

    combined = data_a + data_b + data_c
    verdict, chosen, ranked = evaluate_decision_rules(combined)

    assert verdict == "PASS"
    assert chosen is not None
    # Priority 1: simpler policy -> P0 chosen over P1!
    assert chosen["policy"] == "P0"


def test_decision_rules_run2_preferred_over_run3():
    perfect_scores = {
        "user1": {
            "real_still": 1.0,
            "real_motion": 0.98,
            "spoof_photo_close": 0.95,
            "spoof_photo_far": 0.95,
            "spoof_video": 0.95,
        }
    }
    # Both P1, same margin
    data_run2 = _make_dummy_records("run2", "P1", "w=1.0", {"window_s": 1.0}, perfect_scores)
    data_run3 = _make_dummy_records("run3", "P1", "w=1.0", {"window_s": 1.0}, perfect_scores)

    verdict, chosen, ranked = evaluate_decision_rules(data_run2 + data_run3)
    assert verdict == "PASS"
    assert chosen["model"] == "run2"


def test_decision_rules_one_person_fail():
    scores = {
        "user1": {
            "real_still": 0.98,
            "real_motion": 0.96,
            "spoof_photo_close": 0.95,
            "spoof_photo_far": 0.95,
            "spoof_video": 0.95,
        },
        "user2": {
            "real_still": 0.80,  # Fails! (Must be >= 0.95)
            "real_motion": 0.95,
            "spoof_photo_close": 0.92,
            "spoof_photo_far": 0.93,
            "spoof_video": 0.91,
        },
    }
    data = _make_dummy_records("run2", "P1", "w=1.0", {"window_s": 1.0}, scores)
    verdict, chosen, ranked = evaluate_decision_rules(data)
    assert verdict == "NO_PASS"


def test_decision_rules_insufficient_data():
    scores = {
        "user1": {
            "real_still": 1.0,
            "real_motion": 1.0,
            "spoof_photo_close": 1.0,
            "spoof_photo_far": 1.0,
            "spoof_video": 1.0,
        }
    }
    # Only 5 windows (< 10 threshold)
    data = _make_dummy_records("run2", "P1", "w=1.0", {"window_s": 1.0}, scores, n_windows=5)
    verdict, chosen, ranked = evaluate_decision_rules(data)
    assert verdict == "INSUFFICIENT"
    assert chosen is not None
    assert chosen["is_pass"] is False
