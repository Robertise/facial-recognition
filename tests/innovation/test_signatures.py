"""
tests/innovation/test_signatures.py - Unit tests for defensive signatures S1/S2/S3.

Tests:
1. 8 horizontal stripes occlusion covers the exact 14 px regions across 112x112 crop.
2. Leave-one-out gallery drops the exact query image from target identity.
3. Sign convention: higher value = more suspect of attack for S1, S2, and S3.
"""
from __future__ import annotations

import numpy as np
import pytest

from innovation.attacks.adv_glasses.signatures import (
    bootstrap_auc_ci,
    build_session_streams,
    compute_operating_points,
    compute_s1,
    compute_s2,
    compute_s3_signatures_for_window,
    evaluate_rule_check,
    make_leave_one_out_gallery,
    merge_galleries,
    occlude_stripes,
)
from innovation.eval.metrics import auc


def test_occlude_stripes_covers_correct_regions():
    """Verify that occlude_stripes splits 112x112 into 8 stripes of 14px each,
    occluding only the target stripe while leaving other rows untouched.
    """
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    for y in range(112):
        img[y, :, :] = y  # row value equals row index

    stripes = occlude_stripes(img, num_stripes=8, fill_val=255)
    assert len(stripes) == 8

    total_covered_rows = 0
    for k, occ in enumerate(stripes):
        assert occ.shape == (112, 112, 3)
        y_start = k * 14
        y_end = (k + 1) * 14
        total_covered_rows += (y_end - y_start)

        # Target stripe must be filled with 255
        assert np.all(occ[y_start:y_end, :, :] == 255)

        # Above target stripe must be identical to original img
        if y_start > 0:
            assert np.array_equal(occ[:y_start, :, :], img[:y_start, :, :])

        # Below target stripe must be identical to original img
        if y_end < 112:
            assert np.array_equal(occ[y_end:, :, :], img[y_end:, :, :])

    assert total_covered_rows == 112


def test_make_leave_one_out_gallery_removes_exact_query():
    """Verify leave-one-out drops exactly the query embedding for target person,
    leaving remaining embeddings and other identities untouched.
    """
    # 3 identities, each with 5 embeddings of 4 dims
    embs_A = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    embs_B = np.ones((5, 4), dtype=np.float32) * 42.0
    embs_C = np.ones((5, 4), dtype=np.float32) * 99.0

    gallery = {
        "id_A": embs_A,
        "id_B": embs_B,
        "id_C": embs_C,
    }

    # Remove query index 2 from id_A
    loo = make_leave_one_out_gallery(gallery, target_id="id_A", query_idx=2)

    assert len(loo["id_A"]) == 4
    # The remaining rows must be rows 0, 1, 3, 4 of embs_A
    expected_A = np.delete(embs_A, 2, axis=0)
    assert np.array_equal(loo["id_A"], expected_A)
    # Row 2 must not be present
    assert not any(np.array_equal(row, embs_A[2]) for row in loo["id_A"])

    # id_B and id_C must remain completely unchanged
    assert len(loo["id_B"]) == 5
    assert np.array_equal(loo["id_B"], embs_B)
    assert len(loo["id_C"]) == 5
    assert np.array_equal(loo["id_C"], embs_C)


def test_make_leave_one_out_gallery_cosine_threshold():
    """Verify leave-one-out drops embedding with max cosine similarity only when >= 0.99."""
    # Normalized embeddings
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((5, 16)).astype(np.float32)
    embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    gallery = {"id_A": embs}

    # Query with exact match to index 3 (cosine == 1.0 >= 0.99)
    query_exact = embs[3].copy()
    loo_exact = make_leave_one_out_gallery(gallery, target_id="id_A", query_emb=query_exact, min_cos=0.99)
    assert loo_exact is not None
    assert len(loo_exact["id_A"]) == 4
    assert not any(np.allclose(row, embs[3], atol=1e-5) for row in loo_exact["id_A"])

    # Query with cosine < 0.99 (perturbed query)
    noise = rng.standard_normal((16,)).astype(np.float32)
    query_mismatch = embs[3] + noise * 0.5
    query_mismatch /= np.linalg.norm(query_mismatch)
    max_sim = float(np.max(embs @ query_mismatch))
    assert max_sim < 0.99

    loo_mismatch = make_leave_one_out_gallery(gallery, target_id="id_A", query_emb=query_mismatch, min_cos=0.99)
    assert loo_mismatch is None



def test_signatures_sign_convention_on_synthetic_data():
    """Verify that all signatures follow the convention: higher value = more suspect of attack.
    Checks S1, S2, and S3 on synthetic distributions and verifies AUC > 0.5.
    """
    # --------------------------------------------------------------------------
    # 1. S1: Residual identity
    # In attack: top1 is victim (e.g. 0.70), top2 is attacker (e.g. 0.55).
    # In real face: top1 is owner (e.g. 0.75), top2 is distant other (e.g. 0.20).
    # --------------------------------------------------------------------------
    s1_att = compute_s1(top1_score=0.70, top2_score=0.55)
    s1_real = compute_s1(top1_score=0.75, top2_score=0.20)

    assert s1_att["s1_second"] > s1_real["s1_second"]
    assert s1_att["s1_gap_inv"] > s1_real["s1_gap_inv"]
    assert s1_att["s1_ratio"] > s1_real["s1_ratio"]

    assert auc(np.array([s1_att["s1_second"]]), np.array([s1_real["s1_second"]])) == 1.0
    assert auc(np.array([s1_att["s1_gap_inv"]]), np.array([s1_real["s1_gap_inv"]])) == 1.0
    assert auc(np.array([s1_att["s1_ratio"]]), np.array([s1_real["s1_ratio"]])) == 1.0

    # --------------------------------------------------------------------------
    # 2. S2: Concentrated evidence
    # In attack: occluding stripe 2 (glasses) drops score heavily (0.70 -> 0.20)
    # and flips top-1 or drops below 0.5, while other stripes drop slightly (0.02).
    # In real face: occluding any stripe causes small, even drops (e.g. 0.08), top-1 stays.
    # --------------------------------------------------------------------------
    class MockRecognizer:
        def __init__(self, drops: list[float], flips: list[bool]):
            self.drops = drops
            self.flips = flips

        def extract_embedding(self, img):
            # Return dummy vector
            return np.zeros(8, dtype=np.float32)

    # Synthetic gallery with victim and attacker
    v_victim = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    v_other = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    syn_gallery = {"victim": v_victim, "other": v_other}

    # We test compute_s2 via custom mock recognizer that returns simulated occluded embeddings
    class MockAttackRecognizer:
        def __init__(self):
            self.preprocess = None  # force fallback to extract_embedding
            self.call_count = 0

        def extract_embedding(self, img):
            k = self.call_count % 8
            self.call_count += 1
            # Stripe 2 occludes glasses -> victim similarity collapses, other wins
            if k == 2:
                # low victim sim (0.2), other has 0.6 (flip!)
                return np.array([0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # small drop on other stripes (victim sim 0.68, other 0.1)
                return np.array([0.68, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    class MockRealRecognizer:
        def __init__(self):
            self.preprocess = None
            self.call_count = 0

        def extract_embedding(self, img):
            self.call_count += 1
            # Even drop across all stripes: victim/owner stays strong (0.67), other is 0.1
            return np.array([0.67, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    face_dummy = np.zeros((112, 112, 3), dtype=np.uint8)

    s2_att = compute_s2(face_dummy, "victim", 0.70, MockAttackRecognizer(), syn_gallery)
    s2_real = compute_s2(face_dummy, "victim", 0.75, MockRealRecognizer(), syn_gallery)

    assert s2_att["s2_maxdrop"] > s2_real["s2_maxdrop"]
    assert s2_att["s2_flips"] > s2_real["s2_flips"]
    assert s2_att["s2_conc"] > s2_real["s2_conc"]

    assert auc(np.array([s2_att["s2_maxdrop"]]), np.array([s2_real["s2_maxdrop"]])) == 1.0
    assert auc(np.array([s2_att["s2_flips"]]), np.array([s2_real["s2_flips"]])) == 1.0
    assert auc(np.array([s2_att["s2_conc"]]), np.array([s2_real["s2_conc"]])) == 1.0

    # --------------------------------------------------------------------------
    # 3. S3: Instability over time
    # In attack stream: top-1 flickers between victim and other, score fluctuates.
    # In real face stream: top-1 is stably owner, score is steady.
    # --------------------------------------------------------------------------
    history_att = [
        {"t": 0.0, "top1_id": "victim", "scores": {"victim": 0.72}},
        {"t": 0.2, "top1_id": "other", "scores": {"victim": 0.35}},
        {"t": 0.4, "top1_id": "victim", "scores": {"victim": 0.68}},
        {"t": 0.6, "top1_id": "other", "scores": {"victim": 0.40}},
        {"t": 0.8, "top1_id": "victim", "scores": {"victim": 0.70}},
    ]
    s3_switch_att, s3_std_att = compute_s3_signatures_for_window(history_att, curr_top1_id="victim")

    history_real = [
        {"t": 0.0, "top1_id": "owner", "scores": {"owner": 0.75}},
        {"t": 0.2, "top1_id": "owner", "scores": {"owner": 0.74}},
        {"t": 0.4, "top1_id": "owner", "scores": {"owner": 0.76}},
        {"t": 0.6, "top1_id": "owner", "scores": {"owner": 0.75}},
        {"t": 0.8, "top1_id": "owner", "scores": {"owner": 0.75}},
    ]
    s3_switch_real, s3_std_real = compute_s3_signatures_for_window(history_real, curr_top1_id="owner")

    assert s3_switch_att > s3_switch_real
    assert s3_std_att > s3_std_real

    assert auc(np.array([s3_switch_att]), np.array([s3_switch_real])) == 1.0
    assert auc(np.array([s3_std_att]), np.array([s3_std_real])) == 1.0


def test_bootstrap_auc_ci():
    """Verify bootstrap CI returns valid range bounding the point AUC."""
    pos = np.array([0.8, 0.9, 0.85, 0.95, 0.88])
    neg = np.array([0.2, 0.3, 0.25, 0.35, 0.28])
    pt_auc, ci = bootstrap_auc_ci(pos, neg, n_boot=500, seed=0)
    assert pt_auc == 1.0
    assert ci == [1.0, 1.0]

    # Overlapping distributions
    rng = np.random.default_rng(42)
    pos = rng.normal(loc=1.0, scale=1.0, size=50)
    neg = rng.normal(loc=0.0, scale=1.0, size=50)
    pt_auc, ci = bootstrap_auc_ci(pos, neg, n_boot=500, seed=0)
    assert pt_auc is not None
    assert ci is not None
    assert 0.5 < pt_auc < 1.0
    assert ci[0] <= pt_auc <= ci[1]



def test_merge_galleries_duplicate_id_raises():
    """Verify that merging extra gallery embeddings raises ValueError on duplicate ID,
    and cleanly merges non-overlapping identities.
    """
    base_gallery = {
        "owner": np.ones((5, 16), dtype=np.float32),
        "victim": np.ones((5, 16), dtype=np.float32) * 2,
    }
    extra_valid = {
        "p01": np.ones((5, 16), dtype=np.float32) * 3,
        "p02": np.ones((5, 16), dtype=np.float32) * 4,
    }

    # Clean merge
    merged = merge_galleries(base_gallery, extra_valid)
    assert set(merged.keys()) == {"owner", "victim", "p01", "p02"}
    assert np.array_equal(merged["p01"], extra_valid["p01"])

    # Duplicate ID with base gallery raises ValueError
    extra_duplicate = {
        "victim": np.ones((5, 16), dtype=np.float32) * 99,
        "p03": np.ones((5, 16), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="Duplicate identities"):
        merge_galleries(base_gallery, extra_duplicate)


def test_build_session_streams_people_negative_only():
    """Verify that when include_attacks=False, build_session_streams only produces
    negative streams (label=0) for the specified target_person and no attack streams.
    """
    c3_assets = {"seen": np.zeros(1), "mask": np.zeros(1), "victim": "victim_c3"}
    r2b_assets = {"seen": np.zeros(1), "mask": np.zeros(1), "victim": "victim_r2b"}

    # Extra people session: negative streams only
    streams = build_session_streams(
        c3_assets=c3_assets,
        r2b_assets=r2b_assets,
        target_person="p01",
        set_prefix="R_p01_s2",
        include_attacks=False,
    )

    assert len(streams) == 6
    for s in streams:
        assert s["label"] == 0, f"Stream {s['tier']} must be negative (label=0)"
        assert s["target_id"] == "p01", f"Target ID must be p01, got {s['target_id']}"
        assert not s["set"].startswith("A_"), f"No attack streams allowed, got {s['set']}"

    tiers = [s["tier"] for s in streams]
    assert tiers == [
        "baseline",
        "baseline_EOT",
        "control_random_C3",
        "control_plain_C3",
        "control_random_R2b",
        "control_plain_R2b",
    ]

    # Baseline and EOT belong to R_p01_s2; controls belong to R_p01_s2_controls
    assert streams[0]["set"] == "R_p01_s2"
    assert streams[1]["set"] == "R_p01_s2"
    assert all(s["set"] == "R_p01_s2_controls" for s in streams[2:])

    # Contrast with owner session (include_attacks=True)
    owner_streams = build_session_streams(
        c3_assets=c3_assets,
        r2b_assets=r2b_assets,
        target_person="owner",
        set_prefix="R_owner",
        include_attacks=True,
    )
    attack_streams = [s for s in owner_streams if s["label"] == 1]
    assert len(attack_streams) == 4
    assert {s["set"] for s in attack_streams} == {"A_C3", "A_C3_EOT", "A_R2b", "A_R2b_EOT"}
    assert all(s["target_id"].startswith("victim") for s in attack_streams)


def test_compute_operating_points_and_rule_check_synthetic():
    """Verify that 95% TPR threshold and FPR per negative group are calculated correctly,
    and rule_check enforces min_auc >= 0.90 and fpr <= 0.10 on primary session.
    """
    # 100 positive rows: 50 C3 and 50 R2b
    # Values spaced linearly from 0.50 to 0.99
    pos_vals = np.linspace(0.50, 0.99, 100)
    rows: list[dict] = []
    for i, v in enumerate(pos_vals):
        set_name = "A_C3" if i < 25 else ("A_C3_EOT" if i < 50 else ("A_R2b" if i < 75 else "A_R2b_EOT"))
        rows.append({
            "set": set_name,
            "s2_maxdrop": float(v),
            "s2_conc": float(v * 0.9),
        })

    # Expected 5th percentile on pos_vals:
    expected_thresh_maxdrop = float(np.percentile(pos_vals, 5.0))

    # Add negative groups
    # R_owner: 40 frames, all low (0.10 - 0.20) -> 0 FP
    for v in np.linspace(0.10, 0.20, 40):
        rows.append({"set": "R_owner", "s2_maxdrop": float(v), "s2_conc": float(v * 0.9)})
    # R_owner_controls: 40 frames, all low -> 0 FP
    for v in np.linspace(0.12, 0.25, 40):
        rows.append({"set": "R_owner_controls", "s2_maxdrop": float(v), "s2_conc": float(v * 0.9)})
    # R_lfw: 20 frames, 2 above threshold -> 2 / 20 = 0.10 FPR
    for i, v in enumerate(np.linspace(0.20, 0.60, 20)):
        rows.append({"set": "R_lfw", "s2_maxdrop": float(v), "s2_conc": float(v * 0.9)})
    # R_p01_s2: 50 frames, all low (0.15 - 0.30) -> 0 FP
    for v in np.linspace(0.15, 0.30, 50):
        rows.append({"set": "R_p01_s2", "s2_maxdrop": float(v), "s2_conc": float(v * 0.9)})
    # R_p01_s2_controls: 50 frames, 2 above threshold -> 2 / 50 = 0.04 FPR
    for i, v in enumerate(np.linspace(0.15, 0.60, 50)):
        rows.append({"set": "R_p01_s2_controls", "s2_maxdrop": float(v), "s2_conc": float(v * 0.9)})
    # R_p01_s1: 50 frames, many high (0.40 - 0.70) -> high FPR
    for v in np.linspace(0.40, 0.70, 50):
        rows.append({"set": "R_p01_s1", "s2_maxdrop": float(v), "s2_conc": float(v * 0.9)})
        rows.append({"set": "R_p01_s1_controls", "s2_maxdrop": float(v), "s2_conc": float(v * 0.9)})

    op = compute_operating_points(rows, people_sessions=["p01/s2", "p01/s1"], tpr_target=0.95)

    assert "s2_maxdrop" in op
    assert "s2_conc" in op
    assert op["s2_maxdrop"]["n_pos"] == 100
    assert abs(op["s2_maxdrop"]["threshold_tpr95"] - expected_thresh_maxdrop) < 1e-3

    # Check FPRs for s2_maxdrop
    assert op["s2_maxdrop"]["R_owner"]["fpr"] == 0.0
    assert op["s2_maxdrop"]["R_owner_controls"]["fpr"] == 0.0
    # p01_s2_all has 100 negative frames (50 s2 + 50 controls)
    assert op["s2_maxdrop"]["R_p01_s2_all"]["n_neg"] == 100
    p01_s2_fpr = op["s2_maxdrop"]["R_p01_s2_all"]["fpr"]
    assert p01_s2_fpr <= 0.10, f"Expected FPR <= 0.10, got {p01_s2_fpr}"

    # p01_s1 has many high values -> FPR > 0.10
    p01_s1_fpr = op["s2_maxdrop"]["R_p01_s1_all"]["fpr"]
    assert p01_s1_fpr > 0.10

    # Test evaluate_rule_check
    mock_eval = {
        "A_C3_vs_R_p01_s2_all": {"s2_maxdrop": {"auc": 0.98}, "s2_conc": {"auc": 0.96}},
        "A_R2b_vs_R_p01_s2_all": {"s2_maxdrop": {"auc": 0.99}, "s2_conc": {"auc": 0.97}},
        "A_C3_vs_R_p01_s1_all": {"s2_maxdrop": {"auc": 0.75}, "s2_conc": {"auc": 0.70}},
        "A_R2b_vs_R_p01_s1_all": {"s2_maxdrop": {"auc": 0.78}, "s2_conc": {"auc": 0.72}},
    }
    mock_recog = {
        "p01/s2": {"recognized_frames": 74, "total_face_frames": 75, "recognition_rate": 0.9867},
        "p01/s1": {"recognized_frames": 40, "total_face_frames": 75, "recognition_rate": 0.5333},
    }

    rc = evaluate_rule_check(
        mock_eval,
        op,
        mock_recog,
        people_sessions=["p01/s2", "p01/s1"],
        primary_session="p01/s2",
    )

    assert rc["primary_session"] == "p01/s2"
    # p01/s2: min_auc = 0.98 >= 0.90 and fpr <= 0.10 -> pass
    assert rc["sessions"]["p01/s2"]["s2_maxdrop"]["min_auc"] == 0.98
    assert rc["sessions"]["p01/s2"]["s2_maxdrop"]["pass"] is True
    assert rc["sessions"]["p01/s2"]["recognized_frames"] == 74
    assert rc["sessions"]["p01/s2"]["total_face_frames"] == 75

    # p01/s1: min_auc = 0.75 < 0.90 and fpr > 0.10 -> fail
    assert rc["sessions"]["p01/s1"]["s2_maxdrop"]["pass"] is False

    # s2_holds_for_other_people is determined by primary_session (p01/s2)
    assert rc["s2_holds_for_other_people"] is True

    # If primary_session was p01/s1, it would be False
    rc_fail = evaluate_rule_check(
        mock_eval,
        op,
        mock_recog,
        people_sessions=["p01/s2", "p01/s1"],
        primary_session="p01/s1",
    )
    assert rc_fail["s2_holds_for_other_people"] is False
