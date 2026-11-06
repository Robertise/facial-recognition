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
    compute_s1,
    compute_s2,
    compute_s3_signatures_for_window,
    make_leave_one_out_gallery,
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
