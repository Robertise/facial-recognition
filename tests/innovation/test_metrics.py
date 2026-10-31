"""tests/innovation/test_metrics.py – EER, AUC and FAR/FRR on known score sets."""
import numpy as np

from innovation.eval.metrics import auc, eer, rates_at


def test_perfect_separation():
    g, i = np.array([0.8, 0.9, 0.95]), np.array([0.1, 0.2, 0.3])
    assert eer(g, i)[0] == 0.0
    assert auc(g, i) == 1.0
    assert rates_at(g, i, 0.5) == {"threshold": 0.5, "far": 0.0, "frr": 0.0}


def test_overlap_and_ties():
    g, i = np.array([0.4, 0.6, 0.8, 0.9]), np.array([0.1, 0.5, 0.7, 0.2])
    rate, thr = eer(g, i)
    assert 0.2 <= rate <= 0.3 and 0.5 <= thr <= 0.7
    assert auc(np.array([0.5]), np.array([0.5])) == 0.5
    r = rates_at(g, i, 0.6)
    assert r["far"] == 0.25 and r["frr"] == 0.25
