"""
metrics.py – verification metrics from genuine and impostor scores
(higher score = more likely the same person). numpy only.
"""
from __future__ import annotations

import numpy as np


def roc(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, FAR, FRR) at every distinct score, thresholds ascending."""
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    thr = np.unique(np.concatenate([g, i]))
    far = np.array([(i >= t).mean() for t in thr])
    frr = np.array([(g < t).mean() for t in thr])
    return thr, far, frr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where FAR and FRR cross."""
    thr, far, frr = roc(genuine, impostor)
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(thr[k])


def auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Probability a random genuine score beats a random impostor score (ties count half)."""
    g, i = np.asarray(genuine, float)[:, None], np.asarray(impostor, float)[None, :]
    return float((g > i).mean() + 0.5 * (g == i).mean())


def rates_at(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> dict[str, float]:
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    return {"threshold": float(threshold), "far": float((i >= threshold).mean()), "frr": float((g < threshold).mean())}
