"""ARES-style P@k and MAP@k metrics for multi-tab predictions."""

from __future__ import annotations

import numpy as np


def precision_and_map_at_k(
    y_true: np.ndarray, y_pred_score: np.ndarray, k: int
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(y_pred_score, dtype=np.float64)
    if y_true.ndim != 2 or scores.ndim != 2 or y_true.shape != scores.shape:
        raise ValueError("y_true and y_pred_score must be equal two-dimensional arrays")
    if not len(y_true):
        raise ValueError("metric inputs must not be empty")
    if not 1 <= k <= y_true.shape[1]:
        raise ValueError(f"k must be in [1, {y_true.shape[1]}]")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must be multi-hot")
    if np.any(y_true.sum(axis=1) != k):
        raise ValueError(f"every target must contain exactly {k} positive labels")
    if not np.all(np.isfinite(scores)):
        raise ValueError("prediction scores must be finite")

    top_k = np.argsort(scores, axis=1, kind="stable")[:, -k:][:, ::-1]
    hits = np.take_along_axis(y_true, top_k, axis=1)
    true_positives = np.cumsum(hits, axis=1).sum(axis=0)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    precision_by_rank = true_positives / (len(y_true) * ranks)
    return {
        f"p@{k}": round(float(precision_by_rank[-1]), 4) * 100,
        f"map@{k}": round(float(precision_by_rank.mean()), 4) * 100,
    }
