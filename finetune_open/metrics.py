"""Holmes-style r-precision/recall metrics for Open-World evaluation."""

from __future__ import annotations

import numpy as np


def threshold_curve(
    targets: np.ndarray,
    monitored_predictions: np.ndarray,
    monitored_confidence: np.ndarray,
    open_world_index: int,
    thresholds: np.ndarray,
    ratio: float = 20.0,
) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(monitored_predictions, dtype=np.int64)
    confidence = np.asarray(monitored_confidence, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if not (targets.shape == predictions.shape == confidence.shape):
        raise ValueError("targets, predictions, and confidence must have equal shapes")
    if targets.ndim != 1 or thresholds.ndim != 1:
        raise ValueError("metric inputs must be one-dimensional")
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("monitored confidence must be finite and within [0, 1]")
    if not np.all(np.diff(thresholds) >= 0) or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("thresholds must be sorted and within [0, 1]")
    if ratio <= 0:
        raise ValueError("r must be positive")

    monitored = targets != open_world_index
    unmonitored = ~monitored
    monitored_total = int(monitored.sum())
    unmonitored_total = int(unmonitored.sum())
    if not monitored_total or not unmonitored_total:
        raise ValueError("both monitored and unmonitored test traces are required")

    true_positive = np.empty(len(thresholds), dtype=np.int64)
    wrong_positive = np.empty(len(thresholds), dtype=np.int64)
    false_positive = np.empty(len(thresholds), dtype=np.int64)
    for index, threshold in enumerate(thresholds):
        accepted = confidence >= threshold
        true_positive[index] = np.sum(
            accepted & monitored & (predictions == targets)
        )
        wrong_positive[index] = np.sum(
            accepted & monitored & (predictions != targets)
        )
        false_positive[index] = np.sum(accepted & unmonitored)

    tpr = true_positive / monitored_total
    wpr = wrong_positive / monitored_total
    fpr = false_positive / unmonitored_total
    denominator = tpr + wpr + ratio * fpr
    r_precision = np.divide(
        tpr,
        denominator,
        out=np.zeros_like(tpr, dtype=np.float64),
        where=denominator > 0,
    )
    return {
        "threshold": thresholds,
        "r_precision": r_precision,
        "recall": tpr,
        "tpr": tpr,
        "wpr": wpr,
        "fpr": fpr,
        "true_positive": true_positive,
        "wrong_positive": wrong_positive,
        "false_positive": false_positive,
        "monitored_total": monitored_total,
        "unmonitored_total": unmonitored_total,
        "r": float(ratio),
    }


def serializable_curve(curve: dict) -> dict:
    result = {}
    for key, value in curve.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result
