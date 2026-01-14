from __future__ import annotations

from typing import Tuple
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Internal utilities

def _ensure_vector(outputs, n_rows: int) -> np.ndarray:
    """
    Ensure GP outputs are a vector of length n_rows.
    Handles scalar outputs from GP trees.
    """
    outputs = np.asarray(outputs)
    if outputs.ndim == 0:
        outputs = np.full(n_rows, float(outputs))
    return outputs


def _sanitize_outputs(outputs: np.ndarray) -> np.ndarray:
    """
    Replace NaN / +/-inf values produced by GP trees.
    """
    return np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)


# Prediction helpers

def predict_with_individual(
    toolbox,
    individual,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with a single GP individual.

    Returns
    -------
    y_pred : np.ndarray
        Binary predictions (0/1)
    scores : np.ndarray
        Continuous raw GP outputs (before thresholding)
    """
    func = toolbox.compile(expr=individual)

    scores = func(*x.T)
    scores = _ensure_vector(scores, x.shape[0])
    scores = _sanitize_outputs(scores)

    y_pred = (scores > 0).astype(int)
    return y_pred, scores


# Metrics

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute standard classification metrics.

    Used in:
    - pipeline.py (production results)
    - experiments.py (comparisons)

    Returns
    -------
    dict
        accuracy, precision, recall, f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_metrics_with_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """
    Extended metrics using continuous scores.

    Intended for experiments / analysis only.
    """
    metrics = compute_metrics(y_true, y_pred)

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics
