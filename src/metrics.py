from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _ensure_vector(outputs, n_rows: int) -> np.ndarray:
    outputs = np.asarray(outputs)
    if outputs.ndim == 0:
        outputs = np.full(n_rows, float(outputs))
    return outputs


def predict_with_individual(toolbox, individual, x):
    func = toolbox.compile(expr=individual)
    outputs = func(*x.T)
    outputs = _ensure_vector(outputs, x.shape[0])
    outputs = np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
    return (outputs > 0).astype(int), outputs


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
