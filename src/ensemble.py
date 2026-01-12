from __future__ import annotations

import numpy as np
from deap import tools


def ensemble_predict(
    population,
    toolbox,
    x: np.ndarray,
    ensemble_size: int,
    voting: str = "soft",  # "soft" or "hard"
) -> np.ndarray:
    """
    Ensemble prediction using either soft or hard voting.

    Soft voting:
        - Average raw outputs and threshold at 0

    Hard voting:
        - Majority vote over individual binary predictions
    """
    top = tools.selBest(population, min(ensemble_size, len(population)))

    scores = []
    preds = []

    for ind in top:
        func = toolbox.compile(expr=ind)
        outputs = func(*x.T)
        outputs = np.asarray(outputs)
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

        scores.append(outputs)
        preds.append((outputs > 0).astype(int))

    scores = np.vstack(scores)
    preds = np.vstack(preds)

    if voting == "soft":
        avg_scores = np.mean(scores, axis=0)
        return (avg_scores > 0).astype(int)

    elif voting == "hard":
        avg_votes = np.mean(preds, axis=0)
        return (avg_votes >= 0.5).astype(int)

    else:
        raise ValueError(f"Unknown voting strategy: {voting}")
