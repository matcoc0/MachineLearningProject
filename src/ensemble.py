from __future__ import annotations

import numpy as np
from deap import tools


def _safe_outputs(outputs: np.ndarray) -> np.ndarray:
    outputs = np.asarray(outputs)
    outputs = np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
    return outputs


def _compile_and_predict(individual, toolbox, x: np.ndarray) -> np.ndarray:
    func = toolbox.compile(expr=individual)
    outputs = func(*x.T)
    return _safe_outputs(outputs)


def _soft_voting(raw_scores: list[np.ndarray]) -> np.ndarray:
    scores = np.mean(np.vstack(raw_scores), axis=0)
    return (scores > 0).astype(int)


def _hard_voting(raw_scores: list[np.ndarray]) -> np.ndarray:
    votes = [(s > 0).astype(int) for s in raw_scores]
    votes = np.vstack(votes)
    return (np.mean(votes, axis=0) >= 0.5).astype(int)


def _weighted_voting(raw_scores: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(np.sum(weights))
    if s <= 1e-12:
        return _soft_voting(raw_scores)
    weights = weights / s
    scores = np.average(np.vstack(raw_scores), axis=0, weights=weights)
    return (scores > 0).astype(int)


def _diversity_subset(population, max_k: int):
    selected = []
    seen_heights = set()

    for ind in population:
        h = getattr(ind, "height", None)
        if h is None:
            continue
        if h not in seen_heights:
            selected.append(ind)
            seen_heights.add(h)
        if len(selected) >= max_k:
            break

    if len(selected) < max_k:
        for ind in population:
            if ind not in selected:
                selected.append(ind)
            if len(selected) >= max_k:
                break

    return selected


def ensemble_predict(
    population,
    toolbox,
    x: np.ndarray,
    ensemble_size: int,
    voting: str = "soft",
) -> np.ndarray:
    print("[GA+AL+EL] Using final GA population for ensemble inference")
    if len(population) == 0:
        raise ValueError("Population is empty.")

    k = min(int(ensemble_size), len(population))
    if k <= 0:
        raise ValueError("ensemble_size must be >= 1.")

    # start from best individuals
    best_pool = tools.selBest(population, min(len(population), max(k * 5, k)))

    if voting == "diversity":
        selected = _diversity_subset(best_pool, k)
        raw_scores = [_compile_and_predict(ind, toolbox, x) for ind in selected]
        return _soft_voting(raw_scores)

    selected = best_pool[:k]
    raw_scores = [_compile_and_predict(ind, toolbox, x) for ind in selected]

    if voting == "soft":
        return _soft_voting(raw_scores)
    if voting == "hard":
        return _hard_voting(raw_scores)
    if voting == "weighted":
        weights = np.array([float(ind.fitness.values[0]) for ind in selected], dtype=float)
        return _weighted_voting(raw_scores, weights)

    raise ValueError(
        f"Unknown voting strategy '{voting}'. Choose from {'soft','hard','weighted','diversity'}."
    )
