from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple
import operator
import random
import time
import copy
from functools import partial

import numpy as np
from deap import base, creator, gp, tools
from sklearn.metrics import f1_score


# ============================================================
# Result container
# ============================================================

@dataclass
class GAResult:
    population: list
    best_individual: Any
    log: list[dict]
    train_time_sec: float


# ============================================================
# Protected operators
# ============================================================

def _protected_div(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    out = np.ones(np.broadcast(left, right).shape, dtype=float)
    np.divide(left, right, out=out, where=np.abs(right) > 1e-6)
    return out


def _ensure_vector(outputs, n_rows: int) -> np.ndarray:
    outputs = np.asarray(outputs)
    if outputs.ndim == 0:
        outputs = np.full(n_rows, float(outputs))
    return outputs


def _sanitize(outputs: np.ndarray) -> np.ndarray:
    return np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Toolbox builder
# ============================================================

def build_toolbox(
    n_features: int,
    seed: int,
    max_tree_height: int,
):
    random.seed(seed)
    np.random.seed(seed)

    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(_protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))

    # DEAP creators (safe if re-imported)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)

    # IMPORTANT: ensure clone exists (robust across envs)
    toolbox.register("clone", copy.deepcopy)

    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height),
    )

    return toolbox


# ============================================================
# Evaluation
# ============================================================

def _evaluate_individual(individual, toolbox, data):
    func = toolbox.compile(expr=individual)

    outputs = func(*data["x"].T)
    outputs = _ensure_vector(outputs, data["x"].shape[0])
    outputs = _sanitize(outputs)

    preds = (outputs > 0).astype(int)
    score = f1_score(data["y"], preds, zero_division=0)
    return (score,)


# ============================================================
# Active Learning sampler
# ============================================================

def _active_sample(
    pool_x: np.ndarray,
    pool_y: np.ndarray,
    n_samples: int,
    strategy: str,
    scorer: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
    - remaining pool_x, remaining pool_y
    - new_x, new_y (selected samples)
    """
    if pool_x is None or pool_y is None or len(pool_x) == 0:
        # Keep shapes consistent
        return pool_x, pool_y, np.empty((0, 0)), np.empty((0,), dtype=int)

    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(pool_x))

    if strategy == "random":
        idx = rng.choice(len(pool_x), n_samples, replace=False)
    else:
        # Uncertainty: closest to 0 in absolute score
        scores = scorer(pool_x)
        scores = _sanitize(np.asarray(scores))
        idx = np.argsort(np.abs(scores))[:n_samples]

    mask = np.ones(len(pool_x), dtype=bool)
    mask[idx] = False

    new_x = pool_x[idx]
    new_y = pool_y[idx]

    return pool_x[mask], pool_y[mask], new_x, new_y


# ============================================================
# Main GA runner
# ============================================================

def run_ga(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    population_size: int,
    generations: int,
    crossover_prob: float,
    mutation_prob: float,
    tournament_size: int,
    max_tree_height: int,
    seed: int,
    active_learning: bool = False,
    al_initial_fraction: float = 0.1,
    al_samples_per_round: int = 1000,
    al_interval: int = 2,
    al_strategy: str = "uncertainty",
) -> GAResult:
    toolbox = build_toolbox(x_train.shape[1], seed, max_tree_height)

    # Override tournament size dynamically
    toolbox.unregister("select")
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Data split for Active Learning
    # --------------------------------------------------------
    data = {"x": x_train, "y": y_train}
    pool_x, pool_y = None, None

    if active_learning:
        init_size = max(1, int(len(x_train) * al_initial_fraction))
        indices = rng.permutation(len(x_train))

        labeled_idx = indices[:init_size]
        pool_idx = indices[init_size:]

        data["x"] = x_train[labeled_idx]
        data["y"] = y_train[labeled_idx]

        pool_x = x_train[pool_idx]
        pool_y = y_train[pool_idx]

    toolbox.register("evaluate", _evaluate_individual, toolbox=toolbox, data=data)

    population = toolbox.population(n=population_size)

    start = time.time()

    # Initial evaluation
    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    log: list[dict] = []

    # --------------------------------------------------------
    # Evolution loop
    # --------------------------------------------------------
    for gen in range(1, generations + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < crossover_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        for mutant in offspring:
            if rng.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring

        best = tools.selBest(population, 1)[0]
        fitnesses = [ind.fitness.values[0] for ind in population]

        # Active Learning step
        if active_learning and pool_x is not None and pool_y is not None and gen % al_interval == 0:
            best_func = toolbox.compile(expr=best)

            def scorer(px: np.ndarray) -> np.ndarray:
                out = best_func(*px.T)
                out = _ensure_vector(out, px.shape[0])
                return _sanitize(out)

            pool_x, pool_y, new_x, new_y = _active_sample(
                pool_x=pool_x,
                pool_y=pool_y,
                n_samples=al_samples_per_round,
                strategy=al_strategy,
                scorer=scorer,
                seed=seed + gen,
            )

            if new_x.size > 0:
                data["x"] = np.vstack([data["x"], new_x])
                data["y"] = np.concatenate([data["y"], new_y])

        log.append(
            {
                "generation": gen,
                "best_f1": float(best.fitness.values[0]),
                "mean_f1": float(np.mean(fitnesses)),
                "train_size": int(len(data["x"])),
            }
        )

    train_time = time.time() - start
    best = tools.selBest(population, 1)[0]

    return GAResult(
        population=population,
        best_individual=best,
        log=log,
        train_time_sec=train_time,
    )
