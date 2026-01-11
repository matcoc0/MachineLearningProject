from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import operator
import random
from functools import partial
import time

import numpy as np
from deap import base, creator, gp, tools
from sklearn.metrics import f1_score


@dataclass
class GAResult:
    population: list
    best_individual: Any
    log: list[dict]
    train_time_sec: float


def _protected_div(left, right):
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    out_shape = np.broadcast(left_arr, right_arr).shape
    safe = np.ones(out_shape, dtype=float)
    np.divide(left_arr, right_arr, out=safe, where=np.abs(right_arr) > 1e-6)
    return safe


def _ensure_vector(outputs, n_rows: int) -> np.ndarray:
    outputs = np.asarray(outputs)
    if outputs.ndim == 0:
        outputs = np.full(n_rows, float(outputs))
    return outputs


def build_toolbox(n_features: int, seed: int, max_tree_height: int):
    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(_protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))

    random.seed(seed)
    np.random.seed(seed)
    return toolbox


def _evaluate_individual(individual, toolbox, data):
    func = toolbox.compile(expr=individual)
    outputs = func(*data["x"].T)
    outputs = _ensure_vector(outputs, data["x"].shape[0])
    outputs = np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
    preds = (outputs > 0).astype(int)
    score = f1_score(data["y"], preds, zero_division=0)
    return (score,)


def _active_sample(pool_x, pool_y, n_samples, strategy, scorer, seed):
    if len(pool_x) == 0:
        return pool_x, pool_y, np.empty((0, pool_x.shape[1])), np.empty((0,), dtype=int)

    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(pool_x))

    if strategy == "random":
        idx = rng.choice(len(pool_x), n_samples, replace=False)
    else:
        scores = scorer(pool_x)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.argsort(np.abs(scores))[:n_samples]

    mask = np.ones(len(pool_x), dtype=bool)
    mask[idx] = False
    new_x = pool_x[idx]
    new_y = pool_y[idx]
    return pool_x[mask], pool_y[mask], new_x, new_y


def run_ga(
    x_train: np.ndarray,
    y_train: np.ndarray,
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
    toolbox.unregister("select")
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    data = {"x": x_train, "y": y_train}
    pool_x = None
    pool_y = None
    rng = np.random.default_rng(seed)

    if active_learning:
        initial_size = max(1, int(len(x_train) * al_initial_fraction))
        indices = rng.permutation(len(x_train))
        labeled_idx = indices[:initial_size]
        pool_idx = indices[initial_size:]
        data["x"] = x_train[labeled_idx]
        data["y"] = y_train[labeled_idx]
        pool_x = x_train[pool_idx]
        pool_y = y_train[pool_idx]

    toolbox.register("evaluate", _evaluate_individual, toolbox=toolbox, data=data)

    population = toolbox.population(n=population_size)
    start = time.time()

    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    log = []
    for gen in range(1, generations + 1):
        print(f"Generation {gen}/{generations}", end="\r")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if rng.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        best = tools.selBest(population, 1)[0]

        if active_learning and pool_x is not None and len(pool_x) > 0 and gen % al_interval == 0:
            scorer = lambda px: toolbox.compile(expr=best)(*px.T)
            pool_x, pool_y, new_x, new_y = _active_sample(
                pool_x,
                pool_y,
                al_samples_per_round,
                al_strategy,
                scorer,
                seed + gen,
            )
            if len(new_x) > 0:
                data["x"] = np.vstack([data["x"], new_x])
                data["y"] = np.concatenate([data["y"], new_y])

        log.append(
            {
                "generation": gen,
                "best_f1": float(best.fitness.values[0]),
                "train_size": int(len(data["x"])),
            }
        )

    train_time = time.time() - start
    best = tools.selBest(population, 1)[0]
    return GAResult(population=population, best_individual=best, log=log, train_time_sec=train_time)
