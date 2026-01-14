
from dataclasses import dataclass
from typing import Any, Optional
import operator
import random
import time
import copy
from functools import partial

import numpy as np
from deap import base, creator, gp, tools
from sklearn.metrics import f1_score


# Result container
@dataclass
class GAResult:
    """
    Contains GA training results
    """
    population: list
    best_individual: Any
    log: list[dict]
    train_time_sec: float


# Protected operators
def _protected_div(left, right):
    """
    Protected division operator to avoid numerical instabilities
    """
    left = np.asarray(left)
    right = np.asarray(right)
    out = np.ones(np.broadcast(left, right).shape, dtype=float)
    np.divide(left, right, out=out, where=np.abs(right) > 1e-6)
    return out


def _ensure_vector(outputs, n_rows: int) -> np.ndarray:
    """
    Ensure that GP outputs are vectorized
    """
    outputs = np.asarray(outputs)
    if outputs.ndim == 0:
        outputs = np.full(n_rows, float(outputs))
    return outputs


def _sanitize(outputs: np.ndarray) -> np.ndarray:
    return np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)


# Toolbox builder (DEAP GP)
def build_toolbox(n_features: int, seed: int, max_tree_height: int):
    """
    Build and configure the DEAP toolbox for Genetic Programming.
    """
    random.seed(seed)
    np.random.seed(seed)

    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(_protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))

    # Create classes only once (avoid DEAP re-creation errors)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", copy.deepcopy)

    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Limit bloat
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height),
    )

    return toolbox


# Fitness + scoring helpers
def _evaluate_individual(individual, toolbox, data):
    """
    Evaluate a GP individual using the F1-score.
    """
    
    func = toolbox.compile(expr=individual)
    outputs = func(*data["x"].T)
    outputs = _ensure_vector(outputs, data["x"].shape[0])
    outputs = _sanitize(outputs)
    preds = (outputs > 0).astype(int)
    score = f1_score(data["y"], preds, zero_division=0)
    return (score,)


def _raw_scores(individual, toolbox, x: np.ndarray) -> np.ndarray:
    """
    Compute raw (continuous) GP outputs for a given dataset.
    """
    
    func = toolbox.compile(expr=individual)
    outputs = func(*x.T)
    outputs = _ensure_vector(outputs, x.shape[0])
    outputs = _sanitize(outputs)
    return outputs


def _query_indices_uncertainty(
    rng: np.random.Generator,
    individual,
    toolbox,
    pool_x: np.ndarray,
    n_query: int,
) -> np.ndarray:
    """
    Select samples with highest uncertainty for Active Learning.
    """
    scores = _raw_scores(individual, toolbox, pool_x)
    uncertainty = np.abs(scores)
    idx = np.argsort(uncertainty)[:n_query]
    return idx


def _active_learning_step(
    rng: np.random.Generator,
    population: list,
    toolbox,
    data: dict,
    pool_x: np.ndarray,
    pool_y: np.ndarray,
    n_query: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one Active Learning query step.
    """
    if len(pool_x) == 0:
        return pool_x, pool_y

    best = tools.selBest(population, 1)[0]
    n_query = min(n_query, len(pool_x))
    idx = _query_indices_uncertainty(rng, best, toolbox, pool_x, n_query)

    data["x"] = np.vstack([data["x"], pool_x[idx]])
    data["y"] = np.concatenate([data["y"], pool_y[idx]])

    mask = np.ones(len(pool_x), dtype=bool)
    mask[idx] = False
    pool_x = pool_x[mask]
    pool_y = pool_y[mask]
    return pool_x, pool_y


# Main GA runner
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
    model_name: Optional[str] = None, 
) -> GAResult:
    """
    Run a Genetic Programming-based Genetic Algorithm, either for:
    - Standard GA (fully supervised)
    - GA with Active Learning (GA+AL)
    """
    toolbox = build_toolbox(x_train.shape[1], seed, max_tree_height)

    # Use requested tournament size
    toolbox.unregister("select")
    toolbox.register("select", tools.selTournament, tournsize=int(tournament_size))

    rng = np.random.default_rng(seed)

    # Logging name for this run (used by experiments for clarity)
    if model_name is None:
        model_name = "GA+AL" if active_learning else "GA"

    # Labeled set and unlabeled pool (simulated AL: labels exist but are hidden in the pool)
    data = {"x": x_train, "y": y_train}
    pool_x, pool_y = None, None

    if active_learning:
        init_size = max(1, int(len(x_train) * al_initial_fraction))
        idx = rng.permutation(len(x_train))
        data["x"] = x_train[idx[:init_size]]
        data["y"] = y_train[idx[:init_size]]
        pool_x = x_train[idx[init_size:]]
        pool_y = y_train[idx[init_size:]]

    toolbox.register("evaluate", _evaluate_individual, toolbox=toolbox, data=data)

    population = toolbox.population(n=int(population_size))

    # Initial fitness evaluation
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    start = time.time()
    log: list[dict] = []

    for gen in range(1, int(generations) + 1):

        # Active learning sampling every al_interval generations
        if active_learning and pool_x is not None and (gen % int(al_interval) == 0):
            if al_strategy == "uncertainty":
                pool_x, pool_y = _active_learning_step(
                    rng,
                    population,
                    toolbox,
                    data,
                    pool_x,
                    pool_y,
                    int(al_samples_per_round),
                )
            else:
                # fallback to uncertainty if unknown strategy
                pool_x, pool_y = _active_learning_step(
                    rng,
                    population,
                    toolbox,
                    data,
                    pool_x,
                    pool_y,
                    int(al_samples_per_round),
                )

        # Select + clone
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < float(crossover_prob):
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Mutation
        for mut in offspring:
            if rng.random() < float(mutation_prob):
                toolbox.mutate(mut)
                del mut.fitness.values

        # Evaluate invalid
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring

        best = tools.selBest(population, 1)[0]
        log_row = {
            "generation": int(gen),
            "best_f1": float(best.fitness.values[0]),
            "mean_f1": float(np.mean([i.fitness.values[0] for i in population])),
            "train_size": int(len(data["x"])),
        }
        log.append(log_row)

        print(
            f"[{model_name}] Gen {gen:02d} | "
            f"Best F1={log_row['best_f1']:.4f} | "
            f"Train size={log_row['train_size']}"
        )

    train_time = time.time() - start
    best = tools.selBest(population, 1)[0]
    return GAResult(population, best, log, float(train_time))