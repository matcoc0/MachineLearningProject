from __future__ import annotations

from typing import List, Dict, Tuple, Optional

from config import Config
from ga import run_ga, build_toolbox
from ensemble import ensemble_predict
from metrics import compute_metrics
from reporting import save_metrics, plot_metrics


# ============================================================
# Helpers
# ============================================================

def _as_prod_rows(results_prod: List[Dict]) -> List[Dict]:
    """
    Mark production results so they can coexist with experiments.
    """
    out = []
    for r in results_prod:
        rr = dict(r)
        rr.setdefault("category", "prod")
        rr.setdefault("experiment_group", "PROD")
        out.append(rr)
    return out


# ============================================================
# Generic GA runner for variants
# ============================================================

def _run_ga_variant_and_eval(
    *,
    name: str,
    dataset,
    config: Config,
    active_learning: bool,
    ga_overrides: Optional[dict] = None,
) -> Tuple[Dict, object]:
    ga_overrides = ga_overrides or {}

    ga_result = run_ga(
        dataset.x_train,
        dataset.y_train,
        population_size=ga_overrides.get(
            "population_size", config.population_size
        ),
        generations=ga_overrides.get(
            "generations", config.generations
        ),
        crossover_prob=ga_overrides.get(
            "crossover_prob", config.crossover_prob
        ),
        mutation_prob=ga_overrides.get(
            "mutation_prob", config.mutation_prob
        ),
        tournament_size=ga_overrides.get(
            "tournament_size", config.tournament_size
        ),
        max_tree_height=ga_overrides.get(
            "max_tree_height", config.max_tree_height
        ),
        seed=config.random_seed,
        active_learning=active_learning,
        al_initial_fraction=ga_overrides.get(
            "al_initial_fraction", config.al_initial_fraction
        ),
        al_samples_per_round=ga_overrides.get(
            "al_samples_per_round", config.al_samples_per_round
        ),
        al_interval=ga_overrides.get(
            "al_interval", config.al_interval
        ),
        al_strategy=ga_overrides.get(
            "al_strategy", config.al_strategy
        ),
    )

    toolbox = build_toolbox(
        dataset.x_train.shape[1],
        config.random_seed,
        ga_overrides.get("max_tree_height", config.max_tree_height),
    )

    preds = ensemble_predict(
        [ga_result.best_individual],
        toolbox,
        dataset.x_test,
        ensemble_size=1,
        voting="soft",
    )

    metrics = compute_metrics(dataset.y_test, preds)
    metrics.update(
        {
            "approach": name,
            "category": "experiment",
            "experiment_group": "GA+AL" if active_learning else "GA",
            "train_time_sec": ga_result.train_time_sec,
            "active_learning": active_learning,
            **ga_overrides,
        }
    )

    return metrics, ga_result


# ============================================================
# GA variants
# ============================================================

def _run_ga_experiments_only_variants(
    config: Config,
    dataset,
) -> List[Dict]:
    """
    GA variants (baseline already evaluated in PROD).
    """
    results = []

    # Variant 1: shallow trees
    m, _ = _run_ga_variant_and_eval(
        name="GA variant ... shallow trees (h=3)",
        dataset=dataset,
        config=config,
        active_learning=False,
        ga_overrides={"max_tree_height": 3},
    )
    results.append(m)

    # Variant 2: larger population
    m, _ = _run_ga_variant_and_eval(
        name="GA variant ... large population (x2)",
        dataset=dataset,
        config=config,
        active_learning=False,
        ga_overrides={"population_size": config.population_size * 2},
    )
    results.append(m)

    return results


# ============================================================
# GA + Active Learning variants
# ============================================================

def _run_ga_al_experiments_only_variants(
    config: Config,
    dataset,
) -> List[Dict]:
    """
    GA + Active Learning variants (baseline already evaluated in PROD).
    """
    results = []

    # Variant 1: aggressive AL
    m, _ = _run_ga_variant_and_eval(
        name="GA+AL variant ... aggressive sampling (x2)",
        dataset=dataset,
        config=config,
        active_learning=True,
        ga_overrides={"al_samples_per_round": config.al_samples_per_round * 2},
    )
    results.append(m)

    # Variant 2: conservative AL
    m, _ = _run_ga_variant_and_eval(
        name="GA+AL variant ... conservative interval (x2)",
        dataset=dataset,
        config=config,
        active_learning=True,
        ga_overrides={"al_interval": config.al_interval * 2},
    )
    results.append(m)

    return results


# ============================================================
# Ensemble variants
# ============================================================

def _run_ensemble_variants_only(
    config: Config,
    dataset,
    ga_al_result,
    toolbox_al,
) -> List[Dict]:
    """
    Ensemble variants (soft voting already evaluated in PROD).
    """
    results = []

    for voting in ["hard", "weighted", "diversity"]:
        preds = ensemble_predict(
            ga_al_result.population,
            toolbox_al,
            dataset.x_test,
            config.ensemble_size,
            voting=voting,
        )

        m = compute_metrics(dataset.y_test, preds)
        m.update(
            {
                "approach": f"GA+AL+EL variant ... {voting}",
                "category": "experiment",
                "experiment_group": "GA+AL+EL",
                "voting": voting,
                "ensemble_size": config.ensemble_size,
            }
        )
        results.append(m)

    return results


# ============================================================
# Public API
# ============================================================

def run_experiments(
    config: Config,
    dataset,
    results_prod: List[Dict],
    ga_al_result,
    toolbox_al,
) -> None:
    """
    Run additional experiments AFTER the production pipeline.

    Results are saved in:
    reports/results/experiments/
    """
    out_dir = config.reports_dir / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    # Keep PROD results
    all_results.extend(_as_prod_rows(results_prod))

    # GA variants
    all_results.extend(
        _run_ga_experiments_only_variants(config, dataset)
    )

    # GA + AL variants
    all_results.extend(
        _run_ga_al_experiments_only_variants(config, dataset)
    )

    # Ensemble variants
    all_results.extend(
        _run_ensemble_variants_only(
            config,
            dataset,
            ga_al_result,
            toolbox_al,
        )
    )

    # Save + plot
    save_metrics(
        all_results,
        out_dir,
        filename_prefix="experiments",
    )
    plot_metrics(
        all_results,
        out_dir,
        filename="experiments_metrics_plot.png",
    )

    print("[EXPERIMENTS] Done ... results saved to:", out_dir)
