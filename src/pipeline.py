from __future__ import annotations

import time
from sklearn.metrics import classification_report

from config import Config
from data import load_and_preprocess
from ensemble import ensemble_predict
from ga import build_toolbox, run_ga
from metrics import compute_metrics, predict_with_individual
from reporting import (
    ensure_reports_dir,
    save_log,
    save_metrics,
    plot_metrics,
    plot_ga_history,
)


def run_all(config: Config) -> None:
    ensure_reports_dir(config.reports_dir)

    print("Loading and preprocessing dataset...")
    dataset = load_and_preprocess(
        str(config.data_path), config.test_size, config.random_seed
    )
    print(f"Train size: {len(dataset.x_train)} | Test size: {len(dataset.x_test)}")

    results: list[dict] = []

    # GA baseline (best individual)
    print("\nRunning GA baseline...")
    ga_result = run_ga(
        dataset.x_train,
        dataset.y_train,
        population_size=config.population_size,
        generations=config.generations,
        crossover_prob=config.crossover_prob,
        mutation_prob=config.mutation_prob,
        tournament_size=config.tournament_size,
        max_tree_height=config.max_tree_height,
        seed=config.random_seed,
        active_learning=False,
    )

    toolbox = build_toolbox(
        dataset.x_train.shape[1], config.random_seed, config.max_tree_height
    )

    preds_ga, _ = predict_with_individual(
        toolbox, ga_result.best_individual, dataset.x_test
    )
    metrics_ga = compute_metrics(dataset.y_test, preds_ga)
    metrics_ga.update(
        {
            "approach": "GA (best individual)",
            "train_time_sec": ga_result.train_time_sec,
            "model_type": "best_individual",
        }
    )
    results.append(metrics_ga)

    save_log(ga_result.log, config.reports_dir, "ga_log.csv")
    plot_ga_history(config.reports_dir / "ga_log.csv", config.reports_dir, "ga")

    # GA + Active Learning (best individual)
    print("\nRunning GA + Active Learning...")
    ga_al_result = run_ga(
        dataset.x_train,
        dataset.y_train,
        population_size=config.population_size,
        generations=config.generations,
        crossover_prob=config.crossover_prob,
        mutation_prob=config.mutation_prob,
        tournament_size=config.tournament_size,
        max_tree_height=config.max_tree_height,
        seed=config.random_seed,
        active_learning=True,
        al_initial_fraction=config.al_initial_fraction,
        al_samples_per_round=config.al_samples_per_round,
        al_interval=config.al_interval,
        al_strategy=config.al_strategy,
    )

    toolbox_al = build_toolbox(
        dataset.x_train.shape[1], config.random_seed, config.max_tree_height
    )

    preds_al, _ = predict_with_individual(
        toolbox_al, ga_al_result.best_individual, dataset.x_test
    )
    metrics_al = compute_metrics(dataset.y_test, preds_al)
    metrics_al.update(
        {
            "approach": "GA+AL (best individual)",
            "train_time_sec": ga_al_result.train_time_sec,
            "model_type": "best_individual",
        }
    )
    results.append(metrics_al)

    save_log(ga_al_result.log, config.reports_dir, "ga_al_log.csv")
    plot_ga_history(
        config.reports_dir / "ga_al_log.csv", config.reports_dir, "ga_al"
    )

    # Ensemble learning: soft vs hard voting
    print("\nRunning Ensemble (soft voting)...")
    preds_soft = ensemble_predict(
        ga_al_result.population,
        toolbox_al,
        dataset.x_test,
        config.ensemble_size,
        voting="soft",
    )
    metrics_soft = compute_metrics(dataset.y_test, preds_soft)
    metrics_soft.update(
        {
            "approach": "GA+AL+EL (soft voting)",
            "train_time_sec": ga_al_result.train_time_sec,
            "model_type": "ensemble",
            "voting": "soft",
        }
    )
    results.append(metrics_soft)

    print("\nRunning Ensemble (hard voting)...")
    preds_hard = ensemble_predict(
        ga_al_result.population,
        toolbox_al,
        dataset.x_test,
        config.ensemble_size,
        voting="hard",
    )
    metrics_hard = compute_metrics(dataset.y_test, preds_hard)
    metrics_hard.update(
        {
            "approach": "GA+AL+EL (hard voting)",
            "train_time_sec": ga_al_result.train_time_sec,
            "model_type": "ensemble",
            "voting": "hard",
        }
    )
    results.append(metrics_hard)

    # Save global reports
    save_metrics(results, config.reports_dir)
    plot_metrics(results, config.reports_dir)

    print("\nSaved reports to:", config.reports_dir)
