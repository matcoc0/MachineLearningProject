from __future__ import annotations

import time
from sklearn.metrics import classification_report

from config import Config
from data import load_and_preprocess
from eda import run_eda
from ensemble import ensemble_predict
from ga import build_toolbox, run_ga
from metrics import compute_metrics, predict_with_individual
from reporting import (
    ensure_reports_dir,
    ensure_subdirs,
    save_log,
    save_metrics,
    plot_metrics,
    plot_ga_history,
)


def run_all(config: Config) -> None:
    # ---------------------------------------------------------
    # Reports directories
    # ---------------------------------------------------------
    ensure_reports_dir(config.reports_dir)
    ensure_subdirs(config.reports_dir)

    # ---------------------------------------------------------
    # Exploratory Data Analysis (EDA)
    # ---------------------------------------------------------
    if config.run_eda:
        print("\nRunning Exploratory Data Analysis (EDA)...")
        run_eda(
            data_path=str(config.data_path),
            reports_dir=config.reports_dir,
            seed=config.random_seed,
            test_size=config.test_size,
        )

    # ---------------------------------------------------------
    # Load & preprocess data
    # ---------------------------------------------------------
    print("\nLoading and preprocessing dataset...")
    dataset = load_and_preprocess(
        str(config.data_path),
        config.test_size,
        config.random_seed,
    )
    print(f"Train size: {len(dataset.x_train)} | Test size: {len(dataset.x_test)}")

    results: list[dict] = []

    # =========================================================
    # 1) GA baseline
    # =========================================================
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
        dataset.x_train.shape[1],
        config.random_seed,
        config.max_tree_height,
    )

    t0 = time.time()
    preds_ga, _ = predict_with_individual(
        toolbox,
        ga_result.best_individual,
        dataset.x_test,
    )
    test_time_ga = time.time() - t0

    metrics_ga = compute_metrics(dataset.y_test, preds_ga)
    metrics_ga.update(
        {
            "approach": "GA",
            "train_time_sec": ga_result.train_time_sec,
            "test_time_sec": test_time_ga,
        }
    )
    results.append(metrics_ga)

    save_log(ga_result.log, config.reports_dir / "training", "ga_log.csv")
    plot_ga_history(
        config.reports_dir / "training" / "ga_log.csv",
        config.reports_dir / "training",
        prefix="ga",
    )

    print("GA results:")
    print(classification_report(dataset.y_test, preds_ga, zero_division=0))

    # =========================================================
    # 2) GA + Active Learning
    # =========================================================
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
        dataset.x_train.shape[1],
        config.random_seed,
        config.max_tree_height,
    )

    t0 = time.time()
    preds_al, _ = predict_with_individual(
        toolbox_al,
        ga_al_result.best_individual,
        dataset.x_test,
    )
    test_time_al = time.time() - t0

    metrics_al = compute_metrics(dataset.y_test, preds_al)
    metrics_al.update(
        {
            "approach": "GA+AL",
            "train_time_sec": ga_al_result.train_time_sec,
            "test_time_sec": test_time_al,
        }
    )
    results.append(metrics_al)

    save_log(ga_al_result.log, config.reports_dir / "training", "ga_al_log.csv")
    plot_ga_history(
        config.reports_dir / "training" / "ga_al_log.csv",
        config.reports_dir / "training",
        prefix="ga_al",
    )

    print("GA+AL results:")
    print(classification_report(dataset.y_test, preds_al, zero_division=0))

    # =========================================================
    # 3) Ensemble (PROD = soft voting only)
    # =========================================================
    print("\nRunning Ensemble (GA+AL+EL) ... soft voting (PROD)...")
    t0 = time.time()
    preds_soft = ensemble_predict(
        ga_al_result.population,
        toolbox_al,
        dataset.x_test,
        config.ensemble_size,
        voting="soft",
    )
    test_time_soft = time.time() - t0

    metrics_soft = compute_metrics(dataset.y_test, preds_soft)
    metrics_soft.update(
        {
            "approach": "GA+AL+EL (soft)",
            "train_time_sec": ga_al_result.train_time_sec,
            "test_time_sec": test_time_soft,
            "ensemble_size": config.ensemble_size,
        }
    )
    results.append(metrics_soft)

    save_log(
        [metrics_soft],
        config.reports_dir / "training",
        "ga_al_el_log.csv",
    )

    print("GA+AL+EL (soft voting) results:")
    print(classification_report(dataset.y_test, preds_soft, zero_division=0))

    # =========================================================
    # Save final PROD results
    # =========================================================
    save_metrics(
        results,
        config.reports_dir / "results",
        filename_prefix=None,
    )
    plot_metrics(
        results,
        config.reports_dir / "results",
        filename=None,
    )

    print("\nSaved reports to:", config.reports_dir)
