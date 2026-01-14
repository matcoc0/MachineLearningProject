
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
    plot_training_time_comparison
)

def run_all(config: Config) -> None:
    ensure_reports_dir(config.reports_dir)
    ensure_subdirs(config.reports_dir)

    if config.run_eda:
        run_eda(
            data_path=str(config.data_path),
            reports_dir=config.reports_dir,
            seed=config.random_seed,
            test_size=config.test_size,
        )

    dataset = load_and_preprocess(
        str(config.data_path),
        config.test_size,
        config.random_seed,
    )

    results = []

    print("\nRunning GA...")
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
    )

    save_log(ga_result.log, config.reports_dir / "training", "ga_log.csv")
    plot_ga_history(
        config.reports_dir / "training" / "ga_log.csv",
        config.reports_dir / "training",
        "ga",
    )

    toolbox = build_toolbox(dataset.x_train.shape[1], config.random_seed, config.max_tree_height)
    preds_ga, _ = predict_with_individual(toolbox, ga_result.best_individual, dataset.x_test)
    print("\n[GA] Classification report (test set)")
    
    print(
        classification_report(
            dataset.y_test,
            preds_ga,
            digits=4,
            zero_division=0,
        )
    )
    results.append(
        {
            **compute_metrics(dataset.y_test, preds_ga),
            "approach": "GA",
            "train_time_sec": ga_result.train_time_sec,
        }
    )

    print("\nRunning GA + AL...")
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

    save_log(ga_al_result.log, config.reports_dir / "training", "ga_al_log.csv")
    plot_ga_history(
        config.reports_dir / "training" / "ga_al_log.csv",
        config.reports_dir / "training",
        "ga_al",
    )

    toolbox_al = build_toolbox(dataset.x_train.shape[1], config.random_seed, config.max_tree_height)
    preds_al, _ = predict_with_individual(toolbox_al, ga_al_result.best_individual, dataset.x_test)

    print("\n[GA+AL] Classification report (test set)")
    print(
        classification_report(
            dataset.y_test,
            preds_al,
            digits=4,
            zero_division=0,
        )
    )


    results.append(
        {
            **compute_metrics(dataset.y_test, preds_al),
            "approach": "GA+AL",
            "train_time_sec": ga_al_result.train_time_sec,
        }
    )

    print("\n[GA+AL+EL] Building ensemble from final population (last generation)")
    print("[GA+AL+EL] Evaluating ensemble on test set")

    preds_soft = ensemble_predict(
        ga_al_result.population,
        toolbox_al,
        dataset.x_test,
        config.ensemble_size,
        voting="soft",
    )

    print("\n[GA+AL] Classification report (test set)")
    print(
        classification_report(
            dataset.y_test,
            preds_soft,
            digits=4,
            zero_division=0,
        )
    )


    results.append(
        {
            **compute_metrics(dataset.y_test, preds_soft),
            "approach": "GA+AL+EL (soft)",
            "train_time_sec": ga_al_result.train_time_sec,
        }
    )

    save_metrics(results, config.reports_dir / "results")
    plot_metrics(results, config.reports_dir / "results")

    plot_training_time_comparison(
        results,
        config.reports_dir / "results" / "training_time_comparison.png",
    )
    
    print("\nReports saved in the reports folder.")
