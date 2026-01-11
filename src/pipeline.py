from __future__ import annotations

import time

from sklearn.metrics import classification_report

from config import Config
from data import load_and_preprocess
from ensemble import ensemble_predict
from ga import build_toolbox, run_ga
from metrics import compute_metrics, predict_with_individual
from reporting import ensure_reports_dir, plot_metrics, save_log, save_metrics


def run_all(config: Config) -> None:
    ensure_reports_dir(config.reports_dir)

    print("Loading and preprocessing dataset...")
    dataset = load_and_preprocess(str(config.data_path), config.test_size, config.random_seed)
    print(f"Train size: {len(dataset.x_train)} | Test size: {len(dataset.x_test)}")

    results = []

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
    toolbox = build_toolbox(dataset.x_train.shape[1], config.random_seed, config.max_tree_height)

    start = time.time()
    preds, _ = predict_with_individual(toolbox, ga_result.best_individual, dataset.x_test)
    test_time = time.time() - start
    metrics = compute_metrics(dataset.y_test, preds)
    metrics.update(
        {
            "approach": "GA",
            "train_time_sec": ga_result.train_time_sec,
            "test_time_sec": test_time,
        }
    )
    results.append(metrics)
    save_log(ga_result.log, config.reports_dir, "ga_log.csv")

    print("GA metrics:")
    print(classification_report(dataset.y_test, preds, zero_division=0))

    if config.al_enabled:
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
        toolbox_al = build_toolbox(dataset.x_train.shape[1], config.random_seed, config.max_tree_height)

        start = time.time()
        preds_al, _ = predict_with_individual(toolbox_al, ga_al_result.best_individual, dataset.x_test)
        test_time = time.time() - start
        metrics_al = compute_metrics(dataset.y_test, preds_al)
        metrics_al.update(
            {
                "approach": "GA+AL",
                "train_time_sec": ga_al_result.train_time_sec,
                "test_time_sec": test_time,
            }
        )
        results.append(metrics_al)
        save_log(ga_al_result.log, config.reports_dir, "ga_al_log.csv")

        print("GA+AL metrics:")
        print(classification_report(dataset.y_test, preds_al, zero_division=0))

        print("\nRunning Ensemble (GA+AL+EL)...")
        start = time.time()
        ensemble_preds = ensemble_predict(
            ga_al_result.population,
            toolbox_al,
            dataset.x_test,
            config.ensemble_size,
        )
        test_time = time.time() - start
        metrics_ens = compute_metrics(dataset.y_test, ensemble_preds)
        metrics_ens.update(
            {
                "approach": "GA+AL+EL",
                "train_time_sec": ga_al_result.train_time_sec,
                "test_time_sec": test_time,
                "ensemble_size": config.ensemble_size,
            }
        )
        results.append(metrics_ens)
        save_log([metrics_ens], config.reports_dir, "ga_al_el_log.csv")

        print("GA+AL+EL metrics:")
        print(classification_report(dataset.y_test, ensemble_preds, zero_division=0))

    save_metrics(results, config.reports_dir)
    plot_metrics(results, config.reports_dir)

    print("\nSaved reports to:", config.reports_dir)
