from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_reports_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics_rows: list[dict], reports_dir: Path) -> None:
    df = pd.DataFrame(metrics_rows)
    csv_path = reports_dir / "metrics.csv"
    json_path = reports_dir / "metrics.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)


def save_log(log_rows: list[dict], reports_dir: Path, filename: str) -> None:
    df = pd.DataFrame(log_rows)
    df.to_csv(reports_dir / filename, index=False)


def plot_metrics(metrics_rows: list[dict], reports_dir: Path) -> None:
    df = pd.DataFrame(metrics_rows)
    if df.empty:
        return

    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df["approach"]))
    width = 0.2
    for i, metric in enumerate(metrics):
        ax.bar([pos + (i - 1.5) * width for pos in x], df[metric], width=width, label=metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["approach"])
    ax.set_title("Performance Metrics by Approach")
    ax.set_xlabel("Approach")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "metrics_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["approach"], df["train_time_sec"], color="#4c72b0")
    ax.set_title("Training Time by Approach")
    ax.set_xlabel("Approach")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(reports_dir / "training_time.png")
    plt.close(fig)
