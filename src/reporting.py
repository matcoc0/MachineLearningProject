from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Directory utilities
# ============================================================

def ensure_reports_dir(path: Path) -> None:
    """Create root reports directory."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_subdirs(reports_dir: Path) -> None:
    """
    Create standard subdirectories used by the project.
    """
    for sub in [
        "data",
        "training",
        "results",
        "results/experiments",
    ]:
        (reports_dir / sub).mkdir(parents=True, exist_ok=True)


# ============================================================
# Save utilities
# ============================================================

def save_metrics(
    metrics_rows: List[Dict],
    out_dir: Path,
    filename_prefix: Optional[str] = None,
) -> None:
    """
    Save metrics to CSV + JSON.

    - PROD: filename_prefix=None
      -> metrics.csv / metrics.json
    - EXPERIMENTS: filename_prefix="experiments"
      -> experiments_metrics.csv / experiments_metrics.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_rows)

    if filename_prefix is None:
        csv_name = "metrics.csv"
        json_name = "metrics.json"
    else:
        csv_name = f"{filename_prefix}_metrics.csv"
        json_name = f"{filename_prefix}_metrics.json"

    df.to_csv(out_dir / csv_name, index=False)
    df.to_json(
        out_dir / json_name,
        orient="records",
        indent=2,
    )


def save_log(
    log_rows: List[Dict],
    out_dir: Path,
    filename: str,
) -> None:
    """
    Save logs (GA convergence or ensemble logs) to CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(out_dir / filename, index=False)


# ============================================================
# Metrics plots
# ============================================================

def plot_metrics(
    metrics_rows: List[Dict],
    out_dir: Path,
    filename: Optional[str] = None,
) -> None:
    """
    Plot accuracy / precision / recall / f1 for all approaches.

    - PROD: filename=None -> metrics_plot.png
    - EXPERIMENTS: filename="experiments_metrics_plot.png"
    """
    df = pd.DataFrame(metrics_rows)
    if df.empty:
        return

    if filename is None:
        filename = "metrics_plot.png"

    metrics = ["accuracy", "precision", "recall", "f1"]
    x = range(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, m in enumerate(metrics):
        ax.bar(
            [p + (i - 1.5) * width for p in x],
            df[m],
            width=width,
            label=m,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["approach"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Performance comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


# ============================================================
# GA convergence plots
# ============================================================

def plot_ga_history(
    log_csv: Path,
    out_dir: Path,
    prefix: str,
) -> None:
    """
    Plot GA convergence:
    - best_f1
    - mean_f1
    - training size evolution
    """
    df = pd.read_csv(log_csv)

    # ---- F1 convergence ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["generation"], df["best_f1"], label="Best F1")
    ax.plot(df["generation"], df["mean_f1"], label="Mean F1")
    ax.set_xlabel("Generation")
    ax.set_ylabel("F1-score")
    ax.set_title(f"GA convergence ({prefix})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_ga_convergence.png")
    plt.close(fig)

    # ---- Training size ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["generation"], df["train_size"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Training set size")
    ax.set_title(f"Training set size evolution ({prefix})")

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_training_size.png")
    plt.close(fig)

# training time plot
def plot_training_time_comparison(results, save_path):
    """
    results: list of dicts with keys:
        - approach
        - train_time_sec
    """

    labels = [r["approach"] for r in results]
    times = [r["train_time_sec"] for r in results]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, times)
    plt.ylabel("Training time (seconds)")
    plt.title("Comparison of Training Times")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    