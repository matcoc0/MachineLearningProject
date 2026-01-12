from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Directory utilities
# ---------------------------------------------------------------------

def ensure_reports_dir(path: Path) -> None:
    """Create reports directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Save metrics and logs
# ---------------------------------------------------------------------

def save_metrics(metrics_rows: list[dict], reports_dir: Path) -> None:
    """
    Save final performance metrics (GA / GA+AL / GA+AL+EL)
    in both CSV and JSON formats.
    """
    df = pd.DataFrame(metrics_rows)
    df.to_csv(reports_dir / "metrics.csv", index=False)
    df.to_json(reports_dir / "metrics.json", orient="records", indent=2)


def save_log(log_rows: list[dict], reports_dir: Path, filename: str) -> None:
    """
    Save GA training history (per-generation logs) to CSV.
    """
    df = pd.DataFrame(log_rows)
    df.to_csv(reports_dir / filename, index=False)


# ---------------------------------------------------------------------
# Global metrics plots
# ---------------------------------------------------------------------

def plot_metrics(metrics_rows: list[dict], reports_dir: Path) -> None:
    """
    Plot final performance metrics and training time comparison
    between GA, GA+AL and GA+AL+EL.
    """
    df = pd.DataFrame(metrics_rows)
    if df.empty:
        return

    # --- Performance metrics ---
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df["approach"]))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(
            [p + (i - 1.5) * width for p in x],
            df[metric],
            width=width,
            label=metric,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["approach"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Performance comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(reports_dir / "metrics_plot.png")
    plt.close(fig)

    # --- Training time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["approach"], df["train_time_sec"])
    ax.set_ylabel("Seconds")
    ax.set_title("Training time comparison")

    fig.tight_layout()
    fig.savefig(reports_dir / "training_time.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# GA convergence & Active Learning plots
# ---------------------------------------------------------------------

def plot_ga_history(log_csv: Path, reports_dir: Path, prefix: str) -> None:
    """
    Plot GA convergence (best and mean fitness over generations)
    and training set size evolution.

    Parameters
    ----------
    log_csv : Path
        CSV file containing GA logs (generation, best_f1, mean_f1, train_size).
    reports_dir : Path
        Directory where plots will be saved.
    prefix : str
        Prefix used to identify the experiment (e.g., "ga", "ga_al").
    """
    df = pd.read_csv(log_csv)

    # --- Convergence plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["generation"], df["best_f1"], label="Best F1")
    ax.plot(df["generation"], df["mean_f1"], label="Mean F1")
    ax.set_xlabel("Generation")
    ax.set_ylabel("F1-score")
    ax.set_title(f"GA convergence ({prefix})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(reports_dir / f"{prefix}_ga_convergence.png")
    plt.close(fig)

    # --- Training set size evolution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["generation"], df["train_size"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Training set size")
    ax.set_title(f"Training set growth ({prefix})")

    fig.tight_layout()
    fig.savefig(reports_dir / f"{prefix}_training_size.png")
    plt.close(fig)
