from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Utilities

def _ensure_dir(p: Path) -> None:
    """
    Ensure that a directory exists
    """
    p.mkdir(parents=True, exist_ok=True)


def _safe_float(x) -> float:
    """
    Safe convert to float
    """
    try:
        return float(x)
    except Exception:
        return float("nan")


def _iqr_outlier_rate(x: np.ndarray) -> float:
    """
    Compute the proportion of outliers using the IQR rule
    """
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(np.mean((x < low) | (x > high)))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the K-S Statistic between two samples
    """
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), pooled, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), pooled, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


# Main EDA

def run_eda(
    data_path: str,
    reports_dir: Path,
    seed: int = 42,
    test_size: float = 0.2,
) -> None:
    '''
    Full EDA Pipeline of our project. Includes correlation matrix, missing values, class distributions...
    To run it before the main pipeline of the project, make sure that run_eda=true in config.py. Otherwise, make deactivate it.
    '''
    t0 = time.perf_counter()

    # Directories
    data_dir = reports_dir / "data"
    corr_dir = data_dir / "correlations"
    pca_dir = data_dir / "pca"

    for d in [data_dir, corr_dir, pca_dir]:
        _ensure_dir(d)

    # Load dataset
    print("[EDA] Loading dataset...")
    df = pd.read_csv(data_path, compression="gzip")

    # Drop non-feature columns
    drop_cols = ["EventId", "Weight", "KaggleSet", "KaggleWeight"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "Label" not in df.columns:
        raise ValueError("Expected 'Label' column in dataset.")

    # Dataset overview
    overview = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    (data_dir / "dataset_overview.json").write_text(
        json.dumps(overview, indent=2)
    )

    # Label analysis
    label_counts = df["Label"].value_counts()
    label_counts.to_csv(data_dir / "label_counts.csv")

    fig, ax = plt.subplots()
    label_counts.plot(kind="bar", ax=ax)
    ax.set_title("Label distribution")
    fig.tight_layout()
    fig.savefig(data_dir / "label_distribution.png")
    plt.close(fig)

    # Feature matrix
    X = df.drop(columns=["Label"]).replace(-999.0, np.nan)
    y = df["Label"].map({"b": 0, "s": 1}).astype(int).to_numpy()
    feature_names = list(X.columns)

    # Missing values
    missing_ratio = X.isna().mean().sort_values(ascending=False)
    missing_ratio.to_csv(data_dir / "missing_values_ratio.csv")

    # Feature summary
    rows = []
    for col in feature_names:
        x = X[col].to_numpy()
        rows.append(
            {
                "feature": col,
                "missing_ratio": float(np.mean(np.isnan(x))),
                "mean": _safe_float(np.nanmean(x)),
                "std": _safe_float(np.nanstd(x)),
                "min": _safe_float(np.nanmin(x)),
                "max": _safe_float(np.nanmax(x)),
                "iqr_outlier_rate": _iqr_outlier_rate(x),
            }
        )

    pd.DataFrame(rows).to_csv(
        data_dir / "feature_summary.csv", index=False
    )

    # Correlations
    corr = X.corr()
    corr.to_csv(corr_dir / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(corr_dir / "correlation_heatmap.png")
    plt.close(fig)

    # Train / test distribution shift
    X_np = X.to_numpy()
    X_train, X_test, _, _ = train_test_split(
        X_np, y, test_size=test_size, random_state=seed, stratify=y
    )

    shifts = []
    for i, col in enumerate(feature_names):
        ks = _ks_statistic(X_train[:, i], X_test[:, i])
        shifts.append({"feature": col, "ks_stat": ks})

    pd.DataFrame(shifts).to_csv(
        data_dir / "train_test_shift.csv", index=False
    )

    # PCA (imputed + scaled)
    X_pca = X.fillna(X.mean())
    X_scaled = StandardScaler().fit_transform(X_pca)

    pca = PCA(n_components=10, random_state=seed)
    X_proj = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    ax.plot(pca.explained_variance_ratio_, marker="o")
    ax.set_title("PCA explained variance ratio")
    fig.tight_layout()
    fig.savefig(pca_dir / "pca_explained_variance.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(X_proj[:, 0], X_proj[:, 1], c=y, s=2)
    ax.set_title("PCA projection (2D)")
    fig.tight_layout()
    fig.savefig(pca_dir / "pca_2d_projection.png")
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    print(f"[EDA] Done in {elapsed:.2f}s")
