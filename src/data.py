from __future__ import annotations

from dataclasses import dataclass
from typing import List
import time

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Dataset container

@dataclass
class DatasetSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


# Loader + preprocessing

def load_and_preprocess(
    data_path: str,
    test_size: float,
    seed: int,
) -> DatasetSplit:
    """
    Load the Higgs dataset, apply preprocessing, and return
    a structured DatasetSplit object.

    Preprocessing steps:
    - drop non-feature columns
    - label encoding
    - missing value handling (-999 -> NaN -> mean imputation)
    - standardization
    - stratified train/test split
    """
    t0 = time.perf_counter()

    print("[DATA] Loading dataset...")
    df = pd.read_csv(data_path, compression="gzip")

    # --------------------------------------------------------
    # Drop non-feature columns
    # --------------------------------------------------------
    drop_cols = ["EventId", "Weight", "KaggleSet", "KaggleWeight"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "Label" not in df.columns:
        raise ValueError("Expected 'Label' column in dataset.")

    # Labels
    y = (
        df.pop("Label")
        .map({"b": 0, "s": 1})
        .astype(int)
        .to_numpy()
    )

    # Features
    X = df.replace(-999.0, np.nan).astype(np.float32)
    feature_names = list(X.columns)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Imputation + scaling (fit on train only)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Safety checks
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    elapsed = time.perf_counter() - t0
    print(f"[DATA] Load + preprocess done in {elapsed:.2f}s")
    print(f"[DATA] Features: {X_train.shape[1]}")
    print(f"[DATA] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    return DatasetSplit(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
