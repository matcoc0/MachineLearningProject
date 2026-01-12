from __future__ import annotations
import time

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_and_preprocess(path: str, test_size: float, seed: int) -> DatasetSplit:
    t0 = time.perf_counter()

    print("Downloading the Higgs dataset...")
    df = pd.read_csv(path, compression="gzip")

    print("Drop non feature columns and keep label...")
    drop_cols = ["EventId", "Weight", "KaggleSet", "KaggleWeight"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    if "Label" not in df.columns:
        raise ValueError("Expected 'Label' column in dataset.")

    labels = df.pop("Label")
    y = labels.map({"b": 0, "s": 1}).astype(int).to_numpy()

    print("Replacing common missing value indicator and cast to float32 to reduce memory.")
    x = df.replace(-999.0, np.nan).astype(np.float32)
    feature_names = list(x.columns)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    print(f"Train/test split with test_size={test_size}, seed={seed}, stratify=y")
    x_train, x_test, y_train, y_test = train_test_split(
        x.to_numpy(), y, test_size=test_size, random_state=seed, stratify=y
    )

    print("Implementation of the mean SimpleImputer and StandardScaler.")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    elapsed = time.perf_counter() - t0
    print(f"[LOAD + PREPROCESS] Finished in {elapsed:.2f} seconds")

    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )