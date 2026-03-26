"""Dataset loading helpers for toy and CSV-based experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ToyDatasetName = str


@dataclass(slots=True)
class DatasetBundle:
    """Holds train/test splits and metadata for a classification dataset."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    num_classes: int
    feature_dim: int


def load_dataset(
    dataset_name: str,
    seed: int,
    test_size: float = 0.2,
    standardize: bool = True,
    csv_path: str | None = None,
    target_column: str | None = None,
) -> DatasetBundle:
    """Load either a built-in sklearn dataset or a CSV classification dataset."""
    if dataset_name == "csv":
        if csv_path is None or target_column is None:
            raise ValueError("CSV datasets require both --csv-path and --target-column.")
        return load_csv_dataset(
            csv_path=csv_path,
            target_column=target_column,
            seed=seed,
            test_size=test_size,
            standardize=standardize,
        )
    return load_sklearn_dataset(
        dataset_name=dataset_name,
        seed=seed,
        test_size=test_size,
        standardize=standardize,
    )


def load_sklearn_dataset(
    dataset_name: str,
    seed: int,
    test_size: float = 0.2,
    standardize: bool = True,
) -> DatasetBundle:
    """Load a small sklearn tabular classification dataset."""
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    if dataset_name not in loaders:
        supported = ", ".join(sorted(loaders))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}.")

    raw = loaders[dataset_name]()
    X = np.asarray(raw.data, dtype=np.float32)
    y = np.asarray(raw.target, dtype=np.int64)
    return split_dataset(
        name=dataset_name,
        X=X,
        y=y,
        seed=seed,
        test_size=test_size,
        standardize=standardize,
    )


def load_csv_dataset(
    csv_path: str,
    target_column: str,
    seed: int,
    test_size: float = 0.2,
    standardize: bool = True,
) -> DatasetBundle:
    """Load a tabular CSV classification dataset with a named target column."""
    frame = pd.read_csv(Path(csv_path))
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in {csv_path}.")

    X = frame.drop(columns=[target_column]).to_numpy(dtype=np.float32)
    labels = frame[target_column].to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int64)

    dataset_name = Path(csv_path).stem
    return split_dataset(
        name=dataset_name,
        X=X,
        y=y,
        seed=seed,
        test_size=test_size,
        standardize=standardize,
    )


def split_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    standardize: bool,
) -> DatasetBundle:
    """Split a classification dataset into standardized train/test subsets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    return DatasetBundle(
        name=name,
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        num_classes=int(np.unique(y).size),
        feature_dim=int(X.shape[1]),
    )


def subset_training_data(bundle: DatasetBundle, n_samples: int) -> DatasetBundle:
    """Create a smaller training subset for scalability or learning-curve experiments."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    n_samples = min(n_samples, bundle.X_train.shape[0])
    return DatasetBundle(
        name=f"{bundle.name}[N={n_samples}]",
        X_train=bundle.X_train[:n_samples].copy(),
        X_test=bundle.X_test.copy(),
        y_train=bundle.y_train[:n_samples].copy(),
        y_test=bundle.y_test.copy(),
        num_classes=bundle.num_classes,
        feature_dim=bundle.feature_dim,
    )


def build_class_increment_splits(
    X: np.ndarray,
    y: np.ndarray,
    classes_per_step: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition data by sequentially introducing groups of classes."""
    unique_classes = np.unique(y)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, unique_classes.size, classes_per_step):
        current_classes = unique_classes[start : start + classes_per_step]
        mask = np.isin(y, current_classes)
        splits.append((X[mask], y[mask]))
    return splits


def apply_covariate_shift(X: np.ndarray, shift_strength: float) -> np.ndarray:
    """Apply a simple additive covariate shift to all features."""
    shift_vector = np.full(shape=X.shape[1], fill_value=shift_strength, dtype=np.float32)
    return X + shift_vector
