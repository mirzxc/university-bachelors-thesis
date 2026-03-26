from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture()
def iris_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = load_iris()
    X = raw.data.astype(np.float32)
    y = raw.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test
