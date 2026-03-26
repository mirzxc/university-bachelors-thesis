from __future__ import annotations

import numpy as np
import pytest

from thesis_elm.models import (
    ELMClassifier,
    LogisticRegressionClassifier,
    MLPClassifier,
    OSELMClassifier,
)


@pytest.mark.parametrize(
    ("model_factory", "minimum_accuracy"),
    [
        (
            lambda input_dim, num_classes: LogisticRegressionClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                max_epochs=200,
                patience=25,
                seed=7,
                device="cpu",
            ),
            0.7,
        ),
        (
            lambda input_dim, num_classes: MLPClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                depth=2,
                width=64,
                max_epochs=200,
                patience=30,
                seed=7,
                device="cpu",
            ),
            0.7,
        ),
        (
            lambda input_dim, num_classes: ELMClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                L=100,
                seed=7,
                device="cpu",
            ),
            0.7,
        ),
        (
            lambda input_dim, num_classes: OSELMClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                L=100,
                initial_batch_size=60,
                seed=7,
                device="cpu",
            ),
            0.65,
        ),
    ],
)
def test_models_fit_predict_score(
    iris_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    model_factory,
    minimum_accuracy: float,
) -> None:
    X_train, X_test, y_train, y_test = iris_data
    model = model_factory(X_train.shape[1], np.unique(y_train).size)
    returned = model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)

    assert returned is model
    assert predictions.shape == y_test.shape
    assert score >= minimum_accuracy


def test_elm_shape_invariants(
    iris_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, _, y_train, _ = iris_data
    model = ELMClassifier(
        input_dim=X_train.shape[1],
        num_classes=np.unique(y_train).size,
        L=50,
        seed=11,
        device="cpu",
    )
    model.fit(X_train, y_train)
    H = model.compute_H(X_train)

    assert H.shape == (X_train.shape[0], 50)
    assert tuple(model.W.shape) == (X_train.shape[1], 50)
    assert tuple(model.b.shape) == (50,)
    assert tuple(model.beta.shape) == (50, np.unique(y_train).size)


def test_os_elm_shape_invariants(
    iris_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, _, y_train, _ = iris_data
    model = OSELMClassifier(
        input_dim=X_train.shape[1],
        num_classes=np.unique(y_train).size,
        L=40,
        initial_batch_size=50,
        seed=11,
        device="cpu",
    )
    model.fit(X_train, y_train)
    H = model.compute_H(X_train)

    assert H.shape == (X_train.shape[0], 40)
    assert tuple(model.W.shape) == (X_train.shape[1], 40)
    assert tuple(model.b.shape) == (40,)
    assert tuple(model.beta.shape) == (40, np.unique(y_train).size)
    assert tuple(model.P.shape) == (40, 40)


def test_seed_reproducibility(
    iris_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, X_test, y_train, _ = iris_data
    first = ELMClassifier(
        input_dim=X_train.shape[1],
        num_classes=np.unique(y_train).size,
        L=75,
        seed=99,
        device="cpu",
    )
    second = ELMClassifier(
        input_dim=X_train.shape[1],
        num_classes=np.unique(y_train).size,
        L=75,
        seed=99,
        device="cpu",
    )

    first.fit(X_train, y_train)
    second.fit(X_train, y_train)

    np.testing.assert_allclose(first.beta.cpu().numpy(), second.beta.cpu().numpy(), atol=5e-2)
    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))
