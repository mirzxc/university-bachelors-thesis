"""Base abstractions shared by all thesis classification models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Self

import numpy as np
import torch
from torch import nn

from thesis_elm.utils import (
    ensure_1d_long_tensor,
    ensure_2d_float_tensor,
    seed_everything,
    select_device,
)


class BaseClassifier(nn.Module, ABC):
    """Shared `fit`, `predict`, and `score` API for thesis classifiers."""

    def __init__(self, num_classes: int, seed: int = 42, device: str | None = "auto") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seed = seed
        self.device = select_device(device)
        self._is_fitted = False
        seed_everything(seed)

    @abstractmethod
    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> Self:
        """Fit the classifier to input features and class labels."""

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict integer class labels for a batch of features."""
        self._require_fitted()
        self.eval()
        X_tensor = ensure_2d_float_tensor(X, device=self.device)
        with torch.no_grad():
            logits = self.forward(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        return predictions.detach().cpu().numpy()

    def score(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> float:
        """Compute classification accuracy on a feature matrix and labels."""
        labels = ensure_1d_long_tensor(y, device=torch.device("cpu")).cpu().numpy()
        predictions = self.predict(X)
        return float(np.mean(predictions == labels))

    def _require_fitted(self) -> None:
        """Raise an error when prediction is requested before fitting."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before use.")

    def _feature_tensor(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert features into a tensor on the model device."""
        return ensure_2d_float_tensor(X, device=self.device)

    def _label_tensor(self, y: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert labels into a tensor on the model device."""
        return ensure_1d_long_tensor(y, device=self.device)


class GradientClassifier(BaseClassifier, ABC):
    """Base class for models trained with Adam and early stopping."""

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-2,
        max_epochs: int = 200,
        batch_size: int = 32,
        patience: int = 20,
        validation_fraction: float = 0.2,
        seed: int = 42,
        device: str | None = "auto",
    ) -> None:
        super().__init__(num_classes=num_classes, seed=seed, device=device)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> Self:
        """Train the model with Adam and restore the best validation checkpoint."""
        X_tensor = self._feature_tensor(X)
        y_tensor = self._label_tensor(y)
        self.train()
        self.to(self.device)

        X_train, y_train, X_val, y_val = self._split_validation(X_tensor, y_tensor)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        best_state = deepcopy(self.state_dict())
        best_loss = float("inf")
        epochs_without_improvement = 0

        for _ in range(self.max_epochs):
            permutation = torch.randperm(X_train.shape[0], generator=generator, device="cpu")
            for batch_start in range(0, X_train.shape[0], self.batch_size):
                batch_indices = permutation[
                    batch_start : batch_start + self.batch_size
                ].to(self.device)
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                optimizer.zero_grad()
                logits = self.forward(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            validation_loss = self._validation_loss(X_val, y_val, criterion)
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        self.load_state_dict(best_state)
        self._is_fitted = True
        return self

    def _split_validation(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a reproducible train/validation split from tensors."""
        n_samples = X.shape[0]
        if n_samples < 2 or self.validation_fraction <= 0.0:
            return X, y, X, y

        validation_size = max(1, int(round(n_samples * self.validation_fraction)))
        if validation_size >= n_samples:
            validation_size = n_samples - 1

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        indices = torch.randperm(n_samples, generator=generator, device="cpu")
        val_indices = indices[:validation_size].to(self.device)
        train_indices = indices[validation_size:].to(self.device)

        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def _validation_loss(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        """Compute the current validation loss."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(X_val)
            loss = criterion(logits, y_val)
        self.train()
        return float(loss.item())
