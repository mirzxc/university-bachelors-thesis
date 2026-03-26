"""Logistic Regression baseline implemented with a single linear layer."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from thesis_elm.models.base import GradientClassifier


class LogisticRegressionClassifier(GradientClassifier):
    """Multiclass Logistic Regression using one `nn.Linear` layer."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        learning_rate: float = 1e-2,
        max_epochs: int = 200,
        batch_size: int = 32,
        patience: int = 20,
        validation_fraction: float = 0.2,
        seed: int = 42,
        device: str | None = "auto",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            validation_fraction=validation_fraction,
            seed=seed,
            device=device,
        )
        self.linear = nn.Linear(input_dim, num_classes)
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map features to class logits."""
        return cast(torch.Tensor, self.linear(X))
