"""Configurable multilayer perceptron baseline for thesis experiments."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from thesis_elm.models.base import GradientClassifier


class MLPClassifier(GradientClassifier):
    """Multilayer perceptron with configurable depth, width, and early stopping."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        width: int = 128,
        depth: int = 2,
        learning_rate: float = 1e-3,
        max_epochs: int = 300,
        batch_size: int = 64,
        patience: int = 30,
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
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, width))
            layers.append(nn.ReLU())
            current_dim = width
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map features to class logits."""
        return cast(torch.Tensor, self.network(X))
