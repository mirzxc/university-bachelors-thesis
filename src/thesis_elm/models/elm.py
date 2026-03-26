"""Extreme Learning Machine classifier with a closed-form output layer."""

from __future__ import annotations

import numpy as np
import torch

from thesis_elm.models.base import BaseClassifier
from thesis_elm.utils import one_hot_encode


class ELMClassifier(BaseClassifier):
    """ELM with frozen random `W`, frozen random `b`, and solved output weights `β`."""

    W: torch.Tensor
    b: torch.Tensor
    beta: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        L: int = 200,
        activation: str = "sigmoid",
        l2_reg: float = 1e-6,
        rcond: float | None = None,
        seed: int = 42,
        device: str | None = "auto",
    ) -> None:
        super().__init__(num_classes=num_classes, seed=seed, device=device)
        if L < 1:
            raise ValueError("L must be at least 1.")
        if activation not in {"sigmoid", "relu"}:
            raise ValueError("activation must be either 'sigmoid' or 'relu'.")

        self.input_dim = input_dim
        self.L = L
        self.activation = activation
        self.l2_reg = l2_reg
        self.rcond = rcond

        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer(
            "W",
            torch.randn(input_dim, L, generator=generator, dtype=torch.float32),
        )
        self.register_buffer("b", torch.randn(L, generator=generator, dtype=torch.float32))
        self.register_buffer("beta", torch.zeros(L, num_classes, dtype=torch.float32))
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute class logits from the hidden layer matrix `H` and output weights `β`."""
        H = self.compute_H(X)
        return H @ self.beta

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> ELMClassifier:
        """Solve `β` from `H` and one-hot targets `T` using `torch.linalg.lstsq`."""
        X_tensor = self._feature_tensor(X)
        T = one_hot_encode(y, num_classes=self.num_classes, device=self.device)
        H = self.compute_H(X_tensor)

        if self.l2_reg > 0.0:
            identity = torch.eye(self.L, device=self.device, dtype=torch.float32)
            zeros = torch.zeros((self.L, self.num_classes), device=self.device, dtype=torch.float32)
            ridge = torch.sqrt(torch.tensor(self.l2_reg, device=self.device))
            A = torch.cat([H, ridge * identity], dim=0)
            B = torch.cat([T, zeros], dim=0)
        else:
            A = H
            B = T

        solution = torch.linalg.lstsq(A, B, rcond=self.rcond).solution
        self.beta.copy_(solution)
        self._is_fitted = True
        return self

    def compute_H(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Compute the hidden layer output matrix `H` with shape `(N, L)`."""
        X_tensor = self._feature_tensor(X)
        linear_response = X_tensor @ self.W + self.b
        if self.activation == "sigmoid":
            return torch.sigmoid(linear_response)
        return torch.relu(linear_response)
