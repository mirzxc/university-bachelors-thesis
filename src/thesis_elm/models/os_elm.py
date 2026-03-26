"""Online Sequential Extreme Learning Machine with recursive least squares updates."""

from __future__ import annotations

from typing import Self

import numpy as np
import torch

from thesis_elm.models.base import BaseClassifier
from thesis_elm.utils import one_hot_encode


class OSELMClassifier(BaseClassifier):
    """OS-ELM using a frozen random hidden layer and recursive least squares for `β`."""

    W: torch.Tensor
    b: torch.Tensor
    beta: torch.Tensor
    P: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        L: int = 200,
        activation: str = "sigmoid",
        l2_reg: float = 1e-3,
        initial_batch_size: int | None = None,
        update_chunk_size: int = 1,
        seed: int = 42,
        device: str | None = "auto",
    ) -> None:
        super().__init__(num_classes=num_classes, seed=seed, device=device)
        if L < 1:
            raise ValueError("L must be at least 1.")
        if activation not in {"sigmoid", "relu"}:
            raise ValueError("activation must be either 'sigmoid' or 'relu'.")
        if update_chunk_size < 1:
            raise ValueError("update_chunk_size must be at least 1.")

        self.input_dim = input_dim
        self.L = L
        self.activation = activation
        self.l2_reg = l2_reg
        self.initial_batch_size = initial_batch_size
        self.update_chunk_size = update_chunk_size

        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer(
            "W",
            torch.randn(input_dim, L, generator=generator, dtype=torch.float32),
        )
        self.register_buffer("b", torch.randn(L, generator=generator, dtype=torch.float32))
        self.register_buffer("beta", torch.zeros(L, num_classes, dtype=torch.float32))
        self.register_buffer("P", torch.eye(L, dtype=torch.float32))
        self._is_initialized = False
        self.to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute class logits as `H @ β`."""
        H = self.compute_H(X)
        return H @ self.beta

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> Self:
        """Initialize OS-ELM on a batch, then update `β` and `P` sequentially."""
        X_tensor = self._feature_tensor(X)
        y_tensor = self._label_tensor(y)
        n_samples = X_tensor.shape[0]
        initial_batch_size = self.initial_batch_size or min(
            max(self.L, self.num_classes),
            n_samples,
        )
        initial_batch_size = min(max(1, initial_batch_size), n_samples)

        self.reset_state()
        self._initialize(X_tensor[:initial_batch_size], y_tensor[:initial_batch_size])
        if initial_batch_size < n_samples:
            self.partial_fit(X_tensor[initial_batch_size:], y_tensor[initial_batch_size:])

        self._is_fitted = True
        return self

    def partial_fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> Self:
        """Update `β` and `P` for additional batches without reinitializing `W` or `b`."""
        X_tensor = self._feature_tensor(X)
        y_tensor = self._label_tensor(y)

        if not self._is_initialized:
            self._initialize(X_tensor, y_tensor)
            self._is_fitted = True
            return self

        for batch_start in range(0, X_tensor.shape[0], self.update_chunk_size):
            batch_X = X_tensor[batch_start : batch_start + self.update_chunk_size]
            batch_y = y_tensor[batch_start : batch_start + self.update_chunk_size]
            if batch_X.shape[0] == 0:
                continue
            self._update_batch(batch_X, batch_y)

        self._is_fitted = True
        return self

    def reset_state(self) -> None:
        """Reset the solved output weights `β` and covariance matrix `P`."""
        self.beta.zero_()
        identity = torch.eye(self.L, device=self.device, dtype=torch.float32)
        self.P.copy_(identity)
        self._is_initialized = False
        self._is_fitted = False

    def compute_H(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Compute the hidden layer output matrix `H` with shape `(N, L)`."""
        X_tensor = self._feature_tensor(X)
        linear_response = X_tensor @ self.W + self.b
        if self.activation == "sigmoid":
            return torch.sigmoid(linear_response)
        return torch.relu(linear_response)

    def _initialize(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Compute the initial `β` and `P` from a starting batch."""
        H0 = self.compute_H(X)
        T0 = one_hot_encode(y, num_classes=self.num_classes, device=self.device)
        identity = torch.eye(self.L, device=self.device, dtype=torch.float32)
        gram = H0.T @ H0 + self.l2_reg * identity
        self.P.copy_(torch.linalg.solve(gram, identity))
        beta0 = self.P @ H0.T @ T0
        self.beta.copy_(beta0)
        self._is_initialized = True

    def _update_batch(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Apply a recursive least squares update for one incoming batch."""
        H_k = self.compute_H(X)
        T_k = one_hot_encode(y, num_classes=self.num_classes, device=self.device)
        identity = torch.eye(H_k.shape[0], device=self.device, dtype=torch.float32)
        innovation = identity + H_k @ self.P @ H_k.T
        gain = self.P @ H_k.T @ torch.linalg.solve(innovation, identity)
        residual = T_k - H_k @ self.beta
        updated_beta = self.beta + gain @ residual
        updated_P = self.P - gain @ H_k @ self.P
        self.beta.copy_(updated_beta)
        self.P.copy_(updated_P)
