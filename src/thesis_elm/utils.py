"""Shared utilities for reproducible thesis experiments."""

from __future__ import annotations

import csv
import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


@dataclass(slots=True)
class TimerResult:
    """Stores the duration of a timed block in seconds."""

    duration_s: float = 0.0


@contextmanager
def timed_block() -> Iterator[TimerResult]:
    """Measure elapsed wall-clock time for a code block."""
    timer = TimerResult()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.duration_s = time.perf_counter() - start


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str | None = None) -> torch.device:
    """Choose the compute device, defaulting to CUDA when available."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_2d_float_tensor(X: ArrayLike, device: torch.device) -> torch.Tensor:
    """Convert input features to a 2D float tensor on the target device."""
    tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {tuple(tensor.shape)}.")
    return tensor


def ensure_1d_long_tensor(y: ArrayLike, device: torch.device) -> torch.Tensor:
    """Convert integer class labels to a 1D long tensor on the target device."""
    tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    if tensor.ndim != 1:
        raise ValueError(f"Expected a 1D label vector, got shape {tuple(tensor.shape)}.")
    return tensor


def one_hot_encode(y: ArrayLike, num_classes: int, device: torch.device) -> torch.Tensor:
    """Convert class labels into the one-hot target matrix T."""
    labels = ensure_1d_long_tensor(y, device=device)
    T = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    return T.to(dtype=torch.float32)


def write_results_csv(
    rows: Sequence[dict[str, str | float | int]],
    output_path: str | Path,
) -> None:
    """Write experiment result rows to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "dataset", "metric", "value", "seed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_result_rows(
    model_name: str,
    dataset_name: str,
    seed: int,
    metrics: dict[str, float],
) -> list[dict[str, str | float | int]]:
    """Convert a metric dictionary into the standardized CSV row format."""
    return [
        {
            "model": model_name,
            "dataset": dataset_name,
            "metric": metric_name,
            "value": metric_value,
            "seed": seed,
        }
        for metric_name, metric_value in metrics.items()
    ]
