"""Experiment runners and model factory functions for thesis workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from thesis_elm.data import (
    DatasetBundle,
    apply_covariate_shift,
    build_class_increment_splits,
    load_dataset,
    subset_training_data,
)
from thesis_elm.models.base import BaseClassifier
from thesis_elm.models.elm import ELMClassifier
from thesis_elm.models.logistic_regression import LogisticRegressionClassifier
from thesis_elm.models.mlp import MLPClassifier
from thesis_elm.models.os_elm import OSELMClassifier
from thesis_elm.utils import build_result_rows, timed_block, write_results_csv


def run_command(args: argparse.Namespace) -> Path:
    """Run a single experiment and write its metrics to a CSV file."""
    bundle = load_dataset_bundle(args)
    model = create_model(args, bundle.feature_dim, bundle.num_classes)
    model_label = format_model_name(args)
    metrics = evaluate_model(model, bundle)
    output_path = resolve_output_path(args, suffix="run")
    write_results_csv(build_result_rows(model_label, bundle.name, args.seed, metrics), output_path)
    log_metrics(model_label, bundle.name, metrics, output_path)
    return output_path


def grid_search_command(args: argparse.Namespace) -> Path:
    """Run an `L` grid search for ELM-based models and save all metrics."""
    if args.model not in {"elm", "os_elm"}:
        raise ValueError("grid-search currently supports only 'elm' and 'os_elm'.")

    bundle = load_dataset_bundle(args)
    rows: list[dict[str, str | float | int]] = []
    for L_value in parse_int_list(args.L_values):
        args.L = L_value
        model = create_model(args, bundle.feature_dim, bundle.num_classes)
        model_label = format_model_name(args)
        metrics = evaluate_model(model, bundle)
        rows.extend(build_result_rows(model_label, bundle.name, args.seed, metrics))

    output_path = resolve_output_path(args, suffix="grid_search")
    write_results_csv(rows, output_path)
    print(f"Saved grid-search results to {output_path}")
    return output_path


def scaling_command(args: argparse.Namespace) -> Path:
    """Measure performance as the number of training samples `N` changes."""
    bundle = load_dataset_bundle(args)
    rows: list[dict[str, str | float | int]] = []
    for n_samples in parse_int_list(args.n_values):
        subset = subset_training_data(bundle, n_samples=n_samples)
        model = create_model(args, subset.feature_dim, subset.num_classes)
        model_label = format_model_name(args)
        metrics = evaluate_model(model, subset)
        rows.extend(build_result_rows(model_label, subset.name, args.seed, metrics))

    output_path = resolve_output_path(args, suffix="scaling")
    write_results_csv(rows, output_path)
    print(f"Saved scaling results to {output_path}")
    return output_path


def sequential_command(args: argparse.Namespace) -> Path:
    """Run class-incremental or covariate-shift experiments."""
    bundle = load_dataset_bundle(args)
    rows: list[dict[str, str | float | int]] = []

    if args.scenario == "class_incremental":
        rows.extend(run_class_incremental_sequence(args, bundle))
    elif args.scenario == "covariate_shift":
        rows.extend(run_covariate_shift_sequence(args, bundle))
    else:
        raise ValueError(f"Unsupported scenario '{args.scenario}'.")

    output_path = resolve_output_path(args, suffix=args.scenario)
    write_results_csv(rows, output_path)
    print(f"Saved sequential results to {output_path}")
    return output_path


def load_dataset_bundle(args: argparse.Namespace) -> DatasetBundle:
    """Build a dataset bundle from CLI arguments."""
    return load_dataset(
        dataset_name=args.dataset,
        seed=args.seed,
        test_size=args.test_size,
        standardize=not args.no_standardize,
        csv_path=args.csv_path,
        target_column=args.target_column,
    )


def create_model(args: argparse.Namespace, input_dim: int, num_classes: int) -> BaseClassifier:
    """Instantiate a classifier from CLI configuration."""
    if args.model == "logistic_regression":
        return LogisticRegressionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            device=args.device,
        )
    if args.model == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            width=args.width,
            depth=args.depth,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            device=args.device,
        )
    if args.model == "elm":
        return ELMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            L=args.L,
            activation=args.activation,
            l2_reg=args.l2_reg,
            seed=args.seed,
            device=args.device,
        )
    if args.model == "os_elm":
        return OSELMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            L=args.L,
            activation=args.activation,
            l2_reg=args.l2_reg,
            initial_batch_size=args.initial_batch_size,
            update_chunk_size=args.update_chunk_size,
            seed=args.seed,
            device=args.device,
        )
    raise ValueError(f"Unsupported model '{args.model}'.")


def evaluate_model(model: BaseClassifier, bundle: DatasetBundle) -> dict[str, float]:
    """Fit a model, measure training and inference time, and compute accuracy."""
    with timed_block() as training_timer:
        model.fit(bundle.X_train, bundle.y_train)

    with timed_block() as inference_timer:
        predictions = model.predict(bundle.X_test)

    accuracy = float(np.mean(predictions == bundle.y_test))
    return {
        "accuracy": accuracy,
        "training_time_s": training_timer.duration_s,
        "inference_time_s": inference_timer.duration_s,
    }


def run_class_incremental_sequence(
    args: argparse.Namespace,
    bundle: DatasetBundle,
) -> list[dict[str, str | float | int]]:
    """Evaluate sequential class introduction, updating the model step by step."""
    train_splits = build_class_increment_splits(
        bundle.X_train,
        bundle.y_train,
        args.classes_per_step,
    )
    model = create_model(args, bundle.feature_dim, bundle.num_classes)
    rows: list[dict[str, str | float | int]] = []

    accumulated_X: list[np.ndarray] = []
    accumulated_y: list[np.ndarray] = []

    for step_index, (X_step, y_step) in enumerate(train_splits, start=1):
        accumulated_X.append(X_step)
        accumulated_y.append(y_step)
        seen_classes = np.unique(np.concatenate(accumulated_y, axis=0))
        test_mask = np.isin(bundle.y_test, seen_classes)
        step_bundle = DatasetBundle(
            name=f"{bundle.name}[class_incremental,step={step_index}]",
            X_train=np.concatenate(accumulated_X, axis=0),
            X_test=bundle.X_test[test_mask],
            y_train=np.concatenate(accumulated_y, axis=0),
            y_test=bundle.y_test[test_mask],
            num_classes=bundle.num_classes,
            feature_dim=bundle.feature_dim,
        )

        with timed_block() as training_timer:
            if isinstance(model, OSELMClassifier):
                if step_index == 1:
                    model.fit(X_step, y_step)
                else:
                    model.partial_fit(X_step, y_step)
            else:
                model = create_model(args, bundle.feature_dim, bundle.num_classes)
                model.fit(step_bundle.X_train, step_bundle.y_train)

        with timed_block() as inference_timer:
            predictions = model.predict(step_bundle.X_test)

        accuracy = float(np.mean(predictions == step_bundle.y_test))
        metrics = {
            "accuracy": accuracy,
            "training_time_s": training_timer.duration_s,
            "inference_time_s": inference_timer.duration_s,
        }
        rows.extend(
            build_result_rows(format_model_name(args), step_bundle.name, args.seed, metrics),
        )

    return rows


def run_covariate_shift_sequence(
    args: argparse.Namespace,
    bundle: DatasetBundle,
) -> list[dict[str, str | float | int]]:
    """Evaluate a simple additive covariate shift across sequential batches."""
    X_chunks = np.array_split(bundle.X_train, args.steps)
    y_chunks = np.array_split(bundle.y_train, args.steps)
    model = create_model(args, bundle.feature_dim, bundle.num_classes)
    rows: list[dict[str, str | float | int]] = []

    accumulated_X: list[np.ndarray] = []
    accumulated_y: list[np.ndarray] = []

    for step_index, (X_step, y_step) in enumerate(zip(X_chunks, y_chunks, strict=True), start=1):
        shift_value = args.shift_strength * step_index
        shifted_X_step = apply_covariate_shift(X_step, shift_strength=shift_value)
        accumulated_X.append(shifted_X_step)
        accumulated_y.append(y_step)

        shifted_X_test = apply_covariate_shift(bundle.X_test, shift_strength=shift_value)
        step_bundle = DatasetBundle(
            name=f"{bundle.name}[covariate_shift,step={step_index}]",
            X_train=np.concatenate(accumulated_X, axis=0),
            X_test=shifted_X_test,
            y_train=np.concatenate(accumulated_y, axis=0),
            y_test=bundle.y_test,
            num_classes=bundle.num_classes,
            feature_dim=bundle.feature_dim,
        )

        with timed_block() as training_timer:
            if isinstance(model, OSELMClassifier):
                if step_index == 1:
                    model.fit(shifted_X_step, y_step)
                else:
                    model.partial_fit(shifted_X_step, y_step)
            else:
                model = create_model(args, bundle.feature_dim, bundle.num_classes)
                model.fit(step_bundle.X_train, step_bundle.y_train)

        with timed_block() as inference_timer:
            predictions = model.predict(step_bundle.X_test)

        accuracy = float(np.mean(predictions == step_bundle.y_test))
        metrics = {
            "accuracy": accuracy,
            "training_time_s": training_timer.duration_s,
            "inference_time_s": inference_timer.duration_s,
        }
        rows.extend(
            build_result_rows(
                format_model_name(args),
                step_bundle.name,
                args.seed,
                metrics,
            ),
        )

    return rows


def resolve_output_path(args: argparse.Namespace, suffix: str) -> Path:
    """Choose the output CSV location for an experiment command."""
    if args.output is not None:
        return Path(args.output)
    safe_model = args.model.replace("/", "_")
    safe_dataset = args.dataset.replace("/", "_")
    return Path("results") / f"{suffix}_{safe_model}_{safe_dataset}_seed{args.seed}.csv"


def format_model_name(args: argparse.Namespace) -> str:
    """Format a stable model label for CSV output."""
    if args.model == "mlp":
        return f"mlp[depth={args.depth},width={args.width}]"
    if args.model == "elm":
        return f"elm[L={args.L},activation={args.activation}]"
    if args.model == "os_elm":
        return f"os_elm[L={args.L},activation={args.activation}]"
    return "logistic_regression"


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of integers from the CLI."""
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def log_metrics(
    model_name: str,
    dataset_name: str,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Print a short summary of experiment results."""
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
    print(f"Saved results to {output_path}")


def supported_datasets() -> Sequence[str]:
    """Return the built-in datasets exposed by the CLI."""
    return ("iris", "wine", "breast_cancer", "digits", "csv")
