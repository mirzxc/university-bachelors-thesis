"""Command-line interface for thesis experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from thesis_elm.experiments import (
    grid_search_command,
    run_command,
    scaling_command,
    sequential_command,
    supported_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser and all subcommands."""
    parser = argparse.ArgumentParser(
        description="Run thesis experiments for ELM models in PyTorch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one experiment.")
    add_common_arguments(run_parser)
    run_parser.set_defaults(handler=run_command)

    grid_parser = subparsers.add_parser("grid-search", help="Search over hidden neurons L.")
    add_common_arguments(grid_parser)
    grid_parser.add_argument("--L-values", default="50,100,200,500,1000,2000")
    grid_parser.set_defaults(handler=grid_search_command)

    scaling_parser = subparsers.add_parser("scaling", help="Measure performance as N changes.")
    add_common_arguments(scaling_parser)
    scaling_parser.add_argument("--n-values", default="1000,5000,10000,50000")
    scaling_parser.set_defaults(handler=scaling_command)

    sequential_parser = subparsers.add_parser(
        "sequential",
        help="Run sequential learning experiments.",
    )
    add_common_arguments(sequential_parser)
    sequential_parser.add_argument(
        "--scenario",
        choices=("class_incremental", "covariate_shift"),
        default="class_incremental",
    )
    sequential_parser.add_argument("--classes-per-step", type=int, default=1)
    sequential_parser.add_argument("--steps", type=int, default=5)
    sequential_parser.add_argument("--shift-strength", type=float, default=0.15)
    sequential_parser.set_defaults(handler=sequential_command)

    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared dataset, model, and output options to a parser."""
    parser.add_argument(
        "--model",
        choices=("logistic_regression", "mlp", "elm", "os_elm"),
        default="logistic_regression",
    )
    parser.add_argument("--dataset", choices=tuple(supported_datasets()), default="iris")
    parser.add_argument("--csv-path")
    parser.add_argument("--target-column")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--no-standardize", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)

    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--activation", choices=("sigmoid", "relu"), default="sigmoid")
    parser.add_argument("--l2-reg", type=float, default=1e-6)
    parser.add_argument("--initial-batch-size", type=int)
    parser.add_argument("--update-chunk-size", type=int, default=1)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
