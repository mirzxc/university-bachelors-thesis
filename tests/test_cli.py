from __future__ import annotations

import csv
from pathlib import Path

from thesis_elm.cli import main


def test_cli_run_command_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "single_run.csv"
    exit_code = main(
        [
            "run",
            "--model",
            "logistic_regression",
            "--dataset",
            "iris",
            "--max-epochs",
            "50",
            "--output",
            str(output_path),
        ],
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert exit_code == 0
    assert output_path.exists()
    assert {row["metric"] for row in rows} == {"accuracy", "training_time_s", "inference_time_s"}


def test_cli_grid_search_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "grid_search.csv"
    exit_code = main(
        [
            "grid-search",
            "--model",
            "elm",
            "--dataset",
            "iris",
            "--L-values",
            "20,40",
            "--output",
            str(output_path),
        ],
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert exit_code == 0
    assert output_path.exists()
    assert len(rows) == 6
    assert {row["model"] for row in rows} == {
        "elm[L=20,activation=sigmoid]",
        "elm[L=40,activation=sigmoid]",
    }
