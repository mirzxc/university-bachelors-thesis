from __future__ import annotations

import csv
from pathlib import Path

from thesis_elm.utils import build_result_rows, write_results_csv


def test_write_results_csv_has_expected_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "results.csv"
    rows = build_result_rows(
        model_name="elm[L=100]",
        dataset_name="iris",
        seed=42,
        metrics={"accuracy": 0.95, "training_time_s": 0.1},
    )

    write_results_csv(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        loaded_rows = list(reader)

    assert reader.fieldnames == ["model", "dataset", "metric", "value", "seed"]
    assert len(loaded_rows) == 2
    assert loaded_rows[0]["model"] == "elm[L=100]"
    assert loaded_rows[0]["dataset"] == "iris"
    assert loaded_rows[0]["seed"] == "42"
