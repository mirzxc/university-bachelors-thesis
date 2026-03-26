---
type: command
repo: thesis-elm
status: active
cli_name: run
handler: run_command
source_files:
  - "../../../src/thesis_elm/cli.py"
  - "../../../src/thesis_elm/experiments.py"
tags:
  - cli
  - command
  - baseline
---

# `run`

## Purpose

Runs a single experiment, computes accuracy plus timing metrics, and writes standardized CSV rows.

## Entry points

- Parser: [src/thesis_elm/cli.py](../../../src/thesis_elm/cli.py)
- Handler: [src/thesis_elm/experiments.py](../../../src/thesis_elm/experiments.py)

## Common use

```bash
uv run thesis-elm run --model mlp --dataset breast_cancer --depth 3 --width 128
```

## Output pattern

- Default path: `results/run_<model>_<dataset>_seed<seed>.csv`
- Metrics written: `accuracy`, `training_time_s`, `inference_time_s`

## Related notes

- [[../models/logistic-regression]]
- [[../models/mlp]]
- [[../models/elm]]
- [[../models/os-elm]]
