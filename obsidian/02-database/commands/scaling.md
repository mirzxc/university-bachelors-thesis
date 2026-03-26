---
type: command
repo: thesis-elm
status: active
cli_name: scaling
handler: scaling_command
source_files:
  - "../../../src/thesis_elm/cli.py"
  - "../../../src/thesis_elm/experiments.py"
  - "../../../src/thesis_elm/data.py"
tags:
  - cli
  - command
  - scaling
  - learning-curve
---

# `scaling`

## Purpose

Measures how performance changes as the effective training set size `N` changes. The same command can double as a learning-curve generator.

## Implementation detail

The command calls `subset_training_data(...)` to take the first `n_samples` from the training split, then evaluates the selected model on that subset.

## Common use

```bash
uv run thesis-elm scaling --model elm --dataset digits --n-values 100,300,600,1000 --L 1000
```

## Output pattern

- Default path: `results/scaling_<model>_<dataset>_seed<seed>.csv`

## Related notes

- [[../models/elm]]
- [[../models/mlp]]
- [[../docs/experiment-cookbook]]
