---
type: command
repo: thesis-elm
status: active
cli_name: sequential
handler: sequential_command
source_files:
  - "../../../src/thesis_elm/cli.py"
  - "../../../src/thesis_elm/experiments.py"
  - "../../../src/thesis_elm/data.py"
tags:
  - cli
  - command
  - continual-learning
  - covariate-shift
---

# `sequential`

## Purpose

Runs either class-incremental or covariate-shift experiments step by step and logs metrics after each step.

## Scenarios

- `class_incremental`
- `covariate_shift`

## Important behavior

- `os_elm` updates the same model instance with `partial_fit(...)`
- Other models are recreated and refit on accumulated data

## Common use

```bash
uv run thesis-elm sequential --model os_elm --dataset digits --scenario class_incremental --classes-per-step 2 --L 1000
uv run thesis-elm sequential --model os_elm --dataset wine --scenario covariate_shift --steps 5 --shift-strength 0.15
```

## Output pattern

- Default path: `results/<scenario>_<model>_<dataset>_seed<seed>.csv`

## Related notes

- [[../models/os-elm]]
- [[../docs/experiment-cookbook]]
- [[../docs/reproducibility-notes]]
