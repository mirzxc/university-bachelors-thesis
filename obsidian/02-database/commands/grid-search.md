---
type: command
repo: thesis-elm
status: active
cli_name: grid-search
handler: grid_search_command
source_files:
  - "../../../src/thesis_elm/cli.py"
  - "../../../src/thesis_elm/experiments.py"
tags:
  - cli
  - command
  - hyperparameter-search
---

# `grid-search`

## Purpose

Sweeps over a comma-separated list of hidden-neuron counts `L` and appends result rows for each run.

## Constraints

- Only supports `elm` and `os_elm`
- Mutates `args.L` inside the loop before each model creation

## Common use

```bash
uv run thesis-elm grid-search --model elm --dataset digits --L-values 50,100,200,500,1000,2000
```

## Output pattern

- Default path: `results/grid_search_<model>_<dataset>_seed<seed>.csv`

## Related notes

- [[../models/elm]]
- [[../models/os-elm]]
- [[../docs/experiment-cookbook]]
