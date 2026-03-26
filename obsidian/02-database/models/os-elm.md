---
type: model
repo: thesis-elm
status: active
family: online-closed-form
cli_model: os_elm
supports_partial_fit: true
source_files:
  - "../../../src/thesis_elm/models/os_elm.py"
  - "../../../src/thesis_elm/experiments.py"
  - "../../../docs/modeling_guide.md"
tags:
  - model
  - os-elm
  - continual-learning
---

# OS-ELM

## Role

Online Sequential Extreme Learning Machine with a frozen random hidden layer and recursive least squares updates over `beta` and `P`.

## Implementation

- Model file: [src/thesis_elm/models/os_elm.py](../../../src/thesis_elm/models/os_elm.py)
- Factory wiring: [src/thesis_elm/experiments.py](../../../src/thesis_elm/experiments.py)
- Theory note: [docs/modeling_guide.md](../../../docs/modeling_guide.md)

## Key parameters

- `L`
- `activation`
- `l2_reg`
- `initial_batch_size`
- `update_chunk_size`

## Distinguishing behavior

- Supports `partial_fit(...)` for sequential experiments.
- Reuses the same hidden layer instead of rebuilding the model at each step.
- Tracks the covariance-like matrix `P` for recursive updates.

## Example commands

```bash
uv run thesis-elm run --model os_elm --dataset wine --L 500 --initial-batch-size 200
uv run thesis-elm sequential --model os_elm --dataset digits --scenario class_incremental --classes-per-step 2 --L 1000
```

## Related notes

- [[elm]]
- [[../commands/sequential]]
- [[../docs/reproducibility-notes]]
