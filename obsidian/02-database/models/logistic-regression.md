---
type: model
repo: thesis-elm
status: active
family: gradient
cli_model: logistic_regression
supports_partial_fit: false
source_files:
  - "../../../src/thesis_elm/models/logistic_regression.py"
  - "../../../src/thesis_elm/models/base.py"
  - "../../../src/thesis_elm/experiments.py"
tags:
  - model
  - baseline
  - logistic-regression
---

# Logistic Regression

## Role

The simplest supervised baseline in the repo. It is implemented as a single `nn.Linear` layer and trained with the shared Adam-based training loop.

## Implementation

- Model file: [src/thesis_elm/models/logistic_regression.py](../../../src/thesis_elm/models/logistic_regression.py)
- Training base class: [src/thesis_elm/models/base.py](../../../src/thesis_elm/models/base.py)
- Factory wiring: [src/thesis_elm/experiments.py](../../../src/thesis_elm/experiments.py)

## Key parameters

- `learning_rate`
- `max_epochs`
- `batch_size`
- `patience`
- `validation_fraction`

## Example command

```bash
uv run thesis-elm run --model logistic_regression --dataset iris
```

## Related notes

- [[mlp]]
- [[../commands/run]]
- [[../docs/modeling-guide]]
