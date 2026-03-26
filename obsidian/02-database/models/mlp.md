---
type: model
repo: thesis-elm
status: active
family: gradient
cli_model: mlp
supports_partial_fit: false
source_files:
  - "../../../src/thesis_elm/models/mlp.py"
  - "../../../src/thesis_elm/models/base.py"
  - "../../../src/thesis_elm/experiments.py"
tags:
  - model
  - baseline
  - mlp
---

# MLP

## Role

A configurable multilayer perceptron baseline for tabular classification. It reuses the shared early-stopping training loop from `GradientClassifier`.

## Implementation

- Model file: [src/thesis_elm/models/mlp.py](../../../src/thesis_elm/models/mlp.py)
- Training base class: [src/thesis_elm/models/base.py](../../../src/thesis_elm/models/base.py)
- Factory wiring: [src/thesis_elm/experiments.py](../../../src/thesis_elm/experiments.py)

## Key parameters

- `depth`
- `width`
- `learning_rate`
- `max_epochs`
- `batch_size`
- `patience`

## Example command

```bash
uv run thesis-elm run --model mlp --dataset wine --depth 2 --width 128
```

## Related notes

- [[logistic-regression]]
- [[../commands/run]]
- [[../commands/scaling]]
