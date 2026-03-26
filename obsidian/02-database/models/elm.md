---
type: model
repo: thesis-elm
status: active
family: closed-form
cli_model: elm
supports_partial_fit: false
source_files:
  - "../../../src/thesis_elm/models/elm.py"
  - "../../../src/thesis_elm/experiments.py"
  - "../../../docs/modeling_guide.md"
tags:
  - model
  - elm
  - thesis-core
---

# ELM

## Role

Extreme Learning Machine with a frozen random hidden layer and output weights solved in closed form from the hidden-layer matrix `H` and one-hot targets `T`.

## Implementation

- Model file: [src/thesis_elm/models/elm.py](../../../src/thesis_elm/models/elm.py)
- Factory wiring: [src/thesis_elm/experiments.py](../../../src/thesis_elm/experiments.py)
- Theory note: [docs/modeling_guide.md](../../../docs/modeling_guide.md)

## Key parameters

- `L`
- `activation`
- `l2_reg`
- `rcond`

## Numerical concerns

- Large `L` relative to `N` can hurt conditioning and generalization.
- `l2_reg` is the main stabilization knob exposed by the implementation.
- Feature standardization is part of the normal pipeline and should stay on for comparisons.

## Example commands

```bash
uv run thesis-elm run --model elm --dataset wine --L 500 --activation sigmoid
uv run thesis-elm grid-search --model elm --dataset digits --L-values 50,100,200,500,1000,2000
```

## Related notes

- [[os-elm]]
- [[../commands/grid-search]]
- [[../docs/modeling-guide]]
