---
type: dataset
repo: thesis-elm
status: active
dataset_kind: sklearn_builtin
cli_name: wine
source_files:
  - "../../../src/thesis_elm/data.py"
tags:
  - dataset
  - sklearn
  - wine
---

# Wine

## Status

Built-in sklearn dataset available through the standard loader map in [src/thesis_elm/data.py](../../../src/thesis_elm/data.py).

## Notes

- Used in the seeded examples for MLP, ELM, OS-ELM, and covariate-shift runs.
- Accessible directly through `--dataset wine`.

## Example command

```bash
uv run thesis-elm run --model elm --dataset wine --L 500
```
