---
type: dataset
repo: thesis-elm
status: active
dataset_kind: sklearn_builtin
cli_name: iris
source_files:
  - "../../../src/thesis_elm/data.py"
tags:
  - dataset
  - sklearn
  - iris
---

# Iris

## Status

Built-in sklearn dataset available through the standard loader map in [src/thesis_elm/data.py](../../../src/thesis_elm/data.py).

## Notes

- Good for smoke tests and baseline sanity checks.
- Accessible directly through `--dataset iris`.

## Example command

```bash
uv run thesis-elm run --model logistic_regression --dataset iris
```
