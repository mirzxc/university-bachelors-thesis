---
type: dataset
repo: thesis-elm
status: active
dataset_kind: sklearn_builtin
cli_name: breast_cancer
source_files:
  - "../../../src/thesis_elm/data.py"
tags:
  - dataset
  - sklearn
  - breast-cancer
---

# Breast Cancer

## Status

Built-in sklearn dataset available through the standard loader map in [src/thesis_elm/data.py](../../../src/thesis_elm/data.py).

## Notes

- Used in the docs as a learning-curve style example for the `scaling` command.
- CLI name uses an underscore: `breast_cancer`.

## Example command

```bash
uv run thesis-elm scaling --model mlp --dataset breast_cancer --n-values 50,100,200,300
```
