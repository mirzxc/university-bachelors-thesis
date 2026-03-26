---
type: dataset
repo: thesis-elm
status: active
dataset_kind: sklearn_builtin
cli_name: digits
source_files:
  - "../../../src/thesis_elm/data.py"
tags:
  - dataset
  - sklearn
  - digits
---

# Digits

## Status

Built-in sklearn dataset available through the standard loader map in [src/thesis_elm/data.py](../../../src/thesis_elm/data.py).

## Notes

- Used in the docs for `L` sweeps, scalability experiments, and class-incremental OS-ELM runs.
- Accessible directly through `--dataset digits`.

## Example command

```bash
uv run thesis-elm sequential --model os_elm --dataset digits --scenario class_incremental --classes-per-step 2 --L 1000
```
