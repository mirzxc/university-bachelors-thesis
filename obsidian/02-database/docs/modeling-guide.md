---
type: repo_doc
repo: thesis-elm
status: active
doc_path: "../../../docs/modeling_guide.md"
source_files:
  - "../../../docs/modeling_guide.md"
tags:
  - docs
  - theory
  - modeling
---

# Modeling Guide

## Source

[docs/modeling_guide.md](../../../docs/modeling_guide.md)

## Why it matters

This is the main theory-to-code bridge for the thesis. It defines the notation used across the codebase and highlights the numerical stability concerns around ELM and OS-ELM.

## Key takeaways

- `H`, `W`, `b`, `beta`, `T`, `N`, `L`, and `P` are the main symbols to keep consistent between code and thesis text.
- Logistic Regression and MLP are gradient-trained baselines.
- ELM solves the output layer in closed form.
- OS-ELM updates the same solved output weights sequentially.

## Related notes

- [[../models/elm]]
- [[../models/os-elm]]
