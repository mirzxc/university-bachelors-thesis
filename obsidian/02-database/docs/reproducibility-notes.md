---
type: repo_doc
repo: thesis-elm
status: active
doc_path: "../../../docs/reproducibility.md"
source_files:
  - "../../../docs/reproducibility.md"
  - "../../../src/thesis_elm/utils.py"
tags:
  - docs
  - reproducibility
  - results
---

# Reproducibility Notes

## Source

[docs/reproducibility.md](../../../docs/reproducibility.md)

## Why it matters

This note anchors thesis reporting discipline around seeds, device choices, and result formatting.

## Key takeaways

- Keep seeds explicit and fixed during comparisons.
- Treat CPU as the baseline reproducibility environment.
- Report `L`, preprocessing, training time, and inference time.
- Result rows are standardized as `model,dataset,metric,value,seed`.

## Related notes

- [[../models/elm]]
- [[../models/os-elm]]
- [[../commands/sequential]]
