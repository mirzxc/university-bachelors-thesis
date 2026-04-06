---
type: dashboard
repo: thesis-elm
status: active
tags:
  - obsidian
  - thesis
  - repo
---

# Dashboard

## Start here

- [[01-maps/repository-map]]
- [[01-maps/sync-workflow]]
- [[01-maps/literature-tracker]]
- [[02-database/models/logistic-regression]]
- [[02-database/models/mlp]]
- [[02-database/models/elm]]
- [[02-database/models/os-elm]]

## Repo snapshot

- Thesis scope: tabular classification experiments comparing Logistic Regression, MLP, ELM, and OS-ELM
- Runtime stack: Python 3.12, `uv`, PyTorch, NumPy, pandas, scikit-learn
- CLI entrypoint: [src/thesis_elm/cli.py](../src/thesis_elm/cli.py)
- Experiment orchestration: [src/thesis_elm/experiments.py](../src/thesis_elm/experiments.py)
- Data loading and splits: [src/thesis_elm/data.py](../src/thesis_elm/data.py)
- Shared utilities and CSV writing: [src/thesis_elm/utils.py](../src/thesis_elm/utils.py)

## Database slices

### Models

- [[02-database/models/logistic-regression]]
- [[02-database/models/mlp]]
- [[02-database/models/elm]]
- [[02-database/models/os-elm]]

### Commands

- [[02-database/commands/run]]
- [[02-database/commands/grid-search]]
- [[02-database/commands/scaling]]
- [[02-database/commands/sequential]]

### Datasets

- [[02-database/datasets/iris]]
- [[02-database/datasets/wine]]
- [[02-database/datasets/breast-cancer]]
- [[02-database/datasets/digits]]
- [[02-database/datasets/csv]]

### Repo docs

- [[02-database/docs/modeling-guide]]
- [[02-database/docs/experiment-cookbook]]
- [[02-database/docs/reproducibility-notes]]

### Literature

- [[01-maps/literature-tracker]]
- [[02-database/literature/huang-2006-extreme-learning-machine-theory-and-applications]]
- [[02-database/literature/liang-2006-online-sequential-learning-algorithm]]
- [[02-database/literature/huang-2012-elm-regression-and-multiclass-classification]]
- [[02-database/literature/huang-2014-insight-into-extreme-learning-machines]]
- [[02-database/literature/huang-et-al-2015-trends-in-extreme-learning-machines-review]]
- [[02-database/literature/masana-et-al-2023-class-incremental-learning-survey]]

## Practical next notes

- Use [[03-templates/experiment-note]] for each experiment run you want to discuss in the thesis.
- Use [[03-templates/literature-note]] for papers on ELM, OS-ELM, continual learning, or numerical stability.
- Promote papers from `to_read` to `reading` to `read` directly in their note properties.
- Keep a thesis-specific MOC note in this vault once figures and tables start to stabilize.
