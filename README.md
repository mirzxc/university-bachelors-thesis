# Thesis ELM Experiments

This repository is a reproducible PyTorch workspace for a bachelor thesis on four tabular classification models:

- Logistic Regression
- MLP
- ELM
- OS-ELM

The project is managed with `uv`, uses a `src/` package layout, and exposes CLI entrypoints for single runs, grid search over `L`, scalability experiments over `N`, and sequential experiments for continual learning or covariate shift.

## Quick start

1. Install Python 3.12 for `uv` if needed:

```bash
uv python install 3.12
```

2. Create the virtual environment and install project dependencies:

```bash
uv sync --python 3.12 --extra dev --extra analysis
```

If your shell already has a different virtual environment activated, `uv run ...` may warn that `VIRTUAL_ENV` does not match this project's `.venv`. Deactivate the other environment first with `deactivate` (or `unset VIRTUAL_ENV`) to silence the warning. Use `uv run --active ...` only if you intentionally want to target the currently active environment instead of the project-local `.venv`.

3. Run a baseline experiment:

```bash
uv run thesis-elm run --model logistic_regression --dataset iris
```

4. Run an ELM grid search over hidden neurons `L`:

```bash
uv run thesis-elm grid-search --model elm --dataset wine --L-values 50,100,200,500
```

## Project layout

- `src/thesis_elm/`: models, data loaders, CLI, and experiment runners
- `docs/`: thesis-oriented notes, notation guide, and experiment cookbook
- `tests/`: smoke tests for models, CLI, reproducibility, and CSV outputs
- `results/`: CSV outputs written by experiment commands

## CLI overview

Run one experiment:

```bash
uv run thesis-elm run --model mlp --dataset breast_cancer --depth 3 --width 128
```

Run a scaling experiment over training set size `N`:

```bash
uv run thesis-elm scaling --model elm --dataset digits --n-values 100,300,600,1000 --L 500
```

Run a class-incremental OS-ELM experiment:

```bash
uv run thesis-elm sequential --model os_elm --dataset digits --scenario class_incremental --classes-per-step 2 --L 1000
```

Run a covariate-shift OS-ELM experiment:

```bash
uv run thesis-elm sequential --model os_elm --dataset wine --scenario covariate_shift --steps 5 --shift-strength 0.2
```

Use a CSV dataset:

```bash
uv run thesis-elm run --model elm --dataset csv --csv-path data/my_dataset.csv --target-column label --L 500
```

All experiment commands save CSV rows with the columns `model,dataset,metric,value,seed`.

## CPU and CUDA

The default project setup is CPU-first. If you later want CUDA acceleration, install a CUDA-compatible PyTorch build inside the same `uv` environment and then pass `--device cuda` when running experiments. Keep CPU runs as the reproducibility baseline for the thesis.

## Documentation

- [Modeling Guide](docs/modeling_guide.md)
- [Experiment Cookbook](docs/experiments.md)
- [Reproducibility Notes](docs/reproducibility.md)
- [Obsidian Vault](obsidian/README.md)
