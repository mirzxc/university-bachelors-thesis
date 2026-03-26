# Experiment Cookbook

This guide maps the planned thesis experiments to CLI commands.

## 1. Baselines

Logistic Regression:

```bash
uv run thesis-elm run --model logistic_regression --dataset iris
```

MLP:

```bash
uv run thesis-elm run --model mlp --dataset wine --depth 2 --width 128
```

ELM:

```bash
uv run thesis-elm run --model elm --dataset wine --L 500 --activation sigmoid
```

OS-ELM:

```bash
uv run thesis-elm run --model os_elm --dataset wine --L 500 --initial-batch-size 200
```

## 2. Grid search over `L`

Use this for the thesis hyperparameter sweep:

```bash
uv run thesis-elm grid-search --model elm --dataset digits --L-values 50,100,200,500,1000,2000
```

You can also search OS-ELM:

```bash
uv run thesis-elm grid-search --model os_elm --dataset digits --L-values 50,100,200,500
```

## 3. Scalability over `N`

Measure how training time changes as the training set grows:

```bash
uv run thesis-elm scaling --model elm --dataset digits --n-values 100,300,600,1000 --L 1000
```

The CSV output stores training time and inference time alongside accuracy.

## 4. Generalization and learning curves

Reuse the scaling command as a learning-curve generator:

```bash
uv run thesis-elm scaling --model mlp --dataset breast_cancer --n-values 50,100,200,300
```

Interpret the accuracy rows as learning-curve points.

## 5. Continual learning with sequential class introduction

OS-ELM is the intended model here:

```bash
uv run thesis-elm sequential --model os_elm --dataset digits --scenario class_incremental --classes-per-step 2 --L 1000
```

The command evaluates after each step on the subset of test classes seen so far.

## 6. Covariate shift

Use sequential batches with increasing additive feature shifts:

```bash
uv run thesis-elm sequential --model os_elm --dataset wine --scenario covariate_shift --steps 5 --shift-strength 0.15
```

For non-OS-ELM models, the command refits from accumulated shifted data each step. For OS-ELM, it updates the same model instance with `partial_fit`.
