# Reproducibility Notes

## Seed handling

Every experiment uses one explicit integer seed. The helper `seed_everything` sets:

- `random.seed`
- `numpy.random.seed`
- `torch.manual_seed`
- `torch.cuda.manual_seed_all` when CUDA is available

## Determinism caveats

- CPU runs are the recommended reproducibility baseline for thesis tables and figures.
- GPU runs can differ slightly across devices, PyTorch versions, and kernels.
- ELM and OS-ELM are especially sensitive to the initial random `W` and `b`, so keep seeds fixed during comparisons.

## Reporting guidance

When comparing models:

- keep the same train/test split seed
- keep the same preprocessing settings
- report `L` explicitly for ELM and OS-ELM
- log both training time and inference time

## Results format

Every experiment writes CSV rows with:

- `model`
- `dataset`
- `metric`
- `value`
- `seed`

Typical metrics are:

- `accuracy`
- `training_time_s`
- `inference_time_s`

## CPU vs CUDA

Start on CPU and only move to CUDA once the pipeline is stable. If you later install a CUDA-enabled PyTorch build, pass `--device cuda` explicitly and treat that as a separate experimental condition in the thesis.
