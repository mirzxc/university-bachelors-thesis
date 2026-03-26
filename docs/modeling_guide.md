# Modeling Guide

This project keeps the same notation in code comments and documentation as the thesis.

## ELM notation reference

- `H`: hidden layer output matrix, shape `(N, L)`
- `W`: random input weights, shape `(input_dim, L)`, frozen after initialization
- `b`: random input biases, shape `(L,)`, frozen after initialization
- `β`: output weights, shape `(L, num_classes)`, solved from least squares
- `T`: one-hot target matrix, shape `(N, num_classes)`
- `N`: number of training samples
- `L`: number of hidden neurons
- `P`: covariance matrix for OS-ELM recursive least squares, shape `(L, L)`

## Theory to code mapping

### Logistic Regression

- Implemented as one `nn.Linear(input_dim, num_classes)`
- Trained with cross-entropy loss and Adam
- Serves as the simplest supervised baseline

### MLP

- Stack of configurable hidden layers with ReLU activations
- Trained with Adam
- Uses early stopping on a validation split to avoid overfitting during longer runs

### ELM

- Hidden layer parameters `W` and `b` are sampled once and then frozen
- The hidden layer matrix `H` is computed as `activation(X @ W + b)`
- The output weights `β` are solved in closed form from `H` and `T`
- The implementation uses `torch.linalg.lstsq`
- A small `l2_reg` term is exposed to reduce issues when `H` is ill-conditioned

### OS-ELM

- Starts from an initial batch to compute the first `β` and `P`
- Updates `β` and `P` sequentially with recursive least squares
- Supports `partial_fit` so sequential experiments can update the same model instance over time

## Numerical stability notes

Start with the simple fixes first:

1. Standardize features before fitting any model.
2. Use a small `l2_reg` value for ELM and OS-ELM.
3. Avoid choosing `L` so large that `H` becomes badly conditioned relative to `N`.

Specific issues to watch:

- If `H` is close to rank-deficient, the solved `β` can become unstable.
- If `L` is much larger than `N`, ELM may fit training data extremely well but generalize poorly.
- OS-ELM updates can drift numerically if `P` becomes poorly conditioned over many updates.

The code keeps these mitigations minimal on purpose so the thesis can discuss them clearly.
