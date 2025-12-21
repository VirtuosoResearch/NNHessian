# NNHessian

`NNHessian` is a PyTorch-based utility for estimating the Hessian properties of a neural network model. The goal is to build an interface for utilizing the spectral statistics of Neural Network Hessians to analyze generalization and optimization properties of black-box models such as large language models.
It supports Hutchinson trace estimation, power iteration for maximum eigenvalue, and other spectral analysis tools.
This project is currently work in progress.

---

## Installation

Simply copy the `NNHessian` class into your project or package it into your own library.  
Requires:

- Python 3.8+
- PyTorch

---

## Class: `NNHessianCalculator`

### Initialization

```python
NNHessianCalculator(
    model: nn.Module,
    loss_fn: Callable,
    dataloader: Optional[Iterable] = None,
    external_load_batch_func: Optional[Callable] = None,
    assigned_parameters: Optional[Iterable[ParamSelector]] = None,
    device: Union[str, torch.device] = "cpu",
    aggregate_method: str = "mean",
)
```

**Parameters:**

- `model` – PyTorch model whose Hessian you want to analyze.
- `loss_fn` – Loss function used for Hessian computation.
- `dataloader` – Iterable of data batches (optional if using `external_load_batch_func`).
- `external_load_batch_func` – Custom batch-loading function (optional).
- `assigned_parameters` – Specific parameters to include (optional).
- `device` – `"cpu"` or `"cuda"`.
- `aggregate_method` – How to aggregate Hessian estimates (`"mean"`, `"sum"`, etc.).

------

### Methods

#### `hutchinson_trace`

```
hutchinson_trace(
    num_samples: int = 50,
    distribution: str = "rademacher",
    dataloader=None,
    seed: int = None,
    return_std: bool = False
)
```

Estimates the trace of the Hessian using Hutchinson's method.

------

#### `max_eigenvalue_power`

```
max_eigenvalue_power(
    num_iters: int = 50,
    tol: float = 1e-5,
    dataloader=None,
    init_vec: torch.Tensor = None,
    distribution: str = "rademacher",
    seed: int = None,
    which: str = "lm",
    return_vec: bool = False
)
```

Finds the maximum eigenvalue (and optionally eigenvector) of the Hessian using power iteration.
 `which` can be:

- `"lm"` – largest magnitude eigenvalue
- `"la"` – largest algebraic eigenvalue

------

#### `hutch_pp_trace_estimator`

```
hutch_pp_trace_estimator(m: int)
```

Estimates the Hessian trace using the Hutch++ method with `m` probing vectors.

------

#### `get_full_spectrum`

```
get_full_spectrum(
    n_v: int,
    n_iter: int,
    dataloader=None
)
```

Computes the full eigenvalue spectrum of the Hessian using Lanczos iterations.

------

## Example Usage

```
from your_module import NNHessianCalculator
import torch
import torch.nn as nn

# Example model and loss
model = nn.Linear(10, 1)
loss_fn = nn.MSELoss()

# Example dataloader
data = torch.randn(100, 10)
targets = torch.randn(100, 1)
dataloader = [(data, targets)]

# Create calculator
calc = NNHessianCalculator(model, loss_fn, dataloader)

# Estimate Hessian trace
trace = calc.hutchinson_trace()
print("Estimated trace:", trace)

# Find max eigenvalue
max_eig = calc.max_eigenvalue_power()
print("Max eigenvalue:", max_eig)
```
