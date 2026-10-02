"""
sobol_sampling.py

Standalone utilities for generating Sobol samples in a unit hypercube,
mapping them to physical parameter bounds, evaluating them point-by-point,
and returning the sampled inputs and outputs.

This module contains no Bayesian optimisation, GP fitting, acquisition
functions, or TuRBO logic.

Example
-------
from sobol_sampling import sobol_sample_and_evaluate

X, Y = sobol_sample_and_evaluate(
    evaluate_fn=evaluate_fn,
    bounds_phys=bounds_phys,
    dim=7,
    n_samples=100,
    seed=1,
)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.quasirandom import SobolEngine


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def unnormalize(
    x_unit: torch.Tensor,
    bounds_phys: torch.Tensor,
) -> torch.Tensor:
    """
    Map points from the unit hypercube [0, 1]^d to physical bounds.

    Parameters
    ----------
    x_unit : torch.Tensor
        Unit-cube samples with shape (..., dim).

    bounds_phys : torch.Tensor
        Physical bounds with shape (2, dim), where:
            bounds_phys[0] = lower bounds
            bounds_phys[1] = upper bounds

    Returns
    -------
    torch.Tensor
        Physical-space samples with the same shape as x_unit.
    """
    if bounds_phys.ndim != 2 or bounds_phys.shape[0] != 2:
        raise ValueError(
            f"bounds_phys must have shape (2, dim), but got {tuple(bounds_phys.shape)}"
        )

    lower = bounds_phys[0].to(dtype=x_unit.dtype, device=x_unit.device)
    upper = bounds_phys[1].to(dtype=x_unit.dtype, device=x_unit.device)

    return lower + (upper - lower) * x_unit


def generate_sobol_samples(
    bounds_phys: torch.Tensor,
    dim: int,
    n_samples: int,
    seed: int = 1,
    dtype: torch.dtype = torch.double,
    device: str | torch.device = "cpu",
    scramble: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Sobol samples in both unit and physical parameter spaces.

    Parameters
    ----------
    bounds_phys : torch.Tensor
        Physical bounds with shape (2, dim).

    dim : int
        Number of parameters.

    n_samples : int
        Number of Sobol samples.

    seed : int, default=1
        Seed used by the scrambled Sobol engine.

    dtype : torch.dtype, default=torch.double
        Tensor dtype.

    device : str or torch.device, default="cpu"
        Device on which samples are created.

    scramble : bool, default=True
        Whether to scramble the Sobol sequence.

    Returns
    -------
    x_unit : torch.Tensor
        Sobol samples in [0, 1]^dim, shape (n_samples, dim).

    x_phys : torch.Tensor
        Samples mapped to physical bounds, shape (n_samples, dim).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if bounds_phys.shape != (2, dim):
        raise ValueError(
            f"bounds_phys must have shape (2, {dim}), "
            f"but got {tuple(bounds_phys.shape)}"
        )

    set_seed(seed)

    sobol = SobolEngine(
        dimension=dim,
        scramble=scramble,
        seed=seed,
    )

    x_unit = sobol.draw(n_samples).to(
        dtype=dtype,
        device=device,
    )

    bounds_phys = bounds_phys.to(
        dtype=dtype,
        device=device,
    )

    x_phys = unnormalize(x_unit, bounds_phys)

    return x_unit, x_phys


def sobol_sample_and_evaluate(
    evaluate_fn: Callable[..., tuple[torch.Tensor, Any]],
    bounds_phys: torch.Tensor,
    dim: int,
    n_samples: int,
    seed: int = 1,
    dtype: torch.dtype = torch.double,
    device: str | torch.device = "cpu",
    scramble: bool = True,
    verbose: bool = True,
    return_unit: bool = False,
) -> tuple[torch.Tensor, np.ndarray] | tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Generate Sobol samples, map them to physical bounds, and evaluate them.

    No Bayesian optimisation is performed.

    The function evaluates one point at a time. This matches an evaluation
    function with an interface such as:

        y, details = evaluate_fn(x_phys)

    where x_phys has shape (1, dim).

    Parameters
    ----------
    evaluate_fn : Callable
        Function used to evaluate a physical-space input.

        Expected interface:

            y, details = evaluate_fn(x)

        where x has shape (1, dim), and y is a torch.Tensor containing
        one scalar objective value.

    bounds_phys : torch.Tensor
        Physical bounds with shape (2, dim).

    dim : int
        Number of input parameters.

    n_samples : int
        Number of Sobol samples to evaluate.

    seed : int, default=1
        Random seed.

    dtype : torch.dtype, default=torch.double
        Tensor dtype.

    device : str or torch.device, default="cpu"
        Device used for sample tensors.

    scramble : bool, default=True
        Whether to scramble the Sobol sequence.

    verbose : bool, default=True
        Print progress during evaluation.

    return_unit : bool, default=False
        If False:
            returns x_phys, y

        If True:
            returns x_unit, x_phys, y

    Returns
    -------
    x_phys : torch.Tensor
        Physical-space input samples, shape (n_samples, dim).

    y : np.ndarray
        Evaluated outputs, shape (n_samples,).

    x_unit : torch.Tensor, optional
        Unit-cube Sobol samples. Returned only when return_unit=True.
    """

    x_unit, x_phys = generate_sobol_samples(
        bounds_phys=bounds_phys,
        dim=dim,
        n_samples=n_samples,
        seed=seed,
        dtype=dtype,
        device=device,
        scramble=scramble,
    )

    y_values: list[float] = []

    for i in range(n_samples):
        x_i = x_phys[i : i + 1]
        result = evaluate_fn(x_i)
        # print(x_i.shape)
        # print(result)

        # Breet's tuning for trapezoid A tests, x axis
        # x_i = torch.tensor(
        #     [
        #         0.004,
        #         0.9,
        #         0.0002,
        #         3.7,
        #         100,
        #         0.0,
        #         0.00196,
        #     ]
        # ).reshape(1, 7)

        # Breet's tuning for trapezoid B tests, x axis
        # x_i = torch.tensor(
        #     [
        #         0.004,
        #         0.9,
        #         0.0001,
        #         3.55,
        #         0.0,
        #         0.0,
        #         0.001,
        #     ]
        # ).reshape(1, 7)

        # test the evaluation function with a single point, to see if it works
        # result = evaluate_fn(x_i)
        # print(f"test result: {result}")
        # asd

        if isinstance(result, tuple):  # type: ignore
            y_i = result[0]
        else:
            # This also allows a simpler evaluation function that returns
            # only the objective tensor.
            y_i = result

        if not isinstance(y_i, torch.Tensor):  # type: ignore
            y_i = torch.as_tensor(y_i)

        if y_i.numel() != 1:
            raise ValueError(
                "evaluate_fn must return one scalar objective per sample. "
                f"Sample {i} returned a tensor with shape {tuple(y_i.shape)}."
            )

        y_value = float(y_i.detach().cpu().reshape(-1)[0].item())
        y_values.append(y_value)

        if verbose:
            print(f"Sobol sample {i + 1:4d}/{n_samples} | y = {y_value:.10g}")

    y = np.asarray(y_values, dtype=np.float64)

    # Return CPU tensors so the generated dataset can be used/saved directly.
    x_unit_cpu = x_unit.detach().cpu()
    x_phys_cpu = x_phys.detach().cpu()

    if verbose:
        best_idx = int(np.argmin(y))

        print("\nSobol sampling finished")
        print(f"Number of samples : {n_samples}")
        print(f"Input dimension   : {dim}")
        print(f"Best objective    : {y[best_idx]:.8g}")
        print(
            "Best input        :",
            x_phys_cpu[best_idx].numpy(),
        )

    if return_unit:
        return x_unit_cpu, x_phys_cpu, y

    return x_phys_cpu, y


def save_sobol_dataset(
    filename: str,
    x_phys: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    param_names: list[str] | None = None,
    output_name: str = "objective",
) -> None:
    """
    Save Sobol inputs and outputs to a CSV file.

    Parameters
    ----------
    filename : str
        Output CSV path.

    x_phys : torch.Tensor or np.ndarray
        Inputs with shape (n_samples, dim).

    y : torch.Tensor or np.ndarray
        Outputs with shape (n_samples,) or (n_samples, 1).

    param_names : list[str] or None
        Optional parameter names. If omitted, p0, p1, ... are used.

    output_name : str, default="objective"
        Name of the output column.
    """
    import pandas as pd

    if isinstance(x_phys, torch.Tensor):
        x_np = x_phys.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_phys)

    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    y_np = y_np.reshape(-1)

    if x_np.ndim != 2:
        raise ValueError(f"x_phys must have shape (n_samples, dim), got {x_np.shape}")

    if x_np.shape[0] != y_np.shape[0]:
        raise ValueError(
            "x_phys and y must contain the same number of samples: "
            f"{x_np.shape[0]} vs {y_np.shape[0]}"
        )

    dim = x_np.shape[1]

    if param_names is None:
        param_names = [f"p{i}" for i in range(dim)]

    if len(param_names) != dim:
        raise ValueError(
            f"param_names contains {len(param_names)} names, "
            f"but the input dimension is {dim}"
        )

    df = pd.DataFrame(x_np, columns=param_names)
    df[output_name] = y_np
    df.to_csv(filename, index=False)

    print(f"Saved Sobol dataset to: {filename}")
