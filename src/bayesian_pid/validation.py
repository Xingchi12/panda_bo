from collections.abc import Sequence
from typing import Any

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll  # pyright: ignore[reportUnknownVariableType]
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from bayesian_pid.metrics import following_error_score
from bayesian_pid.sim import Array, simulate_pid_fast

# Order simulate_pid_fast expects its gain vector in.
GAIN_ORDER = ("kp", "kv", "ki", "kd", "kvff", "kaff", "kpff1", "kpff0", "ktot")


def gain_vector(values: dict[str, float]) -> Array:
    """Pack a name->value mapping into simulate_pid_fast's positional vector.

    Anything unspecified is zero, except ktot which defaults to 1.0 (the value
    the device holds), so omitting it is a no-op rather than a mute controller.
    """
    defaults: dict[str, float] = dict.fromkeys(GAIN_ORDER, 0.0)
    defaults["ktot"] = 1.0
    unknown = set(values) - set(GAIN_ORDER)
    if unknown:
        raise ValueError(f"unknown gains {sorted(unknown)}; expected {GAIN_ORDER}")
    defaults.update(values)
    return np.array([defaults[k] for k in GAIN_ORDER], dtype=np.float64)


# ============================================================
# Objective / evaluation function used in the Optimisation loop
# ============================================================


# def evaluate(pid_tensor, trajectory=pos_ref, return_details=False):
def evaluate(
    pid_tensor: torch.Tensor,
    trajectory: Array,
    a1: Array,
    a2: Array,
    b: Array,
    c: Array,
    u_min: Array,
    u_max: Array,
    max_integral: float,
    param_names: list[str],
    fixed_params: dict[str, Any],
    axes: Sequence[int] | None = None,
    dt: float = 1.0,
    dt_inv: float = 1.0,
    return_details: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]] | None]:
    """
    Evaluate PID parameters using the learned second-order linear dynamics.

    Objective = RMS following error in the plateau window, over `axes`.
    Pass the same `axes` here and to HardwareEvaluator so the simulated and
    measured objectives are on the same scale and the warm start transfers.
    """
    if pid_tensor.ndim == 1:
        pid_tensor = pid_tensor.unsqueeze(0)

    # simulate_pid_fast is compiled for float64; anything else would either fail
    # to match its signature or trigger a silent recompile per dtype.
    trajectory = np.ascontiguousarray(trajectory, dtype=np.float64)
    a1 = np.ascontiguousarray(a1, dtype=np.float64)
    a2 = np.ascontiguousarray(a2, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    u_min = np.ascontiguousarray(u_min, dtype=np.float64)
    u_max = np.ascontiguousarray(u_max, dtype=np.float64)

    pid_np = pid_tensor.detach().cpu().numpy()
    values: list[float] = []
    details: list[dict[str, Any]] = []

    for i in range(pid_np.shape[0]):
        bo_params = pid_np[i]

        gains: dict[str, float] = {
            name: float(value)
            for name, value in zip(param_names, bo_params, strict=True)
        }
        gains.update({k: float(v) for k, v in fixed_params.items()})

        actual_pos, error_hist, control_hist = simulate_pid_fast(
            gain_vector(gains),
            trajectory,
            a1,
            a2,
            b,
            c,
            dt,
            dt_inv,
            u_min,
            u_max,
            max_integral,
        )

        plateau_std = following_error_score(trajectory, actual_pos, axes=axes)

        values.append(plateau_std)
        details.append(
            {
                "pid": {k: gains[k] for k in (*param_names, *fixed_params)},
                "plateau_std": float(plateau_std),
                "actual_pos": actual_pos,
                "error_hist": error_hist,
                "control_hist": control_hist,
            }
        )

    y = torch.tensor(values, dtype=torch.double).unsqueeze(-1)
    if return_details:
        return y, details
    else:
        return y, None


def fit_gp_with_fallback(
    gp: Any,
    mll: Any,
    adam_steps: int = 100,
    adam_lr: float = 0.05,
    verbose: bool = True,
) -> Any:
    """
    Fit an Exact GP using your original L-BFGS-B path, with the same Adam fallback.
    """

    try:
        fit_gpytorch_mll(mll)

    except Exception as e:
        if verbose:
            print(
                f"\n[Warning] L-BFGS-B failed ({type(e).__name__}: {e}). "
                f"Falling back to Adam."
            )

        mll.train()
        optimizer = torch.optim.Adam(gp.parameters(), lr=adam_lr)

        train_inputs = gp.train_inputs[0]
        train_targets = gp.train_targets

        for _ in range(adam_steps):
            optimizer.zero_grad()
            with (
                gpytorch.settings.cholesky_max_tries(15),
                gpytorch.settings.cholesky_jitter(1e-3),
            ):
                output = gp(train_inputs)
                loss = -mll(output, train_targets).sum()
                loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        mll.eval()

    return gp


def build_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dim: int,
    verbose: bool = True,
    objective_floor: float = 1e-12,
) -> tuple[Any, torch.Tensor]:
    """
    Build and fit the GP model.

    This intentionally matches the model setup in the existing TuRBO code:
    - objective standardised outside the GP;
    - GaussianLikelihood with small noise constraint;
    - ScaleKernel(MaternKernel nu=2.5, ARD, same lengthscale constraint).
    """
    # Plain log10, not log10(1 + y/100). The objective spans ~8 orders of
    # magnitude, and the old form squashed everything below y~100 to near zero:
    # the useful region (0.055 to 0.52 on the 3.5v plant) occupied 0.044% of the
    # transformed range, a spread of 0.0014 sd, which the GP's own fitted noise
    # (sd 0.0072) then swamped 5x over. That is the "noise floor" — an artefact
    # of the transform, not physics. log10 gives the same region 12.5% of the
    # range and 0.42 sd, ~290x more resolution where it matters.

    train_y_log = torch.log10(train_y.clamp_min(objective_floor))
    y_std = train_y_log.std(correction=0).clamp_min(1e-8)
    y = (train_y_log - train_y_log.mean()) / y_std

    # no log transfer
    # train_y = train_y.clamp_min(objective_floor)
    # y_std = train_y.std(correction=0).clamp_min(1e-8)
    # y = (train_y - train_y.mean()) / y_std

    # The simulator is deterministic, so observation noise exists only for
    # numerical conditioning. The upper bound is kept well below the
    # good-region signal (0.42 sd after the log10 transform) so the GP cannot
    # explain real structure away as noise; raise it if fits stop converging.
    # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-4))
    # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-2))
    likelihood = GaussianLikelihood()

    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )

    # gp = SingleTaskGP(
    #     train_X=train_x,
    #     train_Y=y,
    #     covar_module=covar_module,
    #     likelihood=likelihood,
    #     # Explicitly None. botorch >=0.12 defaults this to DEFAULT, which applies
    #     # Standardize(m=1) internally — so leaving it unset standardised y a
    #     # second time, on top of the log1p-and-standardise done above. The double
    #     # transform very nearly cancels (Standardize inverts itself in
    #     # untransform_posterior), but it put the GP's actual training targets and
    #     # the noise/lengthscale constraints on a different scale than this
    #     # function's docstring claims.
    #     outcome_transform=None,
    # )

    # A more flexible warper
    # warp_tf = MonotoneSplineWarp(d=train_x.shape[-1], num_bins=3, prior_sd=0.5).to(
    #     train_x
    # )

    # a standard warper, worked well on the surrogate, simpler to learn
    warp_tf = Warp(
        d=train_x.shape[-1],
        indices=list(range(train_x.shape[-1])),
        # use a prior with median at 1.
        # when a=1 and b=1, the Kumaraswamy CDF is the identity function
        concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
        concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
        bounds=torch.tensor([[0.0] * 7, [1.0] * 7]),
    )
    gp = SingleTaskGP(
        train_X=train_x,
        train_Y=y,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=warp_tf,
        outcome_transform=None,
        # outcome_transform=Standardize(m=1), # newly added
    )

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp = fit_gp_with_fallback(gp, mll, verbose=verbose)

    return gp, y


def as_pid_vector(x: torch.Tensor | Array, dim: int) -> Array:
    """Return a PID tensor/array as a flat length-`dim` numpy vector.

    The order matches the caller's param_names (currently 7 gains, not 3).
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size != dim:
        raise ValueError(
            f"Expected PID vector with {dim} values, got shape {np.asarray(x).shape}."
        )
    return arr
