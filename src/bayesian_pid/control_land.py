"""
Created on Fri Jun 12 10:21:53 2026

@author: Xingchi Liu

Use the one-direction trajectory as the demand

Here the PID is based on Brett's implementation'

Run with `python -m bayesian_pid.Control_land`. To create sobol smaples
in the defined parameter space. It is defined as 'TOTAL_SAMPLE'.
Pass --sim-only to make every hardware path unreachable, for unattended runs.
"""

# matplotlib types its Axes/Figure methods with `**kwargs: Unknown`, so the
# plotting calls below read as "partially unknown" under strict mode no matter
# what we annotate.
# pyright: reportUnknownMemberType=false

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from bayesian_pid.hardware import HardwareEvaluator
from bayesian_pid.metrics import DEFAULT_WINDOW
from bayesian_pid.optimiser import (
    run_one_optimization,
    select_warm_start,
)
from bayesian_pid.panda import (
    PANDA_HOST,
    PANDA_PID_FIELDS,
    ask_run_hardware,
    ask_run_mode,
    health_check,
    read_pid_values,
    validate_bounds,
)
from bayesian_pid.plotting import plot_single_result
from bayesian_pid.random_scan import save_sobol_dataset, sobol_sample_and_evaluate
from bayesian_pid.sim import Array
from bayesian_pid.utils import DEFAULT_BO_THREADS
from bayesian_pid.validation import as_pid_vector, evaluate

# ============================================================
# The code is for generating sample of parameters combinations
# Then get the corresponding FE and visualise the FE landscape

# ============================================================
# Configuration — this is the only block to edit
# ============================================================

# DATA_PATH = "/workspaces/bayesian_pid/data"
DATA_PATH = "/workspaces/panda_bo/data"

# Which identified plant model to simulate against. The trajectory must be the
# one the model was identified from.
PLANT_SUFFIX = "_3.5v"  # "" for the 1v_1 data set, "_3.5v" for 3.5v_1
TRAJECTORY_FILE = "xonly_trajectory.npy"

# Output and integral limits, read from the live device (BRETT_PID.MAX_OUTPUT_I
# and MAX_INTEGRAL_I on bl99p-mo-panda-03). The simulation must clamp exactly
# where the FPGA does: it previously used 1000 and 1e9, so it saturated the
# control ~3.9x earlier than hardware and left the integral effectively
# unbounded where the device caps it at 28000.
MAX_OUTPUT = 3932.0
MAX_INTEGRAL = 28000.0

# The VHDL's dt_i and dt_inv_i registers. These are INDEPENDENT of each other
# and of pid_period_i (which sets the servo rate, currently 10 kHz). dt scales
# the integral term, dt_inv scales the velocity/desired-velocity/derivative
# terms.
#
# Both are 1.0 so the model reduces to the PMAC C reference, which has no dt at
# all: the integral is Ki*PosError and the velocities are per servo cycle. Set
# these only to what the device actually holds — deriving DT_INV as 1/DT
# rescales kv, kvff, kaff and ki by four orders of magnitude and makes tuned
# gains meaningless on hardware.
DT = 1.0
DT_INV = 1.0

# Which axis columns the objective scores, in both simulation and on hardware.
# (0,) = x only, which is all the hardware measures with a single encoder.
# Set to None to score every axis, or e.g. (0, 1) for x and y — but the hardware
# evaluator then needs a matching position_field per axis.
SCORE_AXES: tuple[int, ...] | None = (0,)

# -------------------------------------------------------
# Declare which PID gains to auto-tune (and their bounds)
# vs which to hold fixed.
#
# Gain units follow brett_pid_ff.vhd with dt_i = dt_inv_i = 1, which is the same
# convention as the PMAC C reference: velocities are per servo cycle and the
# integral increment is ki * error. validate_bounds() checks at startup that
# every bound is writable to the device's fixed-point fields.
#
# Note kpff1/kpff0 act on the raw demand, which reaches 500,000 counts, so with
# MAX_OUTPUT = 3932 anything above ~8e-3 (kpff1) saturates the output on
# feedforward alone. Most of their declared range is a flat, saturated region.
# -------------------------------------------------------
TUNE_PARAMS: dict[str, tuple[float, float]] = {
    # name:  (lo,   hi)     device field limit
    "kp": (0.0, 2.0),  # KP_I     < 64
    "kv": (0.0, 2.0),  # KV_I     < 64
    "ki": (0.0, 2.0),  # KI_I     < 64
    "kvff": (0.0, 10.0),  # KVFF_I   < 64
    "kaff": (0.0, 300.0),  # KAFF_I   < 2048
    "kpff1": (0.0, 1.0),  # KPFF1_I  < 64
    "kpff0": (0.0, 1.0),  # KPFF0_I  < 64
    # Use panda.full_positive_range(name) for the widest writable range, but
    # note these narrower bounds already bracket the known-good operating point
    # (kp~0.87, kv~2.0, ki~0.008, kvff~4.7, kaff~215) at dt = dt_inv = 1.
}

# Hold vals constant whilst others are tuned
FIXED_PARAMS: dict[str, float] = {
    "kd": 0.0,
    # "kaff":  0.0,
    # "kpff1": 0.0,
    # "kpff0": 0.0,
    # "kvff": 3.7,
    # "kaff": 200.0,
    # "kpff1": 0.002,
    # "kpff0": 0.0,
}

METHOD = "Turbo"  # change to "standard_bo" for ordinary BO without trust region
RUN_MC = True  # False: one optimisation run; True: MC runs over different seeds
N_MC_RUNS = 50  # How many MC runs
SEED0 = 1  # MC seeds will be SEED0, SEED0+1, ..., SEED0+N_MC_RUNS-1
TOTAL_BUDGET = 1  # total number of function evaluations per simulated run
LIVE_BUDGET = 1  # total number of function evaluations per live run
N_INIT = 10  # initial Sobol points per run/restart
TOTAL_SAMPLE = 2**17  # number of random samples
# -------------------------------------------------------
# BO loop compute.
#
# BO_THREADS is 1 so that a seeded run replays EXACTLY. Above one thread, BLAS
# sums reductions in a different grouping, the GP fit moves a few hundred ULPs,
# and because every candidate is chosen from a GP fitted to every previous one,
# that compounds: 1e-13 to 1e-2 in thirteen iterations, after which the two runs
# are searching different places. Not a race, and not suppressible — it is what a
# chaotic search does to any perturbation. See utils.DEFAULT_BO_THREADS.
#
# Take parallelism from MC_JOBS instead. Seeds are independent, so they run as
# separate single-threaded processes: same answers, all the cores. n_jobs does
# not change results and a test pins that.
#
# Raising BO_THREADS is ~2.2x on the GP fit only above n~200, and costs
# reproducibility. Do it only for a throwaway run.
# -------------------------------------------------------
BO_THREADS: int | None = None  # None -> utils.DEFAULT_BO_THREADS (1)
MC_JOBS = min(N_MC_RUNS, os.cpu_count() or 1)  # MC seeds run concurrently

# -------------------------------------------------------
# Hardware warm start. The live device must never be given a randomly drawn
# gain set — that is the whole reason the simulation phase exists.
# -------------------------------------------------------
WARM_START_K = 1  # sim points to seed hardware BO with; 1 = the single best
WARM_START_MIN_SEPARATION = 0.05  # min unit-cube distance between those points
# What a TuRBO restart does on hardware once the trust region collapses:
#   "warm_region" -> re-seed near the warm-start points (keeps escaping local minima)
#   "stop"        -> halt and return the best found so far
HW_RESTART_POLICY = "warm_region"

SAVE_MC_CSV = True
RESULTS_ROOT = "xonly_trajectory_turbo_mc_results"
# Each run writes to its own timestamped subdirectory, so re-running never
# overwrites a previous run's CSVs and plots. Set False to write directly into
# RESULTS_ROOT and overwrite in place.
TIMESTAMP_RESULTS = True

# The device configuration the simulation assumes. health_check compares this
# against the live registers and fails loudly on any disagreement.
EXPECTED_DEVICE_CONFIG: dict[str, float] = {
    "dt": DT,
    "dt_inv": DT_INV,
    "ktot": 1.0,
    "max_output": MAX_OUTPUT,
    "max_integral": MAX_INTEGRAL,
    "dir_toggle": 0.0,
}

DEVICE = torch.device("cpu")
DTYPE = torch.double


# ============================================================
# Setup helpers
# ============================================================


def load_plant(
    data_path: str = DATA_PATH, suffix: str = PLANT_SUFFIX
) -> tuple[Array, Array, Array, Array]:
    """Load the identified second-order plant: x' = a1@x1 + a2@x0 + b@u + c."""
    return (
        np.load(f"{data_path}/A1{suffix}.npy"),
        np.load(f"{data_path}/A2{suffix}.npy"),
        np.load(f"{data_path}/B{suffix}.npy"),
        np.load(f"{data_path}/c{suffix}.npy"),
    )


def load_trajectory(
    data_path: str = DATA_PATH, filename: str = TRAJECTORY_FILE
) -> Array:
    """Load the x-only demand and pad it with zero y/z to make it 3-D."""
    x = np.load(f"{data_path}/{filename}")
    return np.concatenate([x.reshape(-1, 1), np.zeros((len(x), 2))], axis=1)


def make_results_dir() -> str:
    """Create this run's output directory."""
    directory = RESULTS_ROOT
    if TIMESTAMP_RESULTS:
        directory = os.path.join(
            RESULTS_ROOT, datetime.now().strftime("run_%Y%m%d_%H%M%S")
        )
    os.makedirs(directory, exist_ok=True)
    return directory


def make_bounds(tune_params: dict[str, tuple[float, float]]) -> torch.Tensor:
    return torch.tensor(
        [
            [lo for lo, _ in tune_params.values()],
            [hi for _, hi in tune_params.values()],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )


def build_sim_objective(
    trajectory: Array,
    plant: tuple[Array, Array, Array, Array],
    param_names: list[str],
) -> Callable[..., Any]:
    """The closure BO calls each iteration: gains -> plateau RMS following error."""
    a1, a2, b, c = plant
    u_max = MAX_OUTPUT * np.ones(3)
    return partial(
        evaluate,
        trajectory=trajectory,
        a1=a1,
        a2=a2,
        b=b,
        c=c,
        u_min=-u_max,
        u_max=u_max,
        max_integral=MAX_INTEGRAL,
        param_names=param_names,
        fixed_params=FIXED_PARAMS,
        axes=SCORE_AXES,
        dt=DT,
        dt_inv=DT_INV,
    )


def _make_progress_fn(names: list[str]) -> Callable[..., None]:
    def _progress(
        eval_count: int,
        total: int,
        x_phys: torch.Tensor,
        y: float,
        best_y: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra = extra or {}
        phase = extra.get("phase", "bo")
        restart = extra.get("restart", None)
        tr = extra.get("tr_length", None)
        x_flat = x_phys.detach().cpu().numpy().flatten()
        params_str = "  ".join(
            f"{n}={v:.4g}" for n, v in zip(names, x_flat, strict=False)
        )
        restart_str = f" R{restart}" if restart else ""
        tr_str = f"  TR={tr:.4f}" if tr is not None else ""
        t_str = f"  t={extra['eval_time']:.1f}s" if "eval_time" in extra else ""
        print(
            f"[{phase:4s} | {restart_str} | {eval_count:4d}/{total}]: "
            f"rms_err={y:.4g} | best={best_y:.4g}{tr_str}{t_str} | {params_str}",
            flush=True,
        )

    return _progress


class _Tee:
    """Write to several streams at once, so stdout is also captured to a log."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def print_config(run_mode: str, results_dir: str, allow_hardware: bool) -> None:
    print("--- Tuned parameters ---")
    for name, (lo, hi) in TUNE_PARAMS.items():
        print(f"  {name:<8}  [{lo}, {hi}]")
    if FIXED_PARAMS:
        print("--- Fixed parameters ---")
        for name, val in FIXED_PARAMS.items():
            print(f"  {name:<8}  {val}")
    print("--- Run config ---")
    print(f"  method       {METHOD}")
    print(f"  mc_runs      {N_MC_RUNS}  (seeds {SEED0}-{SEED0 + N_MC_RUNS - 1})")
    print(f"  sim_budget   {TOTAL_BUDGET} evals  (n_init={N_INIT})")
    print(
        f"  live_budget  {LIVE_BUDGET} evals  "
        f"(warm start k={WARM_START_K}, restart={HW_RESTART_POLICY})"
    )
    print(
        f"  plateau      [{DEFAULT_WINDOW.start_frac}, {DEFAULT_WINDOW.end_frac}] "
        f"(fraction of trajectory)"
    )
    print(f"  score_axes   {'all' if SCORE_AXES is None else SCORE_AXES}")
    threads = DEFAULT_BO_THREADS if BO_THREADS is None else BO_THREADS
    print(
        f"  compute      {MC_JOBS} MC worker(s) x {threads} torch thread(s)"
        f"{'' if threads == 1 else '  [NOT reproducible: threads > 1]'}"
    )
    print(f"  dt / dt_inv  {DT} / {DT_INV}  (VHDL dt_i, dt_inv_i)")
    print(f"  plant        A1{PLANT_SUFFIX} / A2{PLANT_SUFFIX} / B{PLANT_SUFFIX}")
    print(f"  run_mode     {run_mode}")
    print(f"  hardware     {'enabled' if allow_hardware else 'DISABLED (--sim-only)'}")
    print(f"  results_dir  {results_dir}")
    print()


# ============================================================
# Hardware
# ============================================================


def make_hardware_evaluator(
    param_names: list[str], trajectory: Array, debug_plot_path: str
) -> HardwareEvaluator:
    pid_fields = {n: PANDA_PID_FIELDS[n] for n in param_names if n in PANDA_PID_FIELDS}
    # trajectory[:, 0] is already in PGEN integer counts (max ~500,000).
    # round() matches np.round's half-to-even behaviour and stays typed.
    pgen_trajectory = [str(round(float(v))) for v in trajectory[:, 0]]
    return HardwareEvaluator(
        host=PANDA_HOST,
        pid_fields=pid_fields,
        fixed_fields=FIXED_PARAMS,
        trajectory=pgen_trajectory,
        axes=SCORE_AXES,
        debug_plot_path=debug_plot_path,
    )


def run_direct_hardware(
    param_names: list[str],
    bounds_phys: torch.Tensor,
    trajectory: Array,
    debug_plot_path: str,
    progress_fn: Callable[..., None],
) -> dict[str, Any] | None:
    """Tune on hardware directly, warm-started from the device's current gains."""
    if not health_check(
        PANDA_HOST,
        param_names,
        prompt="Continue to hardware tuning? [y/N]: ",
        expected_config=EXPECTED_DEVICE_CONFIG,
    ):
        return None

    print("\n--- Reading current PandA values as warm start ---")
    current_vals = read_pid_values(PANDA_HOST, param_names)
    for name, value in current_vals.items():
        print(f"  {name:<8}  {value}")

    lo = bounds_phys[0].numpy()
    hi = bounds_phys[1].numpy()
    raw = np.array([current_vals.get(n, 0.0) for n in param_names])
    warm_unit = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    warm_start_x = torch.tensor(warm_unit, dtype=DTYPE, device=DEVICE).unsqueeze(0)

    result: dict[str, Any] = run_one_optimization(
        make_hardware_evaluator(param_names, trajectory, debug_plot_path),
        bounds_phys,
        dim=len(param_names),
        dtype=DTYPE,
        device=DEVICE,
        method=METHOD,
        seed=SEED0,
        total_budget=LIVE_BUDGET,
        # Extra initial points are perturbations of the device's current gains,
        # never fresh draws over the full space.
        n_init=N_INIT,
        verbose=True,
        warm_start_x=warm_start_x,
        restart_policy=HW_RESTART_POLICY,
        progress_fn=progress_fn,
        torch_threads=BO_THREADS,
    )
    print(
        "Direct hardware BO best:",
        as_pid_vector(result["best_x_phys"], len(param_names)),
    )
    return result


def run_hardware_phase(
    sim_result: dict[str, Any],
    param_names: list[str],
    bounds_phys: torch.Tensor,
    trajectory: Array,
    debug_plot_path: str,
    progress_fn: Callable[..., None],
) -> dict[str, Any] | None:
    """Continue optimisation on the device, seeded from the simulation's best."""
    if not health_check(
        PANDA_HOST,
        param_names,
        prompt="Continue to hardware tuning? [y/N]: ",
        expected_config=EXPECTED_DEVICE_CONFIG,
    ):
        return None

    # Warm-start hardware BO from the best simulated points, so every live
    # evaluation is a gain set the simulation already vetted.
    warm_start_x = select_warm_start(
        sim_result, k=WARM_START_K, min_separation=WARM_START_MIN_SEPARATION
    )

    result: dict[str, Any] = run_one_optimization(
        make_hardware_evaluator(param_names, trajectory, debug_plot_path),
        bounds_phys,
        dim=len(param_names),
        dtype=DTYPE,
        device=DEVICE,
        method=METHOD,
        seed=SEED0,
        total_budget=LIVE_BUDGET,
        n_init=WARM_START_K,
        verbose=True,
        warm_start_x=warm_start_x,
        restart_policy=HW_RESTART_POLICY,
        progress_fn=progress_fn,
        torch_threads=BO_THREADS,
    )
    print("Hardware BO best:", as_pid_vector(result["best_x_phys"], len(param_names)))
    return result


# ============================================================
# Simulation
# ============================================================


def run_simulation(
    sim_evaluate_fn: Callable[..., Any],
    param_names: list[str],
    bounds_phys: torch.Tensor,
    results_dir: str,
    progress_fn: Callable[..., None],
) -> dict[str, Any]:
    """Run the simulated optimisation and return the best individual run.

    Deliberately a COLD start: no prior knowledge of this stage is fed in, so
    the same code discovers gains for a stage it has never seen. Seeding it with
    values already on a device would bake in stage-specific knowledge and make
    the result look better than the framework actually is.

    The hardware phase is where prior knowledge enters — it warm-starts from
    this run's best points, so live evaluations only ever explore around gains
    the simulation has already vetted.
    """
    dim = len(param_names)
    if RUN_MC:
        # mc_results = run_mc_optimisation(
        #     sim_evaluate_fn,
        #     bounds_phys,
        #     dim=dim,
        #     method=METHOD,
        #     dtype=DTYPE,
        #     device=DEVICE,
        #     n_runs=N_MC_RUNS,
        #     seed0=SEED0,
        #     total_budget=TOTAL_BUDGET,
        #     n_init=N_INIT,
        #     verbose=False,
        #     save_csv=SAVE_MC_CSV,
        #     results_dir=results_dir,
        #     param_names=param_names,
        #     progress_fn=progress_fn,
        #     n_jobs=MC_JOBS,
        #     torch_threads=BO_THREADS,
        # )
        # plot_mc_result(mc_results, uncertainty="std", save_dir=results_dir)
        # # Use the best individual MC run for the final trajectory re-simulation.
        # return min(mc_results["run_results"], key=lambda r: r["best_y"])
        x, y = sobol_sample_and_evaluate(
            evaluate_fn=sim_evaluate_fn,
            bounds_phys=bounds_phys,
            dim=dim,
            n_samples=TOTAL_SAMPLE,
            dtype=DTYPE,
            device=DEVICE,
            seed=SEED0,
        )

        save_sobol_dataset(
            "sobol_data.csv",
            x,
            y,
            param_names=param_names,
        )

    result: dict[str, Any] = run_one_optimization(
        sim_evaluate_fn,
        bounds_phys,
        dim=dim,
        method=METHOD,
        dtype=DTYPE,
        device=DEVICE,
        seed=SEED0,
        total_budget=TOTAL_BUDGET,
        n_init=N_INIT,
        verbose=True,
        progress_fn=progress_fn,
        torch_threads=BO_THREADS,
    )
    plot_single_result(result, save_dir=results_dir)
    return result


def _save(fig: Any, results_dir: str, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_run(
    sim_evaluate_fn: Callable[..., Any],
    result: dict[str, Any],
    trajectory: Array,
    results_dir: str,
) -> None:
    """Re-simulate the best gains and write the tracking/error/voltage plots."""
    best_x_phys = result["best_x_phys"].to(dtype=DTYPE, device=DEVICE)
    _, details = sim_evaluate_fn(best_x_phys, return_details=True)
    best_actual = details[0]["actual_pos"]
    best_voltage = details[0]["control_hist"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trajectory[:, 0], label="Reference x")
    ax.plot(best_actual[:, 0], label="Tracked x")
    ax.set_xlabel("Time step")
    ax.set_ylabel("x")
    ax.set_title("X-axis tracking with BO-tuned PID")
    ax.legend()
    ax.grid(True)
    _save(fig, results_dir, "x_tracking.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.linalg.norm(best_actual - trajectory, axis=1))
    ax.set_title("Position tracking error over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Euclidean error")
    _save(fig, results_dir, "pos_error_euclidean.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(best_actual[:, 0] - trajectory[:, 0])
    ax.set_title("Position tracking error over time (x only)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("x error")
    _save(fig, results_dir, "pos_error_x.png")

    fig, ax = plt.subplots()
    # Skip sample 0: it is seeded open-loop, so no control action was applied.
    ax.hist(best_voltage[1:, 0], bins=100)
    ax.set_title("Control voltage histogram")
    ax.set_xlabel("Voltage")
    _save(fig, results_dir, "voltage_hist.png")

    print(f"\nPlots saved to: {results_dir}/")


# ============================================================
# Entry point
# ============================================================


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimisation of PID gains.")
    parser.add_argument(
        "--sim-only",
        action="store_true",
        help="never contact the PandA; makes every hardware path unreachable",
    )
    parser.add_argument(
        "--no-clear", action="store_true", help="do not clear the terminal on start"
    )
    args = parser.parse_args(argv)

    if not args.no_clear:
        # Clearing the terminal is cosmetic; ignore a non-zero status.
        subprocess.run(["clear"], check=False)  # noqa: S603, S607

    allow_hardware = bool(PANDA_HOST) and not args.sim_only

    param_names = list(TUNE_PARAMS.keys())
    # Fail now rather than after a full BO run producing an unwritable optimum.
    validate_bounds(TUNE_PARAMS)
    validate_bounds({k: (v, v) for k, v in FIXED_PARAMS.items()})

    plant = load_plant()
    trajectory = load_trajectory()
    bounds_phys = make_bounds(TUNE_PARAMS)
    sim_evaluate_fn = build_sim_objective(trajectory, plant, param_names)
    progress_fn = _make_progress_fn(param_names)

    run_mode = "sim"
    if allow_hardware:
        run_mode = ask_run_mode()
        if run_mode == "quit":
            return
        if run_mode == "sim" and not health_check(
            PANDA_HOST,
            param_names,
            prompt="Continue with simulation? [y/N]: ",
            expected_config=EXPECTED_DEVICE_CONFIG,
        ):
            return

    results_dir = make_results_dir()
    debug_plot_path = os.path.join(results_dir, "hw_trajectory_debug.png")

    with open(os.path.join(results_dir, "run.log"), "a") as log_file:
        real_stdout, real_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(real_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(real_stderr, log_file)  # type: ignore[assignment]
        try:
            t_start = time.time()
            print(
                f"\n=== Run started: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            print_config(run_mode, results_dir, allow_hardware)

            if run_mode == "direct":
                run_direct_hardware(
                    param_names,
                    bounds_phys,
                    trajectory,
                    debug_plot_path,
                    progress_fn,
                )
            else:
                result = run_simulation(
                    sim_evaluate_fn,
                    param_names,
                    bounds_phys,
                    results_dir,
                    progress_fn,
                )
                print("\nOptimisation finished.")
                print(
                    "Best PID found:",
                    as_pid_vector(result["best_x_phys"], len(param_names)),
                )
                print("Best average tracking error:", result["best_y"])
                plot_best_run(sim_evaluate_fn, result, trajectory, results_dir)

                if allow_hardware and ask_run_hardware():
                    run_hardware_phase(
                        result,
                        param_names,
                        bounds_phys,
                        trajectory,
                        debug_plot_path,
                        progress_fn,
                    )

            print(f"\n=== Total run time: {time.time() - t_start:.1f}s ===")
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr


if __name__ == "__main__":
    main()
