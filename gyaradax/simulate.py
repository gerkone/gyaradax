"""
Simulation runtime and orchestration.

This module provides high-level utilities to manage the simulation lifecycle,
including configuration loading, state initialization, and periodic dumping.
"""

import os
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from gyaradax.diag import get_diagnostics
from gyaradax.geometry import load_geometry
from gyaradax.integrals import get_integrals
from gyaradax.params import GKParams, gkparams_from_config, load_config
from gyaradax.solver import gksolve, init_f
from gyaradax.params import GKState, default_state
from gyaradax.utils import (
    load_checkpoint,
    load_gkw_k_dump,
    read_gkw_dump_time,
    save_dumps as save_dumps_fn,
)


def _setup_simulation(
    config_path: str,
    checkpoint_interval: Optional[int],
    save_dumps_flag: bool,
    kwargs: Dict[str, Any],
) -> Tuple[Any, Any, Dict[str, jnp.ndarray], int, int, bool]:
    """
    Load configuration and determine simulation-level hyperparameters.
    """
    cfg = load_config(config_path)

    total_steps = int(kwargs.pop("n_steps", getattr(cfg.solver, "n_steps", 120)))

    params = gkparams_from_config(cfg, **kwargs)
    data_dir = cfg.run.data_dir
    geometry = load_geometry(data_dir)

    interval = getattr(cfg.solver, "dump_interval", params.naverage)
    interval = checkpoint_interval or int(kwargs.pop("checkpoint_interval", interval))
    save_dumps_flag = kwargs.pop(
        "save_dumps", getattr(cfg.solver, "save_dumps", save_dumps_flag)
    )

    interval = interval or params.naverage

    return cfg, params, geometry, total_steps, interval, save_dumps_flag


def _init_condition(
    resume_from: Optional[str],
    resume_k_file: Optional[str],
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    verbose: bool,
) -> Tuple[jnp.ndarray, GKState]:
    """
    Initialize the simulation state either from scratch, an internal checkpoint,
    or a GKW reference distribution file.
    """
    nky = len(geometry["krho"])
    state = default_state(nky=nky)

    if resume_from:
        if verbose:
            print(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_checkpoint(resume_from)
        df = ckpt["df"]
        state = GKState(
            time=ckpt["time"],
            step=ckpt["step"],
            accumulated_norm_factor=ckpt["accumulated_norm_factor"],
            window_start_amp=ckpt["window_start_amp"],
            last_growth_rate=ckpt["last_growth_rate"],
        )
    elif resume_k_file:
        if verbose:
            print(f"Resuming from K-file: {resume_k_file}")
        res = (
            len(geometry["intvp"]),
            len(geometry["intmu"]),
            len(geometry["ints"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        df = load_gkw_k_dump(resume_k_file, res)
        dat_path = resume_k_file + ".dat"
        if os.path.exists(dat_path):
            t_start = read_gkw_dump_time(dat_path)
            state = GKState(
                time=jnp.array(t_start, dtype=jnp.float64),
                step=jnp.array(0, dtype=jnp.int32),
                accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
                window_start_amp=jnp.ones(nky, dtype=jnp.float64),
                last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
            )
            if verbose:
                print(f"Loaded start time from {dat_path}: {t_start:.4f}")
    else:
        if verbose:
            print("WARNING: Initializing new simulation from experimental init_f.")
            print("         This profile may not correctly reproduce GKW seed parity.")
        df = init_f(geometry, norm_eps=params.norm_eps)
    return df, state


def simulate(
    config_path: str,
    output_dir: str = "outputs",
    checkpoint_interval: Optional[int] = None,
    resume_from: Optional[str] = None,
    resume_k_file: Optional[str] = None,
    verbose: bool = True,
    save_dumps: bool = True,
    **kwargs,
) -> Tuple[jnp.ndarray, GKState]:
    """
    High-level entry point to run a simulation from a YAML config.

    This function handles the orchestration of the simulation loop, including
    loading geometry, setting initial conditions, executing the time-stepping
    via gksolve, and periodically dumping snapshots and diagnostics.

    Args:
        config_path: Path to the YAML configuration file.
        output_dir: Directory where results and checkpoints will be saved.
        checkpoint_interval: Interval (in small steps) for full state dumping.
        resume_from: Optional path to an internal .npz checkpoint to resume from.
        resume_k_file: Optional path to a GKW K* dump file to initialize from.
        verbose: If True, prints simulation progress to console.
        save_dumps: If True, saves full 5D distribution function snapshots.
        **kwargs: Manual overrides for any GKParams or simulation controls.

    Returns:
        Tuple of (final_df, final_state).
    """
    cfg, params, geometry, total_steps, interval, save_dumps_flag = _setup_simulation(
        config_path, checkpoint_interval, save_dumps, kwargs
    )

    df, state = _init_condition(resume_from, resume_k_file, geometry, params, verbose)
    os.makedirs(output_dir, exist_ok=True)

    phi, fluxes = get_integrals(df, geometry, params=params)

    if verbose:
        print(f"Starting simulation: total_steps={total_steps}, interval={interval}")
        print(f"Initial state: step={int(state.step)}, time={float(state.time):.4f}")

    start_step = int(state.step)
    for _ in range(start_step, total_steps, interval):
        save_dumps_fn(
            output_dir,
            df,
            phi,
            fluxes,
            state,
            geometry,
            save_dumps=save_dumps_flag,
        )

        steps_to_run = min(interval, total_steps - int(state.step))
        if steps_to_run <= 0:
            break

        df, (phi, fluxes), state = gksolve(
            df, geometry, params, state, n_steps=steps_to_run
        )

        if verbose:
            flux = float(fluxes[1])
            growth = float(jnp.mean(state.last_growth_rate))
            print(
                f"[step {int(state.step):06d}/{total_steps}| "
                f"time {float(state.time):.4f} | elux {flux:.4f} | growth {growth:.4f}"
            )

    save_dumps_fn(
        output_dir, df, phi, fluxes, state, geometry, save_dumps=save_dumps_flag
    )

    if verbose:
        print("DONE.")

    return df, state
