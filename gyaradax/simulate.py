"""Simulation runtime and orchestration."""

import os
import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gyaradax.geometry import load_geometry
from gyaradax.analytic_geometry import compute_geometry
from gyaradax.integrals import (
    get_integrals,
    calculate_phi,
    calculate_phi_kinetic,
    geom_tensors,
)
from gyaradax.params import gkparams_from_config, load_config
from gyaradax.solver import (
    gksolve,
    init_f,
    GKState,
    default_state,
    linear_precompute,
    mode_amplitude,
)
from gyaradax.utils import (
    load_checkpoint,
    load_gkw_k_dump,
    read_gkw_dump_time,
    save_dumps as save_dumps_fn,
)


def _geometry_from_config(cfg):
    """build geometry from config when no data_dir is provided.

    defaults are defined in compute_geometry(); we only forward
    values that are actually present in the config.
    """
    gc = getattr(cfg, "geometry", {})
    gr = cfg.grid
    kwargs = {}
    for key, section in [
        ("q", gc), ("shat", gc), ("eps", gc), ("kxmax", gc),
        ("signB", gc), ("Rref", gc),
        ("ns", gr), ("nkx", gr), ("nky", gr), ("nvpar", gr), ("nmu", gr),
        ("vpar_max", gr), ("nperiod", gr), ("krhomax", gr), ("ikxspace", gr),
    ]:
        val = getattr(section, key, None)
        if val is not None:
            kwargs[key] = float(val) if isinstance(val, (int, float)) else val
    return compute_geometry(**kwargs)


def _compute_phi_for_init(df, geometry, params):
    """compute phi for initial amplitude tracking."""
    if params.adiabatic_electrons:
        return calculate_phi(geom_tensors(geometry, params=params), df)
    else:
        return calculate_phi_kinetic(geometry, df)


def _setup_simulation(
    config_path: str,
    checkpoint_interval: Optional[int],
    save_dumps_flag: bool,
    kwargs: Dict[str, Any],
) -> Tuple[Any, Any, Dict[str, jnp.ndarray], int, int, bool]:
    """Load configuration and determine simulation-level hyperparameters."""
    cfg = load_config(config_path)
    total_steps = int(kwargs.pop("n_steps", getattr(cfg.solver, "n_steps", 120)))
    params = gkparams_from_config(cfg, **kwargs)
    data_dir = getattr(cfg.run, "data_dir", None)
    if data_dir:
        geometry = load_geometry(data_dir)
    else:
        geometry = _geometry_from_config(cfg)
    interval = getattr(cfg.solver, "dump_interval", 3) * params.naverage
    if checkpoint_interval:
        interval = checkpoint_interval
    save_dumps_flag = getattr(cfg.solver, "save_dumps", save_dumps_flag)
    return params, geometry, total_steps, interval, save_dumps_flag


def _init_condition(
    resume_from: Optional[str],
    resume_k_file: Optional[str],
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    verbose: bool,
) -> Tuple[jnp.ndarray, GKState]:
    """Initialize from scratch, a checkpoint, or a GKW K-file."""
    nky = len(geometry["krho"])
    state = default_state(nky=nky)
    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    if resume_from:
        if verbose:
            print(f"resuming from checkpoint: {resume_from}")
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
            print(f"resuming from K-file: {resume_k_file}")
        res = (
            len(geometry["intvp"]),
            len(geometry["intmu"]),
            len(geometry["ints"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        df = load_gkw_k_dump(resume_k_file, res, n_species=n_species)
        dat_path = resume_k_file + ".dat"
        if os.path.exists(dat_path):
            t_start = read_gkw_dump_time(dat_path)
            phi0 = _compute_phi_for_init(df, geometry, params)
            amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
            state = GKState(
                time=jnp.array(t_start, dtype=jnp.float64),
                step=jnp.array(0, dtype=jnp.int32),
                accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
                window_start_amp=amp0,
                last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
            )
            if verbose:
                print(f"  start time: {t_start:.4f}")
    else:
        if verbose:
            print("initializing from init_f")
        df = init_f(
            geometry,
            finit=params.finit,
            norm_eps=params.norm_eps,
            n_species=n_species,
        )
        phi0 = _compute_phi_for_init(df, geometry, params)
        amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
        state = GKState(
            time=state.time,
            step=state.step,
            accumulated_norm_factor=state.accumulated_norm_factor,
            window_start_amp=amp0,
            last_growth_rate=state.last_growth_rate,
        )
    return df, state


def simulate(
    config_path: str,
    output_dir: str = "outputs",
    checkpoint_interval: Optional[int] = None,
    resume_from: Optional[str] = None,
    resume_k_file: Optional[str] = None,
    verbose: bool = True,
    save_dumps: bool = False,
    save_last: bool = True,
    **kwargs,
) -> Tuple[jnp.ndarray, GKState, List[Dict[str, float]]]:
    """Run a simulation from a YAML config.

    Handles both adiabatic and kinetic electron configurations.
    """
    params, geometry, total_steps, interval, save_dumps_flag = _setup_simulation(
        config_path, checkpoint_interval, save_dumps, kwargs
    )

    df, state = _init_condition(resume_from, resume_k_file, geometry, params, verbose)
    os.makedirs(output_dir, exist_ok=True)

    phi, fluxes = get_integrals(
        df,
        geometry,
        params=params,
        adiabatic_electrons=params.adiabatic_electrons,
    )
    pre = linear_precompute(geometry, params)

    if verbose:
        sp_label = "kinetic" if not params.adiabatic_electrons else "adiabatic"
        print(
            f"starting: steps={total_steps}, interval={interval}, electrons={sp_label}"
        )
        print(f"  step={int(state.step)}, time={float(state.time):.4f}, dt={params.dt}")

    start_step = int(state.step)
    performance_metrics = []

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

        t_block_start = time.time()
        df, (phi, fluxes), state = gksolve(
            df, geometry, params, state, n_steps=steps_to_run, pre=pre
        )
        t_block_end = time.time()

        block_time = t_block_end - t_block_start
        steps_sec = steps_to_run / max(block_time, 1e-6)

        perf = {"block_time": block_time, "steps_per_sec": steps_sec}
        performance_metrics.append(perf)

        if verbose:
            flux = float(fluxes[1])
            growth = float(jnp.mean(state.last_growth_rate))
            print(
                f"[{int(state.step):8d}/{total_steps}] | t: {float(state.time):.1f} | "
                f"eflux: {flux:.4f} | growth: {growth:.4f} | {steps_sec:.1f} steps/s"
            )

    save_dumps_fn(output_dir, df, phi, fluxes, state, geometry, save_dumps=save_last)

    if performance_metrics:
        perf_path = os.path.join(output_dir, "performance.npz")
        keys = performance_metrics[0].keys()
        perf_data = {k: np.array([p[k] for p in performance_metrics]) for k in keys}
        np.savez(perf_path, **perf_data)

    if verbose:
        print("done\n")

    return df, state, performance_metrics
