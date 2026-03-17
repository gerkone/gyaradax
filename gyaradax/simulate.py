"""Simulation runtime and orchestration."""

import os
import time
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gyaradax.geometry import load_geometry
from gyaradax.integrals import (
    get_integrals,
    calculate_phi,
    calculate_phi_kinetic,
    geom_tensors,
)
from gyaradax.params import gkparams_from_config, load_config, GKParams
from gyaradax.solver import (
    gksolve,
    init_f,
    GKState,
    GKPre,
    default_state,
    linear_precompute,
    mode_amplitude,
)
from gyaradax.utils import save_dumps as save_dumps_fn


def _compute_phi_for_init(df, geometry, params):
    """Compute phi for initial amplitude tracking."""
    if params.adiabatic_electrons:
        return calculate_phi(geom_tensors(geometry, params=params), df)
    else:
        return calculate_phi_kinetic(geometry, df)


def default_log(df, phi, fluxes, state, wall_time: float, block_steps: int = 0):
    """Standard progress printer, passed as log_fn=default_log."""
    fl = jnp.asarray(fluxes)
    growth = float(jnp.mean(state.last_growth_rate))
    if fl.ndim == 2:
        parts = " | ".join(
            f"eflux_{i} {float(fl[i, 1]):>12.4e}" for i in range(fl.shape[0])
        )
        eflux_str = parts
    else:
        eflux_str = f"eflux {float(fl[1]):>12.4e}"
    steps_sec = block_steps / max(wall_time, 1e-6) if block_steps > 0 else 0.0
    print(
        f"  step {int(state.step):>8d} | "
        f"t {float(state.time):>10.3f} | "
        f"{eflux_str} | "
        f"growth {growth:>12.4e} | "
        f"{steps_sec:>8.1f} steps/s"
    )


def gk_init(
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    n_species: int = 1,
) -> Tuple[jnp.ndarray, GKState]:
    """Create initial (df, state) from geometry and params. No IO."""
    df = init_f(
        geometry,
        finit=params.finit,
        norm_eps=params.norm_eps,
        n_species=n_species,
    )
    phi0 = _compute_phi_for_init(df, geometry, params)
    amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
    nky = len(geometry["krho"])
    state = default_state(nky=nky)
    state = GKState(
        time=state.time,
        step=state.step,
        accumulated_norm_factor=state.accumulated_norm_factor,
        window_start_amp=amp0,
        last_growth_rate=state.last_growth_rate,
    )
    return df, state


def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre] = None,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], GKState
]:
    """Run n_steps. Pure, no IO. Returns (df, phi, fluxes, state)."""
    if pre is None:
        pre = linear_precompute(geometry, params)
    final_df, (phi, fluxes), final_state = gksolve(
        df, geometry, params, state, n_steps=n_steps, pre=pre
    )
    return final_df, phi, fluxes, final_state


def gksimulate(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    *,
    pre: Optional[GKPre] = None,
    output_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    save_snapshots: bool = False,
    save_final: bool = True,
    log_fn: Optional[Callable] = None,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], GKState
]:
    """Run n_steps with optional IO checkpointing and logging.

    Returns:
        (df, phi, fluxes, state)
    """
    if pre is None:
        pre = linear_precompute(geometry, params)

    interval = checkpoint_interval if checkpoint_interval else n_steps

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        phi_init, fluxes_init = get_integrals(
            df,
            geometry,
            params=params,
            adiabatic_electrons=params.adiabatic_electrons,
        )
        save_dumps_fn(
            output_dir,
            df,
            phi_init,
            fluxes_init,
            state,
            geometry,
            save_dumps=save_snapshots,
        )

    start_step = int(state.step)
    target_step = start_step + n_steps
    current_df = df
    current_state = state
    current_phi = None
    current_fluxes = None

    while int(current_state.step) < target_step:
        steps_remaining = target_step - int(current_state.step)
        block_steps = min(interval, steps_remaining)
        if block_steps <= 0:
            break

        t0 = time.time()
        current_df, current_phi, current_fluxes, current_state = gk_run(
            current_df, geometry, params, current_state, block_steps, pre=pre
        )
        wall_time = time.time() - t0

        if output_dir is not None:
            is_final = int(current_state.step) >= target_step
            save_dumps_fn(
                output_dir,
                current_df,
                current_phi,
                current_fluxes,
                current_state,
                geometry,
                save_dumps=save_snapshots or (save_final and is_final),
            )

        if log_fn is not None:
            log_fn(
                current_df,
                current_phi,
                current_fluxes,
                current_state,
                wall_time,
                block_steps,
            )

    if current_phi is None:
        current_phi, current_fluxes = get_integrals(
            df,
            geometry,
            params=params,
            adiabatic_electrons=params.adiabatic_electrons,
        )

    return current_df, current_phi, current_fluxes, current_state


def gk_from_config(
    config_path: str,
    **overrides,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], GKParams, GKState, GKPre]:
    """Load YAML config -> (df, geometry, params, state, pre).

    Fresh-start initialization only. Resume from checkpoint/K-file is the
    caller's responsibility using load_checkpoint() or load_gkw_k_dump().
    """
    cfg = load_config(config_path)
    params = gkparams_from_config(cfg, **overrides)
    geometry = load_geometry(cfg.run.data_dir)

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    df, state = gk_init(geometry, params, n_species=n_species)
    pre = linear_precompute(geometry, params)

    return df, geometry, params, state, pre
