"""Simulation runtime."""

import os
import time
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gyaradax.utils import load_geometry
from gyaradax.geometry import compute_geometry
from gyaradax.integrals import (
    get_integrals,
    calculate_phi,
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
    return calculate_phi(geometry, df, params=params)


def _geometry_from_config(cfg):
    """Build geometry from config when no data_dir is provided.

    defaults are defined in compute_geometry(); we only forward
    values that are actually present in the config.
    """
    gc = getattr(cfg, "geometry", {})
    gr = cfg.grid
    kwargs = {}
    _float_keys = {"q", "shat", "eps", "kxmax", "signB", "Rref", "vpar_max", "krhomax"}
    _int_keys = {"ns", "nkx", "nky", "nvpar", "nmu", "nperiod", "ikxspace"}
    for key, section in [
        ("q", gc),
        ("shat", gc),
        ("eps", gc),
        ("kxmax", gc),
        ("signB", gc),
        ("Rref", gc),
        ("ns", gr),
        ("nkx", gr),
        ("nky", gr),
        ("nvpar", gr),
        ("nmu", gr),
        ("vpar_max", gr),
        ("nperiod", gr),
        ("krhomax", gr),
        ("ikxspace", gr),
    ]:
        val = getattr(section, key, None)
        if val is not None:
            kwargs[key] = int(val) if key in _int_keys else float(val)
    # geometry_model: "circ" (Lapillonne) or "s-alpha"
    gm = getattr(gc, "geometry_model", None)
    if gm is not None:
        kwargs["geom_type"] = str(gm)
    return compute_geometry(**kwargs)


def log_step(fluxes, state: GKState, wall_time: float, n_steps: int = 0):
    flx = jnp.asarray(fluxes)
    growth = float(jnp.mean(state.last_growth_rate))
    if flx.ndim == 1:
        flx = flx[jnp.newaxis]
    flx = " | ".join(f"eflux_{i} {float(flx[i, 1]):>8.4f}" for i in range(flx.shape[0]))
    steps_sec = f"{n_steps / max(wall_time, 1e-6):>.2f}" if n_steps > 0 else "N/A"
    print(
        f"[{int(state.step):>8d}] t {float(state.time):>8.2f} | "
        f"{flx} | growth {growth:>8.4f} | {steps_sec} steps/s"
    )


def _ensure_species_arrays(
    geometry: Dict[str, jnp.ndarray], params: GKParams
) -> Dict[str, jnp.ndarray]:
    """Ensure geometry carries multi-species arrays consistent with params.

    ``compute_geometry`` always creates single-element species placeholders
    (``mas=[1.0]``, etc.).  When params describes multiple species the
    downstream flux diagnostics (``calculate_fluxes_kinetic``) need per-species
    arrays in the geometry dict.  This helper copies them from params when the
    geometry arrays are too short.
    """
    _SPECIES_KEYS = ("mas", "signz", "de", "tmp", "vthrat", "rlt", "rln")
    mas = jnp.asarray(params.mas, dtype=jnp.float64)
    nsp = int(mas.shape[0]) if mas.ndim > 0 else 1
    if nsp <= 1:
        return geometry

    geom_nsp = int(jnp.asarray(geometry.get("mas", jnp.ones(1))).shape[0])
    if geom_nsp >= nsp:
        return geometry

    geometry = dict(geometry)  # shallow copy
    for k in _SPECIES_KEYS:
        val = getattr(params, k, None)
        if val is not None:
            geometry[k] = jnp.asarray(val, dtype=jnp.float64)
    return geometry


def gk_init(
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    n_species: int = 1,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], GKState]:
    """Create initial (df, geometry, state) from geometry and params. No IO.

    When *params* indicates kinetic electrons, the geometry dict is
    augmented with per-species arrays from *params* if they are missing
    (e.g. when using ``compute_geometry`` which only creates single-species
    placeholders).  The **returned** geometry must be used for all
    subsequent calls (``gksolve``, ``linear_precompute``, fluxes, etc.).
    """
    if not params.adiabatic_electrons:
        mas = jnp.asarray(params.mas, dtype=jnp.float64)
        n_species = max(n_species, int(mas.shape[0]) if mas.ndim > 0 else 1)

    geometry = _ensure_species_arrays(geometry, params)

    df = init_f(
        geometry,
        finit=params.finit,
        amp_init_real=params.amp_init,
        norm_eps=params.norm_eps,
        n_species=n_species,
        params=params,
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
    return df, geometry, state


def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], GKState]:
    """Run n_steps. Pure, no IO. Returns (df, phi, fluxes, state)."""
    if pre is None:
        pre = linear_precompute(geometry, params)
    final_df, (phi, fluxes), final_state = gksolve(
        df, geometry, params, state, n_steps=n_steps, pre=pre
    )
    return final_df, phi, fluxes, final_state


def gk_run_batched(
    df_batch: jnp.ndarray,
    geometry_batch: Dict[str, jnp.ndarray],
    params_batch: GKParams,
    state_batch: GKState,
    n_steps: int,
    pre_batch: GKPre,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple, GKState]:
    """Batched gk_run: vmap over all per-config arguments.

    All arguments except n_steps carry a leading batch dimension.
    Configs must share the same grid shape and static params
    (adiabatic_electrons, non_linear, finit).
    """

    def _single(df, geom, par, st, pre):
        final_df, (phi, fluxes), final_state = gksolve(df, geom, par, st, n_steps=n_steps, pre=pre)
        return final_df, phi, fluxes, final_state

    return jax.vmap(_single)(df_batch, geometry_batch, params_batch, state_batch, pre_batch)


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
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], GKState]:
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
            params=params,
            pre=pre,
        )

    start_step = int(state.step)
    target_step = start_step + n_steps
    current_df = df
    current_state = state
    current_phi = None
    current_fluxes = None

    # warmup (compilation)
    if n_steps > 0:
        print("warmup (compilation)...")
        w_t0 = time.time()
        _ = gk_run(current_df, geometry, params, current_state, min(interval, n_steps), pre=pre)
        jax.block_until_ready(_[0])
        print(f"compilation: {time.time() - w_t0:.2f}s")

    while int(current_state.step) < target_step:
        steps_remaining = target_step - int(current_state.step)
        block_steps = min(interval, steps_remaining)
        if block_steps <= 0:
            break

        t0 = time.time()
        current_df, current_phi, current_fluxes, current_state = gk_run(
            current_df, geometry, params, current_state, block_steps, pre=pre
        )
        jax.block_until_ready(current_df)
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
                params=params,
                pre=pre,
            )

        log_step(current_fluxes, current_state, wall_time, n_steps=block_steps)

    if current_phi is None:
        current_phi, current_fluxes = get_integrals(
            df,
            geometry,
            params=params,
            adiabatic_electrons=params.adiabatic_electrons,
        )

    return current_df, current_phi, current_fluxes, current_state


def gk_from_gkw_dir(
    gkw_dir: str,
    k_index: int = -1,
    **overrides,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], GKParams, GKState, GKPre]:
    """Load a GKW run directory -> (df, geometry, params, state, pre).

    Builds params from input.dat, loads geometry from geom.dat,
    and resumes from a K-file. No YAML config needed.

    Args:
        k_index: which K-file to load (default -1, i.e. the last one).
    """
    from gyaradax.params import gkparams_from_input_and_geometry
    from gyaradax.utils import K_files, load_gkw_k_dump, read_gkw_dump_time

    input_dat = os.path.join(gkw_dir, "input.dat")
    geometry = load_geometry(gkw_dir)
    params = gkparams_from_input_and_geometry(input_dat, geometry, **overrides)
    geometry = _ensure_species_arrays(geometry, params)

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
    k_files = K_files(gkw_dir)
    if k_files:
        k_path = os.path.join(gkw_dir, k_files[k_index])
        df = jnp.asarray(load_gkw_k_dump(k_path, res, n_species=n_species), dtype=jnp.complex128)
        dat_path = k_path + ".dat"
        t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0
    else:
        df, geometry, _ = gk_init(geometry, params, n_species=n_species)
        t_start = 0.0

    phi0 = _compute_phi_for_init(df, geometry, params)
    amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
    nky = len(geometry["krho"])
    state = GKState(
        time=jnp.array(t_start, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=amp0,
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )
    pre = linear_precompute(geometry, params)
    return df, geometry, params, state, pre


def gk_from_config(
    config_path: str,
    **overrides,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], GKParams, GKState, GKPre]:
    """Load YAML config -> (df, geometry, params, state, pre).

    Fresh-start initialization only. Resume from checkpoint/K-file is the
    caller's responsibility using load_checkpoint() or load_gkw_k_dump().
    Uses analytic geometry when data_dir is absent from the config.
    """
    cfg = load_config(config_path)
    params = gkparams_from_config(cfg, **overrides)

    data_dir = getattr(cfg.run, "data_dir", None)
    if data_dir:
        geometry = load_geometry(data_dir)
    else:
        geometry = _geometry_from_config(cfg)

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    df, geometry, state = gk_init(geometry, params, n_species=n_species)
    pre = linear_precompute(geometry, params)

    return df, geometry, params, state, pre
