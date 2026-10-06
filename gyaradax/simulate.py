"""Simulation runtime."""

import contextlib
import os
import shutil
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast, overload

import jax
import jax.numpy as jnp
import numpy as np

from gyaradax.jax_config import enable_x64

enable_x64()

from gyaradax.utils import load_geometry
from gyaradax.geometry import compute_geometry_from_config
from gyaradax.integrals import (
    get_integrals,
    calculate_phi,
)
from gyaradax.params import gkparams_from_config, load_config, GKParams
from gyaradax.precompute import linear_precompute
from gyaradax.solver import (
    gksolve,
    init_f,
    GKState,
    GKPre,
    default_state,
    mode_amplitude,
)
import gyaradax.telemetry as telemetry_mod
from gyaradax.fields import _compute_fields, g_to_f
from gyaradax.utils import save_dumps as save_dumps_fn


def _compute_phi_for_init(df, geometry, params):
    """Compute phi for initial amplitude tracking."""
    return calculate_phi(geometry, df, params=params)


def _geometry_from_config(cfg):
    """Compatibility wrapper for the public geometry config helper."""
    return compute_geometry_from_config(cfg)


def log_step(fluxes, state: GKState, wall_time: float, n_steps: int = 0, label=None):
    flx = jnp.asarray(fluxes)
    growth = float(jnp.mean(state.last_growth_rate))
    if flx.ndim == 1:
        flx = flx[jnp.newaxis]
    flx = " | ".join(f"eflux_{i} {float(flx[i, 1]):>8.4f}" for i in range(flx.shape[0]))
    steps_sec = f"{n_steps / max(wall_time, 1e-6):>.2f}" if n_steps > 0 else "N/A"
    print(
        f"{f'{label}: ' if label else ''}[{int(state.step):>8d}] t {float(state.time):>8.2f} | "
        f"{flx} | growth {growth:>8.4f} | {steps_sec} steps/s"
    )


def _ensure_species_arrays(
    geometry: Dict[str, jnp.ndarray], params: GKParams
) -> Dict[str, jnp.ndarray]:
    """Ensure geometry carries multi-species arrays consistent with params.

    ``compute_geometry`` always creates single-element placeholders. Multi-species
    runs need per-species arrays in the geometry dict for downstream flux
    diagnostics (``calculate_fluxes_kinetic``); copy them over from params.
    """
    _SPECIES_KEYS = ("mas", "signz", "de", "tmp", "vthrat", "rlt", "rln")
    mas = jnp.asarray(params.mas, dtype=jnp.float64)
    nsp = int(mas.shape[0]) if mas.ndim > 0 else 1
    if nsp <= 1:
        return geometry

    geom_nsp = int(jnp.asarray(geometry.get("mas", jnp.ones(1))).shape[0])
    if geom_nsp >= nsp:
        return geometry

    geometry = dict(geometry)
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

    For kinetic electrons the geometry dict is augmented with per-species arrays
    from ``params`` when they are missing (e.g. when using ``compute_geometry``
    which only creates single-species placeholders). The returned geometry
    must be used for all subsequent calls.
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


@overload
def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre] = None,
    return_dt_info: Literal[False] = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Any, GKState]: ...


@overload
def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre],
    return_dt_info: Literal[True],
) -> Tuple[jnp.ndarray, jnp.ndarray, Any, GKState, Dict[str, Any]]: ...


@overload
def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre] = None,
    *,
    return_dt_info: Literal[True],
) -> Tuple[jnp.ndarray, jnp.ndarray, Any, GKState, Dict[str, Any]]: ...


@overload
def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre],
    return_dt_info: bool,
) -> (
    Tuple[jnp.ndarray, jnp.ndarray, Any, GKState]
    | Tuple[jnp.ndarray, jnp.ndarray, Any, GKState, Dict[str, Any]]
): ...


@overload
def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre] = None,
    *,
    return_dt_info: bool,
) -> (
    Tuple[jnp.ndarray, jnp.ndarray, Any, GKState]
    | Tuple[jnp.ndarray, jnp.ndarray, Any, GKState, Dict[str, Any]]
): ...


def gk_run(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int,
    pre: Optional[GKPre] = None,
    return_dt_info: bool = False,
) -> (
    Tuple[jnp.ndarray, jnp.ndarray, Any, GKState]
    | Tuple[jnp.ndarray, jnp.ndarray, Any, GKState, Dict[str, Any]]
):
    """Run n_steps. Pure, no IO.

    Returns ``(df, phi, fluxes, state)`` by default. When
    ``return_dt_info=True``, returns ``(df, phi, fluxes, state, dt_info)``
    with per-step adaptive-CFL diagnostics from the underlying scan.
    """
    if pre is None:
        pre = linear_precompute(geometry, params)
    if return_dt_info:
        final_df, (phi, fluxes), final_state, dt_info = gksolve(
            df, geometry, params, state, n_steps=n_steps, pre=pre, return_dt_info=True
        )
        return final_df, phi, fluxes, final_state, dt_info
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
    return_dt_info: bool = False,
) -> Any:
    """Batched gk_run: vmap over all per-config arguments.

    All arguments except n_steps carry a leading batch dimension.
    Configs must share the same grid shape and static params
    (adiabatic_electrons, non_linear, finit). ``return_dt_info`` appends the
    batched per-step dt diagnostics, as in ``gk_run``.
    """

    def _single(df, geom, par, st, pre):
        if return_dt_info:
            final_df, (phi, fluxes), final_state, dt_info = gksolve(
                df, geom, par, st, n_steps=n_steps, pre=pre, return_dt_info=True
            )
            return final_df, phi, fluxes, final_state, dt_info
        final_df, (phi, fluxes), final_state = gksolve(df, geom, par, st, n_steps=n_steps, pre=pre)
        return final_df, phi, fluxes, final_state

    return jax.vmap(_single)(df_batch, geometry_batch, params_batch, state_batch, pre_batch)


def _has_diagnostics_at(output_dir: str, step: int) -> bool:
    path = os.path.join(output_dir, "fluxes.npz")
    if not os.path.exists(path):
        return False
    with np.load(path) as data:
        return "step" in data.files and bool(np.any(data["step"] == step))


def _diagnostics(df, geometry, params, pre):
    # fields and fluxes of the physical f (g -> f in EM runs), as gksolve reports them
    if not params.nlapar:
        return get_integrals(
            df, geometry, params=params, adiabatic_electrons=params.adiabatic_electrons
        )
    _, apar, _ = _compute_fields(df, geometry, params, pre)
    return get_integrals(
        g_to_f(df, apar, params, pre),
        geometry,
        params=params,
        adiabatic_electrons=params.adiabatic_electrons,
    )


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
    snapshot_f32: bool = False,
    stop_on_nan: bool = True,
    keep_latest_snapshot: bool = False,
    snapshot_every: int = 1,
    previous_snapshot: Optional[str] = None,
    telemetry: Any = None,
    profile: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Any, GKState]:
    """Run n_steps with optional IO checkpointing and logging.

    ``stop_on_nan`` halts at the first block with a non-finite ``df``, writing a
    ``DIVERGED`` marker into ``output_dir``; the corrupted block is not archived.
    Snapshots (``step_*.npz``) are written every ``snapshot_every`` blocks and
    after the last one: all of them with ``save_snapshots``, only the latest with
    ``keep_latest_snapshot`` (the previous one, or ``previous_snapshot`` such as the
    snapshot a run resumed from, is removed once a newer one exists).
    ``telemetry`` (a ``telemetry.RunTelemetry``) records compile time and per-block
    timings; ``profile`` traces the first compiled block into ``output_dir/profile``.

    Returns:
        (df, phi, fluxes, state)
    """
    if pre is None:
        pre = linear_precompute(geometry, params)

    def run_block(d, s, steps):
        return gk_run(d, geometry, params, s, steps, pre=pre, return_dt_info=True)

    run = {
        "geometry": geometry,
        "params": params,
        "pre": pre,
        "output_dir": output_dir,
        "previous_snapshot": previous_snapshot,
        "telemetry": telemetry,
        "label": None,
    }
    return _simulate(
        run_block,
        df,
        state,
        n_steps,
        [run],
        False,
        checkpoint_interval=checkpoint_interval,
        save_snapshots=save_snapshots,
        save_final=save_final,
        snapshot_f32=snapshot_f32,
        stop_on_nan=stop_on_nan,
        keep_latest_snapshot=keep_latest_snapshot,
        snapshot_every=snapshot_every,
        profile=profile,
    )


def gksimulate_batched(
    df_batch: jnp.ndarray,
    geometry_batch: Dict[str, jnp.ndarray],
    params_batch: GKParams,
    state_batch: GKState,
    n_steps: int,
    *,
    pre_batch: GKPre,
    output_dirs: Sequence[Optional[str]],
    labels: Optional[Sequence[str]] = None,
    previous_snapshots: Optional[Sequence[Optional[str]]] = None,
    telemetries: Optional[Sequence[Any]] = None,
    checkpoint_interval: Optional[int] = None,
    save_snapshots: bool = False,
    save_final: bool = True,
    snapshot_f32: bool = False,
    stop_on_nan: bool = True,
    keep_latest_snapshot: bool = False,
    snapshot_every: int = 1,
    profile: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, Any, GKState]:
    """``gksimulate`` for configs stacked along a leading batch axis (``gk_run_batched``).

    Every member writes its own output directory exactly as ``gksimulate`` would; a
    member that diverges gets its ``DIVERGED`` marker and no further output while
    the others continue.
    """
    n = len(output_dirs)

    def run_block(d, s, steps):
        return gk_run_batched(
            d, geometry_batch, params_batch, s, steps, pre_batch, return_dt_info=True
        )

    runs = [
        {
            "geometry": _member(geometry_batch, i),
            "params": _member(params_batch, i),
            "pre": _member(pre_batch, i),
            "output_dir": output_dirs[i],
            "previous_snapshot": previous_snapshots[i] if previous_snapshots else None,
            "telemetry": telemetries[i] if telemetries else None,
            "label": labels[i] if labels else str(i),
        }
        for i in range(n)
    ]
    return _simulate(
        run_block,
        df_batch,
        state_batch,
        n_steps,
        runs,
        True,
        checkpoint_interval=checkpoint_interval,
        save_snapshots=save_snapshots,
        save_final=save_final,
        snapshot_f32=snapshot_f32,
        stop_on_nan=stop_on_nan,
        keep_latest_snapshot=keep_latest_snapshot,
        snapshot_every=snapshot_every,
        profile=profile,
    )


def _member(tree, i):
    return jax.tree_util.tree_map(lambda x: x[i], tree)


def _simulate(
    run_block,
    df,
    state,
    n_steps: int,
    runs: List[Dict[str, Any]],
    batched: bool,
    *,
    checkpoint_interval: Optional[int],
    save_snapshots: bool,
    save_final: bool,
    snapshot_f32: bool,
    stop_on_nan: bool,
    keep_latest_snapshot: bool,
    snapshot_every: int,
    profile: bool,
):
    """Block loop shared by ``gksimulate`` and ``gksimulate_batched``; I/O is per run."""

    def view(x, i):
        return _member(x, i) if batched else x

    interval = checkpoint_interval if checkpoint_interval else n_steps
    snapshot_every = max(int(snapshot_every), 1)

    # a resumed run keeps the history entry already written for its first step
    for i, run in enumerate(runs):
        out = run["output_dir"]
        st = view(state, i)
        if out is not None and not _has_diagnostics_at(out, int(st.step)):
            os.makedirs(out, exist_ok=True)
            d = view(df, i)
            phi_init, fluxes_init = _diagnostics(d, run["geometry"], run["params"], run["pre"])
            save_dumps_fn(
                out,
                d,
                phi_init,
                fluxes_init,
                st,
                run["geometry"],
                save_dumps=save_snapshots,
                params=run["params"],
                pre=run["pre"],
                snapshot_f32=snapshot_f32,
            )

    start_step = int(view(state, 0).step)
    target_step = start_step + n_steps
    current_df = df
    current_state = state
    current_phi: Any = None
    current_fluxes: Any = None
    alive = [True] * len(runs)
    stopped: List[Optional[Tuple[int, float]]] = [None] * len(runs)
    previous = [run["previous_snapshot"] for run in runs]
    profile_next = profile and runs[0]["output_dir"] is not None
    blocks_done = 0

    # warmup compile with the same return_dt_info as the body loop, otherwise
    # the first block hits a second cache miss for a different specialization
    if n_steps > 0:
        print("warmup (compilation)...")
        w_t0 = time.time()
        _ = run_block(current_df, current_state, min(interval, n_steps))
        jax.block_until_ready(_[0])
        print(f"compilation: {time.time() - w_t0:.2f}s")
        for run in runs:
            if run["telemetry"] is not None:
                run["telemetry"].compiled(time.time() - w_t0)

    run_t0 = time.time()

    while int(view(current_state, 0).step) < target_step and any(alive):
        block_steps = min(interval, target_step - int(view(current_state, 0).step))
        if block_steps <= 0:
            break

        block_start_step = int(view(current_state, 0).step)
        block_start_time = [float(view(current_state, i).time) for i in range(len(runs))]
        profiled = profile_next
        profiler = (
            telemetry_mod.profile_block(
                cast(str, runs[0]["output_dir"]), block_steps, block_start_step
            )
            if profiled
            else contextlib.nullcontext()
        )
        profile_next = False
        t0 = time.time()
        with profiler:
            run_result: Any = run_block(current_df, current_state, block_steps)
            current_df, current_phi, current_fluxes, current_state, dt_info = run_result
            jax.block_until_ready(current_df)
        wall_time = time.time() - t0
        blocks_done += 1
        if profiled:
            _share_profile_summary(runs)

        for i, run in enumerate(runs):
            if not alive[i]:
                continue
            out = run["output_dir"]
            d, st = view(current_df, i), view(current_state, i)
            # every step after the first NaN is wasted; keep the last good snapshot
            if stop_on_nan and not bool(jnp.isfinite(d).all()):
                msg = (
                    f"non-finite df at step {int(st.step)} (t={float(st.time):.4f}); "
                    f"stopping after {int(st.step) - start_step} steps"
                )
                tag = f" ({run['label']})" if run["label"] else ""
                print(f"\n*** DIVERGED{tag}: {msg}")
                if out is not None:
                    with open(os.path.join(out, "DIVERGED"), "w") as fh:
                        fh.write(msg + "\n")
                alive[i] = False
                stopped[i] = (int(st.step) - start_step, time.time() - run_t0)
                continue

            if out is not None:
                is_final = int(st.step) >= target_step
                due = blocks_done % snapshot_every == 0 or is_final
                snapshot = (
                    (save_snapshots and due)
                    or (save_final and is_final)
                    or (keep_latest_snapshot and due)
                )
                save_dumps_fn(
                    out,
                    d,
                    view(current_phi, i),
                    view(current_fluxes, i),
                    st,
                    run["geometry"],
                    save_dumps=snapshot,
                    params=run["params"],
                    pre=run["pre"],
                    dt_info=view(dt_info, i),
                    block_start_step=block_start_step,
                    block_start_time=block_start_time[i],
                    snapshot_f32=snapshot_f32,
                )

                if snapshot and keep_latest_snapshot and not save_snapshots:
                    latest = os.path.join(out, f"step_{int(st.step):06d}.npz")
                    if previous[i] and os.path.abspath(previous[i]) != os.path.abspath(latest):
                        with contextlib.suppress(OSError):
                            os.remove(previous[i])
                    previous[i] = latest

            log_step(
                view(current_fluxes, i), st, wall_time, n_steps=block_steps, label=run["label"]
            )
            if run["telemetry"] is not None:
                run["telemetry"].block(
                    int(st.step),
                    float(st.time),
                    block_steps,
                    wall_time,
                    view(dt_info, i),
                    profiled=profiled,
                )

    if n_steps > 0:
        for i, run in enumerate(runs):
            if run["telemetry"] is not None:
                steps, wall = stopped[i] or (
                    int(view(current_state, i).step) - start_step,
                    time.time() - run_t0,
                )
                run["telemetry"].close(wall, steps, "completed" if alive[i] else "diverged")

    if current_phi is None:
        diags = [
            _diagnostics(view(df, i), run["geometry"], run["params"], run["pre"])
            for i, run in enumerate(runs)
        ]
        if batched:
            current_phi = jnp.stack([p for p, _ in diags])
            current_fluxes = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *[f for _, f in diags])
        else:
            current_phi, current_fluxes = diags[0]

    return current_df, current_phi, current_fluxes, current_state


def _share_profile_summary(runs: List[Dict[str, Any]]) -> None:
    src = os.path.join(cast(str, runs[0]["output_dir"]), "profile_summary.json")
    for run in runs[1:]:
        if run["output_dir"] is not None and os.path.exists(src):
            os.makedirs(run["output_dir"], exist_ok=True)
            shutil.copy(src, os.path.join(run["output_dir"], "profile_summary.json"))


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
        geometry = compute_geometry_from_config(cfg)

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    df, geometry, state = gk_init(geometry, params, n_species=n_species)
    pre = linear_precompute(geometry, params)

    return df, geometry, params, state, pre
