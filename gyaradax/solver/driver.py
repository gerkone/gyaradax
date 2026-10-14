"""Simulation runtime."""

import contextlib
import functools
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast, overload

import jax
import jax.numpy as jnp
import numpy as np

from gyaradax.core.jax_config import enable_x64

enable_x64()

import gyaradax.diagnostics.telemetry as telemetry_mod
from gyaradax.core.params import gkparams_from_config, load_config, GKParams
from gyaradax.core.state import GKPre, GKState
from gyaradax.diagnostics.diag import phi_spectra
from gyaradax.diagnostics.fluxes import get_integrals
from gyaradax.geometry import compute_geometry_from_config
from gyaradax.io.gkw import load_geometry
from gyaradax.io.rundir import save_dumps as save_dumps_fn
from gyaradax.operators.fields import _compute_fields, g_to_f
from gyaradax.operators.poisson import calculate_phi
from gyaradax.operators.precompute import linear_precompute
from gyaradax.solver.rk4 import (
    gksolve,
    gksolve_until,
    init_f,
    default_state,
    mode_amplitude,
)


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


def gk_run_until(df, geometry, params, state, t_end, max_steps: int, pre=None):
    """Step until ``t_end`` (at most ``max_steps``): ``(df, phi, fluxes, state, dt_info)``, see ``gksolve_until``."""
    if pre is None:
        pre = linear_precompute(geometry, params)
    final_df, (phi, fluxes), final_state, dt_info = gksolve_until(
        df, geometry, params, state, t_end, max_steps=max_steps, pre=pre
    )
    return final_df, phi, fluxes, final_state, dt_info


def gk_run_until_batched(df_batch, geometry_batch, params_batch, state_batch, t_end, max_steps: int, pre_batch):
    def _single(df, geom, par, st, pre):
        return gk_run_until(df, geom, par, st, t_end, max_steps, pre=pre)

    return jax.vmap(_single)(df_batch, geometry_batch, params_batch, state_batch, pre_batch)


def frame_step_cap(dump_dt: float, dt: Any) -> int:
    # steps per compiled call in time mode: 8x a frame at the largest time step
    return max(int(np.ceil(8.0 * dump_dt / float(np.min(np.asarray(dt))))), 1)


def frame_target(t: float, origin: float, dump_dt: float) -> float:
    k = int(np.floor((t - origin) / dump_dt + 1e-6))
    return origin + (k + 1) * dump_dt


def _advance(run_until, df, state, t_target: float):
    # calls of at most max_steps steps until every member lands on t_target
    parts, times = [], None
    while True:
        df, phi, fluxes, state, dt_info = run_until(df, state, t_target)
        parts.append(dt_info)
        prev, times = times, np.asarray(state.time)
        if np.all(times >= t_target) or not np.all(np.isfinite(times)):
            break
        if prev is not None and np.array_equal(prev, times):
            break
    if len(parts) > 1:
        dt_info = {
            k: parts[0][k] if k == "dt_input" else jnp.concatenate([p[k] for p in parts], axis=-1)
            for k in parts[0]
        }
    return df, phi, fluxes, state, dt_info


def _steps_taken(dt_info: Dict[str, Any]) -> Dict[str, Any]:
    taken = np.isfinite(np.asarray(dt_info["dt_used"]))
    return {k: v if k == "dt_input" else np.asarray(v)[taken] for k, v in dt_info.items()}


def _has_diagnostics_at(output_dir: str, step: int) -> bool:
    path = os.path.join(output_dir, "fluxes.npz")
    if not os.path.exists(path):
        return False
    with np.load(path) as data:
        return "step" in data.files and bool(np.any(data["step"] == step))


# set from a signal handler: the running block finishes, is snapshotted, and the run stops
STOP_REQUESTED = threading.Event()
# exit code of a run stopped that way: resume it from its run directory
EXIT_INTERRUPTED = 75


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
    dataset: Any = None,
    dump_dt: Optional[float] = None,
    t_end: Optional[float] = None,
    dump_origin: Optional[float] = None,
    monitor: Any = None,
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
    ``dataset`` (a ``dataset.DatasetWriter``) stores one preprocessed frame per block,
    computes the dataset statistics once the run ends and keeps the trajectory's resume
    dump at the state the leg ends on.

    ``dump_dt`` switches to time mode: each block integrates to the next frame time
    ``dump_origin + k * dump_dt`` (default origin: the start time) and lands on it, and the
    run ends on the first frame at or after ``t_end``, or without ``t_end`` on the first
    frame at or after ``n_steps`` steps; ``checkpoint_interval`` is unused.

    ``monitor`` (an ``outcome.RunMonitor``) sees every frame; when it reports a reason to stop
    (decay, banding) the run ends on that frame with its snapshot, and it decides which frames
    go to ``dataset``.

    Returns:
        (df, phi, fluxes, state)
    """
    if pre is None:
        pre = linear_precompute(geometry, params)

    def run_block(d, s, steps):
        return gk_run(d, geometry, params, s, steps, pre=pre, return_dt_info=True)

    def run_until(d, s, t):
        return gk_run_until(d, geometry, params, s, t, frame_step_cap(dump_dt, params.dt), pre=pre)

    run = {
        "geometry": geometry,
        "params": params,
        "pre": pre,
        "output_dir": output_dir,
        "previous_snapshot": previous_snapshot,
        "telemetry": telemetry,
        "dataset": dataset,
        "monitor": monitor,
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
        run_until=run_until,
        frames=_frames(dump_dt, t_end, dump_origin),
    )


def _frames(dump_dt, t_end, dump_origin) -> Optional[Dict[str, Any]]:
    if not dump_dt:
        return None
    if dump_dt <= 0:
        raise ValueError(f"dump_dt must be positive, got {dump_dt}")
    return {"dump_dt": float(dump_dt), "t_end": t_end, "origin": dump_origin}


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
    datasets: Optional[Sequence[Any]] = None,
    dump_dt: Optional[float] = None,
    t_end: Optional[float] = None,
    dump_origin: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Any, GKState]:
    """``gksimulate`` for configs stacked along a leading batch axis (``gk_run_batched``).

    Every member writes its own output directory exactly as ``gksimulate`` would; a
    member that diverges gets its ``DIVERGED`` marker and no further output while
    the others continue. In time mode (``dump_dt``) the members share the frame times.
    """
    n = len(output_dirs)

    def run_block(d, s, steps):
        return gk_run_batched(
            d, geometry_batch, params_batch, s, steps, pre_batch, return_dt_info=True
        )

    def run_until(d, s, t):
        cap = frame_step_cap(dump_dt, params_batch.dt)
        return gk_run_until_batched(d, geometry_batch, params_batch, s, t, cap, pre_batch)

    runs = [
        {
            "geometry": _member(geometry_batch, i),
            "params": _member(params_batch, i),
            "pre": _member(pre_batch, i),
            "output_dir": output_dirs[i],
            "previous_snapshot": previous_snapshots[i] if previous_snapshots else None,
            "telemetry": telemetries[i] if telemetries else None,
            "dataset": datasets[i] if datasets else None,
            "monitor": None,
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
        run_until=run_until,
        frames=_frames(dump_dt, t_end, dump_origin),
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
    run_until: Any = None,
    frames: Optional[Dict[str, Any]] = None,
):
    """Block loop shared by ``gksimulate`` and ``gksimulate_batched``; I/O is per run.

    With ``frames`` (time mode) a block ends on the next frame time instead of after a
    fixed number of steps.
    """

    def view(x, i):
        return _member(x, i) if batched else x

    interval = checkpoint_interval if checkpoint_interval else n_steps
    snapshot_every = max(int(snapshot_every), 1)
    start_step = int(view(state, 0).step)
    target_step = start_step + n_steps
    origin = None
    if frames is not None:
        dump_dt, t_end = frames["dump_dt"], frames["t_end"]
        origin = float(view(state, 0).time) if frames["origin"] is None else float(frames["origin"])
    # the origin rides along in snapshots so a resumed run keeps the frame grid
    grid = {} if origin is None else {"dump_origin": origin}

    def done(st):
        if frames is None:
            return int(st.step) >= target_step
        if t_end is not None:
            return float(st.time) >= t_end - 1e-9 * dump_dt
        return int(st.step) >= target_step

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
                **grid,
            )

    current_df = df
    current_state = state
    current_phi: Any = None
    current_fluxes: Any = None
    alive = [True] * len(runs)
    # why a run left the loop early on its own: decayed, banded (None: diverged)
    ended: List[Optional[str]] = [None] * len(runs)
    stopped: List[Optional[Tuple[int, float]]] = [None] * len(runs)
    previous = [run["previous_snapshot"] for run in runs]
    profile_next = profile and runs[0]["output_dir"] is not None
    blocks_done = 0
    interrupted = False

    # warmup compile with the same return_dt_info as the body loop, otherwise
    # the first block hits a second cache miss for a different specialization
    if not done(view(state, 0)) and (frames is not None or n_steps > 0):
        print("warmup (compilation)...")
        w_t0 = time.time()
        if frames is None:
            _ = run_block(current_df, current_state, min(interval, n_steps))
        else:
            # zero steps: compiles the time-mode solve without advancing
            _ = run_until(current_df, current_state, float(view(current_state, 0).time))
        jax.block_until_ready(_[0])
        print(f"compilation: {time.time() - w_t0:.2f}s")
        for run in runs:
            if run["telemetry"] is not None:
                run["telemetry"].compiled(time.time() - w_t0)

    run_t0 = time.time()

    while not done(view(current_state, 0)) and any(alive):
        block_start_step = int(view(current_state, 0).step)
        if frames is None:
            block_steps = min(interval, target_step - block_start_step)
            if block_steps <= 0:
                break
        else:
            t_target = frame_target(float(view(current_state, 0).time), cast(float, origin), dump_dt)

        block_start_time = [float(view(current_state, i).time) for i in range(len(runs))]
        profiled = profile_next
        profiler = (
            telemetry_mod.profile_block(
                cast(str, runs[0]["output_dir"]),
                block_steps if frames is None else lambda: int(view(current_state, 0).step) - block_start_step,
                block_start_step,
            )
            if profiled
            else contextlib.nullcontext()
        )
        profile_next = False
        t0 = time.time()
        with profiler:
            if frames is None:
                run_result: Any = run_block(current_df, current_state, block_steps)
            else:
                run_result = _advance(run_until, current_df, current_state, t_target)
            current_df, current_phi, current_fluxes, current_state, dt_info = run_result
            jax.block_until_ready(current_df)
        wall_time = time.time() - t0
        blocks_done += 1
        stop_now = STOP_REQUESTED.is_set()
        if profiled:
            _share_profile_summary(runs)
        if frames is not None:
            block_steps = int(view(current_state, 0).step) - block_start_step
            if block_steps == 0 and float(view(current_state, 0).time) < t_target:
                raise RuntimeError(f"time step collapsed at t = {block_start_time[0]:.6f}; no frame reached")

        for i, run in enumerate(runs):
            if not alive[i]:
                continue
            out = run["output_dir"]
            d, st = view(current_df, i), view(current_state, i)
            dt_i = view(dt_info, i) if frames is None else _steps_taken(view(dt_info, i))
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

            steps_i = block_steps if frames is None else len(dt_i["dt_used"])
            monitor = run["monitor"]
            stop_reason = None
            if monitor is not None:
                stop_reason = monitor.observe(st, view(current_fluxes, i), view(current_phi, i))
            if out is not None:
                is_final = done(st) or stop_reason is not None
                due = blocks_done % snapshot_every == 0 or is_final or stop_now
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
                    dt_info=dt_i,
                    block_start_step=block_start_step,
                    block_start_time=block_start_time[i],
                    snapshot_f32=snapshot_f32,
                    **grid,
                )

                if snapshot and keep_latest_snapshot and not save_snapshots:
                    latest = os.path.join(out, f"step_{int(st.step):06d}.npz")
                    if previous[i] and os.path.abspath(previous[i]) != os.path.abspath(latest):
                        with contextlib.suppress(OSError):
                            os.remove(previous[i])
                    previous[i] = latest

            writes = run["dataset"] is not None and (
                monitor is None
                or monitor.write_frame(run["dataset"].n_frames, final=stop_reason is not None)
            )
            if writes:
                _dataset_frame(run, d, view(current_phi, i), view(current_fluxes, i), st)

            log_step(
                view(current_fluxes, i), st, wall_time, n_steps=steps_i, label=run["label"]
            )
            if run["telemetry"] is not None:
                run["telemetry"].block(
                    int(st.step),
                    float(st.time),
                    steps_i,
                    wall_time,
                    dt_i,
                    profiled=profiled,
                )
            if stop_reason is not None:
                tag = f" ({run['label']})" if run["label"] else ""
                print(f"\n*** {stop_reason.upper()}{tag}: {monitor.message}; stopping")
                alive[i] = False
                ended[i] = stop_reason
                stopped[i] = (int(st.step) - start_step, time.time() - run_t0)
        if stop_now and not done(view(current_state, 0)):
            interrupted = True
            step = int(view(current_state, 0).step)
            print(f"\nstop requested: stopped at step {step} after its snapshot")
            break

    if blocks_done or n_steps > 0:
        for i, run in enumerate(runs):
            if run["telemetry"] is not None:
                steps, wall = stopped[i] or (
                    int(view(current_state, i).step) - start_step,
                    time.time() - run_t0,
                )
                status = "completed" if alive[i] and not interrupted else "interrupted"
                status = status if alive[i] else (ended[i] or "diverged")
                run["telemetry"].close(wall, steps, status)
    for i, run in enumerate(runs):
        monitor = run["monitor"]
        if monitor is not None:
            writer = run["dataset"]
            # a run whose frames waited for turbulence that never came keeps its last frame
            if writer is not None and not interrupted and blocks_done and writer.n_frames == 0:
                if alive[i] or ended[i]:
                    _dataset_frame(
                        run,
                        view(current_df, i),
                        view(current_phi, i),
                        view(current_fluxes, i),
                        view(current_state, i),
                    )
            monitor.finish(writer, final=not interrupted)
        if run["dataset"] is None:
            continue
        # an interrupted run leaves the statistics to the leg that completes it
        if not interrupted:
            run["dataset"].finalize()
        if blocks_done and (alive[i] or ended[i]):
            run["dataset"].write_resume(view(current_df, i), view(current_state, i), dump_origin=origin)

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


def _dataset_frame(run: Dict[str, Any], df, phi, fluxes, state) -> None:
    ky_spec, kx_spec = phi_spectra(phi, run["geometry"])
    run["dataset"].add(
        df,
        phi,
        fluxes,
        state,
        ky_spec,
        kx_spec,
        integrals=functools.partial(
            _diagnostics, geometry=run["geometry"], params=run["params"], pre=run["pre"]
        ),
    )


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
    from gyaradax.core.params import gkparams_from_input_and_geometry
    from gyaradax.io.gkw import K_files, load_gkw_k_dump, read_gkw_dump_time

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
