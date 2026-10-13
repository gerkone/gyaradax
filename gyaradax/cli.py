"""gyaradax command-line interface: ``run``, ``bench``, ``growth``, ``info``, ``build-cuda``,
``datagen``.

Run mode (adiabatic/kinetic), electromagnetic terms and velocity-space
sharding all come from the YAML config; flags only override it. See docs/CLI.md.

JAX and gyaradax are imported inside the command handlers, not at module scope,
because ``CUDA_VISIBLE_DEVICES``/``XLA_FLAGS`` must be set before the backend
initialises.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import replace
from typing import Any, Sequence, TypedDict, cast

from omegaconf import OmegaConf

from gyaradax.core.runtime_config import append_xla_flags, configure_runtime_env

_MULTI_GPU_XLA_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_pipelined_all_reduce=true",
    "--xla_gpu_enable_pipelined_all_gather=true",
    "--xla_gpu_enable_while_loop_double_buffering=true",
)


# ---------------------------------------------------------------------------
# config inspection (must stay JAX-free — runs before backend init)
# ---------------------------------------------------------------------------


class ConfigFacts(TypedDict):
    """What we can learn from a config without importing JAX."""

    adiabatic: bool
    nsp: int
    nvpar: int
    nmu: int
    n_gpus: int
    n_gpus_sp: int
    n_gpus_vp: int
    n_gpus_mu: int


def _peek_config(path: str) -> ConfigFacts:
    """Read the handful of switches needed to set up the process environment."""
    cfg = OmegaConf.load(path)
    grid = cfg.get("grid") or {}
    physics = cfg.get("physics") or {}
    shard = cfg.get("sharding") or {}
    adiabatic = bool(grid.get("adiabatic_electrons", True))
    mas = physics.get("mas", 1.0)
    nsp = 1 if adiabatic or not OmegaConf.is_list(mas) else len(mas)
    return {
        "adiabatic": adiabatic,
        "nsp": nsp,
        "nvpar": int(grid.get("nvpar", 0) or 0),
        "nmu": int(grid.get("nmu", 0) or 0),
        "n_gpus": int(shard.get("n_gpus", 0) or 0),
        "n_gpus_sp": int(shard.get("n_gpus_sp", 1)),
        "n_gpus_vp": int(shard.get("n_gpus_vp", 1)),
        "n_gpus_mu": int(shard.get("n_gpus_mu", 1)),
    }


def _auto_mesh(n_devices: int, facts: ConfigFacts) -> tuple[int, int, int]:
    """Split ``n_devices`` over (species, vpar, mu): species first, then mu, then vpar.

    vpar takes what is left and keeps at least two vpar points per shard.
    """
    if facts["nvpar"] <= 0 or facts["nmu"] <= 0:
        raise SystemExit(
            "error: automatic sharding needs grid.nvpar and grid.nmu in the config; "
            "set --n-gpus-sp/--n-gpus-vp/--n-gpus-mu instead"
        )
    sp = math.gcd(n_devices, facts["nsp"])
    mu = math.gcd(n_devices // sp, facts["nmu"])
    vp = n_devices // (sp * mu)
    if facts["nvpar"] % vp or facts["nvpar"] // vp < 2:
        raise SystemExit(
            f"error: cannot split {n_devices} GPUs over species ({facts['nsp']}), "
            f"mu ({facts['nmu']}) and vpar ({facts['nvpar']}); choose a GPU count that "
            "divides nsp * nmu * nvpar / 2"
        )
    return sp, vp, mu


def _visible_gpus(args: argparse.Namespace) -> int | None:
    """GPUs the run will see, when known without initialising JAX."""
    listed = args.device_list or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not listed:
        return None
    return len([d for d in listed.split(",") if d.strip()])


def _mesh_request(args: argparse.Namespace, facts: ConfigFacts) -> tuple[int, int, int]:
    """Resolve the device mesh (species, vpar, mu).

    Precedence: per-axis flags (each overrides the config's value) > ``--n-gpus``
    (automatic layout) > the config's per-axis ``sharding`` > its ``n_gpus`` >
    a multi-GPU ``--device-list`` (automatic layout over the listed GPUs).
    """
    axis_flags = (args.n_gpus_sp, args.n_gpus_vp, args.n_gpus_mu)
    config_axes = (facts["n_gpus_sp"], facts["n_gpus_vp"], facts["n_gpus_mu"])
    if any(axis_flags):
        return cast(
            tuple[int, int, int], tuple(f if f else c for f, c in zip(axis_flags, config_axes))
        )
    if getattr(args, "n_gpus", 0):
        return _auto_mesh(args.n_gpus, facts)
    if math.prod(config_axes) > 1:
        return config_axes
    if facts["n_gpus"] > 1:
        return _auto_mesh(facts["n_gpus"], facts)
    n_listed = len([d for d in (args.device_list or "").split(",") if d.strip()])
    if n_listed > 1:
        return _auto_mesh(n_listed, facts)
    return 1, 1, 1


def _check_mesh(mesh: tuple[int, int, int], facts: ConfigFacts, visible: int | None) -> None:
    """Fail early, with the fix, when a mesh cannot shard the grid or the GPUs are missing."""
    sp, vp, mu = mesh
    n = sp * vp * mu
    problems = []
    if facts["nsp"] % sp:
        problems.append(f"n_gpus_sp={sp} does not divide the {facts['nsp']} species")
    if facts["nmu"] and facts["nmu"] % mu:
        problems.append(f"n_gpus_mu={mu} does not divide nmu={facts['nmu']}")
    if facts["nvpar"] and vp > 1 and (facts["nvpar"] % vp or facts["nvpar"] // vp < 2):
        problems.append(
            f"n_gpus_vp={vp} must divide nvpar={facts['nvpar']} and leave >= 2 vpar points per GPU"
        )
    if visible is not None and n > visible:
        problems.append(
            f"the mesh needs {n} GPUs but only {visible} {'is' if visible == 1 else 'are'} visible"
        )
    if problems:
        hint = ""
        if n > 1 and facts["nvpar"] and facts["nmu"]:
            try:
                hint = f" (e.g. --n-gpus {n} picks sp, vp, mu = {_auto_mesh(n, facts)})"
            except SystemExit:
                pass
        raise SystemExit("error: " + "; ".join(problems) + hint)


def _configure_env(args: argparse.Namespace, facts_list: Sequence[ConfigFacts]) -> tuple[int, ...]:
    """Set device visibility and XLA flags before JAX initialises its backend."""
    meshes = [_mesh_request(args, f) for f in facts_list]
    for m, f in zip(meshes, facts_list):
        _check_mesh(m, f, _visible_gpus(args))
    mesh = max(meshes, key=lambda m: m[0] * m[1] * m[2])
    n_devices = mesh[0] * mesh[1] * mesh[2]

    if n_devices > 1 and args.device is not None and args.device >= 0:
        raise SystemExit(
            f"error: --device {args.device} pins a single GPU but the requested mesh needs "
            f"{n_devices}. Drop --device, or use --device-list."
        )

    if getattr(args, "mem_fraction", None):
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_fraction)
    configure_runtime_env(
        device=args.device if args.device is not None else -1,
        device_list=args.device_list,
        n_gpus_sp=mesh[0],
        n_gpus_vp=mesh[1],
        n_gpus_mu=mesh[2],
    )
    if n_devices > 1:
        # also needed when the mesh came from the config rather than the flags
        append_xla_flags(_MULTI_GPU_XLA_FLAGS)
    return mesh


# ---------------------------------------------------------------------------
# run setup
# ---------------------------------------------------------------------------


class RunSetup(TypedDict):
    df: Any
    geometry: dict[str, Any]
    params: Any
    state: Any
    pre: Any
    output_dir: str
    n_steps: int
    block_size: int
    snapshot_every: int
    save_dumps: bool
    dataset: dict[str, Any] | None
    previous_snapshot: str | None
    stale: list[str]
    name: str
    data_dir: str | None
    mesh: Any
    config_path: str
    effective_config: Any
    resumed_from: str | None
    start_step: int
    telemetry: bool
    profile: bool
    dump_dt: float | None
    t_end: float | None
    dump_origin: float | None
    trajectory: str | None


def _default_output_dir(cfg: Any, args: argparse.Namespace, run_dir: str | None) -> str:
    kinetic = not bool((cfg.get("grid") or {}).get("adiabatic_electrons", True))
    default = f"outputs_{'kinetic' if kinetic else 'adiabatic'}_{cfg.run.name}"
    return args.output_dir or run_dir or default


def _parse_sets(items: Sequence[str] | None) -> dict[str, Any]:
    import yaml

    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"error: --set expects KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def _parse_ns_list(text: str) -> list[int]:
    try:
        values = [int(v) for v in text.replace(",", " ").split()]
    except ValueError:
        raise SystemExit(f"error: expected resolutions like 36 or 24,36, got {text!r}")
    if not values or min(values) < 2:
        raise SystemExit(f"error: expected resolutions like 36 or 24,36, got {text!r}")
    return values


def _progress(every: int = 10) -> Any:
    count = [0]

    def log(msg: str) -> None:
        count[0] += 1
        if count[0] % every == 0:
            print(f"    {msg}", flush=True)

    return log


def _linear_check(args: argparse.Namespace, config_path: str, run_dir: str | None) -> int:
    """Linear growth rates at the run's ns and the reference ones, cached in the run directory."""
    from gyaradax.diagnostics import growth

    cfg = OmegaConf.load(config_path)
    output_dir = _default_output_dir(cfg, args, run_dir)
    path = os.path.join(output_dir, "linear_check.json")
    refs = _parse_ns_list(args.linear_check)
    key = growth.config_key(growth.adjusted_config(cfg))
    report = None
    if os.path.isfile(path):
        with open(path) as fh:
            cached = json.load(fh)
        have = {int(r["ns"]) for r in cached.get("runs", [])}
        if cached.get("config_key") == key and set(refs) | {int(cfg.grid.ns)} <= have:
            report = cached
            print(f"linear check: {path} (computed before)")
    if report is None:
        print(f"linear check: growth rates at ns {int(cfg.grid.ns)} and reference ns {refs}")
        report = growth.check(
            cfg, refs, backend=args.backend, log=_progress(), t_end=args.linear_check_t_end
        )
        os.makedirs(output_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(report, fh, indent=1)
        os.replace(tmp, path)
    print(growth.format_check(report))
    own = report["comparison"][str(report["ns"])]
    tol = args.linear_check_tol
    if tol is not None and own["rel"] is not None and abs(own["rel"]) > tol:
        print(
            f"exit {growth.EXIT_LINEAR_CHECK}: γ at kθρ {own['ky']:.3f} is {own['gamma']:+.4f} "
            f"at ns {report['ns']} against {own['gamma_ref']:+.4f} at ns {report['reference_ns']} "
            f"({own['rel']:+.1%}, tolerance {tol:.1%}); not starting the nonlinear run"
        )
        return growth.EXIT_LINEAR_CHECK
    return 0


def _cmd_growth(args: argparse.Namespace) -> int:
    if args.mem_fraction:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_fraction)
    configure_runtime_env(
        device=args.device if args.device is not None else -1, device_list=args.device_list
    )
    from gyaradax.diagnostics import growth

    cfg = growth.adjusted_config(OmegaConf.load(args.config), sets=_parse_sets(args.set))
    own = int(cfg.grid.ns)
    runs = []
    for ns in sorted(set(args.ns or [own])):
        print(f"ns {ns}: linear run to settled growth rates (t_end {args.t_end:g})", flush=True)
        runs.append(
            growth.growth_rates(cfg, ns, backend=args.backend, t_end=args.t_end, log=_progress())
        )
    report = growth.assemble(runs, own)
    report["config"] = os.path.abspath(args.config)
    report["sets"] = _parse_sets(args.set)
    print(growth.format_check(report))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {args.output}")
    return 0


def _resolve_config(path: str) -> tuple[str, str | None]:
    """A run directory stands for its dumped ``config.yaml`` and is resumed in place."""
    from gyaradax.io.dataset import is_trajectory

    if is_trajectory(path):
        raise SystemExit(f"error: {path} is a dataset trajectory; extend it alone with `gyaradax run {path}`")
    if os.path.isdir(path):
        config = os.path.join(path, "config.yaml")
        if not os.path.isfile(config):
            raise SystemExit(f"error: {path} is a directory without a config.yaml to resume from")
        return config, path
    return path, None


def _snapshots(run_dir: str) -> dict[int, str]:
    steps = {}
    for path in glob.glob(os.path.join(run_dir, "step_*.npz")):
        m = re.search(r"step_(\d+)\.npz$", path)
        if m:
            steps[int(m.group(1))] = path
    return steps


def _latest_snapshot(run_dir: str) -> str | None:
    steps = _snapshots(run_dir)
    return steps[max(steps)] if steps else None


def _in_dir(path: str, directory: str) -> bool:
    return os.path.abspath(os.path.dirname(path)) == os.path.abspath(directory)


_RUN_OUTPUTS = (
    "fluxes.npz",
    "fluxes_em.npz",
    "kxspec.npz",
    "kyspec.npz",
    "growth.npz",
    "dt_history.npz",
    "run_info.jsonl",
    "telemetry.jsonl",
    "profile",
    "profile_summary.json",
    "DIVERGED",
)


def _foreign_outputs(output_dir: str, snapshot: str | None, start_step: int) -> list[str]:
    """Files in ``output_dir`` that do not belong to the trajectory this run continues."""
    snapshots = _snapshots(output_dir)
    if snapshot is not None and _in_dir(snapshot, output_dir):
        return [path for step, path in sorted(snapshots.items()) if step > start_step]
    run_files = [os.path.join(output_dir, name) for name in _RUN_OUTPUTS]
    return [*snapshots.values(), *(p for p in run_files if os.path.exists(p))]


def _confirm(question: str) -> bool:
    try:
        interactive = sys.stdin.isatty()
    except (AttributeError, OSError, ValueError):
        interactive = False
    if not interactive:
        print(f"{question} [y/N] n (not interactive)")
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _has_geom_dat(data_dir: str | None) -> bool:
    return bool(data_dir) and os.path.exists(os.path.join(str(data_dir), "geom.dat"))


def _find_k_file(data_dir: str | None, resume_from: str | None = None) -> str | None:
    """Return a K-dump to resume from, or None to cold-start."""
    from gyaradax.io.gkw import K_files

    if not data_dir:
        return None
    if resume_from:
        for candidate in (resume_from, f"K{resume_from}"):
            path = os.path.join(data_dir, candidate)
            if os.path.exists(path):
                return path
        print(f"  warning: resume file '{resume_from}' not found in {data_dir}")
        return None
    ks = K_files(data_dir)
    if ks:
        return os.path.join(data_dir, ks[0])
    k01 = os.path.join(data_dir, "K01")
    return k01 if os.path.exists(k01) else None


def _setup_run(config_path: str, args: argparse.Namespace, run_dir: str | None = None) -> RunSetup:
    """Build (df, geometry, params, state, pre) and metadata for one config.

    ``run_dir`` (a previous run's directory) resumes from its latest
    ``step_*.npz`` snapshot and keeps writing into it.
    """
    import jax.numpy as jnp
    import numpy as np

    from gyaradax import load_config, sharding as _sharding
    from gyaradax.geometry import compute_geometry_from_config
    from gyaradax.core.params import gkparams_from_config
    from gyaradax.operators.precompute import linear_precompute
    from gyaradax.solver.driver import (
        _compute_phi_for_init,
        _ensure_species_arrays,
        gk_init,
    )
    from gyaradax.solver.rk4 import mode_amplitude
    from gyaradax.core.state import GKState
    from gyaradax.io.gkw import (
        load_geometry,
        load_gkw_k_dump,
        read_gkw_dump_dtim,
        read_gkw_dump_time,
    )

    cfg = load_config(config_path)
    facts = _peek_config(config_path)
    # a named-but-missing reference dir should not fail the run
    data_dir = getattr(cfg.run, "data_dir", None)
    if data_dir and not os.path.isdir(data_dir):
        print(f"  note: data_dir '{data_dir}' not found — computing geometry, no reference report")
        data_dir = None
    name = cfg.run.name
    kinetic = not facts["adiabatic"]

    output_dir = _default_output_dir(cfg, args, run_dir)

    overrides: dict[str, Any] = {}
    if args.mp:
        overrides["mixed_precision"] = True
    if args.dp:
        overrides["mixed_precision"] = False
    if args.z2z is not None:
        overrides["use_z2z"] = args.z2z
    backend = args.backend
    if backend:
        overrides["backend"] = backend
    sp, vp, mu = _mesh_request(args, facts)
    overrides["n_gpus_sp"], overrides["n_gpus_vp"], overrides["n_gpus_mu"] = sp, vp, mu
    params = gkparams_from_config(cfg, **overrides)

    if _has_geom_dat(data_dir):
        geometry = cast(dict[str, Any], load_geometry(data_dir))
    else:
        geometry = cast(dict[str, Any], compute_geometry_from_config(cfg))

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    from gyaradax.io.dataset import is_trajectory, resume_path

    snapshot = None
    rolling = False
    trajectory = None
    if not args.from_scratch:
        if args.resume_from and is_trajectory(args.resume_from):
            trajectory = os.path.abspath(args.resume_from)
            snapshot = resume_path(trajectory)
        elif args.resume_from and args.resume_from.endswith(".npz"):
            snapshot = args.resume_from
        elif run_dir is not None and not args.resume_from:
            snapshot = _latest_snapshot(run_dir)
            rolling = snapshot is not None
    k_path = None
    if not args.from_scratch and snapshot is None:
        k_path = _find_k_file(data_dir, args.resume_from)

    dump_origin = None
    if snapshot is not None:
        if not os.path.isfile(snapshot):
            raise SystemExit(f"error: snapshot {snapshot} not found")
        res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        expected = res if params.adiabatic_electrons else (n_species, *res)
        with np.load(snapshot) as ck:
            if tuple(ck["df"].shape) != expected:
                raise SystemExit(
                    f"error: snapshot df shape {ck['df'].shape} does not match the grid {expected}"
                )
            df = jnp.asarray(ck["df"]).astype(jnp.complex128)
            state = GKState(
                time=jnp.asarray(ck["time"], dtype=jnp.float64),
                step=jnp.asarray(ck["step"], dtype=jnp.int32),
                accumulated_norm_factor=jnp.asarray(ck["accumulated_norm_factor"], dtype=jnp.float64),
                window_start_amp=jnp.asarray(ck["window_start_amp"], dtype=jnp.float64),
                last_growth_rate=jnp.asarray(ck["last_growth_rate"], dtype=jnp.float64),
            )
            if "dump_origin" in ck.files:
                dump_origin = float(ck["dump_origin"])
        geometry = _ensure_species_arrays(geometry, params)
        print(f"  resumed from {snapshot} (step {int(state.step)}, t={float(state.time):.4f})")
    elif k_path is not None:
        res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = load_gkw_k_dump(k_path, res, n_species=n_species)

        dat_path = k_path + ".dat"
        t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0
        actual_dt = read_gkw_dump_dtim(dat_path) if os.path.exists(dat_path) else 0.0
        if 0 < actual_dt < params.dt:
            params = replace(params, dt=actual_dt)

        geometry = _ensure_species_arrays(geometry, params)
        phi0 = _compute_phi_for_init(df, geometry, params)
        amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
        state = GKState(
            time=jnp.array(t_start, dtype=jnp.float64),
            step=jnp.array(0, dtype=jnp.int32),
            accumulated_norm_factor=jnp.ones(len(geometry["krho"]), dtype=jnp.float64),
            window_start_amp=amp0,
            last_growth_rate=jnp.zeros(len(geometry["krho"]), dtype=jnp.float64),
        )
        print(
            f"  resumed from {os.path.basename(k_path)} "
            f"(t={t_start:.4f}, dt={float(params.dt):.4e})"
        )
    else:
        df, geometry, state = gk_init(geometry, params, n_species=n_species)

    stale = _foreign_outputs(output_dir, snapshot, int(state.step)) if args.command == "run" else []
    if stale and not args.overwrite:
        names = sorted(os.path.basename(p) for p in stale)
        listed = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
        own = snapshot is not None and _in_dir(snapshot, output_dir)
        what = "snapshots past the resumed step" if own else "the outputs of another run"
        raise SystemExit(
            f"error: {output_dir} holds {what} ({listed}); resume it with "
            f"`gyaradax run {output_dir}`, write elsewhere with --output-dir, or pass "
            "--overwrite to replace them"
        )

    mesh = _sharding.build_mesh(params)
    if mesh is not None:
        grid = _sharding.grid_shape_from(params, geometry)
        # sharded precompute: full-size arrays never land on one device
        pre = _sharding.precompute_sharded(geometry, params, mesh, grid)
        df = _sharding.shard_df(df, mesh, grid)
    else:
        pre = linear_precompute(geometry, params)

    if not (params.adaptive_dt and params.non_linear):
        from gyaradax.cfl import estimate_linear_timestep

        dt_lin = float(estimate_linear_timestep(pre, params=params))
        if dt_lin < params.dt:
            print(
                f"  linear timestep automatically reduced: dt {params.dt:.4e} -> {dt_lin:.4e} "
                "(linear stability estimate)"
            )
            params = replace(params, dt=dt_lin)

    # checkpoint cadence: --block-size > run.block_size > config's dump_interval x naverage
    run_block = int(getattr(cfg.run, "block_size", 0) or 0)
    if args.block_size:
        block_size = args.block_size
    elif run_block:
        block_size = run_block
    else:
        dump_interval = int(getattr(cfg.solver, "dump_interval", 0) or 0)
        block_size = dump_interval * params.naverage if dump_interval else 120

    # time mode: frames every dump_dt time units, the run ends on the first frame at or after t_end
    dump_dt = float(getattr(args, "dump_dt", None) or getattr(cfg.solver, "dump_dt", 0) or 0) or None
    t_end = _run_end_time(cfg, args, dump_dt, float(state.time))

    # steps: --n-steps more > --n-blocks > up to solver.n_steps (resumed) > solver.n_steps > heuristic
    start_step = int(state.step)
    cfg_steps = int(getattr(cfg.solver, "n_steps", 0) or 0)
    if t_end is not None:
        # an estimate at the largest time step, for reports only
        n_steps = max(int(np.ceil((t_end - float(state.time)) / params.dt)), 0)
    elif args.n_steps:
        n_steps = args.n_steps
    elif args.n_blocks:
        n_steps = args.n_blocks * block_size
    elif snapshot is not None and cfg_steps:
        n_steps = max(cfg_steps - start_step, 0)
    elif cfg_steps:
        n_steps = cfg_steps
    elif kinetic and data_dir:
        try:
            ref_times = np.loadtxt(os.path.join(data_dir, "time.dat"))
            n_steps = max(block_size, int((ref_times[-1] - float(state.time)) / params.dt))
        except (FileNotFoundError, OSError):
            n_steps = 100 * block_size
    else:
        n_steps = 265 * block_size

    # a run shorter than the checkpoint interval is one block
    saved_block_size = block_size
    if dump_dt is None and 0 < n_steps < block_size:
        print(f"  note: checkpoint interval {block_size} > n_steps {n_steps}; this run is one block")
        if _confirm(
            f"  keep {n_steps} as the checkpoint interval when this run is resumed "
            f"(diagnostics and a snapshot every {n_steps} steps)?"
        ):
            saved_block_size = n_steps
        block_size = n_steps

    # restart snapshot every k blocks (and after the last): --snapshot-every > run.snapshot_every
    snapshot_every = int(
        getattr(args, "snapshot_every", 0) or getattr(cfg.run, "snapshot_every", 0) or 1
    )
    dataset = _dataset_settings(cfg, args, name, trajectory)
    archived = bool(getattr(cfg.run, "save_dumps", False))
    save_dumps = archived if args.save_dumps is None else bool(args.save_dumps)
    # the rolling restart file this run continues from goes once a newer one exists
    keep = archived or save_dumps or not rolling or not _in_dir(cast(str, snapshot), output_dir)
    previous_snapshot = None if keep else snapshot

    return {
        "df": df,
        "geometry": geometry,
        "params": params,
        "state": state,
        "pre": pre,
        "output_dir": output_dir,
        "n_steps": n_steps,
        "block_size": block_size,
        "snapshot_every": snapshot_every,
        "save_dumps": save_dumps,
        "dataset": dataset,
        "previous_snapshot": previous_snapshot,
        "stale": stale,
        "name": name,
        "data_dir": data_dir,
        "mesh": mesh,
        "config_path": config_path,
        "effective_config": _effective_config(
            cfg, params, start_step + n_steps, saved_block_size, snapshot_every, save_dumps, dataset,
            dump_dt=dump_dt, t_end=t_end,
        ),
        "resumed_from": snapshot or k_path,
        "start_step": start_step,
        "telemetry": _debug_flag(cfg, args, "telemetry"),
        "profile": _debug_flag(cfg, args, "profile"),
        "dump_dt": dump_dt,
        "t_end": t_end,
        "dump_origin": dump_origin,
        "trajectory": trajectory,
    }


def _run_end_time(cfg: Any, args: argparse.Namespace, dump_dt: float | None, t_now: float) -> float | None:
    """End time of a time-mode run: --t-end > --n-blocks frames > (--n-steps: none) > solver.t_end."""
    t_arg = getattr(args, "t_end", None)
    if dump_dt is None:
        if t_arg is not None or getattr(cfg.solver, "t_end", None) is not None:
            raise SystemExit("error: an end time needs a dump spacing (--dump-dt or solver.dump_dt)")
        return None
    if t_arg is not None:
        return float(t_arg)
    if args.n_blocks:
        return t_now + args.n_blocks * dump_dt
    if args.n_steps:
        return None
    t_cfg = getattr(cfg.solver, "t_end", None)
    return None if t_cfg is None else float(t_cfg)


def _dataset_settings(
    cfg: Any, args: argparse.Namespace, name: str, trajectory: str | None = None
) -> dict[str, Any] | None:
    """Preprocessed-dataset output: command-line flags over the config's ``dataset`` section.

    A run extending a ``trajectory`` from its resume dump writes into it unless --dataset is given.
    """
    from gyaradax.io.dataset import frame_bits

    section = cfg.get("dataset") or {}
    stats_from = getattr(args, "dataset_stats_from", None)
    out = {
        "stats_from": float(section.get("stats_from", 0.0) if stats_from is None else stats_from),
        "resume_dtype": getattr(args, "resume_dtype", None) or str(section.get("resume_dtype", "f32")),
        "takeover": False,
    }
    if trajectory is not None and not getattr(args, "dataset", None):
        suffix = "_ifft_realpotens"
        base = os.path.basename(trajectory.rstrip(os.sep))
        return {
            **out,
            "dir": os.path.dirname(trajectory.rstrip(os.sep)),
            "bits": frame_bits(trajectory),
            "name": base[: -len(suffix)] if base.endswith(suffix) else base,
            "takeover": True,
        }
    root = getattr(args, "dataset", None) or section.get("dir")
    if not root:
        return None
    return {
        **out,
        "dir": str(root),
        "bits": getattr(args, "dataset_bits", None) or str(section.get("bits", "zstd16")),
        "name": getattr(args, "dataset_name", None) or str(section.get("name", name)),
    }


def _dataset_writer(setup: RunSetup) -> Any:
    if setup["dataset"] is None:
        return None
    from gyaradax.io.dataset import DatasetWriter

    ds = setup["dataset"]
    writer = DatasetWriter(
        ds["dir"],
        ds["name"],
        setup["geometry"],
        setup["params"],
        os.path.join(setup["output_dir"], "config.yaml"),
        bits=ds["bits"],
        start_step=setup["start_step"],
        stats_from=ds["stats_from"],
        resume_dtype=ds["resume_dtype"],
        takeover=ds["takeover"],
    )
    print(f"  dataset: {writer.path} ({ds['bits']}, {writer.n_frames} frames so far)")
    return writer


def _debug_flag(cfg: Any, args: argparse.Namespace, name: str) -> bool:
    debug_cfg = cfg.get("debug") or {}
    return bool(getattr(args, name, False) or getattr(args, "debug", False) or debug_cfg.get(name, False))


def _effective_config(
    cfg: Any,
    params: Any,
    target_step: int,
    block_size: int,
    snapshot_every: int,
    save_dumps: bool,
    dataset: dict[str, Any] | None = None,
    dump_dt: float | None = None,
    t_end: float | None = None,
) -> Any:
    """The config with the command-line choices folded in, so the run directory reproduces it."""
    eff = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if dump_dt is not None:
        OmegaConf.update(eff, "solver.dump_dt", float(dump_dt), force_add=True)
        # the end time, or the step budget when the run has none
        if t_end is not None:
            OmegaConf.update(eff, "solver.t_end", float(t_end), force_add=True)
        elif "t_end" in eff.solver:
            eff.solver.pop("t_end")
    for key, value in (
        ("solver.backend", params.backend),
        ("solver.mixed_precision", bool(params.mixed_precision)),
        ("solver.use_z2z", bool(params.use_z2z)),
        ("solver.dt", float(params.dt)),
        ("solver.n_steps", int(target_step) if t_end is None else int(eff.solver.get("n_steps", 0) or 0)),
        ("run.block_size", int(block_size)),
        ("run.snapshot_every", int(snapshot_every)),
        ("run.save_dumps", bool(save_dumps)),
    ):
        OmegaConf.update(eff, key, value, force_add=True)
    if dataset is not None:
        eff.dataset = {k: v for k, v in dataset.items() if k != "takeover"}
    mesh = (params.n_gpus_sp, params.n_gpus_vp, params.n_gpus_mu)
    if "sharding" in eff or math.prod(mesh) > 1:
        eff.sharding = {"n_gpus_sp": mesh[0], "n_gpus_vp": mesh[1], "n_gpus_mu": mesh[2]}
    return eff


def _write_run_record(setup: RunSetup) -> None:
    """config.yaml and geometry.pkl for the run, and one line per invocation in run_info.jsonl."""
    import socket

    from gyaradax.diagnostics.telemetry import _git_revision
    from gyaradax.io.rundir import save_run_metadata

    out = setup["output_dir"]
    save_run_metadata(out, setup["effective_config"], setup["geometry"])
    params = setup["params"]
    record = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "gyaradax": _git_revision(),
        "config": os.path.abspath(setup["config_path"]),
        "resumed_from": setup["resumed_from"],
        "start_step": setup["start_step"],
        "target_step": None if setup["t_end"] is not None else setup["start_step"] + setup["n_steps"],
        "backend": params.backend,
        "mesh": [params.n_gpus_sp, params.n_gpus_vp, params.n_gpus_mu],
    }
    with open(os.path.join(out, "run_info.jsonl"), "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _bracket_name(params: Any) -> str | None:
    if params.backend == "jax" or not _cuda_available():
        return None
    from gyaradax.backends import _cuda

    _cuda._register_ffi()
    if os.environ.get("GYARADAX_BRACKET", "auto") == "v5" or not _cuda._has_bracket_v6:
        return "v5"
    return "v6, v5 for plane sizes without a v6 kernel"


def _describe(setup: RunSetup, config_path: str) -> None:
    """Print a one-screen summary of what is about to run."""
    import jax

    from gyaradax.io.rundir import print_params

    params = setup["params"]
    mode = "adiabatic" if params.adiabatic_electrons else "kinetic"
    em = [n for n, on in (("A_par", params.nlapar), ("B_par", params.nlbpar)) if on]
    mesh = setup["mesh"]

    print("=" * 88)
    print(f"{mode}{' + EM(' + '+'.join(em) + ')' if em else ''}: {setup['name']} ({config_path})")
    print("=" * 88)
    print_params(params, grid_shape=setup["df"].shape)
    backend = params.backend
    if backend == "auto":
        backend = f"auto ({'cuda' if _cuda_available() else 'jax'})"
    print(f"  backend={backend}  mixed_precision={params.mixed_precision}")
    if params.backend == "jax" and _cuda_available():
        print("  note: the CUDA backend is built; --backend cuda runs ~3x faster per step")
    if mesh is not None:
        shape = setup["df"].shape
        axes: tuple[int, ...] = (params.n_gpus_vp, params.n_gpus_mu)
        if len(shape) == 6:
            axes = (params.n_gpus_sp,) + axes
        local = tuple(n // k for n, k in zip(shape, axes)) + tuple(shape[len(axes) :])
        gib = math.prod(local) * 16 / 2**30
        print(
            f"  sharding: sp={params.n_gpus_sp} vp={params.n_gpus_vp} mu={params.n_gpus_mu} "
            f"over {math.prod(mesh.devices.shape)} device(s); df block {local} "
            f"({gib:.2f} GiB) per device"
        )
    else:
        print(f"  sharding: none (single device {jax.devices()[0]})")
    every = setup["snapshot_every"]
    if setup["dump_dt"] is not None:
        end = (
            f"until t={setup['t_end']:g}"
            if setup["t_end"] is not None
            else f"for {setup['n_steps']} steps"
        )
        print(
            f"  frames every {setup['dump_dt']:g} time units {end}, "
            f"snapshot every {every} frame{'s' if every > 1 else ''}"
        )
    else:
        print(
            f"  n_steps={setup['n_steps']}, checkpoint_interval={setup['block_size']}, "
            f"snapshot every {every} block{'s' if every > 1 else ''}"
        )
    print(f"  output: {setup['output_dir']}")
    debug = [n for n in ("telemetry", "profile") if setup[n]]
    if debug:
        print(f"  debug: {', '.join(debug)} (written to the output directory)")


def _cuda_available() -> bool:
    try:
        from gyaradax.backends import _cuda

        return bool(_cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _resolve_args(args: argparse.Namespace) -> list[str | None]:
    """Replace run directories among ``args.configs`` by their config; return the run dirs.

    A single dataset trajectory stands for the config of its resume dump and is extended from it.
    """
    from gyaradax.io.dataset import RESUME_DIR, is_trajectory, resume_path

    if len(args.configs) == 1 and is_trajectory(args.configs[0]):
        traj = args.configs[0]
        if not os.path.isfile(resume_path(traj)):
            raise SystemExit(f"error: {traj} has no resume dump ({resume_path(traj)})")
        if getattr(args, "resume_from", None) or getattr(args, "from_scratch", False):
            raise SystemExit("error: extending a trajectory takes neither --resume-from nor --from-scratch")
        args.resume_from = traj
        args.configs = [os.path.join(traj, RESUME_DIR, "config.yaml")]
        return [None]
    resolved = [_resolve_config(p) for p in args.configs]
    args.configs = [c for c, _ in resolved]
    return [d for _, d in resolved]


def _cmd_run(args: argparse.Namespace) -> int:
    run_dirs = _resolve_args(args)
    facts = [_peek_config(p) for p in args.configs]
    _configure_env(args, facts)

    if len(args.configs) > 1:
        if args.linear_check:
            raise SystemExit(
                "error: --linear-check takes one config; run the configs one at a time"
            )
        return _run_batched(args, run_dirs)
    return _run_single(args, args.configs[0], run_dirs[0])


def _run_telemetry(setup: RunSetup) -> Any:
    if not setup["telemetry"]:
        return None
    import jax

    from gyaradax.diagnostics.telemetry import RunTelemetry

    params = setup["params"]
    mesh = setup["mesh"]
    devices = list(mesh.devices.flat) if mesh is not None else [jax.devices()[0]]
    return RunTelemetry(
        setup["output_dir"],
        {
            "config": os.path.abspath(setup["config_path"]),
            "resumed_from": setup["resumed_from"],
            "grid": list(setup["df"].shape),
            "backend": params.backend,
            "bracket": _bracket_name(params),
            "mixed_precision": bool(params.mixed_precision),
            "mesh": [params.n_gpus_sp, params.n_gpus_vp, params.n_gpus_mu],
            "start_step": setup["start_step"],
            "n_steps": setup["n_steps"],
            "block_size": setup["block_size"],
            "snapshot_every": setup["snapshot_every"],
        },
        devices,
    )


def _start_run(setup: RunSetup) -> Any:
    for path in setup["stale"]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    _write_run_record(setup)
    marker = os.path.join(setup["output_dir"], "DIVERGED")
    if os.path.exists(marker):
        os.remove(marker)
    return _run_telemetry(setup)


def _finish_run(setup: RunSetup) -> int:
    """Reference report; exit code 2 when the run diverged."""
    print(f"output: {setup['output_dir']}")
    if setup["data_dir"]:
        _report(setup["output_dir"], setup["data_dir"], setup["name"], not setup["params"].adiabatic_electrons)
    if os.path.exists(os.path.join(setup["output_dir"], "DIVERGED")):
        print("exit 2: run diverged (see DIVERGED marker)")
        return 2
    return 0


@contextlib.contextmanager
def _stop_on_sigterm() -> Any:
    """SIGTERM ends the run after the running block and its snapshot; a second one exits at once."""
    from gyaradax.solver.driver import STOP_REQUESTED

    def handler(signum: int, frame: Any) -> None:
        if STOP_REQUESTED.is_set():
            raise SystemExit(128 + signum)
        STOP_REQUESTED.set()
        print("\nSIGTERM: stopping after the running block and its snapshot", flush=True)

    STOP_REQUESTED.clear()
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
        STOP_REQUESTED.clear()


def _interrupted(setups: list[RunSetup], final_state: Any) -> bool:
    import numpy as np

    diverged = any(os.path.exists(os.path.join(s["output_dir"], "DIVERGED")) for s in setups)
    t_end, dump_dt = setups[0]["t_end"], setups[0]["dump_dt"]
    if t_end is not None:
        return not diverged and bool(np.min(np.asarray(final_state.time)) < t_end - 1e-9 * dump_dt)
    target = setups[0]["start_step"] + setups[0]["n_steps"]
    return not diverged and bool(np.min(np.asarray(final_state.step)) < target)


def _reached_end(setup: RunSetup) -> bool:
    t_end, dump_dt = setup["t_end"], setup["dump_dt"]
    return t_end is not None and float(setup["state"].time) >= t_end - 1e-9 * cast(float, dump_dt)


def _nothing_to_run(setup: RunSetup) -> None:
    if setup["t_end"] is not None:
        print(
            f"\nnothing to run ({setup['name']}): t={float(setup['state'].time):g} already reaches "
            f"t_end={setup['t_end']:g}; pass --t-end T or --n-blocks N (frames) to continue."
        )
        return
    print(
        f"\nnothing to run ({setup['name']}): step {setup['start_step']} already reaches "
        "solver.n_steps; pass --n-steps N to continue N more steps."
    )


def _run_single(args: argparse.Namespace, config_path: str, run_dir: str | None) -> int:
    import jax

    from gyaradax.solver.driver import EXIT_INTERRUPTED, gksimulate

    if args.linear_check and not args.dry_run:
        code = _linear_check(args, config_path, run_dir)
        if code:
            return code
    setup = _setup_run(config_path, args, run_dir=run_dir)
    _describe(setup, config_path)
    if args.dry_run:
        print("\ndry run: setup complete, not stepping.")
        return 0
    if setup["n_steps"] <= 0 or _reached_end(setup):
        _nothing_to_run(setup)
        return 0

    telemetry = _start_run(setup)
    t0 = time.time()
    with _stop_on_sigterm():
        _, _, _, final_state = gksimulate(
            setup["df"],
            setup["geometry"],
            setup["params"],
            setup["state"],
            setup["n_steps"],
            pre=setup["pre"],
            output_dir=setup["output_dir"],
            checkpoint_interval=setup["block_size"],
            save_snapshots=setup["save_dumps"],
            stop_on_nan=args.stop_on_nan,
            snapshot_f32=(args.snapshot_dtype == "c64"),
            keep_latest_snapshot=True,
            snapshot_every=setup["snapshot_every"],
            previous_snapshot=setup["previous_snapshot"],
            telemetry=telemetry,
            profile=setup["profile"],
            dataset=_dataset_writer(setup),
            dump_dt=setup["dump_dt"],
            t_end=setup["t_end"],
            dump_origin=setup["dump_origin"],
        )
    jax.effects_barrier()
    runtime = time.time() - t0
    if _interrupted([setup], final_state):
        print(f"exit {EXIT_INTERRUPTED}: interrupted; resuming {setup['output_dir']} continues it")
        return EXIT_INTERRUPTED
    print(f"\ncompleted in {runtime:.1f}s ({setup['n_steps'] / runtime:.1f} steps/s)")
    return _finish_run(setup)


def _batch_mismatch(setups: list[RunSetup]) -> str | None:
    """Why the configs cannot share one vmapped solve, or None."""
    import jax
    import numpy as np

    if any(s["mesh"] is not None for s in setups):
        return "sharded runs are not batched"

    def layout(tree: Any) -> Any:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        return treedef, [np.shape(x) for x in leaves]

    for key, what in (
        ("start_step", "start steps"),
        ("n_steps", "step counts"),
        ("block_size", "block sizes"),
        ("snapshot_every", "snapshot cadences"),
        ("save_dumps", "snapshot archiving"),
        ("dump_dt", "dump spacings"),
        ("t_end", "end times"),
        ("dump_origin", "frame grids"),
    ):
        if len({s[key] for s in setups}) > 1:
            return f"{what} differ"
    if setups[0]["dump_dt"] is not None and len({float(s["state"].time) for s in setups}) > 1:
        return "start times differ"
    for key, what in (
        ("df", "grid shapes"),
        ("params", "static parameters"),
        ("geometry", "geometries"),
        ("pre", "precomputed coefficients"),
    ):
        first = layout(setups[0][key])
        if any(layout(s[key]) != first for s in setups[1:]):
            return f"{what} differ"
    return None


def _run_batched(args: argparse.Namespace, run_dirs: list[str | None]) -> int:
    """vmap several same-grid configs over a batch axis; each writes its own run directory."""
    import jax
    import jax.numpy as jnp

    from gyaradax.solver.driver import EXIT_INTERRUPTED, gksimulate_batched

    def member_args(path: str) -> argparse.Namespace:
        # --output-dir names the parent of the members' run directories
        if not args.output_dir:
            return args
        name = OmegaConf.load(path).run.name
        return argparse.Namespace(**{**vars(args), "output_dir": os.path.join(args.output_dir, name)})

    setups = [_setup_run(p, member_args(p), run_dir=d) for p, d in zip(args.configs, run_dirs)]
    outs = [os.path.abspath(s["output_dir"]) for s in setups]
    if len(set(outs)) < len(outs):
        raise SystemExit("error: batched configs would share an output directory; give them distinct run.name")

    reason = _batch_mismatch(setups)
    if reason is not None:
        print(f"{reason}, running the configs one after another")
        del setups
        return max(_run_single(member_args(p), p, d) for p, d in zip(args.configs, run_dirs))

    names = [s["name"] for s in setups]
    print("=" * 88)
    print(f"batched: {len(setups)} configs ({', '.join(names)})")
    print("=" * 88)
    _describe(setups[0], args.configs[0])
    for s in setups[1:]:
        print(f"  output: {s['output_dir']} ({s['name']})")
    if args.dry_run:
        print("\ndry run: setup complete, not stepping.")
        return 0
    if setups[0]["n_steps"] <= 0 or _reached_end(setups[0]):
        for s in setups:
            _nothing_to_run(s)
        return 0

    def _stack(trees: list[Any]) -> Any:
        leaves = [jax.tree_util.tree_leaves(t) for t in trees]
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(trees[0]), [jnp.stack(g) for g in zip(*leaves)]
        )

    telemetries = [_start_run(s) for s in setups]
    t0 = time.time()
    with _stop_on_sigterm():
        _, _, _, final_state = gksimulate_batched(
            jnp.stack([s["df"] for s in setups]),
            _stack([s["geometry"] for s in setups]),
            _stack([s["params"] for s in setups]),
            _stack([s["state"] for s in setups]),
            setups[0]["n_steps"],
            pre_batch=_stack([s["pre"] for s in setups]),
            output_dirs=[s["output_dir"] for s in setups],
            labels=names,
            previous_snapshots=[s["previous_snapshot"] for s in setups],
            telemetries=telemetries,
            checkpoint_interval=setups[0]["block_size"],
            save_snapshots=setups[0]["save_dumps"],
            stop_on_nan=args.stop_on_nan,
            snapshot_f32=(args.snapshot_dtype == "c64"),
            keep_latest_snapshot=True,
            snapshot_every=setups[0]["snapshot_every"],
            profile=any(s["profile"] for s in setups),
            datasets=[_dataset_writer(s) for s in setups],
            dump_dt=setups[0]["dump_dt"],
            t_end=setups[0]["t_end"],
            dump_origin=setups[0]["dump_origin"],
        )
    jax.effects_barrier()
    runtime = time.time() - t0
    if _interrupted(setups, final_state):
        print(f"exit {EXIT_INTERRUPTED}: interrupted; resuming the run directories continues them")
        return EXIT_INTERRUPTED
    n_steps = setups[0]["n_steps"]
    print(f"\ncompleted {len(setups)} configs in {runtime:.1f}s ({n_steps / runtime:.1f} steps/s)")
    return max(_finish_run(s) for s in setups)


def _cmd_bench(args: argparse.Namespace) -> int:
    """Time the solver loop only: same setup as ``run``, no I/O."""
    _resolve_args(args)
    facts = [_peek_config(args.configs[0])]
    _configure_env(args, facts)

    import jax
    import numpy as np

    from gyaradax.solver.rk4 import gksolve
    from gyaradax.diagnostics.telemetry import device_memory, profile_block

    args.from_scratch = True
    setup = _setup_run(args.configs[0], args)
    _describe(setup, args.configs[0])

    df, state = setup["df"], setup["state"]
    geometry, params, pre = setup["geometry"], setup["params"], setup["pre"]

    print(f"\nwarmup (compilation) — {args.steps} steps x {args.blocks} blocks")
    w0 = time.time()
    df_w, _, state_w = gksolve(df, geometry, params, state, n_steps=args.steps, pre=pre)
    jax.block_until_ready(df_w)
    print(f"compilation: {time.time() - w0:.2f}s")

    mesh = setup["mesh"]
    devices = list(mesh.devices.flat) if mesh is not None else [jax.devices()[0]]
    times = []
    for i in range(args.blocks):
            t0 = time.time()
            df, _, state = gksolve(df, geometry, params, state, n_steps=args.steps, pre=pre)
            jax.block_until_ready(df)
            dt = time.time() - t0
            times.append(dt)
            print(
            f"  block {i + 1}/{args.blocks}: {dt:.3f}s "
            f"({args.steps / dt:.2f} steps/s, {dt * 1000 / args.steps:.2f} ms/step)"
        )

    # first block carries lazy-init overhead; report the steady-state median
    steady = args.steps / float(np.median(times[1:] if len(times) > 1 else times))
    peak = max(device_memory(devices)["process_peak_gib"])
    print("\n" + "=" * 60)
    print(f"  steady-state: {steady:.2f} steps/s ({1000 / steady:.2f} ms/step)")
    print(f"  VRAM/device : {peak:.2f} GiB peak since start (busiest of {len(devices)} device(s))")
    print("=" * 60)
    if args.profile:
        import tempfile

        out = args.output_dir or tempfile.mkdtemp(prefix="gyaradax_profile_")
        with profile_block(out, args.steps, int(state.step)):
            df, _, state = gksolve(df, geometry, params, state, n_steps=args.steps, pre=pre)
            jax.block_until_ready(df)
    return 0


def _cmd_dataset_stats(args: argparse.Namespace) -> int:
    """Recompute the df / phi statistics of preprocessed trajectories from a given time on."""
    from gyaradax.io.dataset import write_statistics

    for traj in args.trajectories:
        meta = write_statistics(traj, args.from_time)
        times = meta["timesteps"]
        print(f"{traj}: statistics over t = {times[times >= args.from_time][0]:.3f}-{times[-1]:.3f}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Write a gyaradax YAML config for a GKW run directory."""
    from gyaradax.io.gkw import gkw_to_yaml

    try:
        gkw_to_yaml(args.gkw_dir, args.output)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    return 0


def _cmd_build_cuda(args: argparse.Namespace) -> int:
    from gyaradax.backends import cuda_lib

    def caps(text: str | None) -> list[int]:
        return [int(c.strip().replace(".", "")) for c in (text or "").split(",") if c.strip()]

    archs = caps(args.archs) or cuda_lib.gpu_compute_caps()
    if not archs:
        raise SystemExit("error: no GPU visible here; pass --archs, e.g. --archs 80,90,103")
    print(f"building {cuda_lib.KERNELS_DIR} for sm {archs} into {cuda_lib.env_build_dir(args.prefix)}")
    lib = cuda_lib.build(archs, caps(args.lto_archs) or None, prefix=args.prefix, jobs=args.jobs)
    print(f"installed {lib}")
    return 0


def _cmd_datagen(args: argparse.Namespace) -> int:
    from gyaradax.datagen.campaign import main as datagen_main

    return datagen_main(args.rest)


def _cmd_info(args: argparse.Namespace) -> int:
    configure_runtime_env(device=-1, device_list=args.device_list)

    import jax
    import jaxlib

    import gyaradax
    from gyaradax.backends import _cuda

    print(f"gyaradax   : {getattr(gyaradax, '__version__', 'dev')} ({os.path.dirname(gyaradax.__file__)})")
    print(f"python     : {sys.version.split()[0]} ({sys.executable})")
    print(f"jax/jaxlib : {jax.__version__} / {jaxlib.__version__}")
    try:
        devices = jax.devices()
    except RuntimeError as exc:
        print(f"devices    : UNAVAILABLE — {exc}")
        return 1
    print(f"devices    : {len(devices)} x {devices[0].device_kind if devices else 'none'}")
    for d in devices:
        print(f"             [{d.id}] {d}")
    try:
        cuda_ok = _cuda.is_available()
    except Exception as exc:  # pragma: no cover - diagnostic path
        cuda_ok = False
        print(f"cuda kernels: error — {exc}")
    print(f"cuda backend: {'available' if cuda_ok else 'not available (JAX backend only)'}")
    from gyaradax.backends import cuda_lib

    record = cuda_lib.build_info(_cuda.LIB_PATH)
    print(f"lib path    : {_cuda.LIB_PATH} ({'present' if _cuda.LIB_PATH.exists() else 'missing'})")
    print(f"lib key     : {cuda_lib.build_key()} (kernel sources + jaxlib)")
    if record:
        print(f"lib build   : sm {record.get('archs')}, {record.get('nvcc')}, {record.get('built')}")
    elif _cuda.LIB_PATH.exists():
        print("lib build   : development build in the checkout (no build record)")
    else:
        print("lib build   : none; `gyaradax build-cuda` builds one for this environment")
    if cuda_ok:
        _cuda._register_ffi()
        forced = os.environ.get("GYARADAX_BRACKET")
        bracket = (
            "v6 (cuFFTDx; plane sizes without a v6 kernel run v5)"
            if _cuda._has_bracket_v6
            else "v5 (library built without cuFFTDx)"
        )
        print(f"bracket     : {bracket}{f'; GYARADAX_BRACKET={forced}' if forced else ''}")
    nccl = importlib.util.find_spec("nvidia.nccl") is not None
    missing = "missing — multi-GPU needs nvidia-nccl-cu13 (cuda13 extra)"
    print(f"nccl        : {'found' if nccl else missing}")
    if len(devices) > 1:
        print(f"multi-GPU   : run with --n-gpus {len(devices)} (layout chosen from the grid)")
    return 0


def _report(output_dir: str, data_dir: str, name: str, kinetic: bool) -> None:
    """Compare time-averaged fluxes against a GKW reference directory."""
    import numpy as np

    flux_path = os.path.join(output_dir, "fluxes.npz")
    growth_path = os.path.join(output_dir, "growth.npz")
    if not os.path.exists(flux_path):
        print("diagnostics not found, skipping comparison")
        return

    fluxes_data = np.load(flux_path)["fluxes"]
    sim_times = np.load(growth_path)["time"]
    try:
        ref_fluxes = np.loadtxt(os.path.join(data_dir, "fluxes.dat"))
        ref_time = np.loadtxt(os.path.join(data_dir, "time.dat"))
    except (FileNotFoundError, OSError):
        print("reference data not found, skipping comparison")
        return

    if kinetic and fluxes_data.ndim == 3:
        avg_start = max(0, len(fluxes_data) - len(fluxes_data) // 4)
        n_avg = len(fluxes_data) - avg_start
        print(f"\n{name} time-averaged eflux (last {n_avg} blocks):")
        for sp_idx, (sp_name, ref_col) in {0: ("ion", 1), 1: ("electron", 4)}.items():
            if sp_idx >= fluxes_data.shape[1]:
                continue
            sim_avg = np.mean(fluxes_data[avg_start:, sp_idx, 1])
            ref_avg = np.mean(
                [ref_fluxes[np.argmin(np.abs(ref_time - t)), ref_col] for t in sim_times[avg_start:]]
            )
            rel_err = abs(sim_avg - ref_avg) / max(abs(ref_avg), 1e-15)
            print(f"  {sp_name:>10s}: sim={sim_avg:.4e}  ref={ref_avg:.4e}  rel_err={rel_err:.2e}")
    else:
        avg_count = 80
        sim_avg = np.mean(fluxes_data[-avg_count:, 1])
        ref_avg = np.mean(
            [ref_fluxes[np.argmin(np.abs(ref_time - t)), 1] for t in sim_times[-avg_count * 3 :]]
        )
        print(f"\n{name} time-averaged eflux (last {avg_count}):")
        print(f"  sim={sim_avg:.4e}  ref={ref_avg:.4e}  abs_err={abs(sim_avg - ref_avg):.2e}")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "configs", nargs="+", metavar="CONFIG",
        help="YAML config path(s), a previous run directory to resume, or one dataset trajectory "
             "to extend from its resume dump",
    )
    p.add_argument(
        "--backend", choices=["auto", "jax", "cuda"], default=None,
        help="nonlinear/linear kernel backend (default: from config, else jax)",
    )
    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--mp", action="store_true", help="mixed precision (FP32 nonlinear FFTs)")
    prec.add_argument("--dp", action="store_true", help="full FP64")
    p.add_argument("--z2z", action="store_true", default=None, help="Z2Z FFT for the nonlinear term")
    p.add_argument("--no-z2z", dest="z2z", action="store_false", help="R2C FFT for the nonlinear term")
    p.add_argument("--device", type=int, default=None, help="pin to a single GPU index")
    p.add_argument("--device-list", type=str, default=None, help="comma-separated GPU indices")
    p.add_argument(
        "--n-gpus", type=int, default=0,
        help="shard over this many GPUs; the (species, vpar, mu) layout is chosen from the grid",
    )
    p.add_argument("--n-gpus-sp", type=int, default=0, help="species-axis mesh size (overrides config)")
    p.add_argument("--n-gpus-vp", type=int, default=0, help="vpar-axis mesh size (overrides config)")
    p.add_argument("--n-gpus-mu", type=int, default=0, help="mu-axis mesh size (overrides config)")
    p.add_argument(
        "--mem-fraction", type=float, default=None,
        help="fraction of GPU memory XLA may use (sets XLA_PYTHON_CLIENT_MEM_FRACTION; "
             "e.g. 0.95 for grids near the memory limit)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gyaradax",
        description="Gyrokinetic Vlasov-Poisson solver in JAX.",
        epilog=(
            "Adiabatic vs kinetic electrons, electromagnetic terms and velocity-space "
            "sharding are all taken from the YAML config; flags only override it."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a simulation")
    _add_common(run_p)
    run_p.add_argument("--block-size", type=int, default=0, help="steps between checkpoints")
    run_p.add_argument("--n-blocks", type=int, default=0, help="run this many checkpoint blocks")
    run_p.add_argument(
        "--snapshot-every", type=int, default=0,
        help="write the restart snapshot every K blocks and after the last one (default: 1)",
    )
    run_p.add_argument(
        "--n-steps", type=int, default=0,
        help="run N more steps (default: up to the config's solver.n_steps); with --dump-dt the "
             "run ends on the first frame at or after them",
    )
    run_p.add_argument(
        "--dump-dt", type=float, default=None, metavar="DT",
        help="time mode: a frame (diagnostics, dataset frame) every DT time units, the last step "
             "before each shortened to land on it (overrides solver.dump_dt); --n-blocks counts frames",
    )
    run_p.add_argument(
        "--t-end", type=float, default=None, metavar="T",
        help="time mode: run to the first frame at or after time T (overrides solver.t_end)",
    )
    run_p.add_argument(
        "--from-scratch", action="store_true",
        help="cold start: ignore K-files and the run directory's snapshots",
    )
    run_p.add_argument(
        "--resume-from", type=str, default=None,
        help="resume from a GKW K-file in data_dir (e.g. K03), a step_*.npz snapshot, or the "
             "resume dump of a dataset trajectory directory (frames go on into that trajectory)",
    )
    run_p.add_argument("--output-dir", type=str, default=None, help="override output directory")
    run_p.add_argument(
        "--save-dumps", action=argparse.BooleanOptionalAction, default=None,
        help="keep every restart snapshot instead of only the latest; stored in the run's "
             "config.yaml (--no-save-dumps switches it off)",
    )
    run_p.add_argument(
        "--overwrite", action="store_true",
        help="replace what the output directory holds of another run (or snapshots past "
             "the resumed step) instead of refusing to start",
    )
    run_p.add_argument(
        "--dataset", type=str, default=None, metavar="DIR",
        help="also write the run as a preprocessed dataset trajectory under "
             "DIR/<name>_ifft_realpotens/ (quantized real-space frames every block + metadata)",
    )
    run_p.add_argument(
        "--dataset-bits", choices=["zstd16", "bf16"], default=None,
        help="dataset frame format (default zstd16: lossless zstd over bf16)",
    )
    run_p.add_argument("--dataset-name", type=str, default=None, help="trajectory name (default run.name)")
    run_p.add_argument(
        "--dataset-stats-from", type=float, default=None, metavar="T",
        help="compute the dataset statistics over the frames from time T on (default 0)",
    )
    run_p.add_argument(
        "--resume-dtype", choices=["f32", "f64"], default=None,
        help="df precision of the trajectory's resume dump, replaced at the end of every leg "
             "(default f32; f64 extends bitwise)",
    )
    run_p.add_argument(
        "--snapshot-dtype", choices=["c128", "c64"], default="c128",
        help="archive precision for df/phi in snapshots (solver stays FP64); "
             "c64 halves them (maps to gksimulate's snapshot_f32)",
    )
    run_p.add_argument(
        "--no-stop-on-nan", dest="stop_on_nan", action="store_false", default=True,
        help="keep integrating after df goes non-finite (default: stop and exit 2)",
    )
    run_p.add_argument(
        "--linear-check", type=str, default=None, metavar="NS[,NS...]",
        help="before the run, compare linear growth rates at the run's ns with these reference "
             "resolutions (written to RUN_DIR/linear_check.json, reused on resume)",
    )
    run_p.add_argument(
        "--linear-check-t-end", type=float, default=40.0, metavar="T",
        help="with --linear-check: stop each linear run here if its growth rates have not settled "
             "(default 40)",
    )
    run_p.add_argument(
        "--linear-check-tol", type=float, default=None, metavar="X",
        help="with --linear-check: exit 3 instead of running when γ at the reference's dominant "
             "ky deviates by more than X (relative, e.g. 0.1)",
    )
    run_p.add_argument("--dry-run", action="store_true", help="set up and report, do not step")
    run_p.add_argument(
        "--telemetry", action="store_true",
        help="write telemetry.jsonl (environment, per-block timings, dt, device memory)",
    )
    run_p.add_argument(
        "--profile", action="store_true",
        help="trace one block with the JAX profiler into profile/ and summarise its GPU kernels",
    )
    run_p.add_argument("--debug", action="store_true", help="--telemetry and --profile")
    run_p.set_defaults(func=_cmd_run)

    bench_p = sub.add_parser("bench", help="measure solver throughput")
    _add_common(bench_p)
    bench_p.add_argument("--steps", type=int, default=100, help="steps per timed block")
    bench_p.add_argument("--blocks", type=int, default=5, help="number of timed blocks")
    bench_p.add_argument(
        "--profile", action="store_true",
        help="trace one more block and print its GPU kernel breakdown (trace in --output-dir)",
    )
    bench_p.add_argument("--output-dir", type=str, default=None, help="where --profile writes the trace")
    bench_p.set_defaults(func=_cmd_bench, block_size=0, n_blocks=0, n_steps=0,
                         resume_from=None, save_dumps=False, dry_run=False)

    growth_p = sub.add_parser(
        "growth", help="linear growth rates of a config, at its own or other parallel resolutions"
    )
    growth_p.add_argument("config", metavar="CONFIG", help="YAML config (run linearly)")
    growth_p.add_argument(
        "--ns", type=int, nargs="+", default=None,
        help="parallel resolutions to run (default: the config's); the finest is the reference",
    )
    growth_p.add_argument(
        "--set", action="append", default=None, metavar="KEY=VALUE",
        help="override a config entry, e.g. solver.disp_par=4.0 (repeatable)",
    )
    growth_p.add_argument(
        "--t-end", type=float, default=40.0,
        help="stop here if the growth rates have not settled (default 40)",
    )
    growth_p.add_argument(
        "--output", type=str, default=None, metavar="FILE", help="write the report as JSON"
    )
    growth_p.add_argument("--backend", choices=["auto", "jax", "cuda"], default=None)
    growth_p.add_argument("--device", type=int, default=None, help="pin to a single GPU index")
    growth_p.add_argument(
        "--device-list", type=str, default=None, help="comma-separated GPU indices"
    )
    growth_p.add_argument("--mem-fraction", type=float, default=None)
    growth_p.set_defaults(func=_cmd_growth)

    stats_p = sub.add_parser("dataset-stats", help="recompute the statistics of preprocessed trajectories")
    stats_p.add_argument("trajectories", nargs="+", metavar="TRAJECTORY", help="trajectory directories")
    stats_p.add_argument("--from-time", type=float, default=0.0, help="first time included (default 0)")
    stats_p.set_defaults(func=_cmd_dataset_stats)

    convert_p = sub.add_parser("convert", help="write a YAML config for a GKW run directory")
    convert_p.add_argument("gkw_dir", help="GKW run directory (input.dat, geom.dat, ...)")
    convert_p.add_argument("output", help="YAML config to write")
    convert_p.set_defaults(func=_cmd_convert)

    info_p = sub.add_parser("info", help="show devices, backends and versions")
    info_p.add_argument("--device-list", type=str, default=None)
    info_p.set_defaults(func=_cmd_info)

    build_p = sub.add_parser("build-cuda", help="build the CUDA kernels for this Python environment")
    build_p.add_argument(
        "--archs", type=str, default=None,
        help="compute capabilities, e.g. 80,90,103 (default: the GPUs of this machine)",
    )
    build_p.add_argument(
        "--lto-archs", type=str, default=None,
        help="cuFFT LTO callback compute capabilities (default: the CMake default, 80,90,100)",
    )
    build_p.add_argument("--prefix", type=str, default=None, help="environment prefix (default: this one)")
    build_p.add_argument("-j", "--jobs", type=int, default=None, help="parallel build jobs")
    build_p.set_defaults(func=_cmd_build_cuda)

    datagen_p = sub.add_parser(
        "datagen", add_help=False, help="generate a dataset: run configs over an ssh GPU mesh or Slurm"
    )
    datagen_p.add_argument("rest", nargs=argparse.REMAINDER)
    datagen_p.set_defaults(func=_cmd_datagen)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
