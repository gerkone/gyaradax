"""gyaradax command-line interface: ``run``, ``bench``, ``info``.

Run mode (adiabatic/kinetic), electromagnetic terms and velocity-space
sharding all come from the YAML config; flags only override it. See docs/CLI.md.

JAX and gyaradax are imported inside the command handlers, not at module scope,
because ``CUDA_VISIBLE_DEVICES``/``XLA_FLAGS`` must be set before the backend
initialises.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import replace
from typing import Any, Sequence, TypedDict, cast

from omegaconf import OmegaConf

from gyaradax.runtime_config import append_xla_flags, configure_runtime_env

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
    nlapar: bool
    nlbpar: bool
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
    solver = cfg.get("solver") or {}
    physics = cfg.get("physics") or {}
    shard = cfg.get("sharding") or {}
    adiabatic = bool(grid.get("adiabatic_electrons", True))
    mas = physics.get("mas", 1.0)
    nsp = 1 if adiabatic or not OmegaConf.is_list(mas) else len(mas)
    return {
        "adiabatic": adiabatic,
        "nlapar": bool(solver.get("nlapar", False)),
        "nlbpar": bool(solver.get("nlbpar", False)),
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

    Species and mu shards need no halo; vpar takes what is left and keeps at
    least two vpar points per shard.
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
    name: str
    data_dir: str | None
    mesh: Any
    config_path: str
    effective_config: Any
    resumed_from: str | None
    start_step: int
    telemetry: bool
    profile: bool


def _resolve_config(path: str) -> tuple[str, str | None]:
    """A run directory stands for its dumped ``config.yaml`` and is resumed in place."""
    if os.path.isdir(path):
        config = os.path.join(path, "config.yaml")
        if not os.path.isfile(config):
            raise SystemExit(f"error: {path} is a directory without a config.yaml to resume from")
        return config, path
    return path, None


def _latest_snapshot(run_dir: str) -> str | None:
    steps = {}
    for path in glob.glob(os.path.join(run_dir, "step_*.npz")):
        m = re.search(r"step_(\d+)\.npz$", path)
        if m:
            steps[int(m.group(1))] = path
    return steps[max(steps)] if steps else None


def _has_geom_dat(data_dir: str | None) -> bool:
    return bool(data_dir) and os.path.exists(os.path.join(str(data_dir), "geom.dat"))


def _find_k_file(data_dir: str | None, resume_from: str | None = None) -> str | None:
    """Return a K-dump to resume from, or None to cold-start."""
    from gyaradax.utils import K_files

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
    from gyaradax.params import gkparams_from_config
    from gyaradax.precompute import linear_precompute
    from gyaradax.simulate import (
        _compute_phi_for_init,
        _ensure_species_arrays,
        gk_init,
    )
    from gyaradax.solver import mode_amplitude
    from gyaradax.state import GKState
    from gyaradax.utils import (
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

    output_dir: str = (
        args.output_dir or run_dir or f"outputs_{'kinetic' if kinetic else 'adiabatic'}_{name}"
    )

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

    snapshot = None
    if not args.from_scratch:
        if args.resume_from and args.resume_from.endswith(".npz"):
            snapshot = args.resume_from
        elif run_dir is not None and not args.resume_from:
            snapshot = _latest_snapshot(run_dir)
    k_path = None
    if not args.from_scratch and snapshot is None:
        k_path = _find_k_file(data_dir, args.resume_from)

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

    mesh = _sharding.build_mesh(params)
    if mesh is not None:
        grid = _sharding.grid_shape_from(params, geometry)
        # sharded precompute: full-size arrays never land on one device
        pre = _sharding.precompute_sharded(geometry, params, mesh, grid)
        df = _sharding.shard_df(df, mesh, grid)
    else:
        pre = linear_precompute(geometry, params)

    # checkpoint cadence: --block-size > run.block_size > config's dump_interval x naverage
    run_block = int(getattr(cfg.run, "block_size", 0) or 0)
    if args.block_size:
        block_size = args.block_size
    elif run_block:
        block_size = run_block
    else:
        dump_interval = int(getattr(cfg.solver, "dump_interval", 0) or 0)
        block_size = dump_interval * params.naverage if dump_interval else 120

    # step count: --n-steps > --n-blocks > config solver.n_steps > heuristic; a resumed
    # snapshot runs up to solver.n_steps in total
    start_step = int(state.step)
    cfg_steps = int(getattr(cfg.solver, "n_steps", 0) or 0)
    if args.n_steps:
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

    # clamp so at least one checkpoint is written
    if 0 < n_steps < block_size:
        print(f"  note: checkpoint interval {block_size} > n_steps {n_steps}; using {n_steps}")
        block_size = n_steps

    return {
        "df": df,
        "geometry": geometry,
        "params": params,
        "state": state,
        "pre": pre,
        "output_dir": output_dir,
        "n_steps": n_steps,
        "block_size": block_size,
        "name": name,
        "data_dir": data_dir,
        "mesh": mesh,
        "config_path": config_path,
        "effective_config": _effective_config(cfg, params, start_step + n_steps, block_size, args),
        "resumed_from": snapshot or k_path,
        "start_step": start_step,
        "telemetry": _debug_flag(cfg, args, "telemetry"),
        "profile": _debug_flag(cfg, args, "profile"),
    }


def _debug_flag(cfg: Any, args: argparse.Namespace, name: str) -> bool:
    debug_cfg = cfg.get("debug") or {}
    return bool(getattr(args, name, False) or getattr(args, "debug", False) or debug_cfg.get(name, False))


def _effective_config(
    cfg: Any, params: Any, target_step: int, block_size: int, args: argparse.Namespace
) -> Any:
    """The config with the command-line choices folded in, so the run directory reproduces it."""
    eff = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key, value in (
        ("solver.backend", params.backend),
        ("solver.mixed_precision", bool(params.mixed_precision)),
        ("solver.use_z2z", bool(params.use_z2z)),
        ("solver.dt", float(params.dt)),
        ("solver.n_steps", int(target_step)),
        ("run.block_size", int(block_size)),
        ("debug.telemetry", _debug_flag(cfg, args, "telemetry")),
        ("debug.profile", _debug_flag(cfg, args, "profile")),
    ):
        OmegaConf.update(eff, key, value, force_add=True)
    mesh = (params.n_gpus_sp, params.n_gpus_vp, params.n_gpus_mu)
    if "sharding" in eff or math.prod(mesh) > 1:
        eff.sharding = {"n_gpus_sp": mesh[0], "n_gpus_vp": mesh[1], "n_gpus_mu": mesh[2]}
    return eff


def _write_run_record(setup: RunSetup) -> None:
    """config.yaml and geometry.pkl for the run, and one line per invocation in run_info.jsonl."""
    import socket

    from gyaradax.telemetry import _git_revision
    from gyaradax.utils import save_run_metadata

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
        "target_step": setup["start_step"] + setup["n_steps"],
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
    return "v6"


def _describe(setup: RunSetup, config_path: str) -> None:
    """Print a one-screen summary of what is about to run."""
    import jax

    from gyaradax.utils import print_params

    params = setup["params"]
    mode = "adiabatic" if params.adiabatic_electrons else "kinetic"
    em = [n for n, on in (("A_par", params.nlapar), ("B_par", params.nlbpar)) if on]
    mesh = setup["mesh"]

    print("=" * 88)
    print(f"{mode}{' + EM(' + '+'.join(em) + ')' if em else ''}: {setup['name']} ({config_path})")
    print("=" * 88)
    print_params(params, grid_shape=setup["df"].shape)
    print(f"  backend={params.backend}  mixed_precision={params.mixed_precision}")
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
    print(f"  n_steps={setup['n_steps']}, checkpoint_interval={setup['block_size']}")
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
    """Replace run directories among ``args.configs`` by their config; return the run dirs."""
    resolved = [_resolve_config(p) for p in args.configs]
    args.configs = [c for c, _ in resolved]
    return [d for _, d in resolved]


def _cmd_run(args: argparse.Namespace) -> int:
    run_dirs = _resolve_args(args)
    facts = [_peek_config(p) for p in args.configs]
    _configure_env(args, facts)

    if len(args.configs) > 1:
        return _run_batched(args)

    import jax

    from gyaradax.simulate import gksimulate

    setup = _setup_run(args.configs[0], args, run_dir=run_dirs[0])
    _describe(setup, args.configs[0])
    if args.dry_run:
        print("\ndry run: setup complete, not stepping.")
        return 0
    if setup["n_steps"] <= 0:
        print(
            f"\nnothing to run: step {setup['start_step']} already reaches solver.n_steps; "
            "pass --n-steps N to continue N more steps."
        )
        return 0

    _write_run_record(setup)
    telemetry = None
    if setup["telemetry"]:
        from gyaradax.telemetry import RunTelemetry

        params = setup["params"]
        mesh = setup["mesh"]
        devices = list(mesh.devices.flat) if mesh is not None else [jax.devices()[0]]
        telemetry = RunTelemetry(
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
            },
            devices,
        )

    # drop a marker from an earlier attempt, else a good re-run reports failure
    stale = os.path.join(setup["output_dir"], "DIVERGED")
    if os.path.exists(stale):
        os.remove(stale)

    t0 = time.time()
    gksimulate(
        setup["df"],
        setup["geometry"],
        setup["params"],
        setup["state"],
        setup["n_steps"],
        pre=setup["pre"],
        output_dir=setup["output_dir"],
        checkpoint_interval=setup["block_size"],
        save_snapshots=args.save_dumps,
        stop_on_nan=args.stop_on_nan,
        snapshot_f32=(args.snapshot_dtype == "c64"),
        keep_latest_snapshot=True,
        telemetry=telemetry,
        profile=setup["profile"],
    )
    jax.effects_barrier()
    runtime = time.time() - t0
    print(f"\ncompleted in {runtime:.1f}s ({setup['n_steps'] / runtime:.1f} steps/s)")
    print(f"output: {setup['output_dir']}")

    if setup["data_dir"]:
        _report(setup["output_dir"], setup["data_dir"], setup["name"], not setup["params"].adiabatic_electrons)

    # non-zero exit so a SLURM array task registers the failure
    if os.path.exists(os.path.join(setup["output_dir"], "DIVERGED")):
        print("exit 2: run diverged (see DIVERGED marker)")
        return 2
    return 0


def _run_batched(args: argparse.Namespace) -> int:
    """vmap several same-grid configs over a batch axis."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from gyaradax.simulate import gk_run_batched

    setups = [_setup_run(p, args) for p in args.configs]
    if len({s["df"].shape for s in setups}) > 1:
        print("grid shapes differ, falling back to sequential execution")
        for path in args.configs:
            single = argparse.Namespace(**{**vars(args), "configs": [path]})
            _cmd_run(single)
        return 0

    names = [s["name"] for s in setups]
    n_steps = max(s["n_steps"] for s in setups)
    block_size = setups[0]["block_size"]

    print("=" * 88)
    print(f"batched: {len(setups)} configs ({', '.join(names)})")
    print("=" * 88)
    _describe(setups[0], args.configs[0])
    if args.dry_run:
        print("\ndry run: setup complete, not stepping.")
        return 0

    def _stack(trees: list[Any]) -> Any:
        leaves = [jax.tree_util.tree_leaves(t) for t in trees]
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(trees[0]), [jnp.stack(g) for g in zip(*leaves)]
        )

    df_b = jnp.stack([s["df"] for s in setups])
    geom_b = _stack([s["geometry"] for s in setups])
    params_b = _stack([s["params"] for s in setups])
    state_b = _stack([s["state"] for s in setups])
    pre_b = _stack([s["pre"] for s in setups])

    accum: dict[str, dict[str, list[Any]]] = {
        s["name"]: {"fluxes": [], "growth": [], "times": []} for s in setups
    }
    for out_dir in {s["output_dir"] for s in setups}:
        os.makedirs(out_dir, exist_ok=True)

    print("warmup (compilation)...")
    w0 = time.time()
    warm = gk_run_batched(df_b, geom_b, params_b, state_b, min(block_size, n_steps), pre_b)
    jax.block_until_ready(warm[0])
    print(f"compilation: {time.time() - w0:.2f}s")

    t0 = time.time()
    while int(state_b.step[0]) < n_steps:
        block = min(block_size, n_steps - int(state_b.step[0]))
        if block <= 0:
            break
        bt = time.time()
        df_b, _, fluxes_b, state_b = gk_run_batched(
            df_b, geom_b, params_b, state_b, block, pre_b
        )
        jax.block_until_ready(df_b)
        wall = time.time() - bt

        t_sim = float(state_b.time[0])
        heat = np.asarray(fluxes_b[1])
        growth = np.asarray(state_b.last_growth_rate)
        logs = [
            f"{n} [flx {float(np.mean(heat[i])):.4f}, gr {float(np.mean(growth[i])):.4f}]"
            for i, n in enumerate(names)
        ]
        print(
            f"[{int(state_b.step[0]):>8d}] t {t_sim:>8.2f} | {' | '.join(logs)} | "
            f"{block / wall:.2f} steps/s  x{len(setups)}"
        )

        for i, s in enumerate(setups):
            flx = np.asarray(jax.tree.map(lambda x: x[i], fluxes_b))
            if flx.ndim == 0 or (flx.ndim == 1 and flx.shape[0] != 3):
                flx = np.array([float(fluxes_b[j][i]) for j in range(3)])
            accum[s["name"]]["fluxes"].append(flx)
            accum[s["name"]]["growth"].append(np.asarray(state_b.last_growth_rate[i]))
            accum[s["name"]]["times"].append(t_sim)

    for s in setups:
        a = accum[s["name"]]
        times = np.array(a["times"])
        steps = np.arange(len(a["times"])) * block_size
        np.savez(
            os.path.join(s["output_dir"], "fluxes.npz"),
            fluxes=np.stack(a["fluxes"]), time=times, step=steps,
        )
        np.savez(
            os.path.join(s["output_dir"], "growth.npz"),
            growth=np.stack(a["growth"]), time=times, step=steps,
        )

    print(f"\ncompleted {len(setups)} configs in {time.time() - t0:.1f}s")
    for s in setups:
        if s["data_dir"]:
            _report(
                s["output_dir"], s["data_dir"], s["name"],
                not s["params"].adiabatic_electrons,
            )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    """Time the solver loop only: same setup as ``run``, no I/O."""
    _resolve_args(args)
    facts = [_peek_config(args.configs[0])]
    _configure_env(args, facts)

    import jax
    import numpy as np

    from gyaradax.solver import gksolve

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
    times, peaks = [], []
    for i in range(args.blocks):
        for dev in devices:
            try:
                dev.reset_memory_stats()
            except Exception:
                pass
        t0 = time.time()
        df, _, state = gksolve(df, geometry, params, state, n_steps=args.steps, pre=pre)
        jax.block_until_ready(df)
        dt = time.time() - t0
        times.append(dt)
        peak = 0.0
        for dev in devices:
            try:
                peak = max(peak, (dev.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6)
            except Exception:
                pass
        peaks.append(peak)
        print(
            f"  block {i + 1}/{args.blocks}: {dt:.3f}s "
            f"({args.steps / dt:.2f} steps/s, {dt * 1000 / args.steps:.2f} ms/step, {peak:.0f} MB)"
        )

    # first block carries lazy-init overhead; report the steady-state median
    steady = args.steps / float(np.median(times[1:] if len(times) > 1 else times))
    print("\n" + "=" * 60)
    print(f"  steady-state: {steady:.2f} steps/s ({1000 / steady:.2f} ms/step)")
    print(f"  VRAM/device : {max(peaks):.0f} MB peak (max over {len(devices)} device(s))")
    print("=" * 60)
    if args.profile:
        import tempfile

        from gyaradax.telemetry import profile_block

        out = args.output_dir or tempfile.mkdtemp(prefix="gyaradax_profile_")
        with profile_block(out, args.steps):
            df, _, state = gksolve(df, geometry, params, state, n_steps=args.steps, pre=pre)
            jax.block_until_ready(df)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Write a gyaradax YAML config for a GKW run directory."""
    from gyaradax.utils import gkw_to_yaml

    try:
        gkw_to_yaml(args.gkw_dir, args.output)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    return 0


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
    print(f"lib path    : {_cuda.LIB_PATH} ({'present' if _cuda.LIB_PATH.exists() else 'missing'})")
    if cuda_ok:
        _cuda._register_ffi()
        forced = os.environ.get("GYARADAX_BRACKET")
        bracket = "v6 (cuFFTDx)" if _cuda._has_bracket_v6 else "v5 (library built without cuFFTDx)"
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
        help="YAML config path(s), or a previous run directory to resume",
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
    run_p.add_argument("--n-steps", type=int, default=0, help="total steps (overrides config)")
    run_p.add_argument("--from-scratch", action="store_true", help="ignore K-files, cold start")
    run_p.add_argument(
        "--resume-from", type=str, default=None,
        help="resume from a GKW K-file in data_dir (e.g. K03) or a step_*.npz snapshot",
    )
    run_p.add_argument("--output-dir", type=str, default=None, help="override output directory")
    run_p.add_argument(
        "--save-dumps", action="store_true",
        help="keep a df snapshot every block (default: only the latest, for restarts)",
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

    convert_p = sub.add_parser("convert", help="write a YAML config for a GKW run directory")
    convert_p.add_argument("gkw_dir", help="GKW run directory (input.dat, geom.dat, ...)")
    convert_p.add_argument("output", help="YAML config to write")
    convert_p.set_defaults(func=_cmd_convert)

    info_p = sub.add_parser("info", help="show devices, backends and versions")
    info_p.add_argument("--device-list", type=str, default=None)
    info_p.set_defaults(func=_cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
