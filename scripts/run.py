"""run gyaradax simulations.

automatically detects geometry source and initial condition:
  - if data_dir has geom.dat -> load_geometry; otherwise -> compute_geometry
  - if data_dir has K-files  -> resume from K01; otherwise -> init_f

when multiple configs are passed and share the same grid, they are
batched automatically via jax.vmap for parallel execution on one device.

Usage:
    # single config
    python scripts/run.py configs/iteration_13.yaml --device 5

    # batched (two equilibria, same grid, run in parallel)
    python scripts/run.py configs/iteration_13.yaml configs/iteration_200.yaml --device 5 --n-blocks 10
"""

import argparse
import os
import time
from dataclasses import replace

import numpy as np

# parse --device / --device-list before any jax import so cuda sees the right gpus
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--device", type=int, default=-1)
_parser.add_argument("--device-list", type=str, default=None)
_parser.add_argument("--n-gpus-sp", type=int, default=1)
_parser.add_argument("--n-gpus-vp", type=int, default=1)
_parser.add_argument("--n-gpus-mu", type=int, default=1)
_early_args, _ = _parser.parse_known_args()
if _early_args.device_list:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.device_list
elif _early_args.device != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_early_args.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# latency-hiding + pipelined collective flags: overlap NCCL ops with compute.
# Note: in JAX ≥ 0.9 the old `enable_async_*` flags are removed (async is the
# default). Keep only the pipelining / scheduler hints that still parse.
if _early_args.n_gpus_sp * _early_args.n_gpus_vp * _early_args.n_gpus_mu > 1:
    _async_flags = " ".join([
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_pipelined_all_reduce=true",
        "--xla_gpu_enable_pipelined_all_gather=true",
        "--xla_gpu_enable_while_loop_double_buffering=true",
    ])
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _async_flags).strip()

import jax
import jax.numpy as jnp

from gyaradax import load_config
from gyaradax.utils import load_geometry
from gyaradax.params import gkparams_from_config
from gyaradax.simulate import (
    gk_init,
    gksimulate,
    gk_run_batched,
    _geometry_from_config,
    _compute_phi_for_init,
    _ensure_species_arrays,
)
from gyaradax.solver import GKState, linear_precompute, mode_amplitude
from gyaradax.utils import (
    load_gkw_k_dump,
    read_gkw_dump_time,
    read_gkw_dump_dtim,
    K_files,
    print_params,
)


def _has_geom_dat(data_dir):
    return data_dir and os.path.exists(os.path.join(data_dir, "geom.dat"))


def _find_k_file(data_dir, resume_from=None):
    """Return path to a K-file in data_dir, or None.

    If resume_from is given (e.g. 'K01', '100', 'K03'), use that specific file.
    Otherwise use the first available K-file.
    """
    if not data_dir:
        return None
    if resume_from:
        path = os.path.join(data_dir, resume_from)
        if os.path.exists(path):
            return path
        # try with K prefix
        path = os.path.join(data_dir, f"K{resume_from}")
        if os.path.exists(path):
            return path
        print(f"  warning: resume file '{resume_from}' not found in {data_dir}")
        return None
    ks = K_files(data_dir)
    if ks:
        return os.path.join(data_dir, ks[0])
    k01 = os.path.join(data_dir, "K01")
    return k01 if os.path.exists(k01) else None


def _setup_run(config_path, args):
    """Build (df, geometry, params, state, pre) and metadata for one config."""
    cfg = load_config(config_path)
    data_dir = getattr(cfg.run, "data_dir", None)
    name = cfg.run.name
    kinetic = args.kinetic

    output_dir = args.output_dir or f"validation_{'kinetic' if kinetic else 'outputs'}_{name}"

    overrides = {}
    if args.mp:
        overrides["mixed_precision"] = True
    if args.z2z is not None:
        overrides["use_z2z"] = args.z2z
    if args.backend:
        overrides["backend"] = args.backend
    if kinetic:
        overrides["adaptive_dt"] = True
    if args.n_gpus_sp > 1:
        overrides["n_gpus_sp"] = args.n_gpus_sp
    if args.n_gpus_vp > 1:
        overrides["n_gpus_vp"] = args.n_gpus_vp
    if args.n_gpus_mu > 1:
        overrides["n_gpus_mu"] = args.n_gpus_mu
    params = gkparams_from_config(cfg, **overrides)

    if _has_geom_dat(data_dir):
        geometry = load_geometry(data_dir)
    else:
        geometry = _geometry_from_config(cfg)

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    resume_from = getattr(args, "resume_from", None)
    k_path = None if args.from_scratch else _find_k_file(data_dir, resume_from)

    if k_path is not None:
        res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = load_gkw_k_dump(k_path, res, n_species=n_species)

        dat_path = k_path + ".dat"
        t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0

        actual_dt = read_gkw_dump_dtim(dat_path) if os.path.exists(dat_path) else 0.0
        if actual_dt > 0 and actual_dt < params.dt:
            params = replace(params, dt=actual_dt)

        # ensure geometry has per-species arrays for kinetic runs
        geometry = _ensure_species_arrays(geometry, params)

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
        print(
            f"  resumed from {os.path.basename(k_path)} "
            f"(t={t_start:.4f}, dt={float(params.dt):.4e})"
        )
    else:
        df, geometry, state = gk_init(geometry, params, n_species=n_species)

    from gyaradax import sharding as _sharding
    mesh = _sharding.build_mesh(params)
    if mesh is not None:
        grid = _sharding.grid_shape_from(params, geometry)
        # precompute_sharded compiles linear_precompute with out_shardings so
        # full-size arrays never materialise on a single device — the only safe
        # path when the grid doesn't fit on one GPU.
        pre = _sharding.precompute_sharded(geometry, params, mesh, grid)
        df = _sharding.shard_df(df, mesh, grid)
    else:
        pre = linear_precompute(geometry, params)

    block_size = args.block_size
    if not kinetic:
        block_size = int(getattr(cfg.solver, "dump_interval", 3)) * params.naverage

    if args.n_blocks:
        n_steps = args.n_blocks * block_size
    elif kinetic and data_dir:
        try:
            ref_times = np.loadtxt(os.path.join(data_dir, "time.dat"))
            n_steps = max(block_size, int((ref_times[-1] - float(state.time)) * 1.0 / params.dt))
        except FileNotFoundError:
            n_steps = 100 * block_size
    else:
        n_steps = 265 * block_size

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
    }


def run(config_path, args):
    """Single-config execution path."""
    s = _setup_run(config_path, args)
    kinetic = args.kinetic

    print("=" * 88)
    print(f"{'kinetic' if kinetic else 'adiabatic'}: {s['name']} ({config_path})")
    print("=" * 88)
    print_params(s["params"], grid_shape=s["df"].shape)
    print(f"  n_steps={s['n_steps']}, checkpoint_interval={s['block_size']}")

    t0 = time.time()
    gksimulate(
        s["df"],
        s["geometry"],
        s["params"],
        s["state"],
        s["n_steps"],
        pre=s["pre"],
        output_dir=s["output_dir"],
        checkpoint_interval=s["block_size"],
        save_snapshots=args.save_dumps,
    )
    runtime = time.time() - t0
    print(f"\ncompleted in {runtime:.1f}s ({s['n_steps'] / runtime:.1f} steps/s)")

    if s["data_dir"]:
        report(s["output_dir"], s["data_dir"], s["name"], kinetic)


def run_multi(args):
    """Batched multi-config execution: vmap over configs with the same grid."""
    setups = [_setup_run(inp, args) for inp in args.inputs]
    names = [s["name"] for s in setups]
    n_batch = len(setups)

    # compatibility: same df shape required
    shapes = set(s["df"].shape for s in setups)
    if len(shapes) > 1:
        print(f"grid shapes differ {shapes}, falling back to sequential")
        for inp in args.inputs:
            run(inp, args)
        return

    # use max n_steps so all configs run long enough
    n_steps = max(s["n_steps"] for s in setups)
    block_size = setups[0]["block_size"]

    print("=" * 88)
    print(f"batched: {n_batch} configs ({', '.join(names)})")
    print("=" * 88)
    print_params(setups[0]["params"], grid_shape=setups[0]["df"].shape)
    print(f"  n_steps={n_steps}, checkpoint_interval={block_size}")

    # stack into batched pytrees
    def _stack_pytrees(trees):
        leaves_list = [jax.tree_util.tree_leaves(t) for t in trees]
        stacked_leaves = [jnp.stack(leaf_group) for leaf_group in zip(*leaves_list)]
        treedef = jax.tree_util.tree_structure(trees[0])
        return jax.tree_util.tree_unflatten(treedef, stacked_leaves)

    df_batch = jnp.stack([s["df"] for s in setups])
    geometry_batch = _stack_pytrees([s["geometry"] for s in setups])
    params_batch = _stack_pytrees([s["params"] for s in setups])
    state_batch = _stack_pytrees([s["state"] for s in setups])
    pre_batch = _stack_pytrees([s["pre"] for s in setups])

    # per-config accumulators
    accum = {s["name"]: {"fluxes": [], "growth": [], "times": []} for s in setups}

    for out_dir in set(s["output_dir"] for s in setups):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    target_step = n_steps

    # warmup (compilation)
    if n_steps > 0:
        print("warmup (compilation)...")
        w_t0 = time.time()
        _ = gk_run_batched(
            df_batch,
            geometry_batch,
            params_batch,
            state_batch,
            min(block_size, n_steps),
            pre_batch,
        )
        jax.block_until_ready(_[0])
        print(f"compilation: {time.time() - w_t0:.2f}s")

    while int(state_batch.step[0]) < target_step:
        remaining = target_step - int(state_batch.step[0])
        block = min(block_size, remaining)
        if block <= 0:
            break

        t0 = time.time()
        df_batch, _, fluxes_batch, state_batch = gk_run_batched(
            df_batch, geometry_batch, params_batch, state_batch, block, pre_batch
        )
        jax.block_until_ready(df_batch)
        wall = time.time() - t0

        # log summary
        step = int(state_batch.step[0])
        t_sim = float(state_batch.time[0])

        heat_fluxes = np.asarray(fluxes_batch[1])
        growths = np.asarray(state_batch.last_growth_rate)

        traj_logs = []
        for i, name in enumerate(names):
            eflux_i = float(np.mean(heat_fluxes[i]))
            growth_i = float(np.mean(growths[i]))
            traj_logs.append(f"{name} [flx {eflux_i:.4f}, gr {growth_i:.4f}]")

        print(
            f"[{step:>8d}] t {t_sim:>8.2f} | "
            f"{' | '.join(traj_logs)} | "
            f"{block / wall:.2f} steps/s  x{n_batch}"
        )
        # accumulate per-config diagnostics
        for i, s in enumerate(setups):
            flx_i = np.asarray(jax.tree.map(lambda x: x[i], fluxes_batch))
            if flx_i.ndim == 0 or (flx_i.ndim == 1 and flx_i.shape[0] != 3):
                flx_i = np.array(
                    [
                        float(fluxes_batch[0][i]),
                        float(fluxes_batch[1][i]),
                        float(fluxes_batch[2][i]),
                    ]
                )
            accum[s["name"]]["fluxes"].append(flx_i)
            accum[s["name"]]["growth"].append(np.asarray(state_batch.last_growth_rate[i]))
            accum[s["name"]]["times"].append(t_sim)

    # save per-config in standard format
    for s in setups:
        a = accum[s["name"]]
        common = {"time": np.array(a["times"]), "step": np.arange(len(a["times"])) * block_size}
        np.savez(
            os.path.join(s["output_dir"], "fluxes.npz"), fluxes=np.stack(a["fluxes"]), **common
        )
        np.savez(
            os.path.join(s["output_dir"], "growth.npz"), growth=np.stack(a["growth"]), **common
        )

    runtime = time.time() - t0
    print(f"\ncompleted {n_batch} configs in {runtime:.1f}s")

    for s in setups:
        if s["data_dir"]:
            report(s["output_dir"], s["data_dir"], s["name"], args.kinetic)


def report(output_dir, data_dir, name, kinetic):
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
    except FileNotFoundError:
        print("reference data not found, skipping comparison")
        return

    if kinetic and fluxes_data.ndim == 3:
        avg_start = max(0, len(fluxes_data) - len(fluxes_data) // 4)
        n_avg = len(fluxes_data) - avg_start
        species_cols = {0: ("ion", 1), 1: ("electron", 4)}
        print(f"\n{name} time-averaged eflux (last {n_avg} blocks):")
        for sp_idx, (sp_name, ref_col) in species_cols.items():
            if sp_idx >= fluxes_data.shape[1]:
                continue
            sim_avg = np.mean(fluxes_data[avg_start:, sp_idx, 1])
            ref_samples = [
                ref_fluxes[np.argmin(np.abs(ref_time - t)), ref_col] for t in sim_times[avg_start:]
            ]
            ref_avg = np.mean(ref_samples)
            rel_err = abs(sim_avg - ref_avg) / max(abs(ref_avg), 1e-15)
            print(f"  {sp_name:>10s}: sim={sim_avg:.4e}  ref={ref_avg:.4e}  rel_err={rel_err:.2e}")
    else:
        avg_count = 80
        sim_avg = np.mean(fluxes_data[-avg_count:, 1])
        ref_samples = [
            ref_fluxes[np.argmin(np.abs(ref_time - t)), 1] for t in sim_times[-avg_count * 3 :]
        ]
        ref_avg = np.mean(ref_samples)
        err = abs(sim_avg - ref_avg)
        print(f"\n{name} time-averaged eflux (last {avg_count}):")
        print(f"  sim={sim_avg:.4e}  ref={ref_avg:.4e}  abs_err={err:.2e}")


def main():
    parser = argparse.ArgumentParser(description="run gyaradax simulations.")
    parser.add_argument("inputs", nargs="+", help="yaml config paths")
    parser.add_argument("--kinetic", action="store_true")
    parser.add_argument("--mp", action="store_true", help="mixed precision")
    parser.add_argument(
        "--z2z", action="store_true", default=None, help="use Z2Z FFT for nonlinear term"
    )
    parser.add_argument(
        "--no-z2z", dest="z2z", action="store_false", help="disable Z2Z FFT for nonlinear term"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        choices=["jax", "cuda"],
        help="backend for nonlinear term",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=120)
    parser.add_argument("--n-blocks", type=int, default=None)
    parser.add_argument("--from-scratch", action="store_true", help="ignore K-files, use init_f")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="resume from a specific K-file (e.g. K03, 100, K01)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="override output directory")
    parser.add_argument(
        "--save-dumps", action="store_true", help="save full 5D df snapshots at each checkpoint"
    )
    parser.add_argument("--n-gpus-sp", type=int, default=1, help="species-axis mesh size (>=1)")
    parser.add_argument("--n-gpus-vp", type=int, default=1, help="vpar-axis mesh size (>=1)")
    parser.add_argument("--n-gpus-mu", type=int, default=1, help="mu-axis mesh size (>=1)")
    parser.add_argument(
        "--device-list",
        type=str,
        default=None,
        help="comma-separated CUDA device ids for multi-GPU (overrides --device)",
    )

    args = parser.parse_args()
    if len(args.inputs) > 1:
        run_multi(args)
    else:
        run(args.inputs[0], args)


if __name__ == "__main__":
    main()
