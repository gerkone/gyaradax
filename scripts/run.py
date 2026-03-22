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

# parse --device before any jax import so cuda sees the right gpu
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--device", type=int, default=-1)
_early_args, _ = _parser.parse_known_args()
if _early_args.device != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_early_args.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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


def _find_k_file(data_dir):
    """return path to the first K-file in data_dir, or None."""
    if not data_dir:
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

    output_dir = f"validation_{'kinetic' if kinetic else 'outputs'}_{name}"

    overrides = {"mixed_precision": args.mp}
    if kinetic:
        overrides["adaptive_dt"] = True
    params = gkparams_from_config(cfg, **overrides)

    if _has_geom_dat(data_dir):
        geometry = load_geometry(data_dir)
    else:
        geometry = _geometry_from_config(cfg)

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    k_path = None if args.from_scratch else _find_k_file(data_dir)

    if k_path is not None:
        res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = load_gkw_k_dump(k_path, res, n_species=n_species)

        dat_path = k_path + ".dat"
        t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0
        nky = len(geometry["krho"])

        actual_dt = read_gkw_dump_dtim(dat_path) if os.path.exists(dat_path) else 0.0
        if actual_dt > 0 and actual_dt < params.dt:
            params = replace(params, dt=actual_dt)

        if params.adiabatic_electrons:
            phi0 = _compute_phi_for_init(df, geometry, params)
            amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
        else:
            amp0 = jnp.ones(nky, dtype=jnp.float64)

        state = GKState(
            time=jnp.array(t_start, dtype=jnp.float64),
            step=jnp.array(0, dtype=jnp.int32),
            accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
            window_start_amp=amp0,
            last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
        )
    else:
        df, state = gk_init(geometry, params, n_species=n_species)

    pre = linear_precompute(geometry, params)

    block_size = args.block_size
    if not kinetic:
        block_size = int(getattr(cfg.solver, "dump_interval", 3)) * params.naverage

    if args.n_blocks:
        n_steps = args.n_blocks * block_size
    elif kinetic and data_dir:
        try:
            ref_times = np.loadtxt(os.path.join(data_dir, "time.dat"))
            n_steps = max(block_size, int((ref_times[-1] - float(state.time)) * 0.8 / params.dt))
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
        save_snapshots=False,
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
    df_batch = jnp.stack([s["df"] for s in setups])
    geometry_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *[s["geometry"] for s in setups])
    params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *[s["params"] for s in setups])
    state_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *[s["state"] for s in setups])
    pre_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *[s["pre"] for s in setups])

    # per-config accumulators
    accum = {s["name"]: {"fluxes": [], "growth": [], "times": []} for s in setups}

    for out_dir in set(s["output_dir"] for s in setups):
        os.makedirs(out_dir, exist_ok=True)

    target_step = n_steps
    while int(state_batch.step[0]) < target_step:
        remaining = target_step - int(state_batch.step[0])
        block = min(block_size, remaining)
        if block <= 0:
            break

        t0 = time.time()
        df_batch, phi_batch, fluxes_batch, state_batch = gk_run_batched(
            df_batch, geometry_batch, params_batch, state_batch, block, pre_batch
        )
        wall = time.time() - t0

        # log summary (mean across batch)
        step = int(state_batch.step[0])
        t_sim = float(state_batch.time[0])
        mean_growth = float(jnp.mean(state_batch.last_growth_rate))
        flx_arr = np.asarray(fluxes_batch)
        mean_eflux = float(np.mean(flx_arr[..., 1])) if flx_arr.ndim >= 2 else float(flx_arr[1])
        print(
            f"[{step:>8d}] t {t_sim:>8.2f} | "
            f"eflux(mean) {mean_eflux:>8.4f} | growth(mean) {mean_growth:>8.4f} | "
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

    print(f"\ncompleted {n_batch} configs in {time.time() - t0:.1f}s")

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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=300)
    parser.add_argument("--n-blocks", type=int, default=None)
    parser.add_argument("--from-scratch", action="store_true", help="ignore K-files, use init_f")

    args = parser.parse_args()
    if len(args.inputs) > 1:
        run_multi(args)
    else:
        run(args.inputs[0], args)


if __name__ == "__main__":
    main()
