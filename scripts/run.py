"""run gyaradax simulations and validate against gkw reference data.

Usage:
    python scripts/run.py configs/iteration_13.yaml --device 5
    python scripts/run.py configs/kinetic_991_half_rlt.yaml --kinetic --device 5
    python scripts/run.py configs/iteration_13.yaml --from-scratch
    python scripts/run.py configs/kinetic_991_half_rlt.yaml --kinetic --block-size 300 --n-blocks 265
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

import jax.numpy as jnp

from gyaradax import load_config
from gyaradax.simulate import (
    gk_from_config,
    gksimulate,
    default_log,
    _compute_phi_for_init,
)
from gyaradax.solver import GKState, mode_amplitude
from gyaradax.utils import (
    load_gkw_k_dump,
    read_gkw_dump_time,
    read_gkw_dump_dtim,
    K_files,
    print_params,
)


def run(config_path, args):
    """run a simulation from yaml config, optionally resuming from a k-file."""
    cfg = load_config(config_path)
    data_dir = cfg.run.data_dir
    name = cfg.run.name
    kinetic = args.kinetic

    label = "kinetic" if kinetic else "adiabatic"
    output_dir = f"validation_{'kinetic' if kinetic else 'outputs'}_{name}"

    print("=" * 88)
    print(f"{label}: {name} ({config_path})")
    print("=" * 88)

    # load config, geometry, and build initial condition
    overrides = {"mixed_precision": args.mp}
    if kinetic:
        overrides["adaptive_dt"] = True
    df, geometry, params, state, pre = gk_from_config(config_path, **overrides)

    # determine checkpoint interval and total steps
    block_size = args.block_size
    if not kinetic:
        # adiabatic: interval from config dump_interval * naverage
        block_size = cfg.solver.dump_interval * cfg.solver.naverage

    if args.n_blocks:
        n_steps = args.n_blocks * block_size
    else:
        # default: 265 blocks for adiabatic, auto from ref time for kinetic
        if kinetic:
            try:
                ref_times = np.loadtxt(os.path.join(data_dir, "time.dat"))
                target_time = (ref_times[-1] - float(state.time)) * 0.8
                n_steps = max(block_size, int(target_time / params.dt))
            except FileNotFoundError:
                n_steps = 100 * block_size
        else:
            n_steps = 265 * block_size

    # optionally resume from a gkw k-file instead of fresh init
    if not args.from_scratch:
        n_species = 1
        if not params.adiabatic_electrons:
            n_species = int(jnp.asarray(params.mas).shape[0])
        res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))

        # find the first k-file in data_dir
        ks = K_files(data_dir)
        start_file = ks[0] if ks else None
        if start_file is None and os.path.exists(os.path.join(data_dir, "K01")):
            start_file = "K01"

        if start_file:
            k_path = os.path.join(data_dir, start_file)
            df = load_gkw_k_dump(k_path, res, n_species=n_species)

            # read start time and build state from k-file metadata
            dat_path = k_path + ".dat"
            t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0
            nky = len(geometry["krho"])

            # use GKW's adaptive dtim if it's smaller than config dt
            actual_dt = read_gkw_dump_dtim(dat_path) if os.path.exists(dat_path) else 0.0
            if actual_dt > 0 and actual_dt < params.dt:
                print(
                    f"  using dtim={actual_dt:.6f} from {os.path.basename(dat_path)} (config dt={params.dt:.6f})"
                )
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

    print_params(params, grid_shape=df.shape)
    print(f"  n_steps={n_steps}, checkpoint_interval={block_size}")

    # run simulation with checkpointing and logging
    t0 = time.time()
    gksimulate(
        df,
        geometry,
        params,
        state,
        n_steps,
        pre=pre,
        output_dir=output_dir,
        checkpoint_interval=block_size,
        save_snapshots=False,
    )
    runtime = time.time() - t0
    print(f"\ncompleted in {runtime:.1f}s ({n_steps / runtime:.1f} steps/s)")

    # compare against gkw reference
    report(output_dir, data_dir, name, kinetic)


def report(output_dir, data_dir, name, kinetic):
    """compare time-averaged fluxes against gkw reference."""
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
        # kinetic: (n_entries, nsp, 3) — report per-species eflux
        avg_start = max(0, len(fluxes_data) - len(fluxes_data) // 4)
        n_avg = len(fluxes_data) - avg_start
        # gkw fluxes.dat columns: ion (pflux, eflux, vflux), electron (pflux, eflux, vflux)
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
        # adiabatic: (n_entries, 3)
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
    parser = argparse.ArgumentParser(
        description="run gyaradax simulations and validate against gkw reference data."
    )
    parser.add_argument("inputs", nargs="+", help="yaml config paths")
    parser.add_argument("--kinetic", action="store_true", help="kinetic electron mode")
    parser.add_argument("--mp", action="store_true", help="mixed precision")
    parser.add_argument("--device", type=int, default=0, help="cuda device index")
    parser.add_argument("--block-size", type=int, default=300, help="steps per checkpoint block")
    parser.add_argument(
        "--n-blocks", type=int, default=None, help="number of blocks (auto if omitted)"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="start from init_f instead of k-file",
    )

    args = parser.parse_args()

    for inp in args.inputs:
        run(inp, args)


if __name__ == "__main__":
    main()
