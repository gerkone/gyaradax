"""run gyaradax simulations.

automatically detects geometry source and initial condition:
  - if data_dir has geom.dat -> load_geometry; otherwise -> compute_geometry
  - if data_dir has K-files  -> resume from K01; otherwise -> init_f

Usage:
    # existing GKW run (geometry + K-file from data_dir)
    python scripts/run.py configs/iteration_13.yaml --device 5

    # standalone (analytic geometry, fresh init)
    python scripts/run.py configs/my_new_case.yaml --device 5

    # force fresh init even when K-files exist
    python scripts/run.py configs/iteration_13.yaml --from-scratch --device 5
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
from gyaradax.utils import load_geometry
from gyaradax.params import gkparams_from_config
from gyaradax.simulate import (
    gk_init,
    gksimulate,
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


def run(config_path, args):
    cfg = load_config(config_path)
    data_dir = getattr(cfg.run, "data_dir", None)
    name = cfg.run.name
    kinetic = args.kinetic

    output_dir = f"validation_{'kinetic' if kinetic else 'outputs'}_{name}"

    print("=" * 88)
    print(f"{'kinetic' if kinetic else 'adiabatic'}: {name} ({config_path})")
    print("=" * 88)

    overrides = {"mixed_precision": args.mp}
    if kinetic:
        overrides["adaptive_dt"] = True
    params = gkparams_from_config(cfg, **overrides)

    # geometry: file-based if geom.dat exists, analytic otherwise
    if _has_geom_dat(data_dir):
        geometry = load_geometry(data_dir)
        print(f"geometry: loaded from {data_dir}")
    else:
        geometry = _geometry_from_config(cfg)
        print("geometry: computed from config parameters")

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    # initial condition: K-file if available (and not --from-scratch), else init_f
    k_path = None if args.from_scratch else _find_k_file(data_dir)

    if k_path is not None:
        res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = load_gkw_k_dump(k_path, res, n_species=n_species)

        dat_path = k_path + ".dat"
        t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0
        nky = len(geometry["krho"])

        actual_dt = read_gkw_dump_dtim(dat_path) if os.path.exists(dat_path) else 0.0
        if actual_dt > 0 and actual_dt < params.dt:
            print(f"  using dtim={actual_dt:.6f} from {os.path.basename(dat_path)} (config dt={params.dt:.6f})")
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
        print(f"init: resumed from {os.path.basename(k_path)} (t={t_start:.4f})")
    else:
        df, state = gk_init(geometry, params, n_species=n_species)
        print("init: fresh (init_f)")

    pre = linear_precompute(geometry, params)

    # determine step counts
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

    print_params(params, grid_shape=df.shape)
    print(f"  n_steps={n_steps}, checkpoint_interval={block_size}")

    t0 = time.time()
    gksimulate(
        df, geometry, params, state, n_steps,
        pre=pre,
        output_dir=output_dir,
        checkpoint_interval=block_size,
        save_snapshots=False,
    )
    runtime = time.time() - t0
    print(f"\ncompleted in {runtime:.1f}s ({n_steps / runtime:.1f} steps/s)")

    if data_dir:
        report(output_dir, data_dir, name, kinetic)


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
            ref_fluxes[np.argmin(np.abs(ref_time - t)), 1] for t in sim_times[-avg_count * 3:]
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
    for inp in args.inputs:
        run(inp, args)


if __name__ == "__main__":
    main()
