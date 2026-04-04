#!/usr/bin/env python3
"""benchmark for gyaradax solver with component-level timing."""

import argparse
import os
import time
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

jax.config.update("jax_enable_x64", True)

from gyaradax import load_config, load_geometry
from gyaradax.params import gkparams_from_config
from gyaradax.simulate import (
    gk_init,
    _geometry_from_config,
)
from gyaradax.solver import (
    gksolve,
    GKState,
    linear_precompute,
    _compute_phi,
)
from gyaradax.backends import create_ops
from gyaradax.utils import load_gkw_k_dump, read_gkw_dump_time, read_gkw_dump_dtim


def bench_component(fn, n_iters=5, label=""):
    """time a function over n_iters, return mean ms."""
    # warmup
    result = fn()
    jax.block_until_ready(result)
    times = []
    for _ in range(n_iters):
        t0 = time.time()
        result = fn()
        jax.block_until_ready(result)
        times.append((time.time() - t0) * 1000)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    print(f"  {label:30s}: {mean_ms:8.2f} +/- {std_ms:5.2f} ms")
    return mean_ms


def run_benchmark():
    parser = argparse.ArgumentParser(description="benchmark gyaradax solver")
    parser.add_argument("config", type=str, help="yaml config path")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--kinetic", action="store_true")
    parser.add_argument("--resume", type=str, default="100")
    parser.add_argument("--device", type=int, default=-1, help="GPU device ID")
    parser.add_argument("--components", action="store_true", help="benchmark individual components")
    parser.add_argument("--save-reference", type=str)
    parser.add_argument("--reference", type=str)
    parser.add_argument("--from-scratch", action="store_true", help="ignore K-files, use init_f")
    parser.add_argument("--mp", action="store_true", help="mixed precision")

    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = getattr(cfg.run, "data_dir", None)

    overrides = {"mixed_precision": args.mp}
    params = gkparams_from_config(cfg, **overrides)

    # geometry: file-based if geom.dat exists, analytic otherwise
    if data_dir and os.path.exists(os.path.join(data_dir, "geom.dat")):
        geom = load_geometry(data_dir)
        print(f"geometry: loaded from {data_dir}")
    else:
        geom = _geometry_from_config(cfg)
        print("geometry: computed from config parameters")

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    # determine initial state
    k_path = None
    if not args.from_scratch and data_dir:
        # look for K01 or similar in data_dir
        from gyaradax.utils import K_files

        ks = K_files(data_dir)
        if ks:
            k_path = os.path.join(data_dir, ks[0])

    if k_path is not None:
        shape = tuple(len(geom[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = load_gkw_k_dump(k_path, shape, n_species=n_species)

        dat_path = k_path + ".dat"
        t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0
        nky = len(geom["krho"])

        actual_dt = read_gkw_dump_dtim(dat_path) if os.path.exists(dat_path) else 0.0
        if 0 < actual_dt < params.dt:
            print(f"using dtim={actual_dt:.6f} from {os.path.basename(dat_path)}")
            # params is frozen, but we can update it via some mechanism if needed
            # for benchmark we might just want to use config dt unless explicitly resume-testing

        state = GKState(
            time=jnp.array(t_start, dtype=jnp.float64),
            step=jnp.array(0, dtype=jnp.int32),
            accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
            window_start_amp=jnp.ones(nky, dtype=jnp.float64),  # keep it 1.0 for standard benchmark
            last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
        )
        print(f"init: resumed from {os.path.basename(k_path)} (t={t_start:.4f})")
    else:
        df, state = gk_init(geom, params, n_species=n_species)
        print("init: fresh (init_f)")

    mode = "kinetic" if args.kinetic else "adiabatic"
    print(f"mode: {mode}, grid: {df.shape}, dt: {params.dt}")

    # precompute
    t0 = time.time()
    pre = linear_precompute(geom, params)
    t_pre = time.time() - t0
    print(f"precompute: {t_pre:.3f}s")

    # component-level benchmarks (before full scan warmup)
    if args.components:
        print(f"\ncomponent benchmarks (single evaluation, {mode}):")

        ops = create_ops(pre, df, backend=params.backend, use_z2z=params.use_z2z)

        bench_component(
            lambda: _compute_phi(df, geom, params, pre),
            label="phi solve",
        )

        phi = _compute_phi(df, geom, params, pre)

        bench_component(
            lambda: ops.linear_rhs(df, phi, geom, params, pre),
            label="linear rhs",
        )

        if params.non_linear:
            bench_component(
                lambda: ops.nonlinear_term_iii(df, phi, geom, mixed_precision=params.mixed_precision),
                label="nonlinear rhs (term iii)",
            )
        print()

    # full solver benchmark
    print(f"steps/block: {args.steps}, blocks: {args.blocks}")
    print(f"warmup (compile {args.steps} steps)...")
    t0 = time.time()
    df_w, (phi_w, fluxes_w), state_w = gksolve(df, geom, params, state, n_steps=args.steps, pre=pre)
    jax.block_until_ready(df_w)
    t_warmup = time.time() - t0
    print(f"warmup: {t_warmup:.2f}s")

    df_cur, state_cur = df_w, state_w
    block_times = []
    for i in range(args.blocks):
        t0 = time.time()
        df_cur, (phi, fluxes), state_cur = gksolve(
            df_cur, geom, params, state_cur, n_steps=args.steps, pre=pre
        )
        jax.block_until_ready(df_cur)
        dt_block = time.time() - t0
        block_times.append(dt_block)
        sps = args.steps / dt_block
        print(f"  block {i+1}/{args.blocks}: {dt_block:.3f}s ({sps:.1f} steps/s)")

    times = np.array(block_times)
    mean_sps = args.steps / np.mean(times)
    std_sps = args.steps * np.std(times) / np.mean(times) ** 2
    print(f"\n{'='*50}")
    print(f"  {args.steps * args.blocks} steps in {np.sum(times):.3f}s")
    print(f"  {mean_sps:.2f} +/- {std_sps:.2f} steps/s")
    print(f"  {np.mean(times)*1000/args.steps:.2f} ms/step")
    print(f"{'='*50}")

    if args.reference:
        ref = np.load(args.reference)
        err = float(
            np.linalg.norm(np.array(df_cur) - ref["df"]) / (np.linalg.norm(ref["df"]) + 1e-30)
        )
        print(f"rel_l2 vs reference: {err:.4e}")

    if args.save_reference:
        np.savez(
            args.save_reference,
            df=np.array(df_cur),
            time=np.array(state_cur.time),
            step=np.array(state_cur.step),
        )
        print(f"saved reference to {args.save_reference}")


if __name__ == "__main__":
    run_benchmark()
