#!/usr/bin/env python3
"""benchmark for gyaradax solver with component-level timing."""

import os
import re
import time
import argparse
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gyaradax import load_geometry
from gyaradax.geometry import compute_geometry_from_input
from gyaradax.solver import (
    gksolve,
    GKState,
    linear_precompute,
    _compute_phi,
    _compute_linear_rhs,
    _compute_nonlinear_rhs,
)
from gyaradax.params import gkparams_from_input_dat
from gyaradax.utils import load_gkw_k_dump


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
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--kinetic", action="store_true")
    parser.add_argument("--resume", type=str, default="100")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--components", action="store_true", help="benchmark individual components")
    parser.add_argument("--save-reference", type=str)
    parser.add_argument("--reference", type=str)
    parser.add_argument(
        "--computed-geometry",
        action="store_true",
        help="use analytic geometry instead of loading from files",
    )
    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    data_dir = args.data_dir
    n_species = 2 if args.kinetic else 1

    if args.computed_geometry:
        geom = compute_geometry_from_input(os.path.join(data_dir, "input.dat"))
    else:
        geom = load_geometry(data_dir)
    shape = tuple(len(geom[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
    dump_path = os.path.join(data_dir, args.resume)
    df = load_gkw_k_dump(dump_path, shape, n_species=n_species)

    dat_path = dump_path + ".dat"
    dt_override = {}
    if os.path.exists(dat_path):
        with open(dat_path) as f:
            text = f.read()
        m = re.search(r"DTIM\s*=\s*([0-9eE+\-.]+)", text)
        if m:
            dt_override["dt"] = float(m.group(1))
        m = re.search(r"TIME\s*=\s*([0-9eE+\-.]+)", text)
        t_start = float(m.group(1)) if m else 0.0
    else:
        t_start = 0.0

    params = gkparams_from_input_dat(
        os.path.join(data_dir, "input.dat"),
        non_linear=True,
        adiabatic_electrons=not args.kinetic,
        **dt_override,
    )

    nky = len(geom["krho"])
    state = GKState(
        time=jnp.array(t_start, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

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

        bench_component(
            lambda: _compute_phi(df, geom, params, pre),
            label="phi solve",
        )

        phi = _compute_phi(df, geom, params, pre)

        bench_component(
            lambda: _compute_linear_rhs(df, phi, geom, params, pre),
            label="linear rhs",
        )

        if params.non_linear:
            bench_component(
                lambda: _compute_nonlinear_rhs(df, phi, geom, params, pre),
                label="nonlinear rhs (term iii)",
            )
        print()

    # full solver benchmark
    print(f"steps/block: {args.steps}, blocks: {args.blocks}")
    print(f"warmup (compile {args.steps} steps)...")
    t0 = time.time()
    df_w, _, state_w = gksolve(df, geom, params, state, n_steps=args.steps, pre=pre)
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
