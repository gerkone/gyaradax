#!/usr/bin/env python3
"""C7: gkstep_single — full RK4 time step (linear and nonlinear).
"""
import argparse, os, sys, functools
from pathlib import Path
from dataclasses import replace

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from common import load_setup, BenchTimer, roofline_report, check_accuracy, analyze_cost, BASELINES_DIR
from gyaradax.solver import gkstep_single, default_state, GKPre
from gyaradax.backends import create_ops


def run(config="configs/iteration_13.yaml", mixed_precision=False, backend_forced=None):
    print(f"\n{'='*60}")
    print(f"C7: gkstep_single  (Full RK4 Step)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    state = default_state(nky=df.shape[-1])
    pre_gk = GKPre(pre)
    baseline = BASELINES_DIR / "rk4_step.npz"

    from gyaradax.backends import create_ops
    
    # 1. Linear RK4 Step
    print(f"\n[PHASE 1] Linear RK4 Step")
    for bname in ["jax", "cuda"]:
        if backend_forced and bname != backend_forced: continue
        print(f"\n  -- Backend: {bname.upper()}")
        try:
            ops = create_ops(pre_gk, df, backend=bname)
        except Exception as e:
            print(f"     [SKIP] {bname} not available: {e}")
            continue

        p_var = replace(params, non_linear=False)
        @functools.partial(jax.jit, static_argnames=("ops",))
        def fn(d, s, ops):
            return gkstep_single(d, geom, p_var, s, pre_gk, ops=ops)

        out_df, (out_phi, _), _ = fn(df, state, ops)
        check_accuracy(out_df,  baseline, "out_df_linear")
        check_accuracy(out_phi, baseline, "out_phi_linear")

        print(f"     [XLA] Analyzing cost...")
        flops, bytes_rw = analyze_cost(fn, df, state, ops)
        mean_ms, _ = BenchTimer(lambda: fn(df, state, ops)[0].block_until_ready()).run()
        print(f"     timing: {mean_ms:.3f} ms")
        roofline_report(f"rk4_linear ({bname})", mean_ms, flops, bytes_rw)

    # 2. Nonlinear RK4 Step
    print(f"\n[PHASE 2] Nonlinear RK4 Step")
    for bname in ["jax", "cuda"]:
        if backend_forced and bname != backend_forced: continue
        print(f"\n  -- Backend: {bname.upper()}")
        try:
            ops = create_ops(pre_gk, df, backend=bname)
        except Exception as e:
            print(f"     [SKIP] {bname} not available: {e}")
            continue

        p_var = replace(params, non_linear=True)
        @functools.partial(jax.jit, static_argnames=("ops",))
        def fn(d, s, ops):
            return gkstep_single(d, geom, p_var, s, pre_gk, ops=ops)

        out_df, (out_phi, _), _ = fn(df, state, ops)
        check_accuracy(out_df,  baseline, "out_df_nonlinear")
        check_accuracy(out_phi, baseline, "out_phi_nonlinear")

        print(f"     [XLA] Analyzing cost...")
        flops, bytes_rw = analyze_cost(fn, df, state, ops)
        mean_ms, _ = BenchTimer(lambda: fn(df, state, ops)[0].block_until_ready()).run()
        print(f"     timing: {mean_ms:.3f} ms")
        roofline_report(f"rk4_nonlinear ({bname})", mean_ms, flops, bytes_rw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()
    run(args.config, args.mp, args.backend)

