#!/usr/bin/env python3
"""C7: gkstep_single — full RK4 time step (linear and nonlinear).
"""
import argparse, os, sys
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


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C7: gkstep_single  (Full RK4 Step)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    state = default_state(nky=df.shape[-1])
    pre_gk = GKPre(pre)

    baseline = BASELINES_DIR / "rk4_step.npz"

    def benchmark_variant(name, nl_enabled, df_key, phi_key):
        p = replace(params, non_linear=nl_enabled)

        @jax.jit
        def fn(d, s):
            return gkstep_single(d, geom, p, s, pre_gk)

        print(f"\n  -- {name} RK4 Step")
        out_df, (out_phi, _), _ = fn(df, state)

        check_accuracy(out_df,  baseline, df_key)
        check_accuracy(out_phi, baseline, phi_key)

        print(f"  [XLA] Analyzing cost...")
        flops, bytes_rw = analyze_cost(fn, df, state)

        mean_ms, std_ms = BenchTimer(lambda: fn(df, state)[0].block_until_ready()).run()
        print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
        roofline_report(f"rk4_{name.lower()}", mean_ms, flops, bytes_rw)

    # 1. Linear RK4 Step
    benchmark_variant("Linear",    False, "out_df_linear",    "out_phi_linear")

    # 2. Nonlinear RK4 Step
    benchmark_variant("Nonlinear", True,  "out_df_nonlinear", "out_phi_nonlinear")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
