#!/usr/bin/env python3
"""C2: _apply_vpar — 5-point velocity-space stencil.

OPTIM.md §4.2: AI ≈ 0.11 FLOP/byte, ~87M FLOPs, ~782 MB R+W per call per species.
"""
import argparse, os, sys
from pathlib import Path

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
from gyaradax.solver import _apply_vpar_fn, GKPre
import gyaradax.stencils as stencils


# Internal definition removed; using production _apply_vpar_fn instead.


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C2: _apply_vpar  (5-point vpar stencil)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df
    pre_gk = GKPre(pre)
    _fn = _apply_vpar_fn(pre_gk)
    apply_vpar_jit = jax.jit(_fn)
    baseline = BASELINES_DIR / "apply_vpar.npz"

    for label, coeffs, bkey in [
        ("VPAR_D1 (streaming)", stencils.VPAR_D1, "output_d1"),
        ("VPAR_D4 (dissipation)", stencils.VPAR_D4, "output_d4"),
    ]:
        print(f"\n  -- {label}")
        out = apply_vpar_jit(field, coeffs)
        rel_l2 = check_accuracy(out, baseline, bkey)
        
        print(f"  [XLA] Analyzing cost...")
        flops, bytes_rw = analyze_cost(apply_vpar_jit, field, coeffs)
        
        mean_ms, std_ms = BenchTimer(lambda f=field, c=coeffs: apply_vpar_jit(f, c).block_until_ready()).run()
        print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
        roofline_report(f"_apply_vpar ({label[:6]})", mean_ms, flops, bytes_rw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
