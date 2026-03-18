#!/usr/bin/env python3
"""C5: _compute_phi — kinetic/adiabatic phi solve.

OPTIM.md §4.5: AI ≈ 0.17–0.47 FLOP/byte, ~56M FLOPs, ~119–337 MB R+W per call.
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
from common import load_setup, BenchTimer, roofline_report, check_accuracy, BASELINES_DIR

# best-case (broadcast fused): 56M FLOPs, 119 MB; worst-case: 337 MB
FLOPS         = 56e6
BYTES_RW_BEST = 119e6
BYTES_RW_WORST = 337e6


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C5: _compute_phi  (phi solve)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)

    from gyaradax.solver import _compute_phi
    fn = jax.jit(lambda: _compute_phi(df, geom, params, pre))

    out = fn()
    rel_l2 = check_accuracy(out, BASELINES_DIR / "phi_solve.npz", "output")

    mean_ms, std_ms = BenchTimer(fn).run()
    print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
    roofline_report("_compute_phi (best-case BW)",  mean_ms, FLOPS, BYTES_RW_BEST)
    roofline_report("_compute_phi (worst-case BW)", mean_ms, FLOPS, BYTES_RW_WORST)

    return {"mean_ms": mean_ms, "rel_l2": rel_l2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
