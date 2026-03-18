#!/usr/bin/env python3
"""C3: _compute_linear_rhs — full linear operator (all terms, all species).

OPTIM.md §4.3: AI ≈ 0.087 FLOP/byte, ~635M FLOPs/species, ~7.3 GB R+W/species.
Adiabatic (1 species): ~635M FLOPs, ~7.3 GB R+W per call.
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

FLOPS_PER_SP = 635e6
BYTES_PER_SP = 7.3e9


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C3: _compute_linear_rhs  (full linear operator)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)

    from gyaradax.solver import _compute_linear_rhs
    fn = jax.jit(lambda: _compute_linear_rhs(df, phi, geom, params, pre))

    out = fn()
    rel_l2 = check_accuracy(out, BASELINES_DIR / "linear_rhs.npz", "output")

    mean_ms, std_ms = BenchTimer(fn).run()
    print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")

    n_sp = df.shape[0] if df.ndim == 6 else 1
    roofline_report(
        f"_compute_linear_rhs ({n_sp} sp)",
        mean_ms,
        FLOPS_PER_SP * n_sp,
        BYTES_PER_SP * n_sp,
    )
    return {"mean_ms": mean_ms, "rel_l2": rel_l2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
