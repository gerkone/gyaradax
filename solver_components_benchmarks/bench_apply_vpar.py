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
from common import load_setup, BenchTimer, roofline_report, check_accuracy, BASELINES_DIR

FLOPS    = 87e6
BYTES_RW = 782e6


@jax.jit
def _apply_vpar(field, coeffs):
    nv = field.shape[0]
    out = jnp.zeros_like(field)
    for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):
        idx = jnp.clip(jnp.arange(nv, dtype=jnp.int32) + s, 0, nv - 1)
        valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
        shifted = jnp.take(field, idx, axis=0)
        out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
    return out


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C2: _apply_vpar  (5-point vpar stencil)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df
    from gyaradax import stencils

    baseline = BASELINES_DIR / "apply_vpar.npz"

    for label, coeffs, bkey in [
        ("VPAR_D1 (streaming)", stencils.VPAR_D1, "output_d1"),
        ("VPAR_D4 (dissipation)", stencils.VPAR_D4, "output_d4"),
    ]:
        print(f"\n  -- {label}")
        out = _apply_vpar(field, coeffs)
        rel_l2 = check_accuracy(out, baseline, bkey)
        mean_ms, std_ms = BenchTimer(lambda c=coeffs: _apply_vpar(field, c)).run()
        print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
        roofline_report(f"_apply_vpar ({label[:6]})", mean_ms, FLOPS, BYTES_RW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
