#!/usr/bin/env python3
"""C1: _apply_parallel — 9-point parallel stencil.

OPTIM.md §4.1: AI ≈ 0.08 FLOP/byte, ~157M FLOPs, ~1.96 GB R+W per call per species.
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

# OPTIM.md §4.1 figures (per call, per species, adiabatic)
FLOPS    = 157e6
BYTES_RW = 1.96e9


def build_fn(pre):
    s_shift     = pre["s_shift"]
    kx_shift    = pre["kx_shift"]
    valid_shift = pre["valid_shift"]
    coeffs      = pre["s_total_upar"]

    @jax.jit
    def _apply_parallel_upar(field):
        out = jnp.zeros_like(field)
        nky = field.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
        for i in range(9):
            shifted = jnp.where(
                valid_shift[i][None, None, :, :, :],
                field[:, :, s_shift[i], kx_shift[i], ky_idx],
                0.0,
            )
            out = out + coeffs[i] * shifted
        return out

    return _apply_parallel_upar


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C1: _apply_parallel  (9-point parallel stencil)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df  # 5D adiabatic: (nv, nmu, ns, nkx, nky)

    fn = build_fn(pre)

    # accuracy
    out = fn(field)
    rel_l2 = check_accuracy(out, BASELINES_DIR / "apply_parallel.npz", "output")

    # timing
    timer = BenchTimer(lambda: fn(field))
    mean_ms, std_ms = timer.run()
    print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
    result = roofline_report("_apply_parallel", mean_ms, FLOPS, BYTES_RW)
    result["rel_l2"] = rel_l2
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
