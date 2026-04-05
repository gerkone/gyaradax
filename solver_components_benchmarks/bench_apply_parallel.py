#!/usr/bin/env python3
"""C1: _apply_parallel — 9-point parallel stencil.

Benchmarks the JAX reference implementation against a custom CUDA FFI
kernel side-by-side, with full roofline analysis.
"""
import argparse, os, sys, time, ctypes
from pathlib import Path

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

LIB_PATH = Path(__file__).parent.parent / "cuda_augmentations" / "liblto_bracket.so"

from common import (
    load_setup,
    check_accuracy,
    BASELINES_DIR,
    analyze_cost,
    BenchTimer,
    roofline_report,
    DEFAULT_BW_GBS,
    DEFAULT_FP64_TFLOPS,
)
from gyaradax.solver import GKPre
import gyaradax.stencils as stencils


# Reporters and Main removed; now integrated into run()


# ── Reporting & Main ─────────────────────────────────────────────────────────


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*75}")
    print("C1: _apply_parallel  (9-point parallel stencil)")
    print(f"{'='*75}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df
    pre_gk = GKPre(pre)
    baseline = BASELINES_DIR / "apply_parallel.npz"

    from gyaradax.backends import create_ops

    results = {}
    backends = []
    for b in ["jax", "cuda"]:
        try:
            ops = create_ops(pre_gk, backend=b)
            backends.append((b, ops))
        except Exception as e:
            print(f"  [SKIP] {b} backend not available: {e}")

    coeffs_raw = pre["s_total_upar"]  # (9, nv, 1, ns, nkx, nky)
    target_coeffs_shape = (9, *field.shape)
    coeffs_broadcasted = jnp.broadcast_to(coeffs_raw, target_coeffs_shape)

    # 1. Individual Stencil
    print(f"\n  -- Single Stencil (_apply_parallel)")
    backend_times = {}

    for bname, ops in backends:

        @jax.jit
        def fn(f, c):
            return ops._apply_parallel(f, c)

        # Accuracy check
        out = fn(field, coeffs_broadcasted)
        rel_l2 = check_accuracy(out, baseline, "output")

        # Performance timing
        mean_ms, _ = BenchTimer(
            lambda f=field, c=coeffs_broadcasted: fn(f, c).block_until_ready()
        ).run()
        backend_times[bname] = mean_ms

        print(f"     [{bname.upper():4s}] {mean_ms:7.3f} ms  (rel_l2={rel_l2:.2e})")

        if bname == "cuda":
            flops, bytes_rw = analyze_cost(fn, field, coeffs_broadcasted)
            # Adjust bytes_rw for expected FFI behavior (as in original script)
            ns, nkx, nky = field.shape[2:]
            saved_bytes = 9 * ns * nkx * nky * 4
            bytes_rw_ffi = bytes_rw - saved_bytes
            roofline_report(f"_apply_parallel ({bname})", mean_ms, flops, bytes_rw_ffi)

    if "jax" in backend_times and "cuda" in backend_times:
        print(f"     Speedup: {backend_times['jax']/backend_times['cuda']:.2f}x")

    # 2. Dual Fused Stencil (Merge from bench_apply_parallel_cuda.py)
    print(f"\n  -- Dual Stencil Fusion (_apply_parallel_dual)")

    # Setup inputs for dual stencil
    key = jax.random.PRNGKey(42)
    gyro_phi = jax.random.normal(key, df.shape).astype(df.dtype)
    coeffs1 = pre["s_total_upar"]
    coeffs2 = pre["s_total_t7"]

    dual_times = {}
    for bname, ops in backends:

        @jax.jit
        def fn_dual(f1, f2, c1, c2):
            return ops._apply_parallel_dual(f1, f2, c1, c2)

        out_dual = fn_dual(df, gyro_phi, coeffs1, coeffs2)

        # Accuracy check matches bench_apply_parallel_cuda logic
        @jax.jit
        def ref_two_calls(f1, f2, c1, c2):
            from gyaradax.backends import create_ops

            ops_ref = create_ops(pre_gk, backend="jax")
            return ops_ref._apply_parallel(f1, c1), ops_ref._apply_parallel(f2, c2)

        o_ref1, o_ref2 = ref_two_calls(df, gyro_phi, coeffs1, coeffs2)
        err1 = float(jnp.linalg.norm(out_dual[0] - o_ref1) / jnp.linalg.norm(o_ref1))
        err2 = float(jnp.linalg.norm(out_dual[1] - o_ref2) / jnp.linalg.norm(o_ref2))
        l2_str = f"err1={err1:.1e}, err2={err2:.1e}"

        mean_ms, _ = BenchTimer(
            lambda: jax.block_until_ready(fn_dual(df, gyro_phi, coeffs1, coeffs2))
        ).run()
        dual_times[bname] = mean_ms
        print(f"     [{bname.upper():4s}] {mean_ms:7.3f} ms  ({l2_str})")

        if bname == "cuda":
            flops, bytes_rw = analyze_cost(fn_dual, df, gyro_phi, coeffs1, coeffs2)
            roofline_report(f"_apply_parallel_dual ({bname})", mean_ms, flops, bytes_rw)

    if "jax" in dual_times and "cuda" in dual_times:
        print(f"     Speedup: {dual_times['jax']/dual_times['cuda']:.2f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
