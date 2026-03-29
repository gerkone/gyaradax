#!/usr/bin/env python3
"""Benchmark for dual-fused parallel stencil CUDA kernel.

Compares:
1. Two separate calls to the optimized single-kernel CUDA FFI.
2. One call to the new dual-fused CUDA FFI kernel.
"""
import argparse, os, sys, time, ctypes
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

LIB_PATH = Path(__file__).parent.parent / "cuda_augmentations" / "liblto_bracket.so"

from common import load_setup, BenchTimer, roofline_report, check_accuracy, analyze_cost
from gyaradax.solver import _apply_parallel_fn, GKPre

# Timing helper for multiple outputs
def _bench_dual(fn, n_warmup=10, n_trials=50):
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1e3)
    return np.mean(times), np.std(times)

def register_ffi():
    for name in ["apply_parallel_ffi", "apply_parallel_dual_ffi"]:
        try:
            _lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
            symbol = getattr(_lib, name)
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(symbol), platform="CUDA")
        except Exception as e:
            if "already registered" not in str(e).lower():
                print(f"FFI registration failed for {name}: {e}")

def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*75}")
    print("Optimization: Dual Parallel Stencil Fusion")
    print(f"{'='*75}")

    register_ffi()
    
    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    nv, nmu, ns, nkx, nky = df.shape
    nv_nmu = nv * nmu
    
    # Setup inputs for dual stencil (mirroring _linear_rhs_core)
    phi_b = jnp.reshape(phi, (1, 1, ns, nkx, nky))
    # For testing, we want non-zero gyro_phi to exercise the second path
    key = jax.random.PRNGKey(42)
    gyro_phi = jax.random.normal(key, df.shape).astype(df.dtype)
    
    coeffs1 = pre["s_total_upar"]
    # Ensure coeffs2 is non-zero to test the second path
    coeffs2 = coeffs1 + 0.1
    
    # Indirection maps
    valid_jax = jnp.array(pre["valid_shift"])
    s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
    kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
    packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1)

    # --- 0. JAX Reference Implementation ---
    jax_apply = _apply_parallel_fn(GKPre(pre))

    @jax.jit
    def two_jax_calls(f1, f2, c1, c2):
        # We need to broadcast coefficients for JAX if they aren't already
        c1_b = jnp.broadcast_to(c1, (9, *f1.shape))
        c2_b = jnp.broadcast_to(c2, (9, *f2.shape))
        o1 = jax_apply(f1, c1_b)
        o2 = jax_apply(f2, c2_b)
        return o1, o2

    # --- 1. Single CUDA FFI implementation ---
    @jax.jit
    def apply_single(field, coeffs):
        c_1d = coeffs.view(jnp.float64).reshape(-1)
        res = jax.ffi.ffi_call(
            "apply_parallel_ffi",
            [jax.ShapeDtypeStruct(field.shape, field.dtype)]
        )(field, c_1d, packed_maps,
          nv_nmu=np.int32(nv_nmu), nkx=np.int32(nkx), ns=np.int32(ns), 
          nky=np.int32(nky), nmu=np.int32(nmu))
        return res[0]

    @jax.jit
    def two_separate_calls(f1, f2, c1, c2):
        o1 = apply_single(f1, c1)
        o2 = apply_single(f2, c2)
        return o1, o2

    # --- 2. Dual CUDA FFI implementation ---
    @jax.jit
    def apply_dual(f1, f2, c1, c2):
        c1_1d = c1.view(jnp.float64).reshape(-1)
        c2_1d = c2.view(jnp.float64).reshape(-1)
        res = jax.ffi.ffi_call(
            "apply_parallel_dual_ffi",
            [jax.ShapeDtypeStruct(f1.shape, f1.dtype),
             jax.ShapeDtypeStruct(f2.shape, f2.dtype)]
        )(f1, f2, c1_1d, c2_1d, packed_maps,
          nv_nmu=np.int32(nv_nmu), nkx=np.int32(nkx), ns=np.int32(ns), 
          nky=np.int32(nky), nmu=np.int32(nmu))
        return res

    # Benchmark JAX calls
    print(f"\n  [V0] Measuring JAX production reference (two calls)...")
    t_jax_mean, t_jax_std = _bench_dual(lambda: two_jax_calls(df, gyro_phi, coeffs1, coeffs2))

    # Benchmark separate calls
    print(f"  [V1] Measuring two separate CUDA kernel launches...")
    t_sep_mean, t_sep_std = _bench_dual(lambda: two_separate_calls(df, gyro_phi, coeffs1, coeffs2))

    # Benchmark dual call
    print(f"  [V2] Measuring fused dual-kernel launch...")
    t_dual_mean, t_dual_std = _bench_dual(lambda: apply_dual(df, gyro_phi, coeffs1, coeffs2))

    print(f"\nResults for Grid ({nv_nmu}, {ns}, {nkx}, {nky}):")
    print(f"  JAX Reference: {t_jax_mean:7.3f} ± {t_jax_std:5.3f} ms")
    print(f"  Separate CUDA: {t_sep_mean:7.3f} ± {t_sep_std:5.3f} ms")
    print(f"  Dual-Fused:    {t_dual_mean:7.3f} ± {t_dual_std:5.3f} ms")
    print(f"\nSpeedup vs JAX:  {t_jax_mean / t_dual_mean:7.2f}x")
    print(f"Speedup vs CUDA: {t_sep_mean / t_dual_mean:7.2f}x")

    # Correctness check
    o1_ref, o2_ref = two_jax_calls(df, gyro_phi, coeffs1, coeffs2)
    o1_dual, o2_dual = apply_dual(df, gyro_phi, coeffs1, coeffs2)
    
    norm1_ref = float(jnp.linalg.norm(o1_ref))
    norm2_ref = float(jnp.linalg.norm(o2_ref))
    print(f"\nDebug Norms:")
    print(f"  Norm 1 Ref: {norm1_ref:.2e}")
    print(f"  Norm 2 Ref: {norm2_ref:.2e}")

    err1 = float(jnp.linalg.norm(o1_ref - o1_dual) / norm1_ref) if norm1_ref > 0 else 0.0
    err2 = float(jnp.linalg.norm(o2_ref - o2_dual) / norm2_ref) if norm2_ref > 0 else 0.0
    
    print(f"\nVerification:")
    print(f"  Output 1 Error: {err1:.2e}")
    print(f"  Output 2 Error: {err2:.2e}")
    
    if err1 < 1e-13 and err2 < 1e-13:
        print("  [PASS] Numerical accuracy verified.")
    else:
        print("  [FAIL] Numerical discrepancy detected!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
