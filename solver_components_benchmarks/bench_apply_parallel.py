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
    load_setup, check_accuracy, BASELINES_DIR, analyze_cost,
    DEFAULT_BW_GBS, DEFAULT_FP64_TFLOPS,
)
from gyaradax.solver import _apply_parallel_fn, GKPre
import gyaradax.stencils as stencils


# ── Timing ───────────────────────────────────────────────────────────────────
def _bench(fn, n_warmup=5, n_trials=30):
    for _ in range(n_warmup):
        fn()
    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times_ms)
    return {
        "mean": float(a.mean()),
        "std":  float(a.std()),
        "min":  float(a.min()),
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
    }


# ── JAX Reference Implementation (from Production) ───────────────────────────
def _make_jax_fn(pre_dict):
    pre_gk = GKPre(pre_dict)
    _fn = _apply_parallel_fn(pre_gk)
    
    @jax.jit
    def wrapper(field, coeffs):
        return _fn(field, coeffs)
    
    return wrapper


def _make_opt_cuda_ffi_fn(field_template, pre):
    nv, nmu, ns, nkx, nky = field_template.shape
    nv_nmu = nv * nmu

    valid_jax = jnp.array(pre["valid_shift"])
    s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
    kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
    packed_maps_jax = jnp.stack([s_map_jax, kx_map_jax], axis=-1)

    # Register the FFI target
    name = "apply_parallel_ffi"
    try:
        _lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
        symbol = getattr(_lib, name)
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(symbol), platform="CUDA")
    except Exception as e:
        if "already registered" not in str(e).lower():
            print(f"  [DEBUG] FFI registration failed for {name}: {e}")

    output_shape = field_template.shape

    @jax.jit
    def _fn(f, c_v1):
        c_1d = c_v1.view(jnp.float64).reshape(-1)
        res = jax.ffi.ffi_call(
            "apply_parallel_ffi",
            [jax.ShapeDtypeStruct(output_shape, field_template.dtype)]
        )(f, c_1d, packed_maps_jax,
          nv_nmu=np.int32(nv_nmu), nkx=np.int32(nkx), ns=np.int32(ns), 
          nky=np.int32(nky), nmu=np.int32(nmu))
        return res[0]

    return _fn



# ── Reporting & Main ─────────────────────────────────────────────────────────
_HDR = (
    f"  {'impl':22s}  {'mean':>7}  {'std':>6}  {'min':>6}  {'p50':>6}  {'p95':>6}"
    f"  {'GB/s':>7}  {'%BW':>5}  {'AI':>7}  {'%roof':>6}  accuracy"
)

def _print_row(label, t, flops, bytes_rw, rel_l2):
    mean_ms = t["mean"]
    achieved_gbs = bytes_rw / 1e9 / (mean_ms / 1e3)
    pct_bw   = 100.0 * achieved_gbs / DEFAULT_BW_GBS if DEFAULT_BW_GBS > 0 else float("nan")
    ai       = flops / bytes_rw
    if DEFAULT_BW_GBS > 0 and DEFAULT_FP64_TFLOPS > 0:
        roof_tf  = min(ai * DEFAULT_BW_GBS / 1024, DEFAULT_FP64_TFLOPS)
        pct_roof = 100.0 * (flops / (mean_ms / 1e3)) / (roof_tf * 1e12)
        roof_str = f"{pct_roof:6.1f}"
    else:
        roof_str = "   N/A"

    l2_str = f"{rel_l2:.2e}" if not np.isnan(rel_l2) else "  (no baseline)"
    print(
        f"  {label:22s}  {mean_ms:7.3f}  {t['std']:6.3f}  {t['min']:6.3f}"
        f"  {t['p50']:6.3f}  {t['p95']:6.3f}"
        f"  {achieved_gbs:7.1f}  {pct_bw:5.1f}  {ai:7.4f}  {roof_str}  {l2_str}"
    )

def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*70}")
    print("C1: _apply_parallel  (Template Specialized CUDA Stencil)")
    print(f"{'='*70}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df
    baseline = BASELINES_DIR / "apply_parallel.npz"

    coeffs_raw = pre["s_total_upar"] # (9, nv, 1, ns, nkx, nky)
    target_coeffs_shape = (9, *field.shape)
    coeffs_broadcasted = jnp.broadcast_to(coeffs_raw, target_coeffs_shape)
    
    print(f"\n  [JAX] Compiling production reference function...")
    jax_fn = _make_jax_fn(pre)
    out_jax = jax_fn(field, coeffs_broadcasted)
    rel_l2_jax = check_accuracy(out_jax, baseline, "output")

    print(f"  [XLA] Analyzing cost...")
    flops, bytes_rw = analyze_cost(jax_fn, field, coeffs_broadcasted)
    
    # Correcting bytes_rw for the vectorized map array
    # Old maps (JAX): s_map (4B) + kx_map (4B) + valid (4B) = 12 Bytes per point
    # FFI maps: int2 = 8 Bytes per point.
    ns, nkx, nky = field.shape[2:]
    saved_bytes = 9 * ns * nkx * nky * 4
    bytes_rw_ffi = bytes_rw - saved_bytes

    print(f"    FLOPs/call : {flops/1e6:.1f} M")
    print(f"    Bytes R+W  (JAX):    {bytes_rw/1e9:.3f} GB")
    print(f"    Bytes R+W  (CUDA):   {bytes_rw_ffi/1e9:.3f} GB")

    print(f"  [JAX] Timing (5 warmup, 30 trials)...")
    t_jax = _bench(lambda: jax_fn(field, coeffs_broadcasted).block_until_ready())

    print(f"  [CUDA] Compiling Refined FFI (Templates)...")
    cuda_opt_fn = _make_opt_cuda_ffi_fn(field, pre)
    out_cuda = cuda_opt_fn(field, coeffs_raw)
    rel_l2_cuda = check_accuracy(out_cuda, baseline, "output")

    print(f"  [CUDA] Timing Refined FFI (5 warmup, 30 trials)...")
    t_cuda = _bench(lambda: cuda_opt_fn(field, coeffs_raw).block_until_ready())

    print(f"\n{_HDR}")
    print(f"  {'-'*130}")
    _print_row("JAX (reference)",  t_jax,  flops, bytes_rw, rel_l2_jax)
    _print_row("CUDA (FFI)",        t_cuda, flops, bytes_rw_ffi, rel_l2_cuda)

    speedup = t_jax["mean"] / t_cuda["mean"]
    print(f"\n  Final Speedup vs JAX:    {speedup:.2f}×")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)