#!/usr/bin/env python3
"""C1: _apply_parallel — 9-point parallel stencil.

Benchmarks the JAX reference implementation against a Warp (custom CUDA)
kernel side-by-side, with full roofline analysis.

Kernel structure
----------------
For each output element (iv, imu, is, ikx, iky):
  acc = 0
  for i in 0..8:
    if valid[i, is, ikx, iky]:
      acc += coeffs[i, iv, imu, is, ikx, iky] * field[iv, imu, s_map[i,...], kx_map[i,...], iky]
  out[iv, imu, is, ikx, iky] = acc

FLOP model (per output element, 9 stencil pts):
  real×complex mul: 2 FLOPs (c*re, c*im)
  complex accumulate: 2 FLOPs (add re, add im)
  → 4 FLOPs × 9 pts = 36 FLOPs/element

Memory model (no L2 reuse assumed):
  field reads:   9 × N × 16 B   (complex128)
  coeffs reads:  9 × N × 8 B    (float64, expanded to nv*nmu)
  output write:  1 × N × 16 B
  maps+valid:    3 × 9 × ns×nkx×nky × 4 B   (small)
"""
import argparse, os, sys, time
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

import warp as wp
from warp.jax_experimental.ffi import jax_kernel

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_setup, check_accuracy, BASELINES_DIR,
    DEFAULT_BW_GBS, DEFAULT_FP64_TFLOPS,
)

wp.init()


# ── Warp kernel (compiled once per process) ──────────────────────────────────
#
# To bypass Warp's internal MAX_DIMS=4 limit, all arrays must be strictly <= 4D.
# We flatten the velocity dimensions (nv, nmu) -> nv_nmu.
# We treat JAX's complex128 as a float64 array with a doubled innermost dimension (nky * 2).
#
# Array shapes at launch time:
#   field   : (nv_nmu, ns, nkx, nky * 2)  float64  [strided: ky*2 = real, ky*2+1 = imag]
#   coeffs  : (9, nv_nmu, ns * nkx, nky)  float64  [flattened ns and nkx to stay 4D]
#   s_map   : (9, ns, nkx, nky)           int32
#   kx_map  : (9, ns, nkx, nky)           int32
#   valid   : (9, ns, nkx, nky)           bool
#   out     : (nv_nmu, ns, nkx, nky * 2)  float64
#
# Launch grid: (nv_nmu, ns, nkx, nky) -> Perfectly maps to the 4 spatial dims.

@wp.kernel
def _apply_parallel_warp_k(
    field:   wp.array(dtype=wp.float64, ndim=4), 
    coeffs:  wp.array(dtype=wp.float64, ndim=4), 
    s_map:   wp.array(dtype=wp.int32,   ndim=4), 
    kx_map:  wp.array(dtype=wp.int32,   ndim=4), 
    valid:   wp.array(dtype=wp.bool,   ndim=4), 
    out:     wp.array(dtype=wp.float64, ndim=4), 
):
    v_idx, s, kx, ky = wp.tid()
    
    # Extract nkx from the map shape for coefficient indexing
    nkx = kx_map.shape[2]

    # Accumulators for real and imaginary parts
    acc_r = wp.float64(0.0)
    acc_i = wp.float64(0.0)

    for i in range(9):
        i_wp = wp.static(i)
        if valid[i_wp, s, kx, ky]:
            src_s  = s_map[i_wp, s, kx, ky]
            src_kx = kx_map[i_wp, s, kx, ky]
            
            # coeffs is flattened over ns and nkx to respect MAX_DIMS=4
            c = coeffs[i_wp, v_idx, s * nkx + kx, ky]
            
            # Fetch real and imaginary components (strided by 2)
            val_r = field[v_idx, src_s, src_kx, ky * 2]
            val_i = field[v_idx, src_s, src_kx, ky * 2 + 1]
            
            # Fused multiply-add
            acc_r = acc_r + val_r * c
            acc_i = acc_i + val_i * c

    # Write back the computed complex components to contiguous memory
    out[v_idx, s, kx, ky * 2]     = acc_r
    out[v_idx, s, kx, ky * 2 + 1] = acc_i


# ── Roofline arithmetic ──────────────────────────────────────────────────────

def _roofline(field_shape, coeffs_nmu):
    """Return (flops, bytes_rw) for one _apply_parallel call.

    coeffs_nmu: the nmu dim of the raw coefficient tensor (1 in adiabatic case).
    The expanded coefficients passed to Warp have the full nmu.
    """
    nv, nmu, ns, nkx, nky = field_shape
    N = nv * nmu * ns * nkx * nky    # number of output elements

    flops_per_elem = 9 * 4           # 9 pts × (2 mul + 2 add) real ops
    flops = N * flops_per_elem

    B16, B8, B4 = 16, 8, 4          # bytes per complex128, float64, int32/bool
    # Field: 9 stencil reads (worst-case no L2 reuse)
    b_field  = 9 * N * B16
    # Coeffs: fully indexed (no mu broadcast, already expanded)
    b_coeffs = 9 * N * B8
    # Output write
    b_out    = N * B16
    # Maps and mask (small but counted)
    n_map    = 9 * ns * nkx * nky
    b_maps   = n_map * (2 * B4 + 1)  # s_map, kx_map (int32) + valid (bool)

    bytes_rw = b_field + b_coeffs + b_out + b_maps
    return flops, bytes_rw


# ── Timing ───────────────────────────────────────────────────────────────────

def _bench(fn, n_warmup=5, n_trials=30):
    """Return stats dict (ms) over n_trials after n_warmup."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times_ms.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times_ms)
    return {
        "mean": float(a.mean()),
        "std":  float(a.std()),
        "min":  float(a.min()),
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
    }


# ── Reference JAX implementation (matches solver._apply_parallel exactly) ────

def _make_jax_fn(pre):
    s_shift     = pre["s_shift"]      # (9, ns, nkx, nky)  int32
    kx_shift    = pre["kx_shift"]     # (9, ns, nkx, nky)  int32
    valid_shift = pre["valid_shift"]  # (9, ns, nkx, nky)  bool
    coeffs      = pre["s_total_upar"] # (9, nv, nmu_c, ns, nkx, nky)  float64

    @jax.jit
    def _fn(field):
        out  = jnp.zeros_like(field)
        nky  = field.shape[-1]
        ky_i = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
        for i in range(9):
            shifted = jnp.where(
                valid_shift[i][None, None, :, :, :],
                field[:, :, s_shift[i], kx_shift[i], ky_i],
                0.0,
            )
            out = out + coeffs[i] * shifted
        return out

    return _fn


# ── Warp implementation ───────────────────────────────────────────────────────

def _make_warp_fn(field_shape, pre):
    """Build the JIT-compiled Warp wrapper for the current problem shape."""
    nv, nmu, ns, nkx, nky = field_shape
    nv_nmu = nv * nmu

    coeffs_raw = pre["s_total_upar"]  # (9, nv, nmu_c, ns, nkx, nky)
    s_map      = pre["s_shift"]       # (9, ns, nkx, nky)
    kx_map     = pre["kx_shift"]      # (9, ns, nkx, nky)
    valid      = pre["valid_shift"]   # (9, ns, nkx, nky)

    # Broadcast coefficients to full nmu, then flatten to strictly 4D: (9, nv_nmu, ns*nkx, nky)
    coeffs_4d = jnp.broadcast_to(
        coeffs_raw, (9, nv, nmu, ns, nkx, nky)
    ).astype(jnp.float64).reshape(9, nv_nmu, ns * nkx, nky)

    _kernel = jax_kernel(
        _apply_parallel_warp_k,
        launch_dims=(nv_nmu, ns, nkx, nky),
        in_out_argnames=["out"],
    )

    @jax.jit
    def _fn(field):
        # 1. Reinterpret complex128 to float64 (nky -> nky * 2)
        # 2. Reshape to collapse nv and nmu (5D -> 4D)
        # Resulting shape: (nv_nmu, ns, nkx, nky * 2)
        f_4d = field.view(jnp.float64).reshape(nv_nmu, ns, nkx, nky * 2)
        
        # Allocate float64 output buffer
        out_4d = jnp.zeros_like(f_4d)
        
        # Execute the FFI kernel
        (out_4d,) = _kernel(f_4d, coeffs_4d, s_map, kx_map, valid, out_4d)
        
        # Reinterpret float64 back to complex128 and restore the original 5D layout
        return out_4d.view(jnp.complex128).reshape(nv, nmu, ns, nkx, nky)

    return _fn


# ── Reporting ─────────────────────────────────────────────────────────────────

_HDR = (
    f"  {'impl':20s}  {'mean':>7}  {'std':>6}  {'min':>6}  {'p50':>6}  {'p95':>6}"
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
        f"  {label:20s}  {mean_ms:7.3f}  {t['std']:6.3f}  {t['min']:6.3f}"
        f"  {t['p50']:6.3f}  {t['p95']:6.3f}"
        f"  {achieved_gbs:7.1f}  {pct_bw:5.1f}  {ai:7.4f}  {roof_str}  {l2_str}"
    )
    return achieved_gbs, pct_bw


# ── Main ──────────────────────────────────────────────────────────────────────

def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*70}")
    print("C1: _apply_parallel  (9-point parallel stencil)")
    print(f"{'='*70}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df  # (nv, nmu, ns, nkx, nky)  complex128

    nv, nmu, ns, nkx, nky = field.shape
    coeffs_raw = pre["s_total_upar"]

    print(f"\n  Problem dimensions:")
    print(f"    field      : {field.shape}  {field.dtype}")
    print(f"    coeffs_raw : {coeffs_raw.shape}  {coeffs_raw.dtype}")
    print(f"    s_map      : {pre['s_shift'].shape}  {pre['s_shift'].dtype}")
    print(f"    valid      : {pre['valid_shift'].shape}  {pre['valid_shift'].dtype}")
    print(f"    nmu in coeffs: {coeffs_raw.shape[2]}  →  expanded to {nmu} for Warp kernel")

    flops, bytes_rw = _roofline(field.shape, coeffs_raw.shape[2])
    print(f"\n  Roofline (conservative, no L2 field-read reuse):")
    print(f"    FLOPs/call : {flops/1e6:.1f} M  ({9} pts × 4 FLOPs × {nv*nmu*ns*nkx*nky:,} elements)")
    print(f"    Bytes R+W  : {bytes_rw/1e9:.3f} GB")
    print(f"    Arith. Int.: {flops/bytes_rw:.4f} FLOP/byte  → memory-bound on every GPU")

    baseline = BASELINES_DIR / "apply_parallel.npz"

    # ── JAX reference ────────────────────────────────────────────────────────
    print(f"\n  [JAX] Compiling reference function...")
    jax_fn = _make_jax_fn(pre)
    out_jax = jax_fn(field)          # force compilation
    rel_l2_jax = check_accuracy(out_jax, baseline, "output")

    print(f"  [JAX] Timing ({5} warmup, {30} trials)...")
    t_jax = _bench(lambda: jax_fn(field))

    # ── Warp kernel ───────────────────────────────────────────────────────────
    print(f"\n  [Warp] Compiling custom CUDA kernel (launch grid: {nv*nmu}×{ns}×{nkx}×{nky})...")
    warp_fn = _make_warp_fn(field.shape, pre)
    out_warp = warp_fn(field)        # force compilation
    rel_l2_warp = check_accuracy(out_warp, baseline, "output")

    print(f"  [Warp] Timing ({5} warmup, {30} trials)...")
    t_warp = _bench(lambda: warp_fn(field))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{_HDR}")
    print(f"  {'-'*130}")
    _print_row("JAX (reference)",  t_jax,  flops, bytes_rw, rel_l2_jax)
    _print_row("Warp (custom)",    t_warp, flops, bytes_rw, rel_l2_warp)

    speedup_mean = t_jax["mean"] / t_warp["mean"]
    speedup_min  = t_jax["min"]  / t_warp["min"]
    print(f"\n  Speedup  Warp vs JAX:  {speedup_mean:.2f}× (mean)   {speedup_min:.2f}× (best)")

    return {
        "label":         "_apply_parallel",
        "jax_mean_ms":   t_jax["mean"],
        "warp_mean_ms":  t_warp["mean"],
        "speedup":       speedup_mean,
        "jax_rel_l2":    rel_l2_jax,
        "warp_rel_l2":   rel_l2_warp,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=1,
                        help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true",
                        help="Enable mixed precision in params")
    args = parser.parse_args()
    run(args.config, args.mp)