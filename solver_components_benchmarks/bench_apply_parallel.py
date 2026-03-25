#!/usr/bin/env python3
"""C1: _apply_parallel — 9-point parallel stencil.

Benchmarks the JAX reference implementation against a Warp (custom CUDA)
kernel side-by-side, with full roofline analysis.
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
from warp.jax_experimental import jax_kernel, register_ffi_callback

from common import (
    load_setup, check_accuracy, BASELINES_DIR, analyze_cost,
    DEFAULT_BW_GBS, DEFAULT_FP64_TFLOPS,
)
from gyaradax.solver import _apply_parallel_fn, GKPre
import gyaradax.stencils as stencils

wp.init()

def make_kernels(nv_nmu, nkx, ns, nky, nmu):
    # ── Ultra-Lean Vectorized C++ Snippet ────────────────────────────────────
    cpp_snippet = f"""
        // 128-bit Shared Memory (Conflict-free on Ampere for sequential ky)
        __shared__ double2 smem[{ns}][{nky}];

        const double2* field_ptr = reinterpret_cast<const double2*>(field.data);
        double2* out_ptr = reinterpret_cast<double2*>(out.data);
        const int2* maps_ptr = reinterpret_cast<const int2*>(packed_maps.data);

        // Precompute unified spatial dimensions
        size_t spatial_stride = {ns} * {nkx} * {nky};
        size_t spatial_idx    = size_t(s) * ({nkx} * {nky}) + size_t(kx) * {nky} + ky;
        
        size_t field_idx  = size_t(v_idx) * spatial_stride + spatial_idx;
        
        // 1. Cooperative Load (Single 128-bit transaction)
        smem[s][ky] = __ldg(&field_ptr[field_idx]);

        __syncthreads();

        double acc_r = 0.0;
        double acc_i = 0.0;

        // Strides for Coefficients
        size_t nv_raw = {nv_nmu} / {nmu};
        size_t c_idx_base = size_t(v_idx / {nmu}) * spatial_stride + spatial_idx;
        size_t c_i_stride = nv_raw * spatial_stride;

        #pragma unroll
        for (int i = 0; i < 9; ++i) {{
            size_t map_idx = i * spatial_stride + spatial_idx;
            
            // ONE Vectorized load replaces s_map, kx_map, and valid checks!
            int2 map_val = __ldg(&maps_ptr[map_idx]);
            int src_s = map_val.x;
            
            // Masking check: JAX sets invalid s_shift to -1
            if (src_s >= 0) {{
                int src_kx = map_val.y;
                
                size_t c_idx = i * c_i_stride + c_idx_base;
                double c = __ldg(&coeffs.data[c_idx]);
                
                double2 val;
                if (src_kx == kx) {{
                    val = smem[src_s][ky]; // 86% Path: 128-bit SMEM load
                }} else {{
                    size_t fb_idx = size_t(v_idx) * spatial_stride + size_t(src_s) * ({nkx} * {nky}) + size_t(src_kx) * {nky} + ky;
                    val = __ldg(&field_ptr[fb_idx]); // 14% Path: 128-bit GMEM load
                }}
                
                acc_r += val.x * c;
                acc_i += val.y * c;
            }}
        }}
        
        // Write directly via pointer to bypass Warp AST AST overhead
        out_ptr[field_idx] = make_double2(acc_r, acc_i);
    """

    @wp.func_native(snippet=cpp_snippet)
    def compute_stencil_smem_1d(
        field:       wp.array(dtype=wp.vec2d, ndim=1),
        coeffs:      wp.array(dtype=wp.float64, ndim=1),
        packed_maps: wp.array(dtype=wp.int32, ndim=1),
        out:         wp.array(dtype=wp.vec2d, ndim=1),
        v_idx:       int,
        kx:          int,
        s:           int,
        ky:          int
    ): ...

    @wp.kernel
    def _apply_parallel_1d_k(
        field:       wp.array(dtype=wp.vec2d, ndim=1),
        coeffs:      wp.array(dtype=wp.float64, ndim=1),
        packed_maps: wp.array(dtype=wp.int32, ndim=1),
        out:         wp.array(dtype=wp.vec2d, ndim=1),
    ):
        tid = wp.tid()
        
        _nkx = wp.static(nkx)
        _nky = wp.static(nky)

        block_id  = tid // 512
        local_tid = tid % 512
        
        v_idx = block_id // _nkx
        kx    = block_id % _nkx
        
        s_idx  = local_tid // _nky
        ky_idx = local_tid % _nky

        # We pass 'out' entirely through the C++ snippet now
        compute_stencil_smem_1d(field, coeffs, packed_maps, out,
                                v_idx, kx, s_idx, ky_idx)

    return _apply_parallel_1d_k


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


# ── Pure Warp DLPack Implementation ──────────────────────────────────────────
def _make_warp_fn(field_template, pre, kernel_1d):
    nv, nmu, ns, nkx, nky = field_template.shape
    nv_nmu = nv * nmu

    # Create packed map: int2(s, kx). Use s = -1 for invalid.
    valid_jax = jnp.array(pre["valid_shift"])
    s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
    kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
    
    # Stack on innermost dimension and flatten: [s0, kx0, s1, kx1, ...]
    packed_maps_jax = jnp.stack([s_map_jax, kx_map_jax], axis=-1).reshape(-1)
    wp_packed_maps = wp.from_jax(packed_maps_jax)

    wp_out = wp.zeros(nv_nmu * ns * nkx * nky, dtype=wp.vec2d)

    def _full_fn(field_jax, coeffs_v1_jax):
        f_vec = field_jax.view(jnp.float64).reshape(-1, 2)
        wp_f  = wp.from_jax(f_vec, dtype=wp.vec2d)
        
        c_1d = jnp.array(coeffs_v1_jax.astype(jnp.float64).reshape(-1))
        wp_c = wp.from_jax(c_1d)

        wp.launch(
            kernel=kernel_1d,
            dim=nv_nmu * nkx * 512,
            inputs=[wp_f, wp_c, wp_packed_maps, wp_out],
            block_dim=512 
        )
        wp.synchronize()
        
        out_jax = wp.to_jax(wp_out)
        return out_jax.view(jnp.complex128).reshape(nv, nmu, ns, nkx, nky)

    return _full_fn


# ── JAX + Warp FFI Variants ──────────────────────────────────────────────────

def _make_jax_kernel_fn(field_template, pre, kernel_1d):
    nv, nmu, ns, nkx, nky = field_template.shape
    nv_nmu = nv * nmu
    n_threads = nv_nmu * nkx * 512

    valid_jax = jnp.array(pre["valid_shift"])
    s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
    kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
    packed_maps_jax = jnp.stack([s_map_jax, kx_map_jax], axis=-1).reshape(-1)

    _jk_primitive = jax_kernel(kernel_1d, in_out_argnames=["out"])

    @jax.jit
    def _fn(f, c_v1):
        f_vec = f.view(jnp.float64).reshape(-1, 2)
        c_1d  = c_v1.view(jnp.float64).reshape(-1)
        out_vec = jnp.zeros_like(f_vec)
        
        (out_vec_updated,) = _jk_primitive(
            f_vec, c_1d, packed_maps_jax, out_vec,
            launch_dims=n_threads
        )
        return out_vec_updated.view(jnp.complex128).reshape(nv, nmu, ns, nkx, nky)

    return _fn


def _make_ffi_callback_fn(field_template, pre, kernel_1d):
    nv, nmu, ns, nkx, nky = field_template.shape
    nv_nmu = nv * nmu
    n_threads = nv_nmu * nkx * 512

    valid_jax = jnp.array(pre["valid_shift"])
    s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
    kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
    packed_maps_jax = jnp.stack([s_map_jax, kx_map_jax], axis=-1).reshape(-1)

    wp_packed_maps = wp.from_jax(packed_maps_jax)

    def warp_apply_ffi_callback(inputs, outputs, attrs, ctx):
        f_ptr   = inputs[0].__cuda_array_interface__["data"][0]
        c_ptr   = inputs[1].__cuda_array_interface__["data"][0]
        out_ptr = outputs[0].__cuda_array_interface__["data"][0]
        
        wp_f   = wp.array(ptr=f_ptr,   dtype=wp.vec2d, shape=(n_threads,),   device="cuda")
        n_coeffs = 9 * nv * 1 * ns * nkx * nky
        wp_c   = wp.array(ptr=c_ptr,   dtype=wp.float64, shape=(n_coeffs,), device="cuda")
        wp_out = wp.array(ptr=out_ptr, dtype=wp.vec2d, shape=(n_threads,),   device="cuda")
        
        stream = wp.Stream(device="cuda:0", cuda_stream=ctx.stream)
        with wp.ScopedStream(stream):
            wp.launch(kernel_1d, dim=n_threads,
                      inputs=[wp_f, wp_c, wp_packed_maps, wp_out],
                      block_dim=512)

    callback_name = f"warp_apply_parallel_{id(field_template)}"
    register_ffi_callback(callback_name, warp_apply_ffi_callback)
    
    output_shape = field_template.shape
    jax_ffi_call = jax.ffi.ffi_call(callback_name, [jax.ShapeDtypeStruct(output_shape, field_template.dtype)])

    @jax.jit
    def _fn(f, c_v1):
        c_1d = c_v1.view(jnp.float64).reshape(-1)
        (res,) = jax_ffi_call(f, c_1d)
        return res

    return _fn


# ── Reporting & Main ─────────────────────────────────────────────────────────
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

def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*70}")
    print("C1: _apply_parallel  (9-point parallel stencil)")
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
    # Old maps: s_map (4B) + kx_map (4B) + valid (4B) = 12 Bytes per point
    # New packed map: int2 = 8 Bytes per point. Saved 4 Bytes per point.
    ns, nkx, nky = field.shape[2:]
    saved_bytes = 9 * ns * nkx * nky * 4
    bytes_rw_warp = bytes_rw - saved_bytes

    print(f"    FLOPs/call : {flops/1e6:.1f} M")
    print(f"    Bytes R+W  (JAX): {bytes_rw/1e9:.3f} GB")
    print(f"    Bytes R+W (Warp): {bytes_rw_warp/1e9:.3f} GB")

    print(f"  [JAX] Timing (5 warmup, 30 trials)...")
    t_jax = _bench(lambda: jax_fn(field, coeffs_broadcasted).block_until_ready())

    nv, nmu = field.shape[0:2]
    nv_nmu = nv * nmu
    print(f"\n  [Warp] Specifying kernels for dimensions ns={ns}, nky={nky}, nmu={nmu}...")
    kernel_1d = make_kernels(nv_nmu, nkx, ns, nky, nmu)

    print(f"  [Warp] Compiling custom Tile kernel (Ultra-Lean FFI)...")
    warp_full_fn = _make_warp_fn(field, pre, kernel_1d)
    out_warp = warp_full_fn(field, coeffs_raw)
    rel_l2_warp = check_accuracy(out_warp, baseline, "output")

    print(f"  [Warp] Timing (5 warmup, 30 trials)...")
    t_warp = _bench(lambda: warp_full_fn(field, coeffs_raw))

    print(f"\n  [JAX+Warp] Compiling jax_kernel wrapper...")
    jax_kernel_fn = _make_jax_kernel_fn(field, pre, kernel_1d)
    out_jk = jax_kernel_fn(field, coeffs_raw)
    rel_l2_jk = check_accuracy(out_jk, baseline, "output")

    print(f"  [JAX+Warp] Timing jax_kernel (5 warmup, 30 trials)...")
    t_jk = _bench(lambda: jax_kernel_fn(field, coeffs_raw).block_until_ready())

    print(f"  [JAX+Warp] Compiling ffi_callback wrapper...")
    ffi_cb_fn = _make_ffi_callback_fn(field, pre, kernel_1d)
    out_cb = ffi_cb_fn(field, coeffs_raw)
    rel_l2_cb = check_accuracy(out_cb, baseline, "output")

    print(f"  [JAX+Warp] Timing ffi_callback (5 warmup, 30 trials)...")
    t_cb = _bench(lambda: ffi_cb_fn(field, coeffs_raw).block_until_ready())

    print(f"\n{_HDR}")
    print(f"  {'-'*130}")
    _print_row("JAX (reference)",  t_jax,  flops, bytes_rw, rel_l2_jax)
    _print_row("Warp (custom)",    t_warp, flops, bytes_rw_warp, rel_l2_warp)
    _print_row("JAX+Warp (jk)",    t_jk,   flops, bytes_rw_warp, rel_l2_jk)
    _print_row("JAX+Warp (cb)",    t_cb,   flops, bytes_rw_warp, rel_l2_cb)

    speedup_jk = t_jax["mean"] / t_jk["mean"]
    speedup_cb = t_jax["mean"] / t_cb["mean"]
    print(f"\n  Speedup   jk vs JAX:  {speedup_jk:.2f}× (mean)")
    print(f"  Speedup   cb vs JAX:  {speedup_cb:.2f}× (mean)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)