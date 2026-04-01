#!/usr/bin/env python3
"""Benchmark: Warp kernel launch overhead & CUDA graph amortization.

Measures the Python-level overhead of Warp's FFI callback path by comparing:
  1. Pure JAX kernel  (baseline — native XLA, no callback overhead)
  2. Warp via register_ffi_callback  (Python ctypes callback on every launch)
  3. Each of the above inside lax.scan  (CUDA graph capture amortizes overhead)

Two test sizes are used:
  - MICRO: Trivially small arrays to isolate pure dispatch overhead
  - PRODUCTION: Real solver arrays to measure end-to-end impact
"""
import argparse, os, sys, time
from pathlib import Path
from functools import partial

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

jax.config.update("jax_enable_x64", True)

import warp as wp
from warp.jax_experimental import register_ffi_callback

wp.init()

# ── Timing Utilities ─────────────────────────────────────────────────────────


def _bench_single(fn, n_warmup=10, n_trials=200):
    """Time a single kernel call. Returns μs."""
    for _ in range(n_warmup):
        fn()
    times_us = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times_us.append((time.perf_counter() - t0) * 1e6)
    a = np.array(times_us)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
    }


def _bench_scan(scan_fn, n_iters, n_warmup=5, n_trials=30):
    """Time a lax.scan loop, return per-iteration cost in μs."""
    for _ in range(n_warmup):
        scan_fn()
    times_us = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        scan_fn()
        times_us.append((time.perf_counter() - t0) * 1e6)
    a = np.array(times_us) / n_iters
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
    }


def _stats_str(d):
    return f"{d['mean']:8.1f} ± {d['std']:6.1f}  (min={d['min']:.1f}, p50={d['p50']:.1f})"


# ──────────────────────────────────────────────────────────────────────────────
#  PART A: MICRO-BENCHMARK — trivial compute to isolate dispatch overhead
# ──────────────────────────────────────────────────────────────────────────────


@wp.kernel
def scale_kernel(a: wp.array(dtype=wp.float64), output: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    output[tid] = a[tid] * wp.float64(2.0)


def run_micro():
    """Micro-benchmark: trivial kernel to isolate Python callback overhead."""
    N = 1024  # Tiny array — kernel executes in < 5μs
    print(f"\n{'='*78}")
    print(f"  PART A: MICRO-BENCHMARK (N={N}) — dispatch overhead isolation")
    print(f"{'='*78}")

    a_jax = jnp.ones(N, dtype=jnp.float64)

    # ------- JAX baseline -------
    @jax.jit
    def jax_scale(a):
        return a * 2.0

    _ = jax_scale(a_jax).block_until_ready()

    # ------- Warp via ffi_callback -------
    def warp_scale_cb(inputs, outputs, attrs, ctx):
        in_ptr = inputs[0].__cuda_array_interface__["data"][0]
        out_ptr = outputs[0].__cuda_array_interface__["data"][0]
        wp_in = wp.array(ptr=in_ptr, dtype=wp.float64, shape=(N,), device="cuda")
        wp_out = wp.array(ptr=out_ptr, dtype=wp.float64, shape=(N,), device="cuda")
        stream = wp.Stream(device="cuda:0", cuda_stream=ctx.stream)
        with wp.ScopedStream(stream):
            wp.launch(scale_kernel, dim=N, inputs=[wp_in, wp_out])

    register_ffi_callback("micro_scale", warp_scale_cb)
    ffi_call = jax.ffi.ffi_call("micro_scale", [jax.ShapeDtypeStruct((N,), jnp.float64)])

    @jax.jit
    def warp_scale(a):
        (out,) = ffi_call(a)
        return out

    out_w = warp_scale(a_jax).block_until_ready()
    assert jnp.allclose(out_w, a_jax * 2.0), f"Warp micro result wrong: {out_w[:5]}"

    # ------- lax.scan variants -------
    N_SCAN = 500

    @partial(jax.jit, static_argnums=(1,))
    def jax_scan(a, n):
        def body(carry, _):
            return carry * 2.0, None

        final, _ = lax.scan(body, a, None, length=n)
        return final

    @partial(jax.jit, static_argnums=(1,))
    def warp_scan(a, n):
        def body(carry, _):
            (out,) = ffi_call(carry)
            return out, None

        final, _ = lax.scan(body, a, None, length=n)
        return final

    _ = jax_scan(a_jax, N_SCAN).block_until_ready()
    _ = warp_scan(a_jax, N_SCAN).block_until_ready()

    # ------- Benchmark -------
    print(f"\n  Single-call (200 trials):")
    t_jax = _bench_single(lambda: jax_scale(a_jax).block_until_ready())
    t_wp = _bench_single(lambda: warp_scale(a_jax).block_until_ready())
    overhead_single = t_wp["mean"] - t_jax["mean"]
    print(f"    {'JAX':25s}  {_stats_str(t_jax)}")
    print(f"    {'Warp ffi_callback':25s}  {_stats_str(t_wp)}   +{overhead_single:.0f} μs")

    print(f"\n  lax.scan ({N_SCAN} iters, per-iteration, 30 trials):")
    t_jax_scan = _bench_scan(lambda: jax_scan(a_jax, N_SCAN).block_until_ready(), n_iters=N_SCAN)
    t_wp_scan = _bench_scan(lambda: warp_scan(a_jax, N_SCAN).block_until_ready(), n_iters=N_SCAN)
    scan_overhead = t_wp_scan["mean"] - t_jax_scan["mean"]
    print(f"    {'JAX':25s}  {_stats_str(t_jax_scan)}")
    print(f"    {'Warp ffi_callback':25s}  {_stats_str(t_wp_scan)}   +{scan_overhead:.1f} μs/iter")

    return {
        "single_jax": t_jax,
        "single_warp": t_wp,
        "scan_jax": t_jax_scan,
        "scan_warp": t_wp_scan,
        "overhead_single": overhead_single,
        "overhead_scan": scan_overhead,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  PART B: PRODUCTION BENCHMARK — real solver stencil kernel
# ──────────────────────────────────────────────────────────────────────────────


def make_stencil_kernel(nv_nmu, nkx, ns, nky, nmu):
    cpp_snippet = f"""
        __shared__ double2 smem[{ns}][{nky}];

        const double2* field_ptr = reinterpret_cast<const double2*>(field.data);
        double2* out_ptr = reinterpret_cast<double2*>(out.data);
        const int2* maps_ptr = reinterpret_cast<const int2*>(packed_maps.data);

        size_t spatial_stride = {ns} * {nkx} * {nky};
        size_t spatial_idx    = size_t(s) * ({nkx} * {nky}) + size_t(kx) * {nky} + ky;
        size_t field_idx  = size_t(v_idx) * spatial_stride + spatial_idx;

        smem[s][ky] = __ldg(&field_ptr[field_idx]);
        __syncthreads();

        double acc_r = 0.0;
        double acc_i = 0.0;

        size_t nv_raw = {nv_nmu} / {nmu};
        size_t c_idx_base = size_t(v_idx / {nmu}) * spatial_stride + spatial_idx;
        size_t c_i_stride = nv_raw * spatial_stride;

        #pragma unroll
        for (int i = 0; i < 9; ++i) {{
            size_t map_idx = i * spatial_stride + spatial_idx;
            int2 map_val = __ldg(&maps_ptr[map_idx]);
            int src_s = map_val.x;
            if (src_s >= 0) {{
                int src_kx = map_val.y;
                size_t c_idx = i * c_i_stride + c_idx_base;
                double c = __ldg(&coeffs.data[c_idx]);
                double2 val;
                if (src_kx == kx) {{
                    val = smem[src_s][ky];
                }} else {{
                    size_t fb_idx = size_t(v_idx) * spatial_stride + size_t(src_s) * ({nkx} * {nky}) + size_t(src_kx) * {nky} + ky;
                    val = __ldg(&field_ptr[fb_idx]);
                }}
                acc_r += val.x * c;
                acc_i += val.y * c;
            }}
        }}
        out_ptr[field_idx] = make_double2(acc_r, acc_i);
    """

    @wp.func_native(snippet=cpp_snippet)
    def compute_stencil_smem_1d(
        field: wp.array(dtype=wp.vec2d, ndim=1),
        coeffs: wp.array(dtype=wp.float64, ndim=1),
        packed_maps: wp.array(dtype=wp.int32, ndim=1),
        out: wp.array(dtype=wp.vec2d, ndim=1),
        v_idx: int,
        kx: int,
        s: int,
        ky: int,
    ): ...

    @wp.kernel
    def _apply_parallel_1d_k(
        field: wp.array(dtype=wp.vec2d, ndim=1),
        coeffs: wp.array(dtype=wp.float64, ndim=1),
        packed_maps: wp.array(dtype=wp.int32, ndim=1),
        out: wp.array(dtype=wp.vec2d, ndim=1),
    ):
        tid = wp.tid()
        _nkx = wp.static(nkx)
        _nky = wp.static(nky)
        block_id = tid // 512
        local_tid = tid % 512
        v_idx = block_id // _nkx
        kx = block_id % _nkx
        s_idx = local_tid // _nky
        ky_idx = local_tid % _nky
        compute_stencil_smem_1d(field, coeffs, packed_maps, out, v_idx, kx, s_idx, ky_idx)

    return _apply_parallel_1d_k


def run_production(config, mixed_precision):
    """Production benchmark with real solver data."""
    from common import (
        load_setup,
        check_accuracy,
        BASELINES_DIR,
        DEFAULT_BW_GBS,
        DEFAULT_FP64_TFLOPS,
    )
    from gyaradax.solver import _apply_parallel_fn, GKPre

    print(f"\n{'='*78}")
    print(f"  PART B: PRODUCTION BENCHMARK — real stencil kernel")
    print(f"{'='*78}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df
    baseline = BASELINES_DIR / "apply_parallel.npz"

    coeffs_raw = pre["s_total_upar"]
    target_coeffs_shape = (9, *field.shape)
    coeffs_broadcasted = jnp.broadcast_to(coeffs_raw, target_coeffs_shape)

    nv, nmu, ns, nkx, nky = field.shape
    nv_nmu = nv * nmu

    # JAX reference
    pre_gk = GKPre(pre)
    jax_fn_raw = _apply_parallel_fn(pre_gk)

    @jax.jit
    def jax_single(field, coeffs):
        return jax_fn_raw(field, coeffs)

    _ = jax_single(field, coeffs_broadcasted).block_until_ready()
    check_accuracy(jax_single(field, coeffs_broadcasted), baseline, "output")

    # Warp kernel
    kernel_1d = make_stencil_kernel(nv_nmu, nkx, ns, nky, nmu)

    packed_maps_jax = jnp.stack(
        [
            jnp.where(jnp.array(pre["valid_shift"]), pre["s_shift"], -1).astype(jnp.int32),
            jnp.array(pre["kx_shift"]).astype(jnp.int32),
        ],
        axis=-1,
    ).reshape(-1)
    wp_packed_maps = wp.from_jax(packed_maps_jax)

    n_threads = nv_nmu * nkx * 512

    def warp_stencil_cb(inputs, outputs, attrs, ctx):
        f_ptr = inputs[0].__cuda_array_interface__["data"][0]
        c_ptr = inputs[1].__cuda_array_interface__["data"][0]
        out_ptr = outputs[0].__cuda_array_interface__["data"][0]
        wp_f = wp.array(ptr=f_ptr, dtype=wp.vec2d, shape=(nv_nmu * ns * nkx * nky,), device="cuda")
        n_coeffs = 9 * nv * 1 * ns * nkx * nky
        wp_c = wp.array(ptr=c_ptr, dtype=wp.float64, shape=(n_coeffs,), device="cuda")
        wp_out = wp.array(
            ptr=out_ptr, dtype=wp.vec2d, shape=(nv_nmu * ns * nkx * nky,), device="cuda"
        )
        stream = wp.Stream(device="cuda:0", cuda_stream=ctx.stream)
        with wp.ScopedStream(stream):
            wp.launch(
                kernel_1d, dim=n_threads, inputs=[wp_f, wp_c, wp_packed_maps, wp_out], block_dim=512
            )

    register_ffi_callback("prod_stencil", warp_stencil_cb)
    ffi_call = jax.ffi.ffi_call("prod_stencil", [jax.ShapeDtypeStruct(field.shape, field.dtype)])

    @jax.jit
    def warp_single(f, c_v1):
        c_1d = c_v1.view(jnp.float64).reshape(-1)
        (res,) = ffi_call(f, c_1d)
        return res

    out_wp = warp_single(field, coeffs_raw).block_until_ready()
    check_accuracy(out_wp, baseline, "output")

    # lax.scan variants
    N_SCAN = 100

    @partial(jax.jit, static_argnums=(2,))
    def jax_scan(field_in, coeffs, n):
        def body(carry, _):
            return jax_fn_raw(carry, coeffs), None

        final, _ = lax.scan(body, field_in, None, length=n)
        return final

    @partial(jax.jit, static_argnums=(2,))
    def warp_scan(field_in, c_v1, n):
        c_1d = c_v1.view(jnp.float64).reshape(-1)

        def body(carry, _):
            (out,) = ffi_call(carry, c_1d)
            return out, None

        final, _ = lax.scan(body, field_in, None, length=n)
        return final

    print(f"\n  Compiling lax.scan variants (n={N_SCAN})...")
    _ = jax_scan(field, coeffs_broadcasted, N_SCAN).block_until_ready()
    _ = warp_scan(field, coeffs_raw, N_SCAN).block_until_ready()

    # Benchmark
    print(f"\n  Single-call (200 trials):")
    t_jax = _bench_single(lambda: jax_single(field, coeffs_broadcasted).block_until_ready())
    t_wp = _bench_single(lambda: warp_single(field, coeffs_raw).block_until_ready())
    overhead_single = t_wp["mean"] - t_jax["mean"]
    print(f"    {'JAX':25s}  {_stats_str(t_jax)}")
    print(f"    {'Warp ffi_callback':25s}  {_stats_str(t_wp)}   {overhead_single:+.0f} μs")

    print(f"\n  lax.scan ({N_SCAN} iters, per-iteration, 30 trials):")
    t_jax_scan = _bench_scan(
        lambda: jax_scan(field, coeffs_broadcasted, N_SCAN).block_until_ready(), n_iters=N_SCAN
    )
    t_wp_scan = _bench_scan(
        lambda: warp_scan(field, coeffs_raw, N_SCAN).block_until_ready(), n_iters=N_SCAN
    )
    scan_overhead = t_wp_scan["mean"] - t_jax_scan["mean"]
    print(f"    {'JAX':25s}  {_stats_str(t_jax_scan)}")
    print(f"    {'Warp ffi_callback':25s}  {_stats_str(t_wp_scan)}   {scan_overhead:+.1f} μs/iter")

    return {
        "single_jax": t_jax,
        "single_warp": t_wp,
        "scan_jax": t_jax_scan,
        "scan_warp": t_wp_scan,
        "overhead_single": overhead_single,
        "overhead_scan": scan_overhead,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    # ── Part A: Micro-benchmark ──
    micro = run_micro()

    # ── Part B: Production benchmark ──
    prod = run_production(config, mixed_precision)

    # ── Final Summary ──
    print(f"\n{'='*78}")
    print(f"  FINAL SUMMARY: Warp FFI Callback Overhead")
    print(f"{'='*78}")
    print(f"\n  ┌─────────────────────────┬────────────────────┬────────────────────┐")
    print(f"  │                         │    Single-call     │  lax.scan /iter    │")
    print(f"  ├─────────────────────────┼────────────────────┼────────────────────┤")
    print(f"  │ MICRO (N=1024)          │                    │                    │")
    print(
        f"  │   JAX                   │ {micro['single_jax']['mean']:>8.0f} μs         │ {micro['scan_jax']['mean']:>8.1f} μs/iter  │"
    )
    print(
        f"  │   Warp FFI              │ {micro['single_warp']['mean']:>8.0f} μs         │ {micro['scan_warp']['mean']:>8.1f} μs/iter  │"
    )
    print(
        f"  │   → overhead            │ {micro['overhead_single']:>+8.0f} μs         │ {micro['overhead_scan']:>+8.1f} μs/iter  │"
    )
    print(f"  ├─────────────────────────┼────────────────────┼────────────────────┤")
    print(f"  │ PRODUCTION (stencil)    │                    │                    │")
    print(
        f"  │   JAX                   │ {prod['single_jax']['mean']:>8.0f} μs         │ {prod['scan_jax']['mean']:>8.1f} μs/iter  │"
    )
    print(
        f"  │   Warp FFI              │ {prod['single_warp']['mean']:>8.0f} μs         │ {prod['scan_warp']['mean']:>8.1f} μs/iter  │"
    )
    print(
        f"  │   → overhead            │ {prod['overhead_single']:>+8.0f} μs         │ {prod['overhead_scan']:>+8.1f} μs/iter  │"
    )
    print(f"  └─────────────────────────┴────────────────────┴────────────────────┘")

    # Interpret results
    print(f"\n  Interpretation:")
    if micro["overhead_single"] > 100:
        print(
            f"    • Micro single-call shows +{micro['overhead_single']:.0f}μs Python callback overhead"
        )
    else:
        print(
            f"    • Micro single-call shows {micro['overhead_single']:.0f}μs — callback overhead is minimal"
        )

    if micro["overhead_scan"] < micro["overhead_single"] * 0.3:
        print(
            f"    • CUDA graph capture ELIMINATES overhead: "
            f"{micro['overhead_single']:.0f}μs → {micro['overhead_scan']:.1f}μs in scan"
        )
        print(
            f"      ({micro['overhead_single']/(max(abs(micro['overhead_scan']),0.1)):.0f}× reduction)"
        )
    elif micro["overhead_scan"] < micro["overhead_single"] * 0.7:
        print(
            f"    • CUDA graph capture REDUCES overhead: "
            f"{micro['overhead_single']:.0f}μs → {micro['overhead_scan']:.1f}μs in scan"
        )
    else:
        print(
            f"    • CUDA graph capture did NOT reduce overhead: "
            f"{micro['overhead_single']:.0f}μs → {micro['overhead_scan']:.1f}μs in scan"
        )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Warp kernel launch overhead & CUDA graph amortization"
    )
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
