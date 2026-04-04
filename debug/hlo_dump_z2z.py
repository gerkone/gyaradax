"""Generate HLO dumps for JAX Z2Z fp32 vs fp64 to analyze performance gap."""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np

# Set up environment before importing jax
os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/hlo_z2z_dump"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from gyaradax.backends._jax import JAXOps, _per_s_z2z, _pack_full_z2z
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum


def setup_test_data(nv=16, nmu=8, ns=4, nkx=64, nky=32, dtype=jnp.complex128):
    """Create test data for Z2Z nonlinear term."""
    key = jax.random.PRNGKey(42)
    real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
    df = jax.random.normal(key, (nv, nmu, ns, nkx, nky), dtype=dtype) * 0.1
    phi = jax.random.normal(key, (ns, nkx, nky), dtype=dtype) * 0.1
    
    # Precompute minimal dict for nonlinear term
    mrad, mphi = nkx * 2, nky * 2  # dealiased grid
    mphiw3 = mphi // 3 * 2  # padded for FFT efficiency
    
    # Mode connectivity - jind maps [mrad, mphiw3] linear index to packed [nkx, nky]
    # Create proper 2D wavenumber grids
    kx_1d = jnp.fft.fftfreq(mrad)[:nkx]  # [nkx]
    ky_1d = jnp.fft.fftfreq(mphi)[:nky]  # [nky]
    kx_2d = kx_1d[:, None] * jnp.ones(nky)[None, :]  # [nkx, nky]
    ky_2d = jnp.ones(nkx)[:, None] * ky_1d[None, :]  # [nkx, nky]
    
    # jind: maps linear index in [mrad, mphiw3] to packed index [0, nkx*nky)
    # For simplicity, use row-major mapping
    jind = jnp.arange(mrad * mphiw3, dtype=jnp.int32).reshape(mrad, mphiw3)
    # Only keep valid indices (within nkx x nky region)
    valid_mask = (jind < nkx * nky).ravel()
    jind_flat = jind.ravel()
    
    # Create sparse jind that only maps active modes
    active_indices = jnp.where(jind_flat < nkx * nky, jind_flat, -1)
    jind_final = jnp.arange(mrad * mphiw3, dtype=jnp.int32)
    
    # Simpler approach: just use the actual packed size
    jind_simple = jnp.arange(nkx, dtype=jnp.int32)[:, None] * jnp.ones(nky, dtype=jnp.int32)[None, :]
    jind_simple = jind_simple.ravel()  # [nkx*nky]
    
    pre = {
        "nl_mrad": mrad,
        "nl_mphi": mphi,
        "nl_mphiw3": mphiw3,
        "nl_jind": jind_simple,
        "nl_fft_scale": 1.0 / (mrad * mphi),
        "nl_kx2d": kx_2d,
        "nl_ky2d": ky_2d,
        "nl_dum_s": jnp.ones(ns, dtype=real_dtype),
        "ixzero": mrad // 2,
        "iyzero": mphi // 2,
        "bessel": jnp.ones((nv, nmu, ns, nkx, nky), dtype=real_dtype),
    }
    
    return df, phi, pre


def run_z2z_benchmark(df, phi, pre, mixed_precision=False):
    """Run Z2Z nonlinear term and return result."""
    ops = JAXOps(pre, use_z2z=True)
    
    result = ops.nonlinear_term_iii(
        df, phi, {},
        efun_sign=1.0,
        fft_prefactor=1.0 + 0j,
        exclude_zero_mode=True,
        mixed_precision=mixed_precision,
    )
    return result


def dump_hlo_comparison():
    """Generate HLO dumps for fp32 and fp64 Z2Z paths."""
    print("=" * 60)
    print("JAX Z2Z HLO Dump Comparison")
    print("=" * 60)
    
    # Test parameters (small for faster compilation)
    nv, nmu, ns, nkx, nky = 16, 8, 4, 64, 32
    
    print(f"\nGrid: nv={nv}, nmu={nmu}, ns={ns}, nkx={nkx}, nky={nky}")
    print(f"Dealiased: mrad={nkx*2}, mphi={nky*2}")
    
    # Create test data
    df64, phi64, pre64 = setup_test_data(nv, nmu, ns, nkx, nky, dtype=jnp.complex128)
    df32, phi32, pre32 = setup_test_data(nv, nmu, ns, nkx, nky, dtype=jnp.complex64)
    
    print("\n" + "-" * 60)
    print("Compiling FP64 path...")
    print("-" * 60)
    
    # Warm up and compile
    _ = run_z2z_benchmark(df64, phi64, pre64, mixed_precision=False)
    jax.block_until_ready(_)
    print("FP64 compilation complete")
    
    print("\n" + "-" * 60)
    print("Compiling FP32 path...")
    print("-" * 60)
    
    _ = run_z2z_benchmark(df32, phi32, pre32, mixed_precision=True)
    jax.block_until_ready(_)
    print("FP32 compilation complete")
    
    # Now create fresh dumps with clean compilation
    print("\n" + "=" * 60)
    print("HLO dumps written to: /tmp/hlo_z2z_dump/")
    print("=" * 60)
    print("\nFiles to compare:")
    print("  - module_*.txt (HLO before optimization)")
    print("  - optimized_module_*.txt (HLO after optimization)")
    print("  - compilation_stats.json (timing info)")
    
    # Run with timing
    print("\n" + "-" * 60)
    print("Running timing comparison...")
    print("-" * 60)
    
    # FP64 timing
    start = time.time()
    for _ in range(10):
        result64 = run_z2z_benchmark(df64, phi64, pre64, mixed_precision=False)
        jax.block_until_ready(result64)
    end = time.time()
    time64 = (end - start) / 10
    print(f"FP64 Z2Z: {time64*1000:.2f} ms per call")
    
    # FP32 timing
    start = time.time()
    for _ in range(10):
        result32 = run_z2z_benchmark(df32, phi32, pre32, mixed_precision=True)
        jax.block_until_ready(result32)
    end = time.time()
    time32 = (end - start) / 10
    print(f"FP32 Z2Z: {time32*1000:.2f} ms per call")
    
    print(f"\nSpeedup (FP32 vs FP64): {time64/time32:.2f}x")
    
    # Also dump the lowered HLO text
    print("\n" + "-" * 60)
    print("Generating lowered HLO text...")
    print("-" * 60)
    
    # Lower and dump FP64
    lowered64 = jax.jit(run_z2z_benchmark).lower(df64, phi64, pre64, False)
    hlo_text64 = lowered64.as_text()
    with open("/tmp/hlo_z2z_fp64.txt", "w") as f:
        f.write(hlo_text64)
    print(f"FP64 HLO text: /tmp/hlo_z2z_fp64.txt ({len(hlo_text64)} chars)")
    
    # Lower and dump FP32
    lowered32 = jax.jit(run_z2z_benchmark).lower(df32, phi32, pre32, True)
    hlo_text32 = lowered32.as_text()
    with open("/tmp/hlo_z2z_fp32.txt", "w") as f:
        f.write(hlo_text32)
    print(f"FP32 HLO text: /tmp/hlo_z2z_fp32.txt ({len(hlo_text32)} chars)")
    
    # Count key operations
    def count_ops(hlo_text, name):
        ops = {
            "fft": hlo_text.count("fft"),
            "ifft": hlo_text.count("inverse-fft"),
            "complex_multiply": hlo_text.count("complex multiply"),
            "multiply": hlo_text.count("multiply"),
            "add": hlo_text.count("add"),
            "concatenate": hlo_text.count("concatenate"),
            "dynamic-slice": hlo_text.count("dynamic-slice"),
            "gather": hlo_text.count("gather"),
        }
        print(f"\n{name} operation counts:")
        for op, count in ops.items():
            if count > 0:
                print(f"  {op}: {count}")
        return ops
    
    count_ops(hlo_text64, "FP64")
    count_ops(hlo_text32, "FP32")
    
    print("\n" + "=" * 60)
    print("Done! Check /tmp/hlo_z2z_dump/ for detailed HLO files")
    print("=" * 60)


if __name__ == "__main__":
    dump_hlo_comparison()
