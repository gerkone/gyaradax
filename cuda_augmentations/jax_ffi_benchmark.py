#!/usr/bin/env python3
import argparse
import os
import sys
import time
import ctypes
from pathlib import Path

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
# Use a reasonable preallocation limit for a 20GB MIG
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

# --- Project Imports ---
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "solver_components_benchmarks"))

from common import load_setup, BenchTimer, DEFAULT_BW_GBS, DEFAULT_FP64_TFLOPS, DEVICE_MODEL
from gyaradax.solver import nonlinear_term_iii, GKPre

# --- Enable X64 ---
jax.config.update("jax_enable_x64", True)

# --- FFI Registration ---
def register_ffi():
    lib_path = Path(__file__).parent / "liblto_bracket.so"
    if not lib_path.exists():
        lib_path = Path(__file__).parent / "build" / "liblto_bracket.so"
    
    if not lib_path.exists():
        print(f"  [WARN] liblto_bracket.so not found at {lib_path}. Please build it first.")
        return False

    _lib = ctypes.cdll.LoadLibrary(str(lib_path))
    
    targets = {
        "lto_fft_bracket_ffi": _lib.lto_fft_bracket_ffi,
        "lto_fft_bracket_v1_ffi": _lib.lto_fft_bracket_v1_ffi,
        "lto_fft_bracket_v2_ffi": _lib.lto_fft_bracket_v2_ffi,
    }
    
    for name, symbol in targets.items():
        try:
            ffi.register_ffi_target(name, ffi.pycapsule(symbol), platform="CUDA")
        except Exception:
            pass # ignore re-registration
    return True

# --- FFI Call Wrapper ---
def lto_bracket_ffi_call(df, phi, kx, ky, jind, dum_s, batch, mrad, mphi, nkx, nky, version=0):
    suffix = "" if version == 0 else (f"_v1" if version == 1 else "_v2")
    target_name = f"lto_fft_bracket{suffix}_ffi"
    
    # CRITICAL: The output buffer must match the total batch size (e.g. 4096)
    return ffi.ffi_call(
        target_name,
        jax.ShapeDtypeStruct((batch, nkx, nky), jnp.complex128)
    )(df, phi, kx, ky, jind, dum_s, 
      batch=np.int32(batch), mrad=np.int32(mrad), mphi=np.int32(mphi), 
      nkx=np.int32(nkx), nky=np.int32(nky))

# --- Main Benchmark ---
def main():
    print(f"\n{'='*80}")
    print(f"LTO Bracket Production Baseline")
    print(f"{'='*80}")
    print(f"  device   : {DEVICE_MODEL}")
    print(f"  peak BW  : {DEFAULT_BW_GBS:.0f} GB/s")
    print(f"  peak FP64: {DEFAULT_FP64_TFLOPS:.1f} TFLOP/s")

    if not register_ffi():
        return

    # 1. Load Real Solver State
    config_file = args.config
    print(f"  Loading geometry from {config_file}...")
    df, phi, geom, params, pre = load_setup(config_file)
    
    pre_gk = GKPre(pre)
    mrad, mphi = pre["nl_mrad"], pre["nl_mphi"]
    jind = np.array(pre["nl_jind"]) # packed -> dense map
    nkx, nky = df.shape[-2], df.shape[-1]
    
    # CRITICAL: Build inverse_jind (dense -> packed map) for the FFI callback
    inverse_jind = np.full(mrad, -1, dtype=np.int32)
    for i_pack, i_dense in enumerate(jind):
        if 0 <= i_dense < mrad:
            inverse_jind[i_dense] = i_pack
    inverse_jind = jnp.array(inverse_jind)

    # 2. Reshape Inputs for FFI
    # JAX shape: (nv, nmu, ns, nkx, nky) -> (4096, 85, 32)
    df_lto = df.reshape(-1, nkx, nky)
    # phi is typically (ns, nkx, nky) -> (16, 85, 32)
    phi_lto = phi.reshape(-1, nkx, nky) 
    batch_total = df_lto.shape[0]
    
    kx_vec = pre["nl_kx2d"][:, 0]
    ky_vec = pre["nl_ky2d"][0, :]
    dum_s = pre["nl_dum_s"]
    
    print(f"  df shape   : {df.shape}")
    print(f"  phi shape  : {phi.shape}")
    print(f"  FFI batch  : {batch_total}")
    print(f"  Grid       : mrad={mrad}, mphi={mphi}, nkx={nkx}, nky={nky}")

    # 3. Define Variants
    @jax.jit
    def run_jax_fp64(d, p):
        return nonlinear_term_iii(d, p, geom, pre_gk, mixed_precision=False)

    @jax.jit
    def run_jax_mixed(d, p):
        return nonlinear_term_iii(d, p, geom, pre_gk, mixed_precision=True)

    @jax.jit
    def run_lto_v0(d, p):
        return lto_bracket_ffi_call(d, p, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 0)

    @jax.jit
    def run_lto_v1(d, p):
        return lto_bracket_ffi_call(d, p, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 1)

    @jax.jit
    def run_lto_v2(d, p):
        return lto_bracket_ffi_call(d, p, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 2)

    variants = [
        ("JAX FP64 baseline", run_jax_fp64, (df, phi)),
        ("JAX Mixed baseline", run_jax_mixed, (df, phi)),
        ("LTO cuFFT v0",       run_lto_v0,    (df_lto, phi_lto)),
        ("LTO cuFFT v1",       run_lto_v1,    (df_lto, phi_lto)),
        ("LTO cuFFT v2",       run_lto_v2,    (df_lto, phi_lto)),
    ]

    results = {}
    print(f"\n[DEBUG] Slice of output (batch 0, rad 0, phi 0:5):")
    
    # 4. Calibration & Accuracy
    try:
        ref_out = run_jax_fp64(df, phi)
        ref_flat = ref_out.reshape(-1, nkx, nky)
        print(f"  {variants[0][0]:24s}: {ref_flat[0,0,:5]}")
    except Exception as e:
        print(f"  JAX Reference failed: {e}")
        ref_out = None

    for name, fn, inputs in variants:
        try:
            # Warmup & Accuracy Check
            out = fn(*inputs)
            out_flat = out.reshape(-1, nkx, nky)
            print(f"  {name:24s}: {out_flat[0,0,:5]}")
            
            if ref_out is not None:
                # Compare a subset to avoid slow norms on large arrays
                N = 1000
                rel_err = jnp.linalg.norm((out_flat.ravel() - ref_flat.ravel())[:N]) / jnp.linalg.norm(ref_flat.ravel()[:N])
                print(f"    accuracy: rel_l2 = {rel_err:.3e}")
            
            # Timing
            mean_ms, std_ms = BenchTimer(lambda: fn(*inputs).block_until_ready()).run()
            results[name] = mean_ms
        except Exception as e:
            print(f"  {name:24s}: FAILED ({e})")

    # 5. Final Speedup Table
    print(f"\n{'='*80}")
    print(f"{'Variant':30s} | {'Time (ms)':12s} | {'Speedup':10s} | {'Throughput'}")
    print(f"{'-'*30} | {'-'*12} | {'-'*10} | {'-'*15}")
    base_time = results.get("JAX FP64 baseline", 1.0)
    hbm_bytes = 6.11e9 # Estimated traffic for batch=4096

    for name, t in results.items():
        speedup = base_time / t if t > 0 else 0.0
        bw = (hbm_bytes / 1e9) / (t / 1e3) if t > 0 else 0.0
        print(f"{name:30s} | {t:12.3f} | {speedup:10.2f}x | {bw:8.1f} GB/s")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
