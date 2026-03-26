#!/usr/bin/env python3
import argparse
import os
import sys
import time
import ctypes
from pathlib import Path

# --- Argument Parsing ---
root = Path(__file__).parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--config", type=str, default=str(root / "configs" / "iteration_13.yaml"))
parser.add_argument("--debug", action="store_true", help="Print debug slices and accuracy metrics")
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
from gyaradax.solver import nonlinear_term_iii, GKPre, pack_half_spectrum, unpack_half_spectrum

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
        "lto_fft_bracket_vexp_ffi": _lib.lto_fft_bracket_vexp_ffi,
        "cufft_bracket_ffi": _lib.cufft_bracket_ffi,
    }
    
    for name, symbol in targets.items():
        try:
            ffi.register_ffi_target(name, ffi.pycapsule(symbol), platform="CUDA")
        except Exception:
            pass # ignore re-registration
    return True

# --- FFI Call Wrapper ---
def lto_bracket_ffi_call(df, phi, kx, ky, jind, dum_s, batch, mrad, mphi, nkx, nky, version=0):
    suffixes = {0: "", 1: "_v1", 2: "_v2", "exp": "_vexp"}
    suffix = suffixes.get(version, "")
    target_name = f"lto_fft_bracket{suffix}_ffi"
    
    # All variants (v0, v1, v2) produce a D2Z half-spectrum output: (batch, mrad, mphi//2+1).
    # v2 uses Z2Z internally but the final step is still D2Z.
    return ffi.ffi_call(
        target_name,
        jax.ShapeDtypeStruct((batch, mrad, (mphi // 2 + 1)), jnp.complex128)
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

    # 3. Reference Baselines
    @jax.jit
    def run_jax_fp64(d, p):
        return nonlinear_term_iii(d, p, geom, pre_gk, mixed_precision=False)

    @jax.jit
    def run_jax_mixed(d, p):
        return nonlinear_term_iii(d, p, geom, pre_gk, mixed_precision=True)

    # 4. Shared Physics & Solver Wrapper
    def apply_physics_wrapper(out_raw, is_lto=True):
        # Normalization and Phase Mapping
        # JAX norm="backward" round trip is 1/N.
        # cuFFT round trip is N.
        # LTO v0, v1, v2 kernels are CURRENTLY unscaled (scale=1.0).
        # Standard FFI kernel is ALREADY scaled by 1/N^2.
        N = mrad * mphi
        if is_lto:
            # LTO needs 1/N^2 to match JAX 1/N (N * 1/N^2 = 1/N)
            out_normalized = out_raw / (N * N)
        else:
            # Standard FFI is already 1/N (scaled by 1/N^2 in C++ kernel)
            out_normalized = out_raw
        
        # Apply final phase mapping from solver
        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        nl_half = (fft_prefactor * fft_scale) * out_normalized
        
        # Unpack result back to (batch, nkx, nky)
        nl = unpack_half_spectrum(nl_half, jind, nky)
        
        # Broad Zero-mode exclusion (per solver)
        ixzero, iyzero = pre["ixzero"], pre["iyzero"]
        nl_masked = nl.at[:, ixzero, iyzero].set(0.0)
        
        return nl_masked.reshape(-1, nkx, nky)

    @jax.jit
    def run_lto_v0(d, p):
        # Physics: Bessel Gyro-averaging
        bessel = pre["bessel"] 
        gyro_phi_k = bessel * p.reshape(1, 1, -1, nkx, nky)
        p_lto = gyro_phi_k.reshape(-1, nkx, nky)
        
        out = lto_bracket_ffi_call(d, p_lto, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 0)
        return apply_physics_wrapper(out, is_lto=True)

    @jax.jit
    def run_lto_v1(d, p):
        # Physics: Bessel Gyro-averaging
        bessel = pre["bessel"] 
        gyro_phi_k = bessel * p.reshape(1, 1, -1, nkx, nky)
        p_lto = gyro_phi_k.reshape(-1, nkx, nky)
        
        out = lto_bracket_ffi_call(d, p_lto, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 1)
        return apply_physics_wrapper(out, is_lto=True)

    @jax.jit
    def run_lto_v2(d, p):
        # Physics: Bessel Gyro-averaging
        bessel = pre["bessel"] 
        gyro_phi_k = bessel * p.reshape(1, 1, -1, nkx, nky)
        p_lto = gyro_phi_k.reshape(-1, nkx, nky)
        
        out = lto_bracket_ffi_call(d, p_lto, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 2)
        return apply_physics_wrapper(out, is_lto=True)

    @jax.jit
    def run_lto_vexp(d, p):
        # Physics: Bessel Gyro-averaging
        bessel = pre["bessel"] 
        gyro_phi_k = bessel * p.reshape(1, 1, -1, nkx, nky)
        p_lto = gyro_phi_k.reshape(-1, nkx, nky)
        
        out = lto_bracket_ffi_call(d, p_lto, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, "exp")
        return apply_physics_wrapper(out, is_lto=True)

    # 4. Standard cuFFT Variant (Non-LTO)
    @jax.jit
    def run_cufft_standard(d, p):
        ikx_vec, iky_vec = 1j * kx_vec, 1j * ky_vec
        ns = p.shape[0]
        
        # Physics: Bessel Gyro-averaging
        bessel = pre["bessel"]
        gyro_phi_k = bessel * p.reshape(1, 1, ns, nkx, nky)
        gyro_phi_k_flat = gyro_phi_k.reshape(-1, ns, nkx, nky)
        d_expanded = d.reshape(-1, ns, nkx, nky)
        
        # Pack Gradients
        pk_phi_y = pack_half_spectrum(iky_vec[None, None, None, :] * gyro_phi_k_flat, jind, mrad, pre["nl_mphiw3"])
        pk_f_x   = pack_half_spectrum(ikx_vec[None, None, :, None] * d_expanded,      jind, mrad, pre["nl_mphiw3"])
        pk_phi_x = pack_half_spectrum(ikx_vec[None, None, :, None] * gyro_phi_k_flat, jind, mrad, pre["nl_mphiw3"])
        pk_f_y   = pack_half_spectrum(iky_vec[None, None, None, :] * d_expanded,      jind, mrad, pre["nl_mphiw3"])
        
        out_raw = ffi.ffi_call(
            "cufft_bracket_ffi",
            jax.ShapeDtypeStruct((batch_total, mrad, pre["nl_mphiw3"]), jnp.complex128)
        )(pk_phi_y.reshape(-1, mrad, pre["nl_mphiw3"]), 
          pk_f_x  .reshape(-1, mrad, pre["nl_mphiw3"]), 
          pk_phi_x.reshape(-1, mrad, pre["nl_mphiw3"]), 
          pk_f_y  .reshape(-1, mrad, pre["nl_mphiw3"]), 
          dum_s, # kernel in cufft_bracket side-loads dum_s scaling
          batch=np.int32(batch_total), mrad=np.int32(mrad), mphi=np.int32(mphi), nspec=np.int32(ns))
        
        return apply_physics_wrapper(out_raw, is_lto=False)

    variants = [
        ("JAX FP64 baseline",  run_jax_fp64,      (df, phi)),
        ("JAX Mixed baseline", run_jax_mixed,     (df, phi)),
        ("cuFFT FFI (std)",    run_cufft_standard,(df_lto, phi_lto)),
        ("LTO cuFFT v0",       run_lto_v0,        (df_lto, phi_lto)),
        ("LTO cuFFT v1",       run_lto_v1,        (df_lto, phi_lto)),
        ("LTO cuFFT v2 (Fused)",run_lto_v2,       (df_lto, phi_lto)),
        ("LTO exp (Z2Z)",      run_lto_vexp,      (df_lto, phi_lto)),
    ]

    results = {}
    errors = {}
    if args.debug:
        print(f"\n[DEBUG] Slice of output (batch 0, rad 0, phi 0:5):")
    
    # 4. Calibration & Accuracy
    try:
        ref_out = run_jax_fp64(df, phi)
        ref_flat = ref_out.reshape(-1, nkx, nky)
        if args.debug:
            print(f"  {variants[0][0]:24s}: {ref_flat[0,0,:5]}")
    except Exception as e:
        print(f"  JAX Reference failed: {e}")
        ref_out = None

    for name, fn, inputs in variants:
        try:
            # Warmup & Accuracy Check
            out = fn(*inputs)
            out_flat = out.reshape(-1, nkx, nky)
            if args.debug:
                print(f"  {name:24s}: {out_flat[0,0,:5]}")
            
            if ref_out is not None:
                # Compare a subset to avoid slow norms on large arrays
                N = 1000
                rel_err = jnp.linalg.norm((out_flat.ravel() - ref_flat.ravel())[:N]) / jnp.linalg.norm(ref_flat.ravel()[:N])
                if args.debug:
                    print(f"    accuracy: rel_l2 = {rel_err:.3e}")
                errors[name] = float(rel_err)
            
            # Timing
            mean_ms, std_ms = BenchTimer(lambda: fn(*inputs).block_until_ready()).run()
            results[name] = mean_ms
        except Exception as e:
            print(f"  {name:24s}: FAILED ({e})")

    # 5. Final Speedup Table
    print(f"\n{'='*95}")
    print(f"{'Variant':30s} | {'Time (ms)':12s} | {'Speedup':10s} | {'Rel L2':10s} | {'Throughput'}")
    print(f"{'-'*30} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*15}")
    base_time = results.get("JAX FP64 baseline", 1.0)
    hbm_bytes = 6.11e9 # Estimated traffic for batch=4096

    for name, t in results.items():
        speedup = base_time / t if t > 0 else 0.0
        bw = (hbm_bytes / 1e9) / (t / 1e3) if t > 0 else 0.0
        rel_err = errors.get(name, 0.0)
        print(f"{name:30s} | {t:12.3f} | {speedup:10.2f}x | {rel_err:10.2e} | {bw:8.1f} GB/s")
    print(f"\n{'='*95}")

if __name__ == "__main__":
    main()
