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

    # 4. Standard cuFFT Variant (Non-LTO)
    # This variant requires Python-side packing of gradients
    @jax.jit
    def run_cufft_standard(d, p):
        ikx_vec = 1j * kx_vec
        iky_vec = 1j * ky_vec
        # d is (4096, 85, 32), p is (16, 85, 32)
        # nspec=16, nv_mu=256
        ns = p.shape[0]
        
        # Physics: Bessel Gyro-averaging
        # gyro_phi = bessel_s * phi_s
        bessel = pre["bessel"] # typically (nv, nmu, ns, nkx, nky)
        # Broadcast phi (ns, nkx, nky) to match bessel
        ns = p.shape[0]
        p_expanded = p.reshape(1, 1, ns, nkx, nky)
        gyro_phi_k = bessel * p_expanded
        
        # Now gyro_phi_k is (nv, nmu, ns, nkx, nky) -> reshape to (-1, ns, nkx, nky)
        gyro_phi_k_flat = gyro_phi_k.reshape(-1, ns, nkx, nky)
        d_expanded = d.reshape(-1, ns, nkx, nky)
        
        # Pack Gradients
        # We process each (nv*nmu) "velocity batch" of (ns, nkx, nky)
        phi_y_k = (iky_vec[None, None, None, :] * gyro_phi_k_flat)
        f_x_k   = (ikx_vec[None, None, :, None] * d_expanded)
        phi_x_k = (ikx_vec[None, None, :, None] * gyro_phi_k_flat)
        f_y_k   = (iky_vec[None, None, None, :] * d_expanded)
        
        # Dense packing (cufft_bracket.cu expects dense rows)
        pk_phi_y = pack_half_spectrum(phi_y_k, jind, mrad, pre["nl_mphiw3"])
        pk_f_x   = pack_half_spectrum(f_x_k,   jind, mrad, pre["nl_mphiw3"])
        pk_phi_x = pack_half_spectrum(phi_x_k, jind, mrad, pre["nl_mphiw3"])
        pk_f_y   = pack_half_spectrum(f_y_k,   jind, mrad, pre["nl_mphiw3"])
        
        # Scale: dum_s * efun_sign * fft_scale
        # In FFI, we only use dum_s_eff for the bracket logic. 
        # But real solver applies efun_sign * dum * irfft2(...)
        # And then applies fft_prefactor * fft_scale to the final rfft2 output.
        efun_sign = 1.0
        dum_eff = efun_sign * dum_s 
        
        # The FFI returns (batch, mrad, mphiw3) spectral result
        out_raw = ffi.ffi_call(
            "cufft_bracket_ffi",
            jax.ShapeDtypeStruct((batch_total, mrad, pre["nl_mphiw3"]), jnp.complex128)
        )(pk_phi_y.reshape(-1, mrad, pre["nl_mphiw3"]), 
          pk_f_x  .reshape(-1, mrad, pre["nl_mphiw3"]), 
          pk_phi_x.reshape(-1, mrad, pre["nl_mphiw3"]), 
          pk_f_y  .reshape(-1, mrad, pre["nl_mphiw3"]), 
          dum_eff,
          batch=np.int32(batch_total), mrad=np.int32(mrad), mphi=np.int32(mphi), nspec=np.int32(ns))
        
        # Phase Mapping and Normalization Correction
        # The FFI already includes 1/N^2 in the real-space fused kernel.
        # This matches the JAX irfft2(backward) 1/N scaling for each of the two factors 
        # (resulting in 1/N^2 in real space) and the final rfft2(backward) 1.0 scaling.
        
        # Apply final phase mapping from solver
        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        nl_half = (fft_prefactor * fft_scale) * out_raw
        
        # Unpack result back to (batch, nkx, nky)
        nl = unpack_half_spectrum(nl_half, jind, nky)
        
        # Broad Zero-mode exclusion (per solver)
        ixzero, iyzero = pre["ixzero"], pre["iyzero"]
        nl_masked = nl.at[:, ixzero, iyzero].set(0.0)
        
        return nl_masked.reshape(-1, nkx, nky)

    variants = [
        ("JAX FP64 baseline",  run_jax_fp64,      (df, phi)),
        ("JAX Mixed baseline", run_jax_mixed,     (df, phi)),
        ("cuFFT FFI (std)",    run_cufft_standard,(df_lto, phi_lto)),
        ("LTO cuFFT v0",       run_lto_v0,        (df_lto, phi_lto)),
        ("LTO cuFFT v1",       run_lto_v1,        (df_lto, phi_lto)),
        ("LTO cuFFT v2",       run_lto_v2,        (df_lto, phi_lto)),
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
