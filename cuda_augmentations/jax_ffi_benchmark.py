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
parser.add_argument("--slice", type=int, default=0, help="Run only N batches for debugging")
args = parser.parse_args()

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
from gyaradax.solver import nonlinear_term_iii
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum
from gyaradax.types import GKPre
from gyaradax.backends import create_ops

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
        "lto_fft_bracket_v3_ffi": _lib.lto_fft_bracket_v3_ffi,
        "lto_fft_bracket_vz2z_ffi": _lib.lto_fft_bracket_vz2z_ffi,
        "lto_fft_bracket_vz2z_merged_ffi": _lib.lto_fft_bracket_vz2z_merged_ffi,
        "lto_fft_bracket_v4_ffi": _lib.lto_fft_bracket_v4_ffi,
        "lto_fft_bracket_vexp_ffi": _lib.lto_fft_bracket_vexp_ffi,
        "cufft_bracket_ffi": _lib.cufft_bracket_ffi,
        "cufft_graph_bracket_ffi": _lib.cufft_graph_bracket_ffi,
        "cufft_graph_bracket_fp64_ffi": _lib.cufft_graph_bracket_fp64_ffi,
        "cufft_graph_bracket_fp64_direct_ffi": _lib.cufft_graph_bracket_fp64_direct_ffi,
    }

    for name, symbol in targets.items():
        try:
            ffi.register_ffi_target(name, ffi.pycapsule(symbol), platform="CUDA")
        except Exception:
            pass  # ignore re-registration
    return True


# --- FFI Call Wrapper ---
def lto_bracket_ffi_call(df, phi, kx, ky, jind, dum_s, batch, mrad, mphi, nkx, nky, version=0):
    suffixes = {0: "", 1: "_v1", 2: "_v2", 3: "_v3", "exp": "_vexp"}
    suffix = suffixes.get(version, "")
    target_name = f"lto_fft_bracket{suffix}_ffi"

    # Version 3+ uses sparse store callback writing to packed [batch, nkx, nky]
    if version == 3:
        out_shape = (batch, nkx, nky)
    else:
        out_shape = (batch, mrad, (mphi // 2 + 1))

    return ffi.ffi_call(target_name, jax.ShapeDtypeStruct(out_shape, jnp.complex128))(
        df,
        phi,
        kx,
        ky,
        jind,
        dum_s,
        batch=np.int32(batch),
        mrad=np.int32(mrad),
        mphi=np.int32(mphi),
        nkx=np.int32(nkx),
        nky=np.int32(nky),
    )


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
    jind = np.array(pre["nl_jind"])  # packed -> dense map
    nkx, nky = df.shape[-2], df.shape[-1]

    # CRITICAL: Build inverse_jind (dense -> packed map) for the FFI callback
    inverse_jind = np.full(mrad, -1, dtype=np.int32)
    for i_pack, i_dense in enumerate(jind):
        if 0 <= i_dense < mrad:
            inverse_jind[i_dense] = i_pack
    inverse_jind = jnp.array(inverse_jind)

    # 2. Reshape Inputs for FFI
    # If we have multiple species, we must transpose so species is the fastest batch dimension
    # (matches indexing in CUDA: batch_idx % nspec)
    if df.ndim == 6:  # (nsp, nvpar, nmu, ns, nkx, nky)
        df_ffi = df.transpose(1, 2, 3, 0, 4, 5)
        if phi.ndim == 6:
            phi_ffi = phi.transpose(1, 2, 3, 0, 4, 5)
        else:
            phi_ffi = jnp.broadcast_to(phi[None, ...], df.shape).transpose(1, 2, 3, 0, 4, 5)
    else:
        df_ffi = df
        phi_ffi = phi

    full_batch_size = (
        df_ffi.shape[0]
        * df_ffi.shape[1]
        * df_ffi.shape[2]
        * (df_ffi.shape[3] if df_ffi.ndim == 6 else 1)
    )

    if args.slice > 0:
        batch_to_run = min(full_batch_size, args.slice)
        print(f"  [DEBUG] Slicing benchmark to {batch_to_run} batches")
        df_lto = df_ffi.reshape(-1, nkx, nky)[:batch_to_run]
        phi_lto = phi_ffi.reshape(-1, nkx, nky)[:batch_to_run]
    else:
        df_lto = df_ffi.reshape(-1, nkx, nky)
        phi_lto = phi_ffi.reshape(-1, nkx, nky)

    batch_total = df_lto.shape[0]

    if args.debug:
        print("\n  --- Hermitian Symmetry Check ---")
        for name, arr in [("df", df_lto), ("phi", phi_lto)]:
            col0 = np.zeros(mrad, dtype=complex)
            for ip, id in enumerate(jind):
                col0[id] = arr[0, ip, 0]
            err_sym = np.abs(col0[1] - np.conj(col0[134]))
            print(f"  {name} col0 symmetry error (i=1 vs i=134): {err_sym:.2e}")
            print(f"  {name} col0 DC imag part: {np.abs(col0[0].imag):.2e}")
        print("  -------------------------------\n")

    kx_vec = pre["nl_kx2d"][:, 0]
    ky_vec = pre["nl_ky2d"][0, :]
    dum_s = pre["nl_dum_s"]

    print(f"  df shape   : {df.shape}")
    print(f"  phi shape  : {phi.shape}")
    print(f"  FFI batch  : {batch_total}")
    print(f"  Grid       : mrad={mrad}, mphi={mphi}, nkx={nkx}, nky={nky}")

    # 3. JAX Baselines (R2C and Z2Z)
    from gyaradax.backends._jax import JAXOps

    jax_r2c = JAXOps(pre_gk, use_z2z=False)
    jax_z2z = JAXOps(pre_gk, use_z2z=True)

    @jax.jit
    def run_jax_r2c_fp64(d, p):
        return jax_r2c.nonlinear_term_iii(
            d, p, geom, efun_sign=1.0, fft_prefactor=1.0 + 0.0j, mixed_precision=False
        )

    @jax.jit
    def run_jax_z2z_fp64(d, p):
        return jax_z2z.nonlinear_term_iii(
            d, p, geom, efun_sign=1.0, fft_prefactor=1.0 + 0.0j, mixed_precision=False
        )

    @jax.jit
    def run_jax_z2z_fp32(d, p):
        return jax_z2z.nonlinear_term_iii(
            d, p, geom, efun_sign=1.0, fft_prefactor=1.0 + 0.0j, mixed_precision=True
        )

    # 4. Shared Physics & Solver Wrapper
    def apply_physics_wrapper(out_raw, is_lto=True):
        N = mrad * mphi
        if is_lto:
            out_normalized = out_raw / (N * N)
        else:
            out_normalized = out_raw

        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)

        # Match baseline real-space scaling: dum_s * fft_scale * efun_sign * np.real(fft_prefactor)
        # Note: dum_s was already applied in CUDA, so we only need the rest
        nl_half = (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out_normalized
        nl = unpack_half_spectrum(nl_half, jind, nky)
        ixzero, iyzero = pre["ixzero"], pre["iyzero"]
        nl_masked = nl.at[:, ixzero, iyzero].set(0.0)
        return nl_masked.reshape(-1, nkx, nky)

    @jax.jit
    def run_lto_v2(d, p):
        p_lto = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        out = lto_bracket_ffi_call(
            d, p_lto, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 2
        )
        return apply_physics_wrapper(out, is_lto=True)

    @jax.jit
    def run_lto_v3(d, p):
        p_lto = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        out = lto_bracket_ffi_call(
            d, p_lto, kx_vec, ky_vec, inverse_jind, dum_s, batch_total, mrad, mphi, nkx, nky, 3
        )
        # v3 uses sparse store, but still needs division by N^2 in apply_physics_wrapper
        # wait! v3 returns (batch, nkx, nky).
        # apply_physics_wrapper expects (batch, mrad, mphi_half).
        # Ah! I need a different wrapper for sparse outputs.
        N = mrad * mphi
        out_normalized = out / (N * N)
        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)

        nl = (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out_normalized
        ixzero, iyzero = pre["ixzero"], pre["iyzero"]
        return nl.at[:, ixzero, iyzero].set(0.0).reshape(-1, nkx, nky)

    @jax.jit
    def run_lto_vz2z(d, p):
        p_lto = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        out = ffi.ffi_call(
            "lto_fft_bracket_vz2z_ffi",
            jax.ShapeDtypeStruct((batch_total, nkx, nky), jnp.complex128),
        )(
            d,
            p_lto,
            kx_vec,
            ky_vec,
            inverse_jind,
            dum_s,
            batch=np.int32(batch_total),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
        )
        # vZ2Z is now normalized to 1/N^2 in CUDA
        out_normalized = out
        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)

        nl = (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out_normalized
        ixzero, iyzero = pre["ixzero"], pre["iyzero"]
        return nl.at[:, ixzero, iyzero].set(0.0).reshape(-1, nkx, nky)

    @jax.jit
    def run_lto_vz2z_merged(d, p):
        p_lto = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        out = ffi.ffi_call(
            "lto_fft_bracket_vz2z_merged_ffi",
            jax.ShapeDtypeStruct((batch_total, nkx, nky), jnp.complex128),
        )(
            d,
            p_lto,
            kx_vec,
            ky_vec,
            inverse_jind,
            dum_s,
            batch=np.int32(batch_total),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
        )
        out_normalized = out
        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)

        nl = (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out_normalized
        ixzero, iyzero = pre["ixzero"], pre["iyzero"]
        return nl.at[:, ixzero, iyzero].set(0.0).reshape(-1, nkx, nky)

    @jax.jit
    def run_lto_v5_graph(d, p):
        # nspec = ns (field-line segments)
        nspec_val = dum_s.shape[0]
        # phi at natural shape: (nmu*ns, nkx, nky) — NOT duplicated across nvpar
        # bessel is (1, nmu, ns, nkx, nky) or (nmu, ns, nkx, nky)
        p_phi = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        # p_phi shape: (nmu*ns, nkx, nky) = (128, 85, 32) — 32x smaller than d
        jind_ffi = jnp.array(jind, dtype=jnp.int32)

        out = ffi.ffi_call(
            "cufft_graph_bracket_ffi", jax.ShapeDtypeStruct((batch_total, nkx, nky), jnp.complex128)
        )(
            d,
            p_phi,
            kx_vec,
            ky_vec,
            jind_ffi,
            inverse_jind,
            dum_s,
            batch=np.int32(batch_total // nspec_val),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
            nspec=np.int32(nspec_val),
            ixzero=np.int32(pre["ixzero"]),
            iyzero=np.int32(pre["iyzero"]),
        )

        # Assembly kernel applies 1/N^2 and dum_s; unpack kernel applies zero-mode masking.
        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)
        return (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out

    @jax.jit
    def run_cufft_graph_fp64(d, p):
        nspec_val = dum_s.shape[0]
        p_phi = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        jind_ffi = jnp.array(jind, dtype=jnp.int32)

        out = ffi.ffi_call(
            "cufft_graph_bracket_fp64_ffi", jax.ShapeDtypeStruct((batch_total, nkx, nky), jnp.complex128)
        )(
            d,
            p_phi,
            kx_vec,
            ky_vec,
            jind_ffi,
            inverse_jind,
            dum_s,
            batch=np.int32(batch_total // nspec_val),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
            nspec=np.int32(nspec_val),
            ixzero=np.int32(pre["ixzero"]),
            iyzero=np.int32(pre["iyzero"]),
        )

        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)
        return (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out

    @jax.jit
    def run_cufft_graph_fp64_direct(d, p):
        nspec_val = dum_s.shape[0]
        p_phi = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        jind_ffi = jnp.array(jind, dtype=jnp.int32)

        out = ffi.ffi_call(
            "cufft_graph_bracket_fp64_direct_ffi",
            jax.ShapeDtypeStruct((batch_total, nkx, nky), jnp.complex128),
        )(
            d,
            p_phi,
            kx_vec,
            ky_vec,
            jind_ffi,
            inverse_jind,
            dum_s,
            batch=np.int32(batch_total // nspec_val),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
            nspec=np.int32(nspec_val),
            ixzero=np.int32(pre["ixzero"]),
            iyzero=np.int32(pre["iyzero"]),
        )

        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)
        return (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out

    @jax.jit
    def run_lto_v4(d, p):
        p_lto = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        out = ffi.ffi_call(
            "lto_fft_bracket_v4_ffi", jax.ShapeDtypeStruct((batch_total, nkx, nky), jnp.complex128)
        )(
            d,
            p_lto,
            kx_vec,
            ky_vec,
            inverse_jind,
            dum_s,
            batch=np.int32(batch_total),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
        )
        out_normalized = out
        fft_prefactor = pre.get("nl_fft_prefactor", 1.0 + 0.0j)
        fft_scale = pre["nl_fft_scale"]
        efun = pre.get("efun", 1.0)
        efun_sign = jnp.where(efun > 0, 1.0, -1.0)

        nl = (fft_scale * efun_sign * jnp.real(fft_prefactor)) * out_normalized
        ixzero, iyzero = pre["ixzero"], pre["iyzero"]
        return nl.at[:, ixzero, iyzero].set(0.0).reshape(-1, nkx, nky)

    @jax.jit
    def run_cufft_non_lto(d, p):
        # Match exactly the same structure as run_lto_v2:
        # gyroaverage phi, pack 4 derivative spectra, call FFI, unpack
        p_gyro = (pre["bessel"] * p.reshape(1, 1, -1, nkx, nky)).reshape(-1, nkx, nky)
        d_flat = d.reshape(-1, nkx, nky)

        # Compute Fourier-space derivatives — pack first then multiply (matches JAX baseline)
        jind_jnp = jnp.array(jind)
        mphi_half = mphi // 2 + 1
        ikx = 1j * pack_half_spectrum(
            jnp.broadcast_to(kx_vec[:, None], (nkx, nky)), jind_jnp, mrad, mphi_half
        )
        iky = 1j * pack_half_spectrum(
            jnp.broadcast_to(ky_vec[None, :], (nkx, nky)), jind_jnp, mrad, mphi_half
        )

        phi_y_dense = iky[None, ...] * pack_half_spectrum(p_gyro, jind_jnp, mrad, mphi_half)
        f_x_dense = ikx[None, ...] * pack_half_spectrum(d_flat, jind_jnp, mrad, mphi_half)
        phi_x_dense = ikx[None, ...] * pack_half_spectrum(p_gyro, jind_jnp, mrad, mphi_half)
        f_y_dense = iky[None, ...] * pack_half_spectrum(d_flat, jind_jnp, mrad, mphi_half)

        # dum_s is (ns,); nspec = ns; kernel cycles batch_idx % nspec → correct for (nvpar,nmu,ns) layout
        ns_val = dum_s.shape[0]
        out = ffi.ffi_call(
            "cufft_bracket_ffi",
            jax.ShapeDtypeStruct((batch_total, mrad, mphi_half), jnp.complex128),
        )(
            phi_y_dense,
            f_x_dense,
            phi_x_dense,
            f_y_dense,
            dum_s,
            batch=np.int32(batch_total),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nspec=np.int32(ns_val),
        )

        # Kernel applied inv_n2=1/N^2; apply_physics_wrapper(is_lto=False) skips /N^2
        # then multiplies by fft_scale=N → net = (1/N^2) * N = 1/N, matching JAX irfft norm
        return apply_physics_wrapper(out, is_lto=False)



    variants = [
        ("JAX R2C fp64 baseline", run_jax_r2c_fp64, (df, phi)),
        ("JAX Z2Z fp64", run_jax_z2z_fp64, (df, phi)),
        ("JAX Z2Z fp32", run_jax_z2z_fp32, (df, phi)),
        ("LTO v2 (Standard)", run_lto_v2, (df_lto, phi_lto)),
        ("LTO vZ2Z (Optimized)", run_lto_vz2z, (df_lto, phi_lto)),
        ("LTO vZ2Z-merged", run_lto_vz2z_merged, (df_lto, phi_lto)),
        ("LTO v5 Graph", run_lto_v5_graph, (df_lto, phi_lto)),
        ("cuFFT Graph FP64", run_cufft_graph_fp64, (df_lto, phi_lto)),
        ("cuFFT Graph FP64 direct", run_cufft_graph_fp64_direct, (df_lto, phi_lto)),
        ("LTO v4 (Graph)", run_lto_v4, (df_lto, phi_lto)),
        ("cuFFT (non-LTO fused)", run_cufft_non_lto, (df_lto, phi_lto)),
    ]

    results = {}
    errors = {}
    try:
        ref_out = run_jax_r2c_fp64(df, phi)
        # Use batch_to_run if sliced, else full_batch_size
        limit = batch_to_run if args.slice > 0 else full_batch_size
        ref_flat = ref_out.reshape(-1, nkx, nky)[:limit]
        if args.debug:
            r_off, k_off = 0, 10
            print(f"  [DEBUG] Slice at rad {r_off}, k_off {k_off}:")
            print(f"  {variants[0][0]:24s}: {ref_flat[0, r_off, k_off:k_off+5]}")
    except Exception as e:
        print(f"  JAX Reference failed: {e}")
        ref_out = None

    for name, fn, inputs in variants:
        try:
            out = fn(*inputs)
            out_flat = out.reshape(-1, nkx, nky)
            if args.debug and ref_out is not None:
                print(f"  {name:24s}: {out_flat[0, r_off, k_off:k_off+5]}")

            if ref_out is not None:
                N_acc = 1000
                rel_err = jnp.linalg.norm(
                    (out_flat.ravel() - ref_flat.ravel())[:N_acc]
                ) / jnp.linalg.norm(ref_flat.ravel()[:N_acc])
                errors[name] = float(rel_err)

            mean_ms, std_ms = BenchTimer(lambda: fn(*inputs).block_until_ready(), n_trials=100).run()
            results[name] = (mean_ms, std_ms)
        except Exception as e:
            print(f"  {name:24s}: FAILED ({e})")

    print(f"\n{'='*110}")
    print(
        f"{'Variant':30s} | {'Time (ms)':20s} | {'Speedup':10s} | {'Rel L2':10s} | {'Throughput'}"
    )
    print(f"{'-'*30} | {'-'*20} | {'-'*10} | {'-'*10} | {'-'*15}")
    base_time, _ = results.get("JAX R2C fp64 baseline", (1.0, 0.0))
    hbm_bytes = 6.11e9

    for name, (t, std) in results.items():
        speedup = base_time / t if t > 0 else 0.0
        bw = (hbm_bytes / 1e9) / (t / 1e3) if t > 0 else 0.0
        rel_err = errors.get(name, 0.0)
        time_str = f"{t:7.3f} ± {std:5.3f}"
        print(f"{name:30s} | {time_str:20s} | {speedup:10.2f}x | {rel_err:10.2e} | {bw:8.1f} GB/s")
    print(f"\n{'='*110}")


if __name__ == "__main__":
    main()
