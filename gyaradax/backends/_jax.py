"""JAX backend for solver operations.

Implements the nonlinear ExB bracket (term III) and stencil operations
using pure JAX. This is the direct port of GKW's non_linear_terms.F90
and linear_terms.f90 stencil application.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from gyaradax import stencils
from gyaradax.backends.ops import SolverOps
from gyaradax.params import GKParams
from gyaradax.types import GKPre
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum


@jax.tree_util.register_pytree_node_class
class JAXOps(SolverOps):
    """JAX implementation of solver operations.

    Supports both R2C (real-to-complex) and Z2Z (complex-to-complex) FFTs
    via the use_z2z flag. Mixed precision (FP32 FFTs) is controlled by
    the mixed_precision flag.
    """

    def __init__(self, pre: GKPre, use_z2z: bool = False, mixed_precision: bool = True):
        super().__init__(pre, use_z2z, mixed_precision)

    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        """Apply 5-point vpar stencil (shifts -2..+2) with zero boundary."""
        nv = field.shape[0]
        out = jnp.zeros_like(field)
        for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):
            idx = jnp.clip(jnp.arange(nv, dtype=jnp.int32) + s, 0, nv - 1)
            valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
            valid_mask = valid[:, None, None, None, None]
            shifted = jnp.take(field, idx, axis=0)
            out = out + c * jnp.where(valid_mask, shifted, 0.0)
        return out

    def _apply_vpar_dual(
        self, field: jnp.ndarray, coeffs_d1, coeffs_d4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply d1 and d4 vpar stencils."""
        nv = field.shape[0]
        out_d1 = jnp.zeros_like(field)
        out_d4 = jnp.zeros_like(field)
        for c1, c4, s in zip(coeffs_d1, coeffs_d4, (-2, -1, 0, 1, 2)):
            idx = jnp.clip(jnp.arange(nv, dtype=jnp.int32) + s, 0, nv - 1)
            valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
            valid_mask = valid[:, None, None, None, None]
            shifted = jnp.take(field, idx, axis=0)
            out_d1 = out_d1 + c1 * jnp.where(valid_mask, shifted, 0.0)
            out_d4 = out_d4 + c4 * jnp.where(valid_mask, shifted, 0.0)
        return out_d1, out_d4

    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Apply 9-point parallel stencil using precomputed shift maps."""
        out = jnp.zeros_like(field)
        nky = field.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
        for i in range(9):
            s_map = self.pre["s_shift"][i]
            kx_map = self.pre["kx_shift"][i]
            valid = self.pre["valid_shift"][i]
            valid_mask = valid[None, None, :, :, :]
            shifted = jnp.where(valid_mask, field[:, :, s_map, kx_map, ky_idx], 0.0)
            out = out + coeffs[i] * shifted
        return out

    def _apply_parallel_dual(
        self,
        field1: jnp.ndarray,
        field2: jnp.ndarray,
        coeffs1: jnp.ndarray,
        coeffs2: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply 9-point parallel stencils to two fields."""
        out1 = jnp.zeros_like(field1)
        out2 = jnp.zeros_like(field2)
        nky = field1.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
        for i in range(9):
            s_map = self.pre["s_shift"][i]
            kx_map = self.pre["kx_shift"][i]
            valid = self.pre["valid_shift"][i]
            valid_mask = valid[None, None, :, :, :]
            shifted1 = jnp.where(valid_mask, field1[:, :, s_map, kx_map, ky_idx], 0.0)
            shifted2 = jnp.where(valid_mask, field2[:, :, s_map, kx_map, ky_idx], 0.0)
            out1 = out1 + coeffs1[i] * shifted1
            out2 = out2 + coeffs2[i] * shifted2
        return out1, out2

    # ── nonlinear term III ──────────────────────────────────────────────

    def _nonlinear_term_iii_core(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        bessel: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Nonlinear ExB advection (term III) for 5D df. Shared skeleton for R2C and Z2Z."""
        pre = self.pre
        mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
        fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
        kx2d, ky2d = pre["nl_kx2d"], pre["nl_ky2d"]
        if bessel is None:
            bessel = pre["bessel"]
        dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
        nky = df.shape[-1]

        if self.use_z2z:
            kx_1d = kx2d[:, 0]
            ky_1d = ky2d[0, :]
            rev_jind = (
                jnp.full(mrad, -1, dtype=jnp.int32)
                .at[jind]
                .set(jnp.arange(len(jind), dtype=jnp.int32))
            )

            def _per_s_wrapper(df_s, phi_s, bessel_s, dum):
                return _per_s_z2z(
                    df_s,
                    phi_s,
                    bessel_s,
                    dum,
                    mixed_precision=self.mixed_precision,
                    efun_sign=efun_sign,
                    fft_prefactor=fft_prefactor,
                    fft_scale=fft_scale,
                    kx_vec=kx_1d,
                    ky_vec=ky_1d,
                    jind=jind,
                    rev_jind=rev_jind,
                    mrad=mrad,
                    mphi=mphi,
                    mphiw3=mphiw3,
                    nky=nky,
                )

        else:
            ikx_packed = 1j * pack_half_spectrum(kx2d, jind, mrad, mphiw3)
            iky_packed = 1j * pack_half_spectrum(ky2d, jind, mrad, mphiw3)

            def _per_s_wrapper(df_s, phi_s, bessel_s, dum):
                return _per_s_r2c(
                    df_s,
                    phi_s,
                    bessel_s,
                    dum,
                    mixed_precision=self.mixed_precision,
                    efun_sign=efun_sign,
                    fft_prefactor=fft_prefactor,
                    fft_scale=fft_scale,
                    ikx_packed=ikx_packed,
                    iky_packed=iky_packed,
                    mrad=mrad,
                    mphi=mphi,
                    mphiw3=mphiw3,
                    jind=jind,
                    nky=nky,
                )

        nl = jax.vmap(_per_s_wrapper, in_axes=(2, 0, 2, 0), out_axes=2)(df, phi, bessel, dum_s)
        return nl.at[:, :, :, ixzero, iyzero].set(0.0) if exclude_zero_mode else nl

    def nonlinear_term_iii(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        bessel: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Nonlinear ExB advection with shape dispatch.

        Dispatches on df.ndim: 5D direct, 6D via vmap over species with per-species bessel.
        Mixed precision is controlled by self.mixed_precision (set at construction time).
        """
        if df.ndim == 5:
            return self._nonlinear_term_iii_core(
                df,
                phi,
                geometry,
                efun_sign=efun_sign,
                fft_prefactor=fft_prefactor,
                exclude_zero_mode=exclude_zero_mode,
                bessel=bessel,
            )
        elif df.ndim == 6:
            if bessel is None:
                bessel = self.pre["bessel"]

            def _per_species(df_sp, bes_sp):
                return self._nonlinear_term_iii_core(
                    df_sp,
                    phi,
                    geometry,
                    efun_sign=efun_sign,
                    fft_prefactor=fft_prefactor,
                    exclude_zero_mode=exclude_zero_mode,
                    bessel=bes_sp,
                )

            return jax.vmap(_per_species)(df, bessel)
        else:
            raise ValueError(f"nonlinear_term_iii: expected df with ndim 5 or 6, got {df.ndim}")

    def _linear_rhs_core(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        params: GKParams,
        pre: GKPre,
    ) -> jnp.ndarray:
        """Fused linear RHS for single species (5D df).

        Implements Terms I, II, IV, V, VII, VIII + dissipation.
        Matches GKW linear_terms.f90 and GKW's calc_linear_terms.
        """
        gyro_phi = pre["bessel"] * phi[None, None, :, :, :]

        term_par, term_vii = self._apply_parallel_dual(
            df, gyro_phi, pre["s_total_upar"], pre["s_total_t7"]
        )

        out_d1, out_d4 = self._apply_vpar_dual(df, stencils.VPAR_D1, stencils.VPAR_D4)
        term_iv = pre["utrap"] * out_d1 / params.dvp
        term_vp_diss = params.disp_vp * pre["abs_dum2_vp"] * out_d4 / params.dvp

        kdotvd = pre["drift_x"] * pre["kx_b"] + pre["drift_y"] * pre["ky_b"]

        return (
            term_par
            + term_iv
            + term_vp_diss
            - 1j * kdotvd * df
            + pre["hyper"] * df
            + 1j
            * params.drive_scale
            * (
                pre["dmaxwel_fm_ek"]
                - pre["signz0"] * kdotvd * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1e-15))
            )
            * gyro_phi
            + term_vii
        )

    def linear_rhs(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params: GKParams,
        pre: GKPre,
    ) -> jnp.ndarray:
        """Linear RHS with shape dispatch.

        Implements Terms I, II, IV, V, VII, VIII + dissipation.
        Dispatches on df.ndim: 5D direct, 6D via vmap over species.
        """
        if df.ndim == 5:
            return self._linear_rhs_core(df, phi, params, pre)
        elif df.ndim == 6:
            # Per-species arrays: leading axis is nsp, vmap slices along axis 0.
            # s_total_upar/t7 are (9, nsp, ...) from _fuse_stencils; moveaxis brings
            # nsp to front so each vmap slice is (9, ...) — correct for _apply_parallel_dual.
            sp_arrays = {
                "bessel": pre["bessel"],
                "fmaxwl": pre["fmaxwl"],
                "dmaxwel_fm_ek": pre["dmaxwel_fm_ek"],
                "drift_x": pre["drift_x"],
                "drift_y": pre["drift_y"],
                "utrap": pre["utrap"],
                "abs_dum2_vp": pre["abs_dum2_vp"],
                "tmp0": pre["tmp0"],
                "signz0": pre["signz0"],
                "s_total_upar": jnp.moveaxis(pre["s_total_upar"], 1, 0),
                "s_total_t7": jnp.moveaxis(pre["s_total_t7"], 1, 0),
            }
            sp_in_axes = {k: 0 for k in sp_arrays}

            # kx_b/ky_b in pre are 6D for kinetic (1,1,1,1,nkx,1); squeeze to 5D
            # so that per-species _linear_rhs_core (with 5D df_sp) stays 5D.
            # s_shift/kx_shift/valid_shift are read from self.pre by _apply_parallel_dual
            # and are species-independent, so they don't belong in the vmapped dict.
            shared = {
                "kx_b": pre["kx_b"].ravel().reshape(1, 1, 1, -1, 1),
                "ky_b": pre["ky_b"].ravel().reshape(1, 1, 1, 1, -1),
                "hyper": pre["hyper"],
            }

            def _per_species(df_sp, sp):
                sp_pre = {**sp, **shared}
                return self._linear_rhs_core(df_sp, phi, params, sp_pre)

            return jax.vmap(_per_species, in_axes=(0, sp_in_axes))(df, sp_arrays)
        else:
            raise ValueError(f"linear_rhs: expected df with ndim 5 or 6, got {df.ndim}")


def _pack_full_z2z(field, kx_vec, ky_vec, jind, rev_jind, mrad, mphi, mphiw3, nky, dtype):
    """Scatter from packed [nkx, nky] directly to full z2z workspace [mrad, mphi].

    Gathers field values at each output position via rev_jind — no intermediate
    [mrad, mphiw3] buffer, no scatter ops.

    Packing convention: ws = (i·ky − kx)·F, so Re(IFFT(ws)) = ∂y·F, Im(IFFT(ws)) = ∂x·F.
    Bracket = ∂yϕ · ∂xf − ∂xϕ · ∂yf  (matches R2C path).
    """
    real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    one_j = jnp.array(1j, dtype=dtype)
    nkx = field.shape[-2]

    kx_dense = jnp.zeros(mrad, dtype=real_dtype).at[jind].set(kx_vec.astype(real_dtype))
    ky_half = jnp.zeros(mphiw3, dtype=real_dtype).at[:nky].set(ky_vec[:nky].astype(real_dtype))
    m_mirror = (mrad - jnp.arange(mrad)) % mrad

    # ── Primary half [mrad, mphiw3] ──────────────────────────────────
    m_p = jnp.arange(mrad)
    m_g, j_g = jnp.meshgrid(m_p, jnp.arange(mphiw3), indexing="ij")

    m_src = rev_jind[m_g]
    valid = (m_src >= 0) & (j_g < nky)
    val = jnp.where(valid, field[..., jnp.clip(m_src, 0, nkx - 1), j_g], 0).astype(dtype)

    # Symmetrize ky=0: F(kx,0) = conj(F(-kx,0))
    m_src_mir = rev_jind[m_mirror[m_g]]
    val0_mir = jnp.where(
        (j_g == 0) & (m_src_mir >= 0),
        field[..., jnp.clip(m_src_mir, 0, nkx - 1), 0],
        0,
    ).astype(dtype)
    val = jnp.where(j_g == 0, 0.5 * (val + jnp.conj(val0_mir)), val)

    primary = (one_j * ky_half[j_g] - kx_dense[m_g]).astype(dtype) * val

    # ── Mirror half [mrad, mphi − mphiw3] ────────────────────────────
    j_src = mphi - jnp.arange(mphiw3, mphi)
    m_g2, j_src_g = jnp.meshgrid(m_p, j_src, indexing="ij")

    m_mir_g2 = m_mirror[m_g2]
    m_src_mir2 = rev_jind[m_mir_g2]
    valid_mir2 = (m_src_mir2 >= 0) & (j_src_g < nky)
    val_mir = jnp.where(
        valid_mir2,
        field[..., jnp.clip(m_src_mir2, 0, nkx - 1), jnp.clip(j_src_g, 0, nky - 1)],
        0,
    ).astype(dtype)
    # kx_dense[m_g2]: output position kx (not source/mirror kx)
    mirror = jnp.conj((one_j * ky_half[j_src_g] - kx_dense[m_g2]).astype(dtype) * val_mir)

    return jnp.concatenate([primary, mirror], axis=-1)


def _per_s_r2c(
    df_s,
    phi_s,
    bessel_s,
    dum,
    *,
    mixed_precision,
    efun_sign,
    fft_prefactor,
    fft_scale,
    ikx_packed,
    iky_packed,
    mrad,
    mphi,
    mphiw3,
    jind,
    nky,
):
    """Standard R2C Poisson bracket logic. Port of GKW non_linear_terms.F90.

    Wavenumber arrays are packed to the dealiased grid once before the
    per-s vmap loop. When mixed_precision is True, FFTs run in FP32.
    """
    fft_dtype = jnp.complex64 if mixed_precision else jnp.complex128
    real_dtype = jnp.float32 if mixed_precision else jnp.float64

    gyro_phi = bessel_s * phi_s[None, None, :, :]

    df_packed = pack_half_spectrum(df_s, jind, mrad, mphiw3).astype(fft_dtype)
    phi_packed = pack_half_spectrum(gyro_phi, jind, mrad, mphiw3).astype(fft_dtype)

    # cast wavenumber arrays to fft_dtype to avoid promotion back to fp64
    ikx = ikx_packed.astype(fft_dtype)
    iky = iky_packed.astype(fft_dtype)

    packed_grad_phi_y = iky[None, None, :] * phi_packed
    packed_grad_phi_x = ikx[None, None, :] * phi_packed
    packed_grad_f_x = ikx[None, None, :] * df_packed
    packed_grad_f_y = iky[None, None, :] * df_packed

    def _to_real(packed_spec):
        return jnp.fft.irfft2(packed_spec, s=(mrad, mphi), axes=(-2, -1), norm="backward")

    nl_real = (efun_sign * dum).astype(real_dtype) * (
        _to_real(packed_grad_phi_y) * _to_real(packed_grad_f_x)
        - _to_real(packed_grad_phi_x) * _to_real(packed_grad_f_y)
    )

    nl_half_raw = jnp.fft.rfft2(nl_real, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    if mixed_precision:
        nl_half_raw = nl_half_raw.astype(jnp.complex128)

    nl_half = (
        jnp.asarray(fft_prefactor, dtype=jnp.complex128)
        * jnp.asarray(fft_scale, dtype=jnp.complex128)
        * nl_half_raw
    )
    return unpack_half_spectrum(nl_half, jind, nky)


def _per_s_z2z(
    df_s,
    phi_s,
    bessel_s,
    dum,
    *,
    mixed_precision,
    efun_sign,
    fft_prefactor,
    fft_scale,
    kx_vec,
    ky_vec,
    jind,
    rev_jind,
    mrad,
    mphi,
    mphiw3,
    nky,
):
    """Z2Z 2-for-1 Poisson bracket logic.

    Packs two spectral derivatives into one complex field and uses
    ifft2 (C2C) instead of irfft2 (C2R), halving the inverse FFT count.
    _pack_full_z2z gathers directly from [nkx, nky] into [mrad, mphi]
    via rev_jind — no intermediate half-spectrum buffer, no scatter ops.
    """
    fft_dtype = jnp.complex64 if mixed_precision else jnp.complex128
    real_dtype = jnp.float32 if mixed_precision else jnp.float64

    gyro_phi = bessel_s * phi_s[None, None, :, :]

    ws_df = _pack_full_z2z(df_s, kx_vec, ky_vec, jind, rev_jind, mrad, mphi, mphiw3, nky, fft_dtype)
    ws_phi = _pack_full_z2z(
        gyro_phi, kx_vec, ky_vec, jind, rev_jind, mrad, mphi, mphiw3, nky, fft_dtype
    )

    z2z_df = jnp.fft.ifft2(ws_df, axes=(-2, -1), norm="backward")
    z2z_phi = jnp.fft.ifft2(ws_phi, axes=(-2, -1), norm="backward")

    nl_real = (efun_sign * dum).astype(real_dtype) * (
        jnp.real(z2z_phi) * jnp.imag(z2z_df) - jnp.imag(z2z_phi) * jnp.real(z2z_df)
    )

    nl_half_raw = jnp.fft.rfft2(nl_real, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    if mixed_precision:
        nl_half_raw = nl_half_raw.astype(jnp.complex128)

    nl_half = (
        jnp.asarray(fft_prefactor, dtype=jnp.complex128)
        * jnp.asarray(fft_scale, dtype=jnp.complex128)
        * nl_half_raw
    )
    return unpack_half_spectrum(nl_half, jind, nky)
