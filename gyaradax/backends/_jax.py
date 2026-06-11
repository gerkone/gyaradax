"""JAX backend for solver operations.

Implements the nonlinear ExB bracket (term III) and stencil operations
using pure JAX. This is the direct port of GKW's non_linear_terms.F90
and linear_terms.f90 stencil application.
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

import gyaradax.stencils as stencils
from gyaradax.backends.ops import SolverOps
from gyaradax.collisions import collision_rhs, conservation_correction
from gyaradax.params import GKParams
from gyaradax.state import GKPre
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

    def _nonlinear_term_iii_core(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        bessel: jnp.ndarray | None = None,
        chi_correction: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Nonlinear ExB advection (term III) for 5D df. Shared skeleton for R2C and Z2Z.

        When chi_correction is provided (EM mode), it is added to gyro_phi
        to form the generalized potential chi = J0*phi + chi_correction.
        chi_correction shape: (nvpar, nmu, ns, nkx, nky).
        """
        pre = self.pre
        mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
        fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
        kx2d, ky2d = pre["nl_kx2d"], pre["nl_ky2d"]
        if bessel is None:
            bessel = pre["bessel"]
        dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
        nky = df.shape[-1]

        # chi_correction per-s slices: vmap axis 2 (s-axis in 5D arrays)
        if chi_correction is not None:
            chi_corr_vmap = chi_correction
            chi_in_axis = 2
        else:
            chi_corr_vmap = jnp.zeros(1)  # dummy, ignored
            chi_in_axis = None

        if self.use_z2z:
            kx_1d = kx2d[:, 0]
            ky_1d = ky2d[0, :]
            rev_jind = (
                jnp.full(mrad, -1, dtype=jnp.int32)
                .at[jind]
                .set(jnp.arange(len(jind), dtype=jnp.int32))
            )

            def _per_s_wrapper(df_s, phi_s, bessel_s, dum, chi_s):
                return _per_s_z2z(
                    df_s,
                    phi_s,
                    bessel_s,
                    dum,
                    chi_correction_s=chi_s if chi_correction is not None else None,
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

            def _per_s_wrapper(df_s, phi_s, bessel_s, dum, chi_s):
                return _per_s_r2c(
                    df_s,
                    phi_s,
                    bessel_s,
                    dum,
                    chi_correction_s=chi_s if chi_correction is not None else None,
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

        nl = jax.vmap(_per_s_wrapper, in_axes=(2, 0, 2, 0, chi_in_axis), out_axes=2)(
            df, phi, bessel, dum_s, chi_corr_vmap
        )
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
        bessel: jnp.ndarray | None = None,
        chi_correction: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Nonlinear ExB advection with shape dispatch.

        Dispatches on df.ndim: 5D direct, 6D via vmap over species with per-species bessel.
        Mixed precision is controlled by self.mixed_precision (set at construction time).
        chi_correction: velocity-dependent EM correction added to gyro_phi to form chi.
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
                chi_correction=chi_correction,
            )
        elif df.ndim == 6:
            if bessel is None:
                bessel = self.pre["bessel"]

            def _per_species(df_sp, bes_sp, chi_sp):
                return self._nonlinear_term_iii_core(
                    df_sp,
                    phi,
                    geometry,
                    efun_sign=efun_sign,
                    fft_prefactor=fft_prefactor,
                    exclude_zero_mode=exclude_zero_mode,
                    bessel=bes_sp,
                    chi_correction=chi_sp if chi_correction is not None else None,
                )

            if chi_correction is not None:
                return jax.vmap(_per_species)(df, bessel, chi_correction)
            else:
                dummy = jnp.zeros((df.shape[0],))
                return jax.vmap(_per_species)(df, bessel, dummy)
        else:
            raise ValueError(f"nonlinear_term_iii: expected df with ndim 5 or 6, got {df.ndim}")

    def _linear_rhs_terms(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        params: GKParams,
        pre: GKPre | dict[str, Any],
        apar: jnp.ndarray | None = None,
        bpar: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Return each linear-RHS term as a dict entry.

        Implements Terms I, II, IV, V, VII, VIII, X, XI + dissipation + collisions.
        The total RHS is the sum of all values in the returned dict.

        chi = J0*phi - 2*vR*vpar*J0*apar + (2*mu*T/Z)*J1hat*bpar
        Drive terms V, VIII use chi/phi; Term VII uses gyro_phi only.

        Term I (streaming) and parallel dissipation share a fused 9-point stencil
        ('s_total_upar') — exposed together in the dict as 'I_par_streaming_plus_diss'.
        """
        gyro_phi = pre["bessel"] * phi[None, None, :, :, :]

        gyro_chi = gyro_phi
        if apar is not None and "apar_chi_factor" in pre:
            apar_b = apar[None, None, :, :, :]
            gyro_chi = gyro_chi + pre["apar_chi_factor"] * apar_b
        if bpar is not None and "bpar_chi_factor" in pre:
            bpar_b = bpar[None, None, :, :, :]
            gyro_chi = gyro_chi + pre["bpar_chi_factor"] * bpar_b

        # parallel stencil (fused: streaming + parallel dissipation + Landau)
        term_I_par, term_VII_landau = self._apply_parallel_dual(
            df, gyro_phi, pre["s_total_upar"], pre["s_total_t7"]
        )

        # vpar stencil (trapping + vpar dissipation; 5-point central)
        out_d1, out_d4 = self._apply_vpar_dual(df, stencils.VPAR_D1, stencils.VPAR_D4)
        term_IV_trapping = pre["utrap"] * out_d1 / pre["dvp"]
        term_IV_vp_diss = params.disp_vp * pre["abs_dum2_vp"] * out_d4 / pre["dvp"]

        kdotvd = pre["drift_x"] * pre["kx_b"] + pre["drift_y"] * pre["ky_b"]

        # term II: drift on perturbed distribution (β-independent)
        term_II_drift_df = -1j * kdotvd * df

        # term V: gradient drive on chi
        term_V_drive_chi = 1j * params.drive_scale * pre["dmaxwel_fm_ek"] * gyro_chi

        # term VIII: curvature drift × F_M × phi (NOT chi -- verified vs GKW iphi_ga)
        term_VIII_curv_phi = (
            -1j
            * params.drive_scale
            * pre["signz0"]
            * kdotvd
            * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1e-15))
            * gyro_phi
        )

        # term X: parallel derivative of gyro-averaged bpar (B_par compression)
        term_X_bpar_par = jnp.zeros_like(df)
        # term XI: curvature drift × F_M × bpar coupling (B_par analog of VIII).
        # GKW vd_grad_phi_fm elem2 at linear_terms.f90:2664-2667; equivalent
        # to VIII with gyro_phi replaced by bpar_chi_factor*bpar.
        term_XI_curv_bpar = jnp.zeros_like(df)
        if bpar is not None and "bpar_chi_factor" in pre:
            bpar_b = bpar[None, None, :, :, :]
            gyro_bpar_scaled = pre["bpar_chi_factor"] * bpar_b
            term_X_bpar_par = self._apply_parallel(gyro_bpar_scaled, pre["s_total_t7"])
            term_XI_curv_bpar = (
                -1j
                * params.drive_scale
                * pre["signz0"]
                * kdotvd
                * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1e-15))
                * gyro_bpar_scaled
            )

        term_hyper_diss = pre["hyper"] * df

        term_collisions = jnp.zeros_like(df)
        if params.collisions and "coll_stencil" in pre:
            term_collisions = collision_rhs(df, pre["coll_stencil"])
            if (
                params.coll_mom_conservation or params.coll_ene_conservation
            ) and "coll_mom_factor" in pre:
                term_collisions = term_collisions + conservation_correction(
                    term_collisions,
                    pre["coll_mom_factor"],
                    pre["coll_ene_factor"],
                    pre["coll_vpar_weight"],
                    pre["coll_vsq_weight"],
                )

        return {
            "I_par_streaming_plus_diss": term_I_par,
            "II_drift_df": term_II_drift_df,
            "IV_trapping": term_IV_trapping,
            "IV_vp_diss": term_IV_vp_diss,
            "V_drive_chi": term_V_drive_chi,
            "VII_landau_phi": term_VII_landau,
            "VIII_curv_phi": term_VIII_curv_phi,
            "X_bpar_par": term_X_bpar_par,
            "XI_curv_bpar": term_XI_curv_bpar,
            "hyper_diss": term_hyper_diss,
            "collisions": term_collisions,
        }

    def _linear_rhs_core(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        params: GKParams,
        pre: GKPre | dict[str, Any],
        apar: jnp.ndarray | None = None,
        bpar: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Total linear RHS = sum of all linear terms.

        Convenience wrapper around _linear_rhs_terms; numerically identical to
        the fused expression. JAX/XLA will fuse term computations under JIT.
        """
        terms = self._linear_rhs_terms(df, phi, params, pre, apar=apar, bpar=bpar)
        total = terms["I_par_streaming_plus_diss"]
        for k, v in terms.items():
            if k == "I_par_streaming_plus_diss":
                continue
            total = total + v
        return total

    def linear_rhs(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params: GKParams,
        pre: GKPre,
        apar: jnp.ndarray | None = None,
        bpar: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Linear RHS with shape dispatch.

        Implements Terms I, II, IV, V, VII, VIII, X, XI + dissipation.
        Dispatches on df.ndim: 5D direct, 6D via vmap over species.
        When apar/bpar are provided, includes EM coupling terms.
        """
        if df.ndim == 5:
            return self._linear_rhs_core(df, phi, params, pre, apar=apar, bpar=bpar)
        elif df.ndim == 6:
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
            # chi factors for em
            if apar is not None and "apar_chi_factor" in pre:
                sp_arrays["apar_chi_factor"] = pre["apar_chi_factor"]
            if bpar is not None and "bpar_chi_factor" in pre:
                sp_arrays["bpar_chi_factor"] = pre["bpar_chi_factor"]
            # per-species collision stencil shape (nsp, 9, nv, nmu, ns); axis 0 mapped
            if params.collisions and "coll_stencil" in pre:
                sp_arrays["coll_stencil"] = pre["coll_stencil"]
                if "coll_mom_factor" in pre:
                    sp_arrays["coll_mom_factor"] = pre["coll_mom_factor"]
                    sp_arrays["coll_ene_factor"] = pre["coll_ene_factor"]
                    sp_arrays["coll_vpar_weight"] = pre["coll_vpar_weight"]
                    sp_arrays["coll_vsq_weight"] = pre["coll_vsq_weight"]
            sp_in_axes = {k: 0 for k in sp_arrays}

            shared = {
                "kx_b": pre["kx_b"].ravel().reshape(1, 1, 1, -1, 1),
                "ky_b": pre["ky_b"].ravel().reshape(1, 1, 1, 1, -1),
                "hyper": pre["hyper"],
                # static grid spacing (python float; species-independent)
                "dvp": pre["dvp"],
            }

            def _per_species(df_sp, sp):
                sp_pre = {**sp, **shared}
                return self._linear_rhs_core(df_sp, phi, params, sp_pre, apar=apar, bpar=bpar)

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
    chi_correction_s=None,
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
    if chi_correction_s is not None:
        gyro_phi = gyro_phi + chi_correction_s

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
    chi_correction_s=None,
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
    if chi_correction_s is not None:
        gyro_phi = gyro_phi + chi_correction_s

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
