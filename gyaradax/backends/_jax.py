"""JAX backend for solver operations.

implements the nonlinear ExB bracket (term III) and stencil operations
using pure JAX. this is the direct port of GKW's non_linear_terms.F90
and linear_terms.f90 stencil application.
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from gyaradax.backends.ops import SolverOps
from gyaradax.types import GKPre
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum


@jax.tree_util.register_pytree_node_class
class JAXOps(SolverOps):
    """JAX implementation of solver operations."""

    def __init__(self, pre: GKPre, field_template: Optional[jnp.ndarray] = None):
        self.pre = pre
        if field_template is not None:
            self.template_meta = (field_template.shape, field_template.dtype)
        else:
            self.template_meta = (None, None)

    def tree_flatten(self):
        return (self.pre,), self.template_meta

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (pre,) = children
        obj = cls(pre, None)
        obj.template_meta = aux_data
        return obj

    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        """apply 5-point vpar stencil (shifts -2..+2) with boundary clamping."""
        nv = field.shape[0]
        out = jnp.zeros_like(field)
        for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):
            idx = jnp.clip(jnp.arange(nv, dtype=jnp.int32) + s, 0, nv - 1)
            valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
            shifted = jnp.take(field, idx, axis=0)
            out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
        return out

    def _apply_vpar_dual(
        self, field: jnp.ndarray, coeffs_d1, coeffs_d4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """apply d1 and d4 vpar stencils in two passes."""
        return self._apply_vpar(field, coeffs_d1), self._apply_vpar(field, coeffs_d4)

    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        """apply 9-point parallel stencil using precomputed shift maps."""
        out = jnp.zeros_like(field)
        nky = field.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
        for i in range(9):
            s_map = self.pre["s_shift"][i]
            kx_map = self.pre["kx_shift"][i]
            valid = self.pre["valid_shift"][i]
            shifted = jnp.where(valid[None, None, :, :, :], field[:, :, s_map, kx_map, ky_idx], 0.0)
            out = out + coeffs[i] * shifted
        return out

    def _apply_parallel_dual(
        self,
        field1: jnp.ndarray,
        field2: jnp.ndarray,
        coeffs1: jnp.ndarray,
        coeffs2: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """apply parallel stencils to two fields."""
        return self._apply_parallel(field1, coeffs1), self._apply_parallel(field2, coeffs2)

    def nonlinear_term_iii(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        mixed_precision: bool = True,
    ) -> jnp.ndarray:
        """nonlinear ExB advection (term III) via pseudospectral Poisson bracket.

        computes {phi, f} on the dealiased real-space grid using the 3/2-rule
        for anti-aliasing. port of GKW non_linear_terms.F90.

        the bracket is evaluated per s-slice with jax.vmap over the parallel
        coordinate axis, avoiding the moveaxis overhead of the naive approach.

        optimization: wavenumber arrays are packed to the dealiased grid once
        before the per-s loop, so spectral derivatives are computed on the
        smaller packed arrays rather than the full (nkx, nky) grid.

        mixed precision: inverse FFTs run in FP32 for the real-space product,
        then the forward FFT result is cast back to FP64 for accumulation.

        args:
            df: distribution function (nvpar, nmu, ns, nkx, nky).
            phi: electrostatic potential (ns, nkx, nky).
            geometry: geometry dict (unused, kept for interface compatibility).
            efun_sign: sign of the ExB function (geometry-dependent).
            fft_prefactor: complex prefactor for the Poisson bracket.
            exclude_zero_mode: zero out the (kx=0, ky=0) zonal mode.
            mixed_precision: use FP32 for inverse FFTs.

        returns:
            nonlinear RHS contribution (nvpar, nmu, ns, nkx, nky).
        """
        pre = self.pre
        mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
        fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
        kx2d, ky2d, bessel = pre["nl_kx2d"], pre["nl_ky2d"], pre["bessel"]
        dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
        nky = df.shape[-1]

        ikx_packed = 1j * pack_half_spectrum(kx2d, jind, mrad, mphiw3)
        iky_packed = 1j * pack_half_spectrum(ky2d, jind, mrad, mphiw3)

        def _per_s(df_s, phi_s, bessel_s, dum):
            fft_dtype = jnp.complex64 if mixed_precision else jnp.complex128
            real_dtype = jnp.float32 if mixed_precision else jnp.float64

            gyro_phi = bessel_s * phi_s[None, None, :, :]

            df_packed = pack_half_spectrum(df_s, jind, mrad, mphiw3).astype(fft_dtype)
            phi_packed = pack_half_spectrum(gyro_phi, jind, mrad, mphiw3).astype(fft_dtype)

            packed_grad_phi_y = iky_packed[None, None, :] * phi_packed
            packed_grad_phi_x = ikx_packed[None, None, :] * phi_packed
            packed_grad_f_x = ikx_packed[None, None, :] * df_packed
            packed_grad_f_y = iky_packed[None, None, :] * df_packed

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

        nl = jax.vmap(_per_s, in_axes=(2, 0, 2, 0), out_axes=2)(df, phi, bessel, dum_s)
        return nl.at[:, :, :, ixzero, iyzero].set(0.0) if exclude_zero_mode else nl

    def nonlinear_term_iii_z2z(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        mixed_precision: bool = True,
    ) -> jnp.ndarray:
        """nonlinear bracket using z2z 2-for-1 packing (experimental).

        packs two spectral derivatives into one complex field and uses
        ifft2 (C2C) instead of irfft2 (C2R), halving the inverse FFT count.
        hermitian symmetrization at ky=0 corrects the bessel-induced
        symmetry defect that would leak across packed channels.
        """
        pre = self.pre
        mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
        fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
        kx2d, ky2d, bessel = pre["nl_kx2d"], pre["nl_ky2d"], pre["bessel"]
        dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
        nky = df.shape[-1]

        kx_1d = kx2d[:, 0]
        ky_1d = ky2d[0, :]

        def _pack_full_z2z(field_packed_half, kx_vec, ky_vec, jind, mrad, mphi, nky):
            """pack half-spectrum into full z2z input with hermitian extension."""
            mphiw3 = mphi // 2 + 1

            kx_dense = jnp.zeros(mrad, dtype=jnp.float64)
            kx_dense = kx_dense.at[jind].set(kx_vec)

            iky_bcast = jnp.zeros((1, mphiw3), dtype=jnp.complex128)
            iky_bcast = iky_bcast.at[:, :nky].set(1j * ky_vec[:nky])
            ikx_bcast = 1j * kx_dense[:, None]

            fy_half = iky_bcast * field_packed_half
            fx_half = ikx_bcast * field_packed_half

            m_mirror = (mrad - jnp.arange(mrad)) % mrad

            def _symmetrize_col0(arr):
                col0 = arr[..., :, 0]
                col0_sym = 0.5 * (col0 + jnp.conj(col0[..., m_mirror]))
                return arr.at[..., :, 0].set(col0_sym)

            fy_half = _symmetrize_col0(fy_half)
            fx_half = _symmetrize_col0(fx_half)

            ws_full = jnp.zeros(field_packed_half.shape[:-2] + (mrad, mphi), dtype=jnp.complex128)

            ws_primary = fy_half + 1j * fx_half
            ws_full = ws_full.at[..., :, :mphiw3].set(ws_primary)

            j_mirror_range = jnp.arange(mphiw3, mphi)
            j_source = mphi - j_mirror_range

            fy_mirror = jnp.conj(fy_half[..., m_mirror[:, None], j_source[None, :]])
            fx_mirror = jnp.conj(fx_half[..., m_mirror[:, None], j_source[None, :]])
            ws_mirror = fy_mirror + 1j * fx_mirror
            ws_full = ws_full.at[..., :, mphiw3:].set(ws_mirror)

            return ws_full

        def _per_s(df_s, phi_s, bessel_s, dum):
            gyro_phi = bessel_s * phi_s[None, None, :, :]

            df_packed = pack_half_spectrum(df_s, jind, mrad, mphiw3)
            phi_packed = pack_half_spectrum(gyro_phi, jind, mrad, mphiw3)

            ws_df = _pack_full_z2z(df_packed, kx_1d, ky_1d, jind, mrad, mphi, nky)
            ws_phi = _pack_full_z2z(phi_packed, kx_1d, ky_1d, jind, mrad, mphi, nky)

            z2z_df = jnp.fft.ifft2(ws_df, axes=(-2, -1), norm="backward")
            z2z_phi = jnp.fft.ifft2(ws_phi, axes=(-2, -1), norm="backward")

            nl_real = (efun_sign * dum) * (
                jnp.real(z2z_phi) * jnp.imag(z2z_df) - jnp.imag(z2z_phi) * jnp.real(z2z_df)
            )

            nl_half_raw = jnp.fft.rfft2(nl_real, s=(mrad, mphi), axes=(-2, -1), norm="backward")

            nl_half = (
                jnp.asarray(fft_prefactor, dtype=jnp.complex128)
                * jnp.asarray(fft_scale, dtype=jnp.complex128)
                * nl_half_raw
            )
            return unpack_half_spectrum(nl_half, jind, nky)

        nl = jax.vmap(_per_s, in_axes=(2, 0, 2, 0), out_axes=2)(df, phi, bessel, dum_s)
        return nl.at[:, :, :, ixzero, iyzero].set(0.0) if exclude_zero_mode else nl

    def linear_rhs(self, df, phi, geometry, params, pre) -> Optional[jnp.ndarray]:
        """not implemented in JAX backend; solver falls back to vmap."""
        return None
