import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional

from gyaradax.types import GKPre
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum
from gyaradax.backends.ops import SolverOps


@jax.tree_util.register_pytree_node_class
class JAXOps(SolverOps):
    """ JAX backend for solver operations """

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
        pre, = children
        obj = cls(pre, None)
        obj.template_meta = aux_data
        return obj


    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
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
        return self._apply_vpar(field, coeffs_d1), self._apply_vpar(field, coeffs_d4)


    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
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
        self, field1: jnp.ndarray, field2: jnp.ndarray, coeffs1: jnp.ndarray, coeffs2: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        """Nonlinear ExB advection via pseudospectral method. df is 5D."""
        pre = self.pre
        mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]

        fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
        kx2d, ky2d, bessel = pre["nl_kx2d"], pre["nl_ky2d"], pre["bessel"]
        dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
        nky = df.shape[-1]

        # Pre-pack the wavenumbers once (very cheap, 2D arrays)
        ikx_packed = 1j * pack_half_spectrum(kx2d, jind, mrad, mphiw3)
        iky_packed = 1j * pack_half_spectrum(ky2d, jind, mrad, mphiw3)

        def _per_s(df_s, phi_s, bessel_s, dum):
            fft_dtype = jnp.complex64 if mixed_precision else jnp.complex128
            real_dtype = jnp.float32 if mixed_precision else jnp.float64

            gyro_phi = bessel_s * phi_s[None, None, :, :]

            # 1. PACK FIRST: 2 massive scatters instead of 4
            df_packed = pack_half_spectrum(df_s, jind, mrad, mphiw3).astype(fft_dtype)
            phi_packed = pack_half_spectrum(gyro_phi, jind, mrad, mphiw3).astype(fft_dtype)

            # 2. MULTIPLY AFTER: 4 elementwise multiplications on the much smaller packed arrays
            packed_grad_phi_y = iky_packed[None, None, :] * phi_packed
            packed_grad_phi_x = ikx_packed[None, None, :] * phi_packed
            packed_grad_f_x   = ikx_packed[None, None, :] * df_packed
            packed_grad_f_y   = iky_packed[None, None, :] * df_packed

            def _to_real(packed_spec):
                return jnp.fft.irfft2(packed_spec, s=(mrad, mphi), axes=(-2, -1), norm="backward")

            # 3. REAL SPACE BRACKET
            nl_real = (efun_sign * dum).astype(real_dtype) * (
                _to_real(packed_grad_phi_y) * _to_real(packed_grad_f_x)
                - _to_real(packed_grad_phi_x) * _to_real(packed_grad_f_y)
            )

            # 4. FORWARD FFT (O5 Optimization)
            if mixed_precision:
                # Execute in FP32, then cast the smaller spectral result to FP64
                nl_half_raw = jnp.fft.rfft2(nl_real, s=(mrad, mphi), axes=(-2, -1), norm="backward")
                nl_half_raw = nl_half_raw.astype(jnp.complex128)
            else:
                nl_half_raw = jnp.fft.rfft2(nl_real, s=(mrad, mphi), axes=(-2, -1), norm="backward")

            nl_half = (
                jnp.asarray(fft_prefactor, dtype=jnp.complex128)
                * jnp.asarray(fft_scale, dtype=jnp.complex128)
                * nl_half_raw
            )
            return unpack_half_spectrum(nl_half, jind, nky)

        # 5. Eliminate `moveaxis` overhead by mapping directly over axis 2
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
        """Nonlinear bracket using Z2Z 2-for-1 packing.

        Packs two spectral derivatives into one complex field:
            ws = field_y + i*field_x
        and uses ifft2 (C2C) instead of irfft2 (C2R).  After the inverse FFT
        Re(ws) = field_y_real, Im(ws) = field_x_real.  This halves the IRFFT
        count while keeping phi at its natural (nmu, ns) batch.

        Hermitian symmetrization at ky=0 corrects the Bessel-induced
        symmetry defect that would otherwise leak across the packed channels.
        """
        pre = self.pre
        mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
        fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
        kx2d, ky2d, bessel = pre["nl_kx2d"], pre["nl_ky2d"], pre["bessel"]
        dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
        nky = df.shape[-1]

        # Wavenumber vectors (1-D, for the packed kx / ky axes)
        kx_1d = kx2d[:, 0]   # (nkx,)
        ky_1d = ky2d[0, :]   # (nky,)

        def _pack_full_z2z(field_packed_half, kx_vec, ky_vec, jind, mrad, mphi, nky):
            """Pack half-spectrum field into full Z2Z input with 2-for-1 packing.

            Returns ws[..., mrad, mphi] complex where:
                ws = (i*ky*field) + i*(i*kx*field) = field_y + i*field_x
            with Hermitian extension for j > mphi/2 and symmetrization at j=0.
            """
            mphiw3 = mphi // 2 + 1
            # field_packed_half: [..., mrad, mphiw3]

            # Wavenumber arrays broadcast to half-spectrum shape
            ikx = kx_vec[jind]                              # (nkx,) → (mrad,) via jind inverse
            # Actually we need kx for each dense m position.
            # Build kx_dense[mrad]: kx value at each dense radial position, 0 for padded
            kx_dense = jnp.zeros(mrad, dtype=jnp.float64)
            kx_dense = kx_dense.at[jind].set(kx_vec)       # (mrad,)

            # Compute derivatives on the half spectrum
            # field_y = i*ky*field, field_x = i*kx*field
            iky_half = 1j * ky_vec[None, :mphiw3]                          # (1, mphiw3) padded
            iky_bcast = jnp.zeros((1, mphiw3), dtype=jnp.complex128)
            iky_bcast = iky_bcast.at[:, :nky].set(1j * ky_vec[:nky])

            ikx_bcast = 1j * kx_dense[:, None]                             # (mrad, 1)

            fy_half = iky_bcast * field_packed_half   # [..., mrad, mphiw3]
            fx_half = ikx_bcast * field_packed_half   # [..., mrad, mphiw3]

            # ── Hermitian symmetrization at ky=0 ──────────────────────
            # For each mirror pair (m, m'), average: val_sym = (val + conj(val_mirror)) / 2
            m_mirror = (mrad - jnp.arange(mrad)) % mrad   # mirror indices

            def _symmetrize_col0(arr):
                col0 = arr[..., :, 0]                      # [..., mrad]
                col0_mirror = col0[..., m_mirror]           # [..., mrad]
                col0_sym = 0.5 * (col0 + jnp.conj(col0_mirror))
                return arr.at[..., :, 0].set(col0_sym)

            fy_half = _symmetrize_col0(fy_half)
            fx_half = _symmetrize_col0(fx_half)

            # ── Hermitian extension to full spectrum ──────────────────
            # For j > mphi/2: F(m, j) = conj(F(m_mirror, mphi-j))
            # Build full spectrum array
            ws_full = jnp.zeros(field_packed_half.shape[:-2] + (mrad, mphi), dtype=jnp.complex128)

            # Primary half: j = 0..mphiw3-1
            # Pack: ws = fy + i*fx
            ws_primary = fy_half - fx_half * 1j   # Note: i*(a+ib) = -b+ia, so fy + i*fx has ws.re = fy.re - fx.im, ws.im = fy.im + fx.re
            # Actually: ws = fy + i*fx where fy, fx are complex arrays
            # (fy + i*fx) is NOT standard complex addition — it's "pack" where
            # Re(ws) should become fy_real after IFFT and Im(ws) should become fx_real.
            # For this to work: ws(k) = fy(k) + i*fx(k) as complex addition.
            ws_primary = fy_half + 1j * fx_half
            ws_full = ws_full.at[..., :, :mphiw3].set(ws_primary)

            # Mirror half: j = mphiw3..mphi-1
            # ws(m, j) = conj(fy(m', j')) + i*conj(fx(m', j'))
            # where m' = m_mirror[m], j' = mphi - j
            # conj(a) + i*conj(b) has:
            #   Re = Re(conj(a)) - Im(conj(b)) = Re(a) - (-Im(b)) = Re(a) + Im(b)...
            # No, just: conj(fy) + i*conj(fx) as complex addition.
            j_mirror_range = jnp.arange(mphiw3, mphi)       # j = 49..95
            j_source = mphi - j_mirror_range                  # j' = 47..1

            fy_mirror = jnp.conj(fy_half[..., m_mirror[:, None], j_source[None, :]])  # [..., mrad, len(j_mirror)]
            fx_mirror = jnp.conj(fx_half[..., m_mirror[:, None], j_source[None, :]])
            ws_mirror = fy_mirror + 1j * fx_mirror
            ws_full = ws_full.at[..., :, mphiw3:].set(ws_mirror)

            return ws_full

        def _per_s(df_s, phi_s, bessel_s, dum):
            # df_s: (nvpar, nmu, nkx, nky), phi_s: (nkx, nky), bessel_s: (nmu, nkx, nky)
            gyro_phi = bessel_s * phi_s[None, None, :, :]     # (nmu, 1, nkx, nky) ... broadcast

            # Pack to half-spectrum (scatter via jind)
            df_packed = pack_half_spectrum(df_s, jind, mrad, mphiw3)      # (nvpar, nmu, mrad, mphiw3)
            phi_packed = pack_half_spectrum(gyro_phi, jind, mrad, mphiw3) # (nmu, 1, mrad, mphiw3)

            # 2-for-1 Z2Z packing with Hermitian extension + j=0 symmetrization
            ws_df  = _pack_full_z2z(df_packed, kx_1d, ky_1d, jind, mrad, mphi, nky)
            ws_phi = _pack_full_z2z(phi_packed, kx_1d, ky_1d, jind, mrad, mphi, nky)

            # Z2Z inverse FFT (complex-to-complex)
            z2z_df  = jnp.fft.ifft2(ws_df, axes=(-2, -1), norm="backward")
            z2z_phi = jnp.fft.ifft2(ws_phi, axes=(-2, -1), norm="backward")

            # Extract: Re = field_y_real, Im = field_x_real
            df_y_r  = jnp.real(z2z_df)   # (nvpar, nmu, mrad, mphi)
            df_x_r  = jnp.imag(z2z_df)
            phi_y_r = jnp.real(z2z_phi)  # (nmu, 1, mrad, mphi)
            phi_x_r = jnp.imag(z2z_phi)

            # Bracket: phi_y * df_x - phi_x * df_y  (phi broadcasts over nvpar)
            nl_real = (efun_sign * dum) * (
                phi_y_r * df_x_r - phi_x_r * df_y_r
            )

            # Forward FFT
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
        return None
