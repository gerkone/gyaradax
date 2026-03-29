import jax
import jax.numpy as jnp
from typing import Dict, Tuple

from gyaradax.types import GKPre
from gyaradax.solver import pack_half_spectrum, unpack_half_spectrum
from gyaradax.backends.ops import SolverOps


class JAXOps(SolverOps):
    """ JAX backend for solver operations """

    def __init__(self, pre: GKPre, field_template: jnp.ndarray):
        self.pre = pre

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


    def nonlinear_term_iii(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        pre: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        mixed_precision: bool = True,
    ) -> jnp.ndarray:
        """Nonlinear ExB advection via pseudospectral method. df is 5D."""
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
