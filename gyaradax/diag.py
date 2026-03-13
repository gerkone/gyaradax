import jax.numpy as jnp
from typing import Dict, Tuple, Any
from gyaradax.integrals import get_integrals
from gyaradax.solver import (
    linear_precompute,
    nonlinear_term_iii,
    extended_firstdim_fft_size,
    extended_seconddim_fft_size,
    build_jind,
    pack_half_spectrum,
    unpack_half_spectrum,
)
from gyaradax.params import GKParams


def kx0_mode_columns(
    mode_label: jnp.ndarray, kxrh: jnp.ndarray
) -> Tuple[int, jnp.ndarray]:
    """
    Identify the wavevector columns corresponding to the kx=0 baseline.

    Used to map global spectral mode labels to the 1D growth/frequency
    diagnostics that traditionally focus on the kx=0 slice.
    """
    mode_label = jnp.asarray(mode_label, dtype=jnp.int32)
    kxrh = jnp.asarray(kxrh, dtype=jnp.float64)
    kx_line = kxrh[0] if kxrh.ndim == 2 else kxrh
    ixzero = int(jnp.argmin(jnp.abs(kx_line)).item())
    # correct for fortran 1-based indexing in mode_label files
    cols = mode_label[ixzero].astype(jnp.int32) - 1
    return ixzero, cols


def project_all_modes_to_kx0(
    all_modes: jnp.ndarray, mode_label: jnp.ndarray, kxrh: jnp.ndarray
) -> jnp.ndarray:
    """Project a flattened diagnostic array back to the kx=0 wavevector slice."""
    _, cols = kx0_mode_columns(mode_label, kxrh)
    return jnp.asarray(all_modes, dtype=jnp.complex128)[:, cols]


def term_iii_rhs(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams | None = None,
    *,
    efun_sign: float = 1.0,
    fft_prefactor: complex = 1.0 + 0.0j,
    exclude_zero_mode: bool = True,
) -> jnp.ndarray:
    """
    Public diagnostic interface for the Nonlinear Term III contribution.
    """
    if params is None:
        params = GKParams()
    pre = linear_precompute(geometry, params)
    phi, _ = get_integrals(df, geometry, params=params, include_fluxes=False)
    return nonlinear_term_iii(
        df,
        phi,
        geometry,
        pre,
        efun_sign=efun_sign,
        fft_prefactor=fft_prefactor,
        exclude_zero_mode=exclude_zero_mode,
    )


def term_iii_fft_pack_roundtrip(
    spec_kxky: jnp.ndarray, geometry: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """
    Verify the dealiased packing and FFT roundtrip for spectral modes.
    """
    nkx = spec_kxky.shape[-2]
    nky = spec_kxky.shape[-1]
    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    if kx.ndim > 1:
        kx = kx[0]
    ixzero = int(jnp.asarray(geometry.get("ixzero", jnp.argmin(jnp.abs(kx)))).item())
    mphi, mphiw3 = extended_firstdim_fft_size(nky)
    mrad = extended_seconddim_fft_size(nkx)
    jind = build_jind(nkx, mrad, ixzero)
    packed = pack_half_spectrum(spec_kxky, jind, mrad, mphiw3)
    real = jnp.fft.irfft2(packed, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    repacked = jnp.fft.rfft2(real, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    return unpack_half_spectrum(repacked, jind, nky)


def get_diagnostics(
    phi: jnp.ndarray,
    fluxes: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    state: Any,
) -> Dict[str, jnp.ndarray]:
    """
    Calculate high-level diagnostics from current field and state.
    Returns a dictionary containing:
        - fluxes: pflux, eflux, vflux (scalars)
        - kx_spec: 1D spectrum over kx (summed over s, ky)
        - ky_spec: 1D spectrum over ky (summed over s, kx)
        - ky_growth: 1D growth rate per ky mode (from state)
        - time, step: Scalars from state
    """
    phi_sq = jnp.abs(phi) ** 2
    # kx spectrum: sum over s and ky
    kx_spec = jnp.sum(phi_sq, axis=(0, 2))
    # ky spectrum: sum over s and kx
    ky_spec = jnp.sum(phi_sq, axis=(0, 1))

    return {
        "pflux": fluxes[0],
        "eflux": fluxes[1],
        "vflux": fluxes[2],
        "kx_spec": kx_spec,
        "ky_spec": ky_spec,
        "ky_growth": state.last_growth_rate,
        "time": state.time,
        "step": state.step,
    }
