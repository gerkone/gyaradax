"""Diagnostics: potential spectra, growth-rate projections and Term III checks."""

import jax.numpy as jnp
from typing import Dict, Tuple, Any
from gyaradax.core.params import GKParams
from gyaradax.operators.precompute import (
    build_jind,
    extended_firstdim_fft_size,
    extended_seconddim_fft_size,
    linear_precompute,
    pack_half_spectrum,
    unpack_half_spectrum,
)
from gyaradax.solver.rk4 import nonlinear_term_iii


def kx0_mode_columns(mode_label: jnp.ndarray, kxrh: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    """Index of kx = 0 and the mode columns of the chains through it.

    Args:
        mode_label: Mode labels, (nkx, nky), 1-based as in GKW.
        kxrh: Radial wavenumbers.

    Returns:
        ``(ixzero, cols)``: the kx = 0 index and the 0-based column of each ky.
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
    """Select the kx = 0 chain columns of a per-mode diagnostic array.

    Args:
        all_modes: Array with the modes on its second axis.
        mode_label: Mode labels, (nkx, nky).
        kxrh: Radial wavenumbers.

    Returns:
        The columns of the kx = 0 chains.
    """
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
    """Nonlinear Term III of a 5D distribution, with the potential solved from it.

    Args:
        df: Distribution, (nvpar, nmu, ns, nkx, nky).
        geometry: Geometry dict.
        params: Run parameters; defaults when None.
        efun_sign: Sign of the ExB bracket.
        fft_prefactor: Prefactor of the FFTs.
        exclude_zero_mode: Zero the (kx, ky) = (0, 0) mode.

    Returns:
        Term III, same shape as ``df``.
    """
    if params is None:
        params = GKParams()
    pre = linear_precompute(geometry, params)
    from gyaradax.operators.poisson import calculate_phi

    phi = calculate_phi(geometry, df, params=params)
    return nonlinear_term_iii(
        df,
        phi,
        geometry,
        pre,
        efun_sign=efun_sign,
        fft_prefactor=fft_prefactor,
        exclude_zero_mode=exclude_zero_mode,
        mixed_precision=params.mixed_precision,
        backend=params.backend,
        use_z2z=params.use_z2z,
    )


def term_iii_fft_pack_roundtrip(
    spec_kxky: jnp.ndarray, geometry: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Spectral modes packed into the dealiased half spectrum and transformed there and back.

    Args:
        spec_kxky: Modes, (..., nkx, nky).
        geometry: Geometry dict.

    Returns:
        The modes after the round trip, same shape as ``spec_kxky``.
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
    """Fluxes, spectra and growth rates of the current state as a dict.

    Args:
        phi: Potential, (ns, nkx, nky).
        fluxes: ``(pflux, eflux, vflux)``.
        state: GKState.

    Returns:
        Dict with ``pflux``, ``eflux``, ``vflux``, ``kx_spec`` (summed over s and ky),
        ``ky_spec`` (summed over s and kx), ``ky_growth``, ``time`` and ``step``.
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


def phi_spectra(phi, geometry):
    """ky and kx spectra of the potential, field-line averaged as in GKW.

    The kx spectrum weights ky = 0 by 1 and the other ky by 2 (Parseval), by
    index, so it assumes the first ky is the zonal mode.

    Args:
        phi: Potential, (ns, nkx, nky).
        geometry: Geometry dict (``ints``).

    Returns:
        ``(ky_spec, kx_spec)``.
    """
    # ky_spec = ds * sum_{s,kx} |phi|^2; kx_spec weights ky > 0 twice (parseval [1, 2, 2, ...])
    ds = float(jnp.asarray(geometry["ints"])[0])
    nky = phi.shape[-1]
    parseval_ky = jnp.array([1.0] + [2.0] * (nky - 1))
    phi_sq = jnp.abs(phi) ** 2
    ky_spec = jnp.sum(ds * phi_sq, axis=(0, 1))
    kx_spec = jnp.sum(ds * phi_sq * parseval_ky[None, None, :], axis=(0, 2))
    return ky_spec, kx_spec
