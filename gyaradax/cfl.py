"""Adaptive CFL and timestep estimates for gyrokinetic solves."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

import jax
import jax.numpy as jnp

from gyaradax.constants import EPS
from gyaradax.jax_config import enable_x64
from gyaradax.state import GKPre, Precompute
from gyaradax.utils import pack_half_spectrum

if TYPE_CHECKING:
    from gyaradax.params import GKParams


enable_x64()


def estimate_nl_timestep(
    phi: jnp.ndarray,
    pre: Precompute,
    bessel: jnp.ndarray,
    dt_input: Any,
    safety_factor: Any = 0.95,
    apar: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """CFL-adaptive timestep from nonlinear ExB velocity.

    Computes max|grad(phi)| and (when apar is provided) max|grad(apar)|*vpmax
    in real space, matching GKW non_linear_terms.F90:1530-1800. The y/x
    correction factors absorb FFTW's unnormalized magnitude (N·|∂phi|_real)
    so `irfft2(norm="backward")` outputs match GKW's `max|ar|·mrad·lxinv`.
    """
    mrad = cast(int, pre["nl_mrad"])
    mphi = cast(int, pre["nl_mphi"])
    mphiw3 = cast(int, pre["nl_mphiw3"])
    jind = pre["nl_jind"]
    kx2d, ky2d = pre["nl_kx2d"], pre["nl_ky2d"]
    ycorr = mrad * mrad * mphi * pre["nl_lxinv"]
    xcorr = mrad * mphi * mphi * pre["nl_lyinv"]

    def _max_grad_per_s(field_s):
        grad_y_k = 1j * ky2d * field_s
        grad_x_k = 1j * kx2d * field_s

        def _to_real(spec):
            return jnp.fft.irfft2(
                pack_half_spectrum(spec[None, None, :, :], jind, mrad, mphiw3),
                s=(mrad, mphi),
                axes=(-2, -1),
                norm="backward",
            )

        max_y = jnp.max(jnp.abs(_to_real(grad_y_k))) * ycorr
        max_x = jnp.max(jnp.abs(_to_real(grad_x_k))) * xcorr
        return jnp.maximum(max_y, max_x)

    max_value = jnp.max(jax.vmap(_max_grad_per_s)(phi))

    # em: 2*vthrat_max * max|grad(apar)| * vpmax (non_linear_terms.F90:1241,1790)
    if apar is not None:
        vpmax = jnp.asarray(pre.get("vpmax", 3.0), dtype=jnp.float64)
        vthrat_max = jnp.asarray(pre.get("vthrat_max", 1.0), dtype=jnp.float64)
        max_grad_apar = jnp.max(jax.vmap(_max_grad_per_s)(apar))
        max_value = jnp.maximum(max_value, 2.0 * vthrat_max * max_grad_apar * vpmax)

    dt_est = jnp.where(
        max_value > EPS,
        jnp.asarray(safety_factor, dtype=jnp.float64) * 2.0 / max_value,
        jnp.asarray(dt_input, dtype=jnp.float64),
    )
    return jnp.minimum(dt_est, jnp.asarray(dt_input, dtype=jnp.float64))


def estimate_linear_timestep(
    pre: GKPre,
    params: Optional["GKParams"] = None,
    fac_dtim_est: float = 0.95,
    safety_factor: Optional[float] = None,
) -> jnp.ndarray:
    """Von Neumann stability timestep estimate (matdat.F90:1440-1510).

    Separates CFL by derivative order with RK4 stability factors:
      tmax = max(tmax1/2.4, tmax4/2.4, 40)
      dt   = fac_dtim_est / tmax

    When *safety_factor* is given, falls back to safety_factor * dx / max|u|.
    """
    sgr_dist = jnp.asarray(pre["sgr_dist"], dtype=jnp.float64)
    dvp = jnp.asarray(pre["dvp"], dtype=jnp.float64)
    max_upar = jnp.max(jnp.abs(pre["upar"]))
    max_utrap = jnp.max(jnp.abs(pre["utrap"]))

    if safety_factor is not None:
        dt_par = jnp.where(max_upar > EPS, safety_factor * sgr_dist / max_upar, 1e10)
        dt_trap = jnp.where(max_utrap > EPS, safety_factor * dvp / max_utrap, 1e10)
        return jnp.minimum(dt_par, dt_trap)

    # max stencil coefficients: boundary D1/D4 = 24/12 = 2.0,
    # interior VPAR_D1 = 8/12, VPAR_D4 = 6/12
    _D1S = jnp.asarray(2.0, dtype=jnp.float64)
    _D4S = jnp.asarray(2.0, dtype=jnp.float64)
    _D1V = jnp.asarray(8.0 / 12.0, dtype=jnp.float64)
    _D4V = jnp.asarray(6.0 / 12.0, dtype=jnp.float64)

    # ideriv=1: streaming + trapping
    tmax1 = jnp.maximum(
        jnp.where(max_upar > EPS, max_upar * _D1S / sgr_dist, 0.0),
        jnp.where(max_utrap > EPS, max_utrap * _D1V / dvp, 0.0),
    )

    # ideriv=4: parallel and velocity dissipation
    disp_par_val = jnp.abs(
        jnp.asarray(params.disp_par if params is not None else 1.0, dtype=jnp.float64)
    )
    disp_vp_val = jnp.abs(
        jnp.asarray(params.disp_vp if params is not None else 0.2, dtype=jnp.float64)
    )
    max_abs_par = jnp.max(jnp.abs(pre["abs_dum2_par"]))
    max_abs_vp = jnp.max(jnp.abs(pre["abs_dum2_vp"]))
    tmax4 = jnp.maximum(
        disp_par_val * jnp.where(max_abs_par > EPS, max_abs_par * _D4S / sgr_dist, 0.0),
        disp_vp_val * jnp.where(max_abs_vp > EPS, max_abs_vp * _D4V / dvp, 0.0),
    )

    # ideriv=2: collision 2nd-derivative; kept undivided in the RK4 max (matdat.F90:1498)
    tmax2 = jnp.asarray(0.0, dtype=jnp.float64)
    if "coll_stencil" in pre:
        tmax2 = jnp.max(jnp.abs(pre["coll_stencil"][0]))

    # field CFL: ES mode frequency (time_est_field), kinetic only
    tmax1 = jnp.maximum(tmax1, jnp.asarray(pre.get("tmax_field", 0.0), dtype=jnp.float64))

    em_streaming_factor = jnp.asarray(pre.get("em_streaming_cfl", 1.0), dtype=jnp.float64)
    tmax1 = tmax1 * em_streaming_factor

    # RK4 von Neumann (meth=2, matdat.F90:1498): tmax = max(tmax1/2.4, tmax2, tmax4/2.4)
    rk4 = jnp.asarray(2.4, dtype=jnp.float64)
    tmax = jnp.maximum(
        jnp.maximum(jnp.maximum(tmax1 / rk4, tmax2), tmax4 / rk4),
        jnp.asarray(40.0, dtype=jnp.float64),
    )

    if params is not None:
        fac_dtim_est = float(getattr(params, "fac_dtim_est", fac_dtim_est))
    fac = jnp.asarray(fac_dtim_est, dtype=jnp.float64)
    return jnp.where(tmax > EPS, fac / tmax, jnp.asarray(1e10, dtype=jnp.float64))


def estimate_timestep(
    phi: jnp.ndarray,
    pre: GKPre,
    bessel: jnp.ndarray,
    dt_input: float,
    safety_factor: float = 1.0,
    params: Optional["GKParams"] = None,
    apar: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Combined CFL: min(nonlinear ExB + EM apar, linear von Neumann)."""
    if params is not None:
        safety_factor = float(getattr(params, "fac_dtim_nl", safety_factor))
    dt_nl = estimate_nl_timestep(phi, pre, bessel, dt_input, safety_factor, apar=apar)
    if params is not None:
        dt_lin = estimate_linear_timestep(pre, params=params)
    else:
        dt_lin = estimate_linear_timestep(pre, safety_factor=1.0 / 3.0)
    return jnp.minimum(dt_nl, dt_lin)
