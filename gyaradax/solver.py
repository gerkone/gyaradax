"""
Gyrokinetic Vlasov-Poisson solver for the local flux-tube limit.

Supports both adiabatic-electron (single species) and kinetic-electron
(multi-species) configurations.

Implemented Equations:
The solver evolves the perturbed distribution function `f` in phase space.
Adiabatic: (vpar, mu, s, kx, ky).  Kinetic: (nsp, vpar, mu, s, kx, ky).

Active RHS Terms from the GKW formulation:
1. Term I   — Parallel Advection: v_par nabla_par f
2. Term II  — Drift Advection: v_d . nabla_perp f
3. Term III — Nonlinear ExB Advection: v_E . nabla_perp f (pseudospectral)
4. Term IV  — Trapping/Mirror: parallel velocity space advection
5. Term V   — Equilibrium Drive: v_E . nabla F_M
6. Term VII — Parallel Field Drive: v_par nabla_par phi coupling
7. Term VIII— Drift Field Drive: v_d . nabla phi coupling

Dissipation: parallel (4th order), velocity space, perpendicular hyper-diffusion.

Time Integration: Explicit RK4 with optional per-ky normalization (linear mode).
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import math
import functools
from typing import Dict, Tuple, Optional

from gyaradax import _EPS, stencils
from gyaradax.integrals import (
    get_integrals,
    j0,
    geom_tensors,
    calculate_phi,
    precompute_phi_kinetic,
    precompute_phi_adiabatic,
    calculate_phi_adiabatic,
)
from gyaradax.backends import create_ops
from gyaradax.params import GKParams
from gyaradax.types import GKPre, GKState
from gyaradax.backends.ops import SolverOps
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum  # noqa: F401
from einops import rearrange


def default_state(nky: int = 1) -> GKState:
    return GKState(
        time=jnp.array(0.0, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )


def kx_ky_grids(geometry: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    ky = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    if kx.ndim == 2:
        kx = kx[0]
    if ky.ndim == 2:
        ky = ky[:, 0]
    return kx, ky


def mode_amplitude(phi: jnp.ndarray, geometry: Dict[str, jnp.ndarray], eps: float) -> jnp.ndarray:
    """
    Per-ky mode amplitude over the connected kx chain containing kx=0.

    Matches GKW convention (diagnos_growth_freq.f90): only kx modes sharing
    the same mode_label as kx=0 contribute to the amplitude for each ky.
    """
    ds = jnp.asarray(geometry["ints"], dtype=jnp.float64)[0]
    mode_label = jnp.asarray(geometry["mode_label"], dtype=jnp.int32)  # (nkx, nky)
    ixzero = jnp.asarray(geometry["ixzero"], dtype=jnp.int32)

    # mask: kx modes in the same chain as kx=0, per ky
    chain_mask = mode_label == mode_label[ixzero, :]  # (nkx, nky)

    # sum |phi|^2 over s and chain kx only
    amp2 = ds * jnp.sum(jnp.abs(phi) ** 2 * chain_mask[None, :, :], axis=(0, 1))
    return jnp.sqrt(jnp.maximum(amp2, eps))


def normalize_per_ky(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    pre: Optional[Dict] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi = calculate_phi(geometry, df, params=params, pre=pre)
    amp_per_ky = mode_amplitude(phi, geometry, params.norm_eps)
    # only normalize modes with meaningful amplitude; leave dormant modes unchanged
    active = amp_per_ky > jnp.sqrt(params.norm_eps)
    inv = jnp.where(active, 1.0 / amp_per_ky, 1.0)
    inv_shape = (1,) * (df.ndim - 1) + (-1,)
    return df * jnp.reshape(inv, inv_shape), inv, amp_per_ky


def prime_factors_smallereq_than(number: int, max_prime: int) -> bool:
    i = 2
    n = int(number)
    while True:
        if n % i == 0:
            n //= i
        elif i == max_prime:
            return n == 1
        else:
            i += 1


def extended_firstdim_fft_size(nmod: int) -> Tuple[int, int]:
    posspace_size = 3 * nmod - 2
    if posspace_size % 2 != 0:
        posspace_size += 1
    while not prime_factors_smallereq_than(posspace_size, 7):
        posspace_size += 2
    for i in range(1, 9):
        cand = posspace_size + 2 * i
        if prime_factors_smallereq_than(cand, 2):
            posspace_size = cand
            break
    return posspace_size, int(math.floor(posspace_size / 2.0) + 1)


def extended_seconddim_fft_size(nx: int) -> int:
    dum = int(math.ceil(1.5 * float(nx + 1)) + 1)
    while not prime_factors_smallereq_than(dum, 7):
        dum += 1
    for i in range(1, 9):
        cand = dum + i
        if prime_factors_smallereq_than(cand, 2):
            dum = cand
            break
    return dum


def build_jind(nkx: int, mrad: int, ixzero: int) -> jnp.ndarray:
    ix = jnp.arange(nkx, dtype=jnp.int32)
    return jnp.where(ix >= ixzero, ix - ixzero, mrad + ix - ixzero)


def nonlinear_term_iii(
    df: jnp.ndarray,
    phi: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    pre: GKPre,
    efun_sign: float = 1.0,
    fft_prefactor: complex = 1.0 + 0.0j,
    exclude_zero_mode: bool = True,
    mixed_precision: bool = True,
    ops: Optional[SolverOps] = None,
    backend: str = "jax",
    use_z2z: bool = False,
) -> jnp.ndarray:
    """Nonlinear ExB advection via pseudospectral method. df is 5D."""
    if ops is None:
        ops = create_ops(pre, backend=backend, use_z2z=use_z2z, mixed_precision=mixed_precision)

    return ops.nonlinear_term_iii(
        df,
        phi,
        geometry,
        efun_sign=efun_sign,
        fft_prefactor=fft_prefactor,
        exclude_zero_mode=exclude_zero_mode,
    )


def estimate_nl_timestep(
    phi: jnp.ndarray,
    pre: Dict[str, jnp.ndarray],
    bessel: jnp.ndarray,
    dt_input: float,
    safety_factor: float = 0.95,
) -> jnp.ndarray:
    """CFL-adaptive timestep estimate from the nonlinear ExB velocity.

    Computes max|grad phi| in real space (dealiased grid) and returns
    dt_est = safety_factor * 2 / max_value, clamped to dt_input.

    Matches GKW's spectral CFL: non_linear_terms.F90 lines 1530-1777.

    Args:
        phi: Electrostatic potential (ns, nkx, nky).
        pre: Precomputed dict with FFT metadata and Bessel functions.
        bessel: Bessel J0 array for gyro-averaging phi. For adiabatic this
            is pre["bessel"] (nv, nmu, ns, nkx, nky). For kinetic, pass
            the ion Bessel (first species) since it has the largest FLR.
        dt_input: Maximum allowed timestep.
        safety_factor: CFL safety factor (default 0.95).

    Returns:
        Scalar dt estimate.
    """
    mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
    jind = pre["nl_jind"]
    kx2d, ky2d = pre["nl_kx2d"], pre["nl_ky2d"]

    bessel_s0 = bessel[0, 0, :, :, :]  # (ns, nkx, nky)

    def _max_grad_per_s(phi_s, bes_s):
        gyro_phi = bes_s * phi_s
        grad_y_k = 1j * ky2d * gyro_phi
        grad_x_k = 1j * kx2d * gyro_phi

        def _to_real(spec):
            return jnp.fft.irfft2(
                pack_half_spectrum(spec[None, None, :, :], jind, mrad, mphiw3),
                s=(mrad, mphi),
                axes=(-2, -1),
                norm="backward",
            )

        max_y = jnp.max(jnp.abs(_to_real(grad_y_k))) * mrad
        max_x = jnp.max(jnp.abs(_to_real(grad_x_k))) * mphi
        return jnp.maximum(max_y, max_x)

    max_vals = jax.vmap(_max_grad_per_s)(phi, bessel_s0)
    max_value = jnp.max(max_vals)

    dt_est = jnp.where(
        max_value > _EPS,
        jnp.asarray(safety_factor, dtype=jnp.float64) * 2.0 / max_value,
        jnp.asarray(dt_input, dtype=jnp.float64),
    )
    return jnp.minimum(dt_est, jnp.asarray(dt_input, dtype=jnp.float64))


def estimate_linear_timestep(
    pre: GKPre,
    params: "GKParams" = None,
    fac_dtim_est: float = 0.95,
    safety_factor: float = None,
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
        dt_par = jnp.where(max_upar > _EPS, safety_factor * sgr_dist / max_upar, 1e10)
        dt_trap = jnp.where(max_utrap > _EPS, safety_factor * dvp / max_utrap, 1e10)
        return jnp.minimum(dt_par, dt_trap)

    # max stencil coefficients: boundary D1/D4 = 24/12 = 2.0,
    # interior VPAR_D1 = 8/12, VPAR_D4 = 6/12
    _D1S = jnp.asarray(2.0, dtype=jnp.float64)
    _D4S = jnp.asarray(2.0, dtype=jnp.float64)
    _D1V = jnp.asarray(8.0 / 12.0, dtype=jnp.float64)
    _D4V = jnp.asarray(6.0 / 12.0, dtype=jnp.float64)

    # ideriv=1: streaming + trapping
    tmax1 = jnp.maximum(
        jnp.where(max_upar > _EPS, max_upar * _D1S / sgr_dist, 0.0),
        jnp.where(max_utrap > _EPS, max_utrap * _D1V / dvp, 0.0),
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
        disp_par_val * jnp.where(max_abs_par > _EPS, max_abs_par * _D4S / sgr_dist, 0.0),
        disp_vp_val * jnp.where(max_abs_vp > _EPS, max_abs_vp * _D4V / dvp, 0.0),
    )

    # field CFL: ES mode frequency (time_est_field), kinetic only
    tmax1 = jnp.maximum(tmax1, jnp.asarray(pre.get("tmax_field", 0.0), dtype=jnp.float64))

    # RK4 von Neumann (meth=2): divide by stability boundary, floor at 40
    rk4 = jnp.asarray(2.4, dtype=jnp.float64)
    tmax = jnp.maximum(jnp.maximum(tmax1 / rk4, tmax4 / rk4), jnp.asarray(40.0, dtype=jnp.float64))

    fac = jnp.asarray(fac_dtim_est, dtype=jnp.float64)
    return jnp.where(tmax > _EPS, fac / tmax, jnp.asarray(1e10, dtype=jnp.float64))


def estimate_timestep(
    phi: jnp.ndarray,
    pre: GKPre,
    bessel: jnp.ndarray,
    dt_input: float,
    safety_factor: float = 0.95,
    params: "GKParams" = None,
) -> jnp.ndarray:
    """Combined CFL: min(nonlinear ExB, linear von Neumann)."""
    dt_nl = estimate_nl_timestep(phi, pre, bessel, dt_input, safety_factor)
    if params is not None:
        dt_lin = estimate_linear_timestep(pre, params=params)
    else:
        dt_lin = estimate_linear_timestep(pre, safety_factor=1.0 / 3.0)
    return jnp.minimum(dt_nl, dt_lin)


def _precompute_shared(
    geometry, params, kx, ky, ns, nkx, nky, vpgr, mugr, bn, ffun, gfun, dfun, efun
):
    """Species-independent precomputed quantities shared by both paths."""
    pos_par = jnp.asarray(geometry["pos_par_grid_class"], dtype=jnp.int32)
    ixzero = jnp.asarray(geometry.get("ixzero", jnp.argmin(jnp.abs(kx))), dtype=jnp.int32)
    iyzero = jnp.asarray(geometry.get("iyzero", jnp.argmin(jnp.abs(ky))), dtype=jnp.int32)
    mphi, mphiw3 = extended_firstdim_fft_size(nky)
    mrad = extended_seconddim_fft_size(nkx)

    def _parallel_coefficients(pos_par_class, table):
        idx = jnp.clip(jnp.asarray(pos_par_class, dtype=jnp.int32) + 2, 0, 4)
        return jnp.moveaxis(jnp.asarray(table)[idx] / 12.0, -1, 0)

    kx_b = jnp.reshape(kx, (1, 1, 1, -1, 1))
    ky_b = jnp.reshape(ky, (1, 1, 1, 1, -1))

    # kxmax is the maximum absolute kx value, used for hyper-dissipation normalisation
    kxmax = jnp.max(jnp.abs(kx))

    hyper = -(
        jnp.abs(params.disp_y)
        * (ky_b / jnp.maximum(params.kymax, _EPS)) ** jnp.where(params.disp_y < 0.0, 2.0, 4.0)
        + jnp.abs(params.disp_x)
        * (kx_b / jnp.maximum(kxmax, _EPS)) ** jnp.where(params.disp_x < 0.0, 2.0, 4.0)
    )

    return {
        "kx_b": kx_b,
        "ky_b": ky_b,
        "hyper": hyper,
        "kxmax": kxmax,
        "s_d1_ipos": _parallel_coefficients(pos_par, stencils.D1_IPW_POS),
        "s_d1_ineg": _parallel_coefficients(pos_par, stencils.D1_IPW_NEG),
        "s_d4_ipos": _parallel_coefficients(pos_par, stencils.D4_IPW_POS),
        "s_d4_ineg": _parallel_coefficients(pos_par, stencils.D4_IPW_NEG),
        "dvp": geometry["dvp"],
        "sgr_dist": geometry["sgr_dist"],
        "ixzero": ixzero,
        "iyzero": iyzero,
        "nl_mphi": mphi,
        "nl_mphiw3": mphiw3,
        "nl_mrad": mrad,
        "nl_fft_scale": jnp.asarray(float(mrad * mphi), dtype=jnp.float64),
        "nl_jind": build_jind(nkx, mrad, ixzero),
        "nl_kx2d": jnp.broadcast_to(jnp.reshape(kx, (nkx, 1)), (nkx, nky)),
        "nl_ky2d": jnp.broadcast_to(jnp.reshape(ky, (1, nky)), (nkx, nky)),
        "nl_dum_s": -jnp.asarray(efun, dtype=jnp.float64),
        "s_shift": jnp.asarray(geometry["s_shift"], dtype=jnp.int32),
        "kx_shift": jnp.asarray(geometry["kx_shift"], dtype=jnp.int32),
        "valid_shift": jnp.asarray(geometry["valid_shift"], dtype=jnp.bool_),
    }


def _fuse_stencils(
    upar,
    abs_par,
    term7_fac,
    disp_par,
    sgr_dist,
    s_d1_ipos,
    s_d1_ineg,
    s_d4_ipos,
    s_d4_ineg,
    stencil_ndim,
):
    """Compute fused streaming + dissipation stencils.

    Works for both 5D (adiabatic) and 6D (kinetic) coefficient arrays.
    stencil_ndim is the number of dimensions for the stencil coefficient
    rearrange pattern: 5 for adiabatic, 6 for kinetic.
    """
    if stencil_ndim == 5:
        # adiabatic: arrays are (nv, nmu, ns, nkx, nky)
        pat_coeff = "v m s x y -> 1 v m s x y"
        pat_stencil = "i s x y -> i 1 1 s x y"
    else:
        # kinetic: arrays are (nsp, nv, nmu, ns, nkx, nky)
        pat_coeff = "sp v m s x y -> 1 sp v m s x y"
        pat_stencil = "i s x y -> i 1 1 1 s x y"

    s_d1p = rearrange(s_d1_ipos, pat_stencil)
    s_d1n = rearrange(s_d1_ineg, pat_stencil)
    s_d4p = rearrange(s_d4_ipos, pat_stencil)
    s_d4n = rearrange(s_d4_ineg, pat_stencil)

    upar_sign = rearrange(jnp.sign(upar), pat_coeff)
    s_d1_upar = jnp.where(upar_sign > 0, s_d1p, s_d1n)
    s_d4_upar = jnp.where(upar_sign > 0, s_d4p, s_d4n)

    t7_sign = rearrange(jnp.sign(term7_fac), pat_coeff)
    s_d1_t7 = jnp.where(t7_sign < 0, s_d1p, s_d1n)

    s_total_upar = (
        rearrange(upar, pat_coeff) * s_d1_upar
        + jnp.asarray(disp_par, dtype=jnp.float64) * rearrange(abs_par, pat_coeff) * s_d4_upar
    ) / jnp.asarray(sgr_dist, dtype=jnp.float64)

    s_total_t7 = (rearrange(term7_fac, pat_coeff) * s_d1_t7) / jnp.asarray(
        sgr_dist, dtype=jnp.float64
    )

    return s_total_upar, s_total_t7


def _compute_species_coeffs(
    mas,
    signz,
    vthrat,
    tmp,
    de,
    rln,
    rlt,
    vpgr,
    mugr,
    bn,
    ffun,
    gfun,
    efun,
    dfun,
    kx,
    ky,
    little_g,
    params,
    ndim,
    dgrid=1.0,
    tgrid=1.0,
):
    """Compute species-dependent RHS coefficients.

    When ndim=5, species params are scalars and output arrays are 5D.
    When ndim=6, species params have shape (nsp,) and output arrays are 6D.
    """
    if ndim == 5:
        # adiabatic: scalar species params, 5D grid arrays
        vp2 = jnp.reshape(vpgr**2, (-1, 1, 1, 1, 1))
        vp = jnp.reshape(vpgr, (-1, 1, 1, 1, 1))
        mu = jnp.reshape(mugr, (1, -1, 1, 1, 1))
        bn_b = jnp.reshape(bn, (1, 1, -1, 1, 1))
        ffun_b = jnp.reshape(ffun, (1, 1, -1, 1, 1))
        gfun_b = jnp.reshape(gfun, (1, 1, -1, 1, 1))
        efun_b = jnp.reshape(efun, (1, 1, -1, 1, 1))
        kx_b = jnp.reshape(kx, (1, 1, 1, -1, 1))
        ky_b = jnp.reshape(ky, (1, 1, 1, 1, -1))

        def g_shape(arr):
            return jnp.reshape(arr, (1, 1, -1, 1, 1))

        def d_shape(arr):
            return jnp.reshape(arr, (1, 1, -1, 1, 1))

        sz = jnp.where(jnp.abs(signz) < _EPS, 1.0, signz)
    else:
        # kinetic: per-species params, 6D arrays
        nsp = mas.shape[0]

        def r6(arr):
            return arr.reshape(nsp, 1, 1, 1, 1, 1)

        mas, signz, vthrat, tmp, de, rln, rlt = (
            r6(mas),
            r6(signz),
            r6(vthrat),
            r6(tmp),
            r6(de),
            r6(rln),
            r6(rlt),
        )
        vp2 = jnp.reshape(vpgr**2, (1, -1, 1, 1, 1, 1))
        vp = jnp.reshape(vpgr, (1, -1, 1, 1, 1, 1))
        mu = jnp.reshape(mugr, (1, 1, -1, 1, 1, 1))
        bn_b = jnp.reshape(bn, (1, 1, 1, -1, 1, 1))
        ffun_b = jnp.reshape(ffun, (1, 1, 1, -1, 1, 1))
        gfun_b = jnp.reshape(gfun, (1, 1, 1, -1, 1, 1))
        efun_b = jnp.reshape(efun, (1, 1, 1, -1, 1, 1))
        kx_b = jnp.reshape(kx, (1, 1, 1, 1, -1, 1))
        ky_b = jnp.reshape(ky, (1, 1, 1, 1, 1, -1))

        def g_shape(arr):
            return jnp.reshape(arr, (1, 1, 1, -1, 1, 1))

        def d_shape(arr):
            return jnp.reshape(arr, (1, 1, 1, -1, 1, 1))

        sz = jnp.where(jnp.abs(signz) < _EPS, 1.0, signz)

    # krloc
    krloc_sq = (
        ky_b**2 * g_shape(little_g[:, 0])
        + 2.0 * ky_b * kx_b * g_shape(little_g[:, 1])
        + kx_b**2 * g_shape(little_g[:, 2])
    )
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, _EPS))

    # Bessel J0
    b_arg = (
        mas * vthrat * krloc * jnp.sqrt(jnp.maximum(2.0 * mu / jnp.maximum(bn_b, _EPS), _EPS)) / sz
    )
    bessel = j0(b_arg)

    # Maxwellian
    t_rat = tmp / jnp.asarray(tgrid, dtype=jnp.float64)
    fmax = (
        (de / jnp.asarray(dgrid, dtype=jnp.float64))
        * jnp.exp(-(vp2 + 2.0 * bn_b * mu) / t_rat)
        / (jnp.sqrt(t_rat * jnp.pi) ** 3)
    )
    et = (vp2 + 2.0 * bn_b * mu) / t_rat - 1.5
    dmax_ek = (rln + rlt * et) * fmax * (efun_b * ky_b)

    # drifts
    ed = vp2 + bn_b * mu
    drift_x = ed * d_shape(dfun[:, 0]) / sz
    drift_y = ed * d_shape(dfun[:, 1]) / sz

    # characteristic speeds
    upar = -ffun_b * vthrat * vp
    utrap = vthrat * mu * bn_b * gfun_b

    # dissipation speeds for idisp=2: RMS velocity (matches GKW linear_terms.f90:643,911)
    vp_rms = jnp.asarray(
        params.vpgr_rms if hasattr(params, "vpgr_rms") else jnp.sqrt(jnp.mean(vpgr**2)),
        dtype=jnp.float64,
    )
    mu_rms = jnp.asarray(
        params.mugr_rms if hasattr(params, "mugr_rms") else jnp.sqrt(jnp.mean(mugr**2)),
        dtype=jnp.float64,
    )
    idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
    use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))
    abs_par = jnp.where(use_abs, jnp.abs(upar), jnp.abs(ffun_b * vthrat * vp_rms))
    abs_vp = jnp.where(use_abs, jnp.abs(utrap), jnp.abs(vthrat * bn_b * gfun_b * mu_rms))

    term7_fac = -signz * ffun_b * vthrat * vp * fmax / tmp

    return {
        "bessel": bessel,
        "fmaxwl": fmax,
        "dmaxwel_fm_ek": dmax_ek,
        "drift_x": drift_x,
        "drift_y": drift_y,
        "upar": upar,
        "utrap": utrap,
        "abs_dum2_par": abs_par,
        "abs_dum2_vp": abs_vp,
        "term7_fac": term7_fac,
        "tmp0": float(tmp) if ndim == 5 else jnp.asarray(tmp.squeeze(), dtype=jnp.float64),
        "signz0": float(signz) if ndim == 5 else jnp.asarray(signz.squeeze(), dtype=jnp.float64),
    }


def linear_precompute(geometry: Dict[str, jnp.ndarray], params: GKParams) -> "GKPre":
    """Precompute static geometry-dependent coefficients and Bessel terms."""
    kx, ky = kx_ky_grids(geometry)
    ns, nkx, nky = len(geometry["ints"]), int(kx.shape[0]), int(ky.shape[0])

    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)
    ffun = jnp.asarray(geometry["ffun"], dtype=jnp.float64)
    gfun = jnp.asarray(geometry.get("gfun", jnp.zeros_like(bn)), dtype=jnp.float64)
    dfun = jnp.asarray(
        geometry.get("dfun", jnp.zeros((ns, 3), dtype=jnp.float64)), dtype=jnp.float64
    )
    efun = jnp.asarray(geometry.get("efun", jnp.ones_like(bn)), dtype=jnp.float64)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)

    # normalisation scalars for the Maxwellian — sourced from GKW data files when
    # available, otherwise default to 1.0 (no rescaling)
    dgrid = float(geometry.get("dgrid", 1.0))
    tgrid = float(geometry.get("tgrid", 1.0))

    # species-independent shared quantities
    out = _precompute_shared(
        geometry, params, kx, ky, ns, nkx, nky, vpgr, mugr, bn, ffun, gfun, dfun, efun
    )

    if not params.adiabatic_electrons:
        # kinetic: per-species arrays with leading nsp dimension
        mas_arr = jnp.asarray(params.mas, dtype=jnp.float64)
        tmp_arr = jnp.asarray(params.tmp, dtype=jnp.float64)
        nsp = int(mas_arr.shape[0])
        # vthrat is derived from species temperatures and masses
        vthrat_arr = jnp.sqrt(tmp_arr / jnp.maximum(mas_arr, _EPS))
        sp = _compute_species_coeffs(
            mas_arr,
            jnp.asarray(params.signz, dtype=jnp.float64),
            vthrat_arr,
            tmp_arr,
            jnp.asarray(params.de, dtype=jnp.float64),
            jnp.asarray(params.rln, dtype=jnp.float64),
            jnp.asarray(params.rlt, dtype=jnp.float64),
            vpgr,
            mugr,
            bn,
            ffun,
            gfun,
            efun,
            dfun,
            kx,
            ky,
            little_g,
            params,
            ndim=6,
            dgrid=dgrid,
            tgrid=tgrid,
        )
        # override vpgr_rms/mugr_rms if available
        if "vpgr_rms" in geometry:
            vp_rms = jnp.asarray(geometry["vpgr_rms"], dtype=jnp.float64)
            mu_rms = jnp.asarray(geometry.get("mugr_rms", 1.0), dtype=jnp.float64)
            vthrat_6 = vthrat_arr.reshape(nsp, 1, 1, 1, 1, 1)
            ffun_6 = jnp.reshape(ffun, (1, 1, 1, -1, 1, 1))
            bn_6 = jnp.reshape(bn, (1, 1, 1, -1, 1, 1))
            gfun_6 = jnp.reshape(gfun, (1, 1, 1, -1, 1, 1))
            idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
            use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))
            sp["abs_dum2_par"] = jnp.where(
                use_abs, jnp.abs(sp["upar"]), jnp.abs(ffun_6 * vthrat_6 * vp_rms)
            )
            sp["abs_dum2_vp"] = jnp.where(
                use_abs,
                jnp.abs(sp["utrap"]),
                jnp.abs(vthrat_6 * bn_6 * gfun_6 * mu_rms),
            )

        sp["s_total_upar"], sp["s_total_t7"] = _fuse_stencils(
            sp["upar"],
            sp["abs_dum2_par"],
            sp["term7_fac"],
            params.disp_par,
            geometry["sgr_dist"],
            out["s_d1_ipos"],
            out["s_d1_ineg"],
            out["s_d4_ipos"],
            out["s_d4_ineg"],
            stencil_ndim=6,
        )
        out.update(sp)
        out["geom_tensors"] = None
        out["nsp"] = nsp
        # kx_b/ky_b computed in _compute_species_coeffs but not returned - add them here
        # For kinetic case, need 6D shape (1, 1, 1, 1, nkx, 1) to broadcast with 6D drift arrays
        out["kx_b"] = jnp.reshape(kx, (1, 1, 1, 1, -1, 1))
        out["ky_b"] = jnp.reshape(ky, (1, 1, 1, 1, 1, -1))

        # precompute kinetic phi solve arrays (avoids recomputing bessel/gamma per RHS call)
        phi_w, phi_d = precompute_phi_kinetic(geometry)
        out["phi_weight"] = phi_w
        out["phi_diag"] = phi_d

        # field CFL: ES limit of Alfvén wave (time_est_field, matdat.F90:1859)
        signz_arr = jnp.asarray(params.signz, dtype=jnp.float64)
        de_arr = jnp.asarray(params.de, dtype=jnp.float64)
        mir = jnp.sum(jnp.where(signz_arr > 0, mas_arr * de_arr, 0.0))
        mer = jnp.sum(jnp.where(signz_arr < 0, mas_arr / jnp.maximum(de_arr, _EPS), 0.0))
        ky_min = jnp.where(nky > 1, ky[1], ky[0])
        kmin2 = ky_min**2 * little_g[:, 0]
        q_val = jnp.asarray(geometry.get("q", getattr(params, "q", 1.0)), dtype=jnp.float64)
        field_period = (
            2.0
            * jnp.pi
            * q_val
            * jnp.asarray(geometry["sgr_dist"], dtype=jnp.float64)
            * bn
            * jnp.sqrt(jnp.maximum(mir * kmin2 * mer, _EPS))
        )
        time_field = jnp.min(jnp.where(field_period > _EPS, field_period, 1e30))
        out["tmax_field"] = jnp.where(time_field < 1e20, 1.0 / time_field, 0.0)
    else:
        # adiabatic: scalar species params, 5D arrays
        mas_val = jnp.asarray(params.mas, dtype=jnp.float64)
        tmp_val = jnp.asarray(params.tmp, dtype=jnp.float64)
        # vthrat is derived from species temperatures and masses
        vthrat_val = jnp.sqrt(tmp_val / jnp.maximum(mas_val, _EPS))
        sp = _compute_species_coeffs(
            params.mas,
            params.signz,
            vthrat_val,
            params.tmp,
            params.de,
            params.rln,
            params.rlt,
            vpgr,
            mugr,
            bn,
            ffun,
            gfun,
            efun,
            dfun,
            kx,
            ky,
            little_g,
            params,
            ndim=5,
            dgrid=dgrid,
            tgrid=tgrid,
        )
        # override vpgr_rms/mugr_rms if available
        if "vpgr_rms" in geometry:
            vp_rms = jnp.asarray(geometry["vpgr_rms"], dtype=jnp.float64)
            mu_rms = jnp.asarray(geometry.get("mugr_rms", 1.0), dtype=jnp.float64)
            ffun_b = jnp.reshape(ffun, (1, 1, -1, 1, 1))
            bn_b = jnp.reshape(bn, (1, 1, -1, 1, 1))
            gfun_b = jnp.reshape(gfun, (1, 1, -1, 1, 1))
            idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
            use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))
            sp["abs_dum2_par"] = jnp.where(
                use_abs, jnp.abs(sp["upar"]), jnp.abs(ffun_b * vthrat_val * vp_rms)
            )
            sp["abs_dum2_vp"] = jnp.where(
                use_abs,
                jnp.abs(sp["utrap"]),
                jnp.abs(vthrat_val * bn_b * gfun_b * mu_rms),
            )

        sp["s_total_upar"], sp["s_total_t7"] = _fuse_stencils(
            sp["upar"],
            sp["abs_dum2_par"],
            sp["term7_fac"],
            params.disp_par,
            geometry["sgr_dist"],
            out["s_d1_ipos"],
            out["s_d1_ineg"],
            out["s_d4_ipos"],
            out["s_d4_ineg"],
            stencil_ndim=5,
        )
        out.update(sp)
        out["geom_tensors"] = geom_tensors(geometry, params=params)

        # precompute adiabatic phi solve arrays
        pw, pcw, tmp, de, signz, gamma, ints, has_zonal, ixzero, iyzero = precompute_phi_adiabatic(
            geometry, params
        )
        out["phi_weight"] = pw
        out["phi_corr_weight"] = pcw
        out["phi_tmp"] = tmp
        out["phi_de"] = de
        out["phi_signz"] = signz
        out["phi_gamma"] = gamma
        out["phi_ints"] = ints
        out["phi_has_zonal"] = has_zonal
        out["phi_ixzero"] = ixzero
        out["phi_iyzero"] = iyzero

    return GKPre(out)


def init_f(
    geometry: Dict[str, jnp.ndarray],
    finit: str = "cosine2",
    amp_init_real: float = 1.0e-4,
    amp_init_imag: float = 0.0,
    normalize_per_toroidal_mode: bool = False,
    norm_eps: float = 1.0e-14,
    n_species: int = 1,
    seed: int = 42,
) -> jnp.ndarray:
    """Initialize the distribution function.

    Supported finit modes (matching GKW):
        cosine2 (default): amp * (cos(2*pi*s) + 1), flat in velocity space
        cosine:  amp * cos(2*pi*s), flat in velocity space
        cosine3: like cosine2 but weighted by exp(-E) in velocity space
        sine:    amp * de(is) * (sin(2*pi*s) + 1), density-weighted
        noise:   uniform random on [-1, 1] (real + imag)
        gnoise:  gaussian random (Box-Muller transform)
        zonal:   Rosenbluth-Hinton test — only ky=0, kx=±1 with Maxwellian weight
    """
    nv, nmu, ns, nkx, nky = (
        len(geometry["intvp"]),
        len(geometry["intmu"]),
        len(geometry["ints"]),
        len(geometry["kxrh"]),
        len(geometry["krho"]),
    )
    sgrid = jnp.asarray(geometry.get("sgrid", jnp.linspace(-0.5, 0.5, ns)), dtype=jnp.float64)
    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)

    amp = jnp.asarray(amp_init_real, dtype=jnp.float64) + 1j * jnp.asarray(
        amp_init_imag, dtype=jnp.float64
    )

    shape_5d = (nv, nmu, ns, nkx, nky)
    shape_6d = (n_species, nv, nmu, ns, nkx, nky)
    full_shape = shape_6d if n_species > 1 else shape_5d

    # velocity-space Maxwellian: (n/n_grid) * exp(-(vpar^2 + 2*mu*B)/T) / (sqrt(T*pi))^3
    # matches GKW's fmaxwl in components.f90 (with dens=dref=tref=1).
    vp2 = vpgr**2  # (nv,)
    tmp_val = jnp.asarray(geometry.get("tmp", jnp.ones(1)), dtype=jnp.float64)
    if tmp_val.ndim > 0:
        tmp_val = tmp_val[0]
    tgrid_val = jnp.asarray(geometry.get("tgrid", jnp.ones(1)), dtype=jnp.float64)
    if tgrid_val.ndim > 0:
        tgrid_val = tgrid_val[0]
    t_rat = tmp_val / tgrid_val
    energy = vp2[:, None, None] + 2.0 * mugr[None, :, None] * bn[None, None, :]
    maxwellian_env = jnp.exp(-energy / t_rat) / (jnp.sqrt(t_rat * jnp.pi) ** 3)  # (nv, nmu, ns)

    if finit in ("noise", "gnoise"):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        if finit == "gnoise":
            noise_real = jax.random.normal(k1, full_shape)
            noise_imag = jax.random.normal(k2, full_shape)
        else:
            noise_real = jax.random.uniform(k1, full_shape, minval=-1.0, maxval=1.0)
            noise_imag = jax.random.uniform(k2, full_shape, minval=-1.0, maxval=1.0)
        df = amp * (noise_real + 1j * noise_imag)

    elif finit == "cosine2":
        # amp * (cos(2*pi*s) + 1), uniform in velocity
        prof_s = amp * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)  # (ns,)
        df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "cosine":
        prof_s = amp * jnp.cos(2.0 * jnp.pi * sgrid)
        df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "cosine3":
        # amp * (cos(2*pi*s) + 1) * exp(-(vpar^2 + 2*mu*B))
        prof_s = amp * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)
        df = _broadcast_profile(prof_s, maxwellian_env, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "sine":
        # amp * de(is) * (sin(2*pi*s) + 1)
        de = jnp.asarray(geometry.get("de", jnp.ones(max(n_species, 1))), dtype=jnp.float64)
        prof_s = amp * (jnp.sin(2.0 * jnp.pi * sgrid) + 1.0)
        if n_species > 1 and de.ndim >= 1 and de.shape[0] > 1:
            # (nsp, ns)
            prof_2d = prof_s[None, :] * de[:, None]
            df = jnp.broadcast_to(prof_2d[:, None, None, :, None, None], shape_6d)
        else:
            de_val = float(de) if de.ndim == 0 else float(de[0])
            prof_s = prof_s * de_val
            df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "zonal":
        # Rosenbluth-Hinton test: only the zonal mode (ky=0) is initialized.
        # in spectral space, set kx = ±1 around kx=0 with ±i*amp*fmaxwl/2
        # to produce a sin(kx*x) radial density perturbation.  Only ions
        # (signz > 0) are initialized.  (GKW init.f90:1471-1514)
        kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
        ixzero = int(jnp.argmin(jnp.abs(kxrh)).item())
        iy0 = int(
            jnp.asarray(
                geometry.get("iyzero", jnp.argmin(jnp.abs(jnp.asarray(geometry["krho"]))))
            ).item()
        )

        df = jnp.zeros(full_shape, dtype=jnp.complex128)

        if n_species > 1:
            signz = jnp.asarray(geometry.get("signz", jnp.ones(n_species)), dtype=jnp.float64)
            for isp in range(n_species):
                if float(signz[isp]) > 0:
                    if ixzero > 0:
                        df = df.at[isp, :, :, :, ixzero - 1, iy0].set(
                            -1j * amp * maxwellian_env / 2.0
                        )
                    if ixzero < nkx - 1:
                        df = df.at[isp, :, :, :, ixzero + 1, iy0].set(
                            1j * amp * maxwellian_env / 2.0
                        )
        else:
            if ixzero > 0:
                df = df.at[:, :, :, ixzero - 1, iy0].set(-1j * amp * maxwellian_env / 2.0)
            if ixzero < nkx - 1:
                df = df.at[:, :, :, ixzero + 1, iy0].set(1j * amp * maxwellian_env / 2.0)
        return df.astype(jnp.complex128)

    else:
        raise ValueError(f"unknown finit: {finit}")

    df = df.astype(jnp.complex128)

    # zero out the zonal mode (ky=0) — not for zonal init which IS the zonal mode
    if nky > 1:
        iy0 = int(
            jnp.asarray(
                geometry.get("iyzero", jnp.argmin(jnp.abs(jnp.asarray(geometry["krho"]))))
            ).item()
        )
        df = df.at[..., iy0].set(0.0)

    if normalize_per_toroidal_mode:
        df, _, _ = normalize_per_ky(df, geometry, GKParams(norm_eps=norm_eps))
    return df


def _broadcast_profile(prof_s, vel_env, n_species, nv, nmu, ns, nkx, nky):
    """Broadcast a parallel profile (and optional velocity envelope) to full shape."""
    if vel_env is not None:
        # vel_env: (nv, nmu, ns), prof_s: (ns,)
        base = vel_env * prof_s[None, None, :]  # (nv, nmu, ns)
        if n_species > 1:
            return jnp.broadcast_to(
                base[None, :, :, :, None, None], (n_species, nv, nmu, ns, nkx, nky)
            )
        else:
            return jnp.broadcast_to(base[:, :, :, None, None], (nv, nmu, ns, nkx, nky))
    else:
        # flat in velocity
        if n_species > 1:
            prof = jnp.broadcast_to(prof_s[None, :], (n_species, ns))
            return jnp.broadcast_to(
                prof[:, None, None, :, None, None], (n_species, nv, nmu, ns, nkx, nky)
            )
        else:
            return jnp.broadcast_to(prof_s[None, None, :, None, None], (nv, nmu, ns, nkx, nky))


def advance_state(
    state: GKState,
    params: GKParams,
    is_window_end: jnp.ndarray,
    per_mode_amp: jnp.ndarray,
    per_mode_norm_fac: jnp.ndarray,
    dt_used: Optional[jnp.ndarray] = None,
) -> GKState:
    dt = dt_used if dt_used is not None else jnp.array(params.dt, dtype=jnp.float64)
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    new_time = state.time + dt
    valid_growth = jnp.logical_and(
        state.window_start_amp > params.norm_eps, per_mode_amp > params.norm_eps
    )
    steps_in_window = jnp.mod(new_step - 1, params.naverage) + 1
    growth_dt = jnp.array(params.dt * steps_in_window, dtype=jnp.float64)
    growth_rate = jnp.where(
        valid_growth,
        jnp.log(per_mode_amp / state.window_start_amp) / growth_dt,
        state.last_growth_rate,
    )
    # post-normalization amplitude: linear → amp*(1/amp)=1, nonlinear → amp*1=amp
    new_window_start_amp = jnp.where(
        is_window_end, per_mode_amp * per_mode_norm_fac, state.window_start_amp
    )
    return GKState(
        time=new_time,
        step=new_step,
        accumulated_norm_factor=state.accumulated_norm_factor * per_mode_norm_fac,
        window_start_amp=new_window_start_amp,
        last_growth_rate=growth_rate,
    )


def _compute_phi(df, geometry, params, pre):
    """Compute phi via the appropriate solver."""
    if params.adiabatic_electrons and "phi_weight" in pre and "phi_corr_weight" in pre:
        return calculate_phi_adiabatic(
            df,
            phi_weight=pre["phi_weight"],
            phi_corr_weight=pre["phi_corr_weight"],
            tmp=pre["phi_tmp"],
            de=pre["phi_de"],
            signz=pre["phi_signz"],
            gamma=pre["phi_gamma"],
            ints=pre["phi_ints"],
            has_zonal=pre["phi_has_zonal"],
            ixzero=pre["phi_ixzero"],
            iyzero=pre["phi_iyzero"],
        )
    else:
        return calculate_phi(geometry, df, params=params, pre=pre)


def gkstep_single(
    prev_df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    pre: GKPre,
    ops: Optional[SolverOps] = None,
    dt_override: Optional[jnp.ndarray] = None,
) -> Tuple[
    jnp.ndarray,
    Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    GKState,
]:
    """Single small-step RK4 time integration with backend dispatch."""
    if ops is None:
        ops = create_ops(
            pre,
            backend=params.backend,
            use_z2z=params.use_z2z,
            mixed_precision=params.mixed_precision,
        )

    dt = dt_override if dt_override is not None else jnp.array(params.dt, dtype=jnp.float64)

    def _rhs(df):
        phi_local = _compute_phi(df, geometry, params, pre)
        rhs = ops.linear_rhs(df, phi_local, geometry, params, pre)
        if params.non_linear:
            rhs = rhs + ops.nonlinear_term_iii(df, phi_local, geometry)
        return rhs

    # RK4 — expanded accumulation lets XLA read k1..k4 and prev_df in one fused kernel
    k1 = _rhs(prev_df)
    k2 = _rhs(prev_df + 0.5 * dt * k1)
    k3 = _rhs(prev_df + 0.5 * dt * k2)
    k4 = _rhs(prev_df + dt * k3)
    dt6 = dt / 6.0
    dt3 = dt / 3.0
    next_df_raw = prev_df + dt6 * k1 + dt3 * k2 + dt3 * k3 + dt6 * k4

    # post-step: normalization and amplitude tracking
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    is_window_end = jnp.equal(jnp.mod(new_step, params.naverage), 0)

    if params.non_linear:
        phi = _compute_phi(next_df_raw, geometry, params, pre)
        current_amp = mode_amplitude(phi, geometry, params.norm_eps)
        next_df = next_df_raw
        norm_factor = jnp.ones_like(state.accumulated_norm_factor)
    else:

        def _apply_norm(_):
            return normalize_per_ky(next_df_raw, geometry, params, pre=pre)

        def _skip_norm(_):
            phi_curr = _compute_phi(next_df_raw, geometry, params, pre)
            amp_curr = mode_amplitude(phi_curr, geometry, params.norm_eps)
            return (next_df_raw, jnp.ones_like(state.accumulated_norm_factor), amp_curr)

        next_df, norm_factor, current_amp = jax.lax.cond(
            is_window_end, _apply_norm, _skip_norm, operand=None
        )
        phi = _compute_phi(next_df, geometry, params, pre)

    z = jnp.array(0.0, dtype=jnp.float64)
    next_state = advance_state(state, params, is_window_end, current_amp, norm_factor, dt_used=dt)
    return next_df, (phi, (z, z, z)), next_state


@functools.partial(jax.jit, static_argnames=("n_steps",))
def gksolve(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int = 1,
    pre: Optional[GKPre] = None,
) -> Tuple[
    jnp.ndarray,
    Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    GKState,
]:
    """Gyrokinetics solver forward.

    Executes multiple time steps via jax.lax.scan.
    When params.adaptive_dt is True, uses CFL-adaptive timestep with
    one-step lag (current step uses CFL estimate from previous step's phi).
    Returns (final_df, (final_phi, final_fluxes), final_state).
    """
    if pre is None:
        pre = linear_precompute(geometry, params)

    ops = create_ops(
        pre, backend=params.backend, use_z2z=params.use_z2z, mixed_precision=params.mixed_precision
    )

    if params.adaptive_dt and params.non_linear:
        # adaptive CFL path: carry dt as part of scan state
        dt_input = jnp.array(params.dt, dtype=jnp.float64)
        cfl_safety = jnp.array(params.cfl_safety, dtype=jnp.float64)

        def _scan_body(carry, _):
            curr_df, curr_state, curr_dt = carry
            next_df, out, next_state = gkstep_single(
                curr_df, geometry, params, curr_state, pre, ops, dt_override=curr_dt
            )
            # estimate next dt from current phi (one-step lag)
            phi_for_cfl = out[0]  # phi from gkstep_single
            bessel_for_cfl = pre["bessel"]
            if not params.adiabatic_electrons:
                bessel_for_cfl = bessel_for_cfl[0]  # ion Bessel (largest FLR), drop species dim
            next_dt = estimate_timestep(
                phi_for_cfl,
                pre,
                bessel_for_cfl,
                dt_input=dt_input,
                safety_factor=cfl_safety,
                params=params,
            )
            return (next_df, next_state, next_dt), None

        # conservative cap for the first step (one-step lag has no phi CFL yet)
        init_dt = jnp.minimum(dt_input, estimate_linear_timestep(pre, safety_factor=0.5))
        (final_df, final_state, _), _ = jax.lax.scan(
            _scan_body, (df, state, init_dt), None, length=n_steps
        )
    else:
        # fixed dt path
        def _scan_body(carry, _):
            curr_df, curr_state = carry
            next_df, out, next_state = gkstep_single(
                curr_df, geometry, params, curr_state, pre, ops
            )
            return (next_df, next_state), None

        (final_df, final_state), _ = jax.lax.scan(_scan_body, (df, state), None, length=n_steps)

    phi, fluxes = get_integrals(
        final_df,
        geometry,
        params=params,
        pre=pre,
        adiabatic_electrons=params.adiabatic_electrons,
    )
    return final_df, (phi, fluxes), final_state
