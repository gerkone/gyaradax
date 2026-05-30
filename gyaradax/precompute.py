"""Linear precompute helpers for the gyrokinetic solver."""

from __future__ import annotations

import math
from typing import Dict, Tuple, cast

import jax.numpy as jnp
from einops import rearrange

import gyaradax.stencils as stencils
from gyaradax.collisions import precompute_collisions
from gyaradax.constants import EPS
from gyaradax.integrals import (
    geom_tensors,
    j0,
    precompute_bpar,
    precompute_phi_adiabatic,
    precompute_phi_kinetic,
)
from gyaradax.jax_config import enable_x64
from gyaradax.params import GKParams
from gyaradax.state import GKPre

enable_x64()


def kx_ky_grids(geometry: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    ky = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    if kx.ndim == 2:
        kx = kx[0]
    if ky.ndim == 2:
        ky = ky[:, 0]
    return kx, ky


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

    # NL CFL length-inverse factors: lxinv = 1/lx with lx = 2π/kx_min
    # (GKW mode.f90:740). Single-mode runs fall back to the grid max.
    ixz_arr = jnp.asarray(geometry.get("ixzero", 0), dtype=jnp.int32)
    if nkx > 1:
        idx = jnp.clip(ixz_arr + 1, 0, nkx - 1)
        lxinv = jnp.where(
            ixz_arr + 1 < nkx,
            jnp.abs(kx[idx]) / (2.0 * jnp.pi),
            jnp.asarray(params.kxmax / (2.0 * jnp.pi), dtype=jnp.float64),
        )
    else:
        lxinv = jnp.asarray(params.kxmax / (2.0 * jnp.pi), dtype=jnp.float64)
    lyinv = (
        ky[1] / (2.0 * jnp.pi)
        if nky > 1
        else jnp.asarray(params.kymax / (2.0 * jnp.pi), dtype=jnp.float64)
    )

    hyper = -(
        jnp.abs(params.disp_y)
        * (ky_b / jnp.maximum(params.kymax, EPS)) ** jnp.where(params.disp_y < 0.0, 2.0, 4.0)
        + jnp.abs(params.disp_x)
        * (kx_b / jnp.maximum(params.kxmax, EPS)) ** jnp.where(params.disp_x < 0.0, 2.0, 4.0)
    )

    return {
        "kx_b": kx_b,
        "ky_b": ky_b,
        "hyper": hyper,
        "s_d1_ipos": _parallel_coefficients(pos_par, stencils.D1_IPW_POS),
        "s_d1_ineg": _parallel_coefficients(pos_par, stencils.D1_IPW_NEG),
        "s_d4_ipos": _parallel_coefficients(pos_par, stencils.D4_IPW_POS),
        "s_d4_ineg": _parallel_coefficients(pos_par, stencils.D4_IPW_NEG),
        "dvp": params.dvp,
        "vpmax": jnp.max(jnp.abs(vpgr)),
        "vthrat_max": jnp.max(jnp.abs(jnp.asarray(params.vthrat, dtype=jnp.float64))),
        "sgr_dist": params.sgr_dist,
        "ixzero": ixzero,
        "iyzero": iyzero,
        "nl_mphi": mphi,
        "nl_mphiw3": mphiw3,
        "nl_mrad": mrad,
        "nl_fft_scale": jnp.asarray(float(mrad * mphi), dtype=jnp.float64),
        "nl_lxinv": jnp.asarray(lxinv, dtype=jnp.float64),
        "nl_lyinv": jnp.asarray(lyinv, dtype=jnp.float64),
        "nl_jind": build_jind(nkx, mrad, cast(int, ixzero)),
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

        sz = jnp.where(jnp.abs(signz) < EPS, 1.0, signz)
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

        sz = jnp.where(jnp.abs(signz) < EPS, 1.0, signz)

    krloc_sq = (
        ky_b**2 * g_shape(little_g[:, 0])
        + 2.0 * ky_b * kx_b * g_shape(little_g[:, 1])
        + kx_b**2 * g_shape(little_g[:, 2])
    )
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, EPS))

    b_arg = (
        mas * vthrat * krloc * jnp.sqrt(jnp.maximum(2.0 * mu / jnp.maximum(bn_b, EPS), EPS)) / sz
    )
    bessel = j0(b_arg)

    # Maxwellian (GKW init.f90:1977-1979): fmax = (de/dgrid)*exp(-vp²-2*B*mu)/π^{3/2}.
    # GKW always sets tmp/tgrid = 1, so no t_rat appears. vpgr is in species-normalized
    # units (v_ths); streaming = vthrat*vpgr gives v in v_thi.
    fmax = (
        (de / jnp.asarray(params.dgrid, dtype=jnp.float64))
        * jnp.exp(-(vp2 + 2.0 * bn_b * mu))
        / jnp.pi**1.5
    )
    et = (vp2 + 2.0 * bn_b * mu) - 1.5
    dmax_ek = (rln + rlt * et) * fmax * (efun_b * ky_b)

    # drifts: GKW drift() scales by tgrid(is) = tmp(is) (linear_terms.f90:4154-4156).
    ed = vp2 + bn_b * mu
    drift_x = tmp * ed * d_shape(dfun[:, 0]) / sz
    drift_y = tmp * ed * d_shape(dfun[:, 1]) / sz

    upar = -ffun_b * vthrat * vp
    utrap = vthrat * mu * bn_b * gfun_b

    # idisp=2: RMS velocity (GKW linear_terms.f90:643,911)
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


def _linear_precompute_core(geometry: Dict[str, jnp.ndarray], params: GKParams) -> "GKPre":
    """Core implementation of linear_precompute (no auto-sharding logic)."""
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

    out = _precompute_shared(
        geometry, params, kx, ky, ns, nkx, nky, vpgr, mugr, bn, ffun, gfun, dfun, efun
    )

    if not params.adiabatic_electrons:
        mas_arr = jnp.asarray(params.mas, dtype=jnp.float64)
        nsp = int(mas_arr.shape[0])
        sp = _compute_species_coeffs(
            mas_arr,
            jnp.asarray(params.signz, dtype=jnp.float64),
            jnp.asarray(params.vthrat, dtype=jnp.float64),
            jnp.asarray(params.tmp, dtype=jnp.float64),
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
        )
        if "vpgr_rms" in geometry:
            vp_rms = jnp.asarray(geometry["vpgr_rms"], dtype=jnp.float64)
            mu_rms = jnp.asarray(geometry.get("mugr_rms", 1.0), dtype=jnp.float64)
            vthrat_6 = jnp.asarray(params.vthrat, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
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
            params.sgr_dist,
            out["s_d1_ipos"],
            out["s_d1_ineg"],
            out["s_d4_ipos"],
            out["s_d4_ineg"],
            stencil_ndim=6,
        )
        out.update(sp)
        out.update(precompute_collisions(geometry, params))
        out["geom_tensors"] = None
        out["nsp"] = nsp
        # 6D broadcast shape so kx_b/ky_b align with kinetic drift arrays
        out["kx_b"] = jnp.reshape(kx, (1, 1, 1, 1, -1, 1))
        out["ky_b"] = jnp.reshape(ky, (1, 1, 1, 1, 1, -1))

        # ensure multi-species arrays present (compute_geometry_from_input may only
        # have single-species defaults)
        geom_sp = dict(geometry)
        geom_sp["mas"] = jnp.asarray(params.mas, dtype=jnp.float64).reshape(-1)
        geom_sp["signz"] = jnp.asarray(params.signz, dtype=jnp.float64).reshape(-1)
        geom_sp["tmp"] = jnp.asarray(params.tmp, dtype=jnp.float64).reshape(-1)
        geom_sp["de"] = jnp.asarray(params.de, dtype=jnp.float64).reshape(-1)
        geom_sp["vthrat"] = jnp.asarray(params.vthrat, dtype=jnp.float64).reshape(-1)
        phi_w, phi_d = precompute_phi_kinetic(geom_sp)
        out["phi_weight"] = phi_w
        out["phi_diag"] = phi_d

        if params.nlapar:
            from gyaradax.integrals import precompute_apar

            apar_w, _apar_d_analytical, kperp_sq = precompute_apar(geometry, params)
            out["apar_weight"] = apar_w
            out["kperp_sq"] = kperp_sq

            signz_6 = jnp.asarray(params.signz, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
            tmp_6 = jnp.asarray(params.tmp, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
            vthrat_6 = jnp.asarray(params.vthrat, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
            vpgr_6 = jnp.reshape(vpgr, (1, -1, 1, 1, 1, 1))
            mas_6 = jnp.asarray(params.mas, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
            de_6 = jnp.asarray(params.de, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
            intmu_6 = jnp.asarray(geometry["intmu"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
            intvp_6 = jnp.asarray(geometry["intvp"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
            bn_6 = jnp.reshape(bn, (1, 1, 1, -1, 1, 1))

            # Ampere denominator (GKW ampere_dia, linear_terms.f90:3760-3785):
            # gamma_num = sum(2*B*intmu*intvp * J0^2 * vpgr^2 * fmaxwl) per species,
            # shared velocity grid for all species (velocitygrid.f90:197).
            gamma_num = jnp.sum(
                2.0 * bn_6 * intmu_6 * intvp_6 * sp["bessel"] ** 2 * vpgr_6**2 * sp["fmaxwl"],
                axis=(1, 2),
                keepdims=True,
            )
            diag_per_sp = signz_6**2 * de_6 * gamma_num / mas_6
            diag_em = params.beta * jnp.sum(diag_per_sp, axis=0)
            diag_em = diag_em.reshape(diag_em.shape[-3], diag_em.shape[-2], diag_em.shape[-1])
            kperp_sq_3d = kperp_sq.reshape(
                kperp_sq.shape[-3], kperp_sq.shape[-2], kperp_sq.shape[-1]
            )
            apar_d = kperp_sq_3d + diag_em
            apar_d = jnp.where(jnp.abs(apar_d) < EPS, 1.0, apar_d)
            out["apar_diag"] = apar_d

            # g2f_factor matches GKW g2f_correct: -2*signz*vthrat*vpgr*J0*fmaxwl/tmp
            g2f = -2.0 * signz_6 * vthrat_6 * vpgr_6 * sp["bessel"] * sp["fmaxwl"] / tmp_6
            out["g2f_factor"] = g2f
            out["apar_g2f_correction"] = jnp.einsum("avmjkl,avmjkl->jkl", apar_w, g2f)
            # chi factor: gyro_chi = gyro_phi + apar_chi_factor*apar with
            # chi = phi - 2*v_R*v_par*A_par (generalized EM potential)
            out["apar_chi_factor"] = -2.0 * vthrat_6 * vpgr_6 * sp["bessel"]

        if params.nlbpar:
            # phi/bpar form a coupled 2x2 system; override phi_weight/phi_diag
            # with the behavior-preserving kinetic multi-species B_parallel helper.
            out.update(precompute_bpar(geom_sp, params, sp))

        # field CFL: Alfvén wave limit (time_est_field, matdat.F90:1859-1919).
        # ES: sqrt(mir*kmin2*mer); EM: sqrt(mir*(beta+kmin2*mer)) -- beta adds Alfven coupling
        signz_arr = jnp.asarray(params.signz, dtype=jnp.float64)
        de_arr = jnp.asarray(params.de, dtype=jnp.float64)
        mir = jnp.sum(jnp.where(signz_arr > 0, mas_arr * de_arr, 0.0))
        mer = jnp.sum(jnp.where(signz_arr < 0, mas_arr / jnp.maximum(de_arr, EPS), 0.0))
        ky_min = jnp.where(nky > 1, ky[1], ky[0])
        kmin2 = ky_min**2 * little_g[:, 0]
        # matdat.F90:1911-1914: fall back to 2π*lxinv = kx_min when smaller than ky_min²·g_yy
        ixz_arr = jnp.asarray(geometry["ixzero"], dtype=jnp.int32)
        if nkx > 1:
            idx = jnp.clip(ixz_arr + 1, 0, nkx - 1)
            kx_min_abs = jnp.abs(kx[idx])
            in_range = (ixz_arr + 1 < nkx) & (kx_min_abs > EPS)
            kmin2 = jnp.where(in_range, jnp.minimum(kx_min_abs, kmin2), kmin2)
        q_val = jnp.asarray(geometry.get("q", getattr(params, "q", 1.0)), dtype=jnp.float64)
        beta_cfl = jnp.asarray(params.beta, dtype=jnp.float64)
        field_cfl_arg = mir * (beta_cfl + kmin2 * mer)
        field_period = (
            2.0 * jnp.pi * q_val * params.sgr_dist * bn * jnp.sqrt(jnp.maximum(field_cfl_arg, EPS))
        )
        time_field = jnp.min(jnp.where(field_period > EPS, field_period, 1e30))
        out["tmax_field"] = jnp.where(time_field < 1e20, 1.0 / time_field, 0.0)

        # g2f -> A_par Alfven coupling is captured by tmax_field; em_streaming_cfl
        # is kept as a pass-through so dt matches GKW's dtim_est.
        if params.nlapar:
            out["em_streaming_cfl"] = jnp.asarray(1.0, dtype=jnp.float64)

    else:
        sp = _compute_species_coeffs(
            params.mas,
            params.signz,
            params.vthrat,
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
        )
        if "vpgr_rms" in geometry:
            vp_rms = jnp.asarray(geometry["vpgr_rms"], dtype=jnp.float64)
            mu_rms = jnp.asarray(geometry.get("mugr_rms", 1.0), dtype=jnp.float64)
            ffun_b = jnp.reshape(ffun, (1, 1, -1, 1, 1))
            bn_b = jnp.reshape(bn, (1, 1, -1, 1, 1))
            gfun_b = jnp.reshape(gfun, (1, 1, -1, 1, 1))
            idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
            use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))
            sp["abs_dum2_par"] = jnp.where(
                use_abs, jnp.abs(sp["upar"]), jnp.abs(ffun_b * params.vthrat * vp_rms)
            )
            sp["abs_dum2_vp"] = jnp.where(
                use_abs,
                jnp.abs(sp["utrap"]),
                jnp.abs(params.vthrat * bn_b * gfun_b * mu_rms),
            )

        sp["s_total_upar"], sp["s_total_t7"] = _fuse_stencils(
            sp["upar"],
            sp["abs_dum2_par"],
            sp["term7_fac"],
            params.disp_par,
            params.sgr_dist,
            out["s_d1_ipos"],
            out["s_d1_ineg"],
            out["s_d4_ipos"],
            out["s_d4_ineg"],
            stencil_ndim=5,
        )
        out.update(sp)
        out.update(precompute_collisions(geometry, params))
        out["geom_tensors"] = geom_tensors(geometry, params=params)

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

        if params.nlapar:
            from gyaradax.integrals import precompute_apar

            apar_w, apar_d, kperp_sq = precompute_apar(geometry, params)
            out["apar_weight"] = apar_w
            out["apar_diag"] = apar_d
            out["kperp_sq"] = kperp_sq
            signz_b = jnp.asarray(params.signz, dtype=jnp.float64)
            tmp_b = jnp.asarray(params.tmp, dtype=jnp.float64)
            vthrat_b = jnp.asarray(params.vthrat, dtype=jnp.float64)
            vpgr_5 = jnp.reshape(vpgr, (-1, 1, 1, 1, 1))
            g2f = -2.0 * signz_b * vthrat_b * vpgr_5 * sp["bessel"] * sp["fmaxwl"] / tmp_b
            out["g2f_factor"] = g2f
            out["apar_g2f_correction"] = jnp.einsum("vmjkl,vmjkl->jkl", apar_w[0], g2f)
            out["apar_chi_factor"] = -2.0 * vthrat_b * vpgr_5 * sp["bessel"]

    return GKPre(out)


def linear_precompute(geometry: Dict[str, jnp.ndarray], params: GKParams) -> "GKPre":
    """Precompute static geometry-dependent coefficients and Bessel terms.

    Automatically uses sharded computation if params indicates multi-GPU config.
    """
    n_gpus_sp = int(getattr(params, "n_gpus_sp", 1))
    n_gpus_vp = int(getattr(params, "n_gpus_vp", 1))
    n_gpus_mu = int(getattr(params, "n_gpus_mu", 1))
    if n_gpus_sp * n_gpus_vp * n_gpus_mu > 1:
        import gyaradax.sharding as sharding

        mesh = sharding.build_mesh(params)
        if mesh is not None:
            grid = sharding.grid_shape_from(params, geometry)
            return sharding.precompute_sharded(geometry, params, mesh, grid)
    return _linear_precompute_core(geometry, params)
