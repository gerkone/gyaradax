"""Linearized Fokker-Planck collision operator.

Full multi-species port of GKW's collision_differential_numu
(collisionop.f90:1547-2228). Uniform-in-v_perp mu grid.

Operator pieces (each independently toggleable):
- pitch-angle scattering (D_theta_theta)
- energy diffusion (D_vv)
- friction / slowing-down (F_v)

For each target species a, the operator is C_a(f) = sum_b C^{a/b}(f),
with coefficients evaluated using `vtb = v * vthrat_a / vthrat_b` and
species-pair prefactor `Γ^{a/b}`. Sum runs over all background species
(kinetic + adiabatic).

Prefactor paths:
- `freq_override=True`: Γ = Z_a²·Z_b²·coll_freq·de_b·(L_ab/L_ref)/T_a²
  where L_ref is the i-i Coulomb log for the first species pair, and
  L_ab is the species-pair Coulomb log.
- `freq_override=False`: Γ = 6.5141e-5·(R_ref·n_ref/T_ref²)·de_b·Z_a²·
  Z_b²·L_ab/T_a² (full Coulomb log per species pair).

Optional Xu-style momentum / energy conservation corrections
(`cons_momentum` in collisionop.f90:2234) on top of the base operator.
"""

from typing import Dict

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from gyaradax import _EPS
from gyaradax.params import GKParams


def _erfp(x):
    """Derivative of erf: 2/sqrt(pi) * exp(-x^2)."""
    return 2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))


def _D_thth(v, gamma_pref, vtb_scale=1.0):
    """Pitch-angle diffusion coefficient (freq_override single-species prefactor)."""
    v_safe = jnp.maximum(v, _EPS)
    vtb = jnp.maximum(v_safe * vtb_scale, _EPS)
    return gamma_pref * ((2.0 - 1.0 / vtb**2) * erf(vtb) + _erfp(vtb) / vtb) / (4.0 * v_safe)


def _D_vv(v, gamma_pref, vtb_scale=1.0):
    """Energy diffusion coefficient."""
    v_safe = jnp.maximum(v, _EPS)
    vtb = jnp.maximum(v_safe * vtb_scale, _EPS)
    return gamma_pref * (erf(vtb) / vtb**2 - _erfp(vtb) / vtb) / (2.0 * v_safe)


def _F_v(v, gamma_pref, vtb_scale=1.0, mrat=1.0):
    """Friction coefficient. mrat = m_a / m_b (target over background)."""
    v_safe = jnp.maximum(v, _EPS)
    vtb = jnp.maximum(v_safe * vtb_scale, _EPS)
    return gamma_pref * mrat * (erf(vtb) - _erfp(vtb) * vtb) / v_safe**2


def _coulomb_log_pair(signz_a, signz_b, tmp_a, tmp_b, mas_a, mas_b, de_a, de_b, ne19, nref, tref):
    """Species-pair Coulomb log per GKW collision_init (lines 460-526).

    All inputs are normalized GKW quantities except `ne19` which is the
    electron density in 10^19 m^-3 (ne19 = sum_e de_e * nref).
    """
    tae = tmp_a * tref  # dim-ful temperature
    tbe = tmp_b * tref
    # e-e
    Lee = 14.9 - 0.5 * jnp.log(jnp.maximum(0.1 * ne19, _EPS)) + jnp.log(jnp.maximum(tae, _EPS))
    # e-i (scattered is electron, background is ion with charge Z_b)
    Lei_low = (
        17.2
        - 0.5 * jnp.log(jnp.maximum(0.1 * signz_b**2 * ne19, _EPS))
        + 1.5 * jnp.log(jnp.maximum(tae, _EPS))
    )
    Lei_high = 14.8 - 0.5 * jnp.log(jnp.maximum(0.1 * ne19, _EPS)) + jnp.log(jnp.maximum(tae, _EPS))
    Lei = jnp.where(tae < 0.01 * signz_b**2, Lei_low, Lei_high)
    # i-e (scattered is ion with charge Z_a, background is electron); NRL p.34
    Lie_low = (
        17.2
        - 0.5 * jnp.log(jnp.maximum(0.1 * signz_a**2 * ne19, _EPS))
        + 1.5 * jnp.log(jnp.maximum(tbe, _EPS))
    )
    Lie_high = 14.8 - 0.5 * jnp.log(jnp.maximum(0.1 * ne19, _EPS)) + jnp.log(jnp.maximum(tbe, _EPS))
    Lie = jnp.where(tbe < 0.01 * signz_a**2, Lie_low, Lie_high)
    # i-i (full NRL formula, collisionop.f90:514-517)
    Lii = (
        17.3
        - jnp.log(jnp.maximum(jnp.abs(signz_a * signz_b) * (mas_a + mas_b), _EPS))
        + jnp.log(jnp.maximum((mas_a * tmp_b + mas_b * tmp_a) * tref, _EPS))
        - 0.5 * jnp.log(jnp.maximum(0.1 * nref / tref, _EPS))
        - 0.5
        * jnp.log(
            jnp.maximum(
                de_a * signz_a**2 / jnp.maximum(tmp_a, _EPS)
                + de_b * signz_b**2 / jnp.maximum(tmp_b, _EPS),
                _EPS,
            )
        )
    )

    a_is_e = signz_a < 0
    b_is_e = signz_b < 0
    a_is_i = jnp.logical_not(a_is_e)
    b_is_i = jnp.logical_not(b_is_e)
    return jnp.where(
        a_is_e & b_is_e, Lee, jnp.where(a_is_e & b_is_i, Lei, jnp.where(a_is_i & b_is_e, Lie, Lii))
    )


def _gamma_pair(params, signz_a, mas_a, tmp_a, de_a, signz_b, mas_b, tmp_b, de_b, ne19, L_ref):
    """Γ^(a/b) species-pair prefactor.

    freq_override=True: Γ = Z_a² Z_b² coll_freq de_b (L_ab/L_ref) / T_a²
    freq_override=False: Γ = 6.5141e-5 (R_ref n_ref / T_ref²) de_b Z_a² Z_b² L_ab / T_a²
    """
    L_ab = _coulomb_log_pair(
        signz_a,
        signz_b,
        tmp_a,
        tmp_b,
        mas_a,
        mas_b,
        de_a,
        de_b,
        ne19,
        params.coll_nref,
        params.coll_tref,
    )
    z2 = signz_a**2 * signz_b**2
    tmp_a_safe = jnp.maximum(tmp_a, _EPS)
    if params.coll_freq_override:
        return z2 * params.coll_freq * de_b * (L_ab / jnp.maximum(L_ref, _EPS)) / tmp_a_safe**2
    c = 6.5141e-5 * params.coll_rref * params.coll_nref / params.coll_tref**2
    return c * de_b * z2 * L_ab / tmp_a_safe**2


def _gamma_pref_self(params, de, tmp, signz):
    """Scalar self-collision prefactor for single-species tests/back-compat."""
    if params.coll_freq_override:
        return params.coll_freq * de / jnp.maximum(tmp, _EPS) ** 2
    L = _coulomb_log_pair(
        signz,
        signz,
        tmp,
        tmp,
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        de,
        de,
        jnp.asarray(params.coll_nref, dtype=jnp.float64),
        params.coll_nref,
        params.coll_tref,
    )
    c = 6.5141e-5 * params.coll_rref * params.coll_nref / params.coll_tref**2
    return c * de * signz**4 * L / jnp.maximum(tmp, _EPS) ** 2


def _coulomb_log_ii(signz, tmp, de, nref, tref):
    """Back-compat wrapper for the old ion-ion self Coulomb log.

    Unchanged interface used by tests; delegates to `_coulomb_log_pair` for
    the (a=b) ion-ion case.
    """
    return _coulomb_log_pair(
        signz,
        signz,
        tmp,
        tmp,
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        de,
        de,
        jnp.asarray(nref, dtype=jnp.float64),
        nref,
        tref,
    )


def _build_stencil(
    vpgr,
    mugr,
    bn,
    dvp,
    dvperp,
    gamma_pref,
    vthrat_target,
    vtb_scale,
    mrat,
    pitch_angle,
    en_scatter,
    friction,
    mass_conserve,
):
    """Assemble the 9-point (v_par, mu) stencil for one (target, background) pair.

    Returns an array of shape (9, nv, nmu, ns).
    """
    nv = int(vpgr.shape[0])
    nmu = int(mugr.shape[0])
    ns = int(bn.shape[0])

    vp = vpgr.reshape(nv, 1, 1)
    vperp = jnp.sqrt(jnp.maximum(2.0 * mugr, 0.0)).reshape(1, nmu, 1)
    bn_b = bn.reshape(1, 1, ns)
    sqrtB = jnp.sqrt(jnp.maximum(bn_b, _EPS))
    dvrp = sqrtB * dvperp
    vperp_phys = vperp * sqrtB

    def Dth(v):
        return _D_thth(v, gamma_pref, vtb_scale)

    def Dvv(v):
        return _D_vv(v, gamma_pref, vtb_scale)

    def Fv(v):
        return _F_v(v, gamma_pref, vtb_scale, mrat)

    # block A: vpar + 1/2
    vpar_A = vp + 0.5 * dvp
    v_A = jnp.sqrt(jnp.maximum(vpar_A**2 + vperp_phys**2, _EPS))
    denom_A = jnp.maximum(vpar_A**2 + vperp_phys**2, _EPS)
    fac_A = jnp.where(pitch_angle, vperp_phys**2 * Dth(v_A) / (denom_A * dvp**2), 0.0)
    fad_A = jnp.where(en_scatter, vpar_A**2 * Dvv(v_A) / (denom_A * dvp**2), 0.0)
    faf_A = jnp.where(friction, vpar_A * Fv(v_A) / (jnp.sqrt(denom_A) * dvp), 0.0)
    mask_top = (jnp.arange(nv).reshape(nv, 1, 1) < nv - 1) | (not mass_conserve)
    fac_A = jnp.where(mask_top, fac_A, 0.0)
    fad_A = jnp.where(mask_top, fad_A, 0.0)
    faf_A = jnp.where(mask_top, faf_A, 0.0)

    # block B: vpar - 1/2
    vpar_B = vp - 0.5 * dvp
    v_B = jnp.sqrt(jnp.maximum(vpar_B**2 + vperp_phys**2, _EPS))
    denom_B = jnp.maximum(vpar_B**2 + vperp_phys**2, _EPS)
    fac_B = jnp.where(pitch_angle, vperp_phys**2 * Dth(v_B) / (denom_B * dvp**2), 0.0)
    fad_B = jnp.where(en_scatter, vpar_B**2 * Dvv(v_B) / (denom_B * dvp**2), 0.0)
    faf_B = jnp.where(friction, vpar_B * Fv(v_B) / (jnp.sqrt(denom_B) * dvp), 0.0)
    mask_bot = (jnp.arange(nv).reshape(nv, 1, 1) > 0) | (not mass_conserve)
    fac_B = jnp.where(mask_bot, fac_B, 0.0)
    fad_B = jnp.where(mask_bot, fad_B, 0.0)
    faf_B = jnp.where(mask_bot, faf_B, 0.0)

    # block C: v_perp + 1/2
    vperp_C = vperp_phys + 0.5 * dvrp
    v_C = jnp.sqrt(jnp.maximum(vp**2 + vperp_C**2, _EPS))
    denom_C = jnp.maximum(vp**2 + vperp_C**2, _EPS)
    vzero = jnp.maximum(vperp_phys, _EPS)
    fac_C = jnp.where(pitch_angle, vperp_C * vp**2 * Dth(v_C) / (denom_C * vzero * dvrp**2), 0.0)
    fad_C = jnp.where(en_scatter, vperp_C**3 * Dvv(v_C) / (denom_C * vzero * dvrp**2), 0.0)
    faf_C = jnp.where(friction, vperp_C**2 * Fv(v_C) / (jnp.sqrt(denom_C) * vzero * dvrp), 0.0)
    mask_muhi = (jnp.arange(nmu).reshape(1, nmu, 1) < nmu - 1) | (not mass_conserve)
    fac_C = jnp.where(mask_muhi, fac_C, 0.0)
    fad_C = jnp.where(mask_muhi, fad_C, 0.0)
    faf_C = jnp.where(mask_muhi, faf_C, 0.0)

    # block D: v_perp - 1/2 (natural zero flux at mu=0)
    vperp_D = vperp_phys - 0.5 * dvrp
    v_D = jnp.sqrt(jnp.maximum(vp**2 + vperp_D**2, _EPS))
    denom_D = jnp.maximum(vp**2 + vperp_D**2, _EPS)
    fac_D = jnp.where(pitch_angle, vperp_D * vp**2 * Dth(v_D) / (denom_D * vzero * dvrp**2), 0.0)
    fad_D = jnp.where(en_scatter, vperp_D**3 * Dvv(v_D) / (denom_D * vzero * dvrp**2), 0.0)
    faf_D = jnp.where(friction, vperp_D**2 * Fv(v_D) / (jnp.sqrt(denom_D) * vzero * dvrp), 0.0)

    # cross blocks E, F (vperp half-points; no friction)
    fac_cross_E = jnp.where(pitch_angle, -(vperp_C**2) * vp * Dth(v_C) / (vzero * denom_C), 0.0)
    fad_cross_E = jnp.where(en_scatter, vperp_C**2 * vp * Dvv(v_C) / (vzero * denom_C), 0.0)
    fac_cross_E = jnp.where(mask_muhi, fac_cross_E, 0.0)
    fad_cross_E = jnp.where(mask_muhi, fad_cross_E, 0.0)

    mask_mulo = (jnp.arange(nmu).reshape(1, nmu, 1) > 0) | (not mass_conserve)
    fac_cross_F = jnp.where(pitch_angle, -(vperp_D**2) * vp * Dth(v_D) / (vzero * denom_D), 0.0)
    fad_cross_F = jnp.where(en_scatter, vperp_D**2 * vp * Dvv(v_D) / (vzero * denom_D), 0.0)
    fac_cross_F = jnp.where(mask_mulo, fac_cross_F, 0.0)
    fad_cross_F = jnp.where(mask_mulo, fad_cross_F, 0.0)

    # cross blocks G, H (vpar half-points; no friction)
    fac_cross_G = jnp.where(pitch_angle, -vperp_phys * vpar_A * Dth(v_A) / denom_A, 0.0)
    fad_cross_G = jnp.where(en_scatter, vperp_phys * vpar_A * Dvv(v_A) / denom_A, 0.0)
    fac_cross_G = jnp.where(mask_top, fac_cross_G, 0.0)
    fad_cross_G = jnp.where(mask_top, fad_cross_G, 0.0)

    fac_cross_H = jnp.where(pitch_angle, -vperp_phys * vpar_B * Dth(v_B) / denom_B, 0.0)
    fad_cross_H = jnp.where(en_scatter, vperp_phys * vpar_B * Dvv(v_B) / denom_B, 0.0)
    fac_cross_H = jnp.where(mask_bot, fac_cross_H, 0.0)
    fad_cross_H = jnp.where(mask_bot, fad_cross_H, 0.0)

    # assemble 9-point stencil; ccdelta = 0.5 (collisionop.f90:3552)
    d = 0.5
    one_minus_d = 1.0 - d
    sum_AB = fac_A + fad_A
    sum_BB = fac_B + fad_B
    sum_CC = fac_C + fad_C
    sum_DD = fac_D + fad_D
    cross_factor = 1.0 / (4.0 * dvp * dvrp)
    sum_E = (fac_cross_E + fad_cross_E) * cross_factor
    sum_F = (fac_cross_F + fad_cross_F) * cross_factor
    sum_G = (fac_cross_G + fad_cross_G) * cross_factor
    sum_H = (fac_cross_H + fad_cross_H) * cross_factor

    c_self = -(sum_AB + sum_BB + sum_CC + sum_DD) + d * (faf_A - faf_B + faf_C - faf_D)
    c_vpar_p = sum_AB + one_minus_d * faf_A + sum_E - sum_F
    c_vpar_m = sum_BB - one_minus_d * faf_B - sum_E + sum_F
    c_mu_p = sum_CC + one_minus_d * faf_C + sum_G - sum_H
    c_mu_m = sum_DD - one_minus_d * faf_D - sum_G + sum_H
    c_pp = sum_E + sum_G
    c_mp = -sum_E - sum_H
    c_pm = -sum_F - sum_G
    c_mm = sum_F + sum_H

    stencil = vthrat_target * jnp.stack(
        [c_self, c_vpar_p, c_vpar_m, c_mu_p, c_mu_m, c_pp, c_mp, c_pm, c_mm],
        axis=0,
    )
    return stencil


def _collect_species_arrays(params):
    """Return target and background species arrays as (ntgt,) and (nbg,).

    - target: species the operator applies to (kinetic species).
    - background: species acting as scatterers (all species, kinetic + adiabatic).

    If `coll_bg_signz` etc. are None, falls back to self-collision only
    (backgrounds == targets).
    """

    def _arr(x):
        return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))

    tgt_mas = _arr(params.mas)
    tgt_signz = _arr(params.signz)
    tgt_tmp = _arr(params.tmp)
    tgt_de = _arr(params.de)
    # vthrat is derived from species temperature and mass: sqrt(T_s/m_s)
    tgt_vthrat = jnp.sqrt(tgt_tmp / jnp.maximum(tgt_mas, 1e-30))

    if params.coll_bg_signz is None:
        bg_mas, bg_signz, bg_tmp, bg_de, bg_vthrat = (
            tgt_mas,
            tgt_signz,
            tgt_tmp,
            tgt_de,
            tgt_vthrat,
        )
    else:
        bg_mas = _arr(params.coll_bg_mas)
        bg_signz = _arr(params.coll_bg_signz)
        bg_tmp = _arr(params.coll_bg_tmp)
        bg_de = _arr(params.coll_bg_de)
        bg_vthrat = _arr(params.coll_bg_vthrat)

    # electron density (ne19 = sum_e de_b * nref), for Coulomb log
    ne19 = jnp.sum(jnp.where(bg_signz < 0, bg_de, 0.0)) * params.coll_nref
    ne19 = jnp.maximum(ne19, _EPS)
    return (
        tgt_mas,
        tgt_signz,
        tgt_tmp,
        tgt_de,
        tgt_vthrat,
        bg_mas,
        bg_signz,
        bg_tmp,
        bg_de,
        bg_vthrat,
        ne19,
    )


def precompute_collisions(geometry: Dict, params: GKParams) -> Dict[str, jnp.ndarray]:
    """Build the 9-point collision stencil.

    Output stencil shape: (9, nv, nmu, ns) for a single target species (adiabatic
    MVP), (nsp, 9, nv, nmu, ns) otherwise. Each target stencil is a sum over all
    background species (kinetic + adiabatic), each contributing the full
    species-pair Fokker-Planck operator with Gamma^(a/b), vtb = v*vthrat_a/vthrat_b,
    and mass ratio m_a/m_b (friction).
    """
    if not params.collisions:
        return {}

    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)
    dvp = float(geometry["dvp"])
    vperp_grid = jnp.sqrt(jnp.maximum(2.0 * mugr, 0.0))
    dvperp = float(vperp_grid[0] * 2.0)

    flags = (
        params.coll_pitch_angle,
        params.coll_en_scatter,
        params.coll_friction,
        params.coll_mass_conserve,
    )

    (
        tgt_mas,
        tgt_signz,
        tgt_tmp,
        tgt_de,
        tgt_vthrat,
        bg_mas,
        bg_signz,
        bg_tmp,
        bg_de,
        bg_vthrat,
        ne19,
    ) = _collect_species_arrays(params)

    # reference Coulomb log for freq_override normalization: first ion-ion pair
    # in the species list, falling back to (0,0) when all species are electrons.
    def _first_ion(signz_arr):
        idx = jnp.argmax(signz_arr > 0)
        return jnp.where(jnp.any(signz_arr > 0), idx, 0)

    iref = _first_ion(bg_signz)
    L_ref = _coulomb_log_pair(
        bg_signz[iref],
        bg_signz[iref],
        bg_tmp[iref],
        bg_tmp[iref],
        bg_mas[iref],
        bg_mas[iref],
        bg_de[iref],
        bg_de[iref],
        ne19,
        params.coll_nref,
        params.coll_tref,
    )
    L_ref = jnp.maximum(L_ref, _EPS)

    def stencil_for_target(mas_a, signz_a, tmp_a, de_a, vthrat_a):
        def per_bg(mas_b, signz_b, tmp_b, de_b, vthrat_b):
            gp = _gamma_pair(
                params, signz_a, mas_a, tmp_a, de_a, signz_b, mas_b, tmp_b, de_b, ne19, L_ref
            )
            vtb_scale = vthrat_a / jnp.maximum(vthrat_b, _EPS)
            mrat = mas_a / jnp.maximum(mas_b, _EPS)
            return _build_stencil(
                vpgr, mugr, bn, dvp, dvperp, gp, vthrat_a, vtb_scale, mrat, *flags
            )

        bg_stencils = jax.vmap(per_bg)(bg_mas, bg_signz, bg_tmp, bg_de, bg_vthrat)
        return jnp.sum(bg_stencils, axis=0)

    if params.adiabatic_electrons and tgt_signz.shape[0] == 1:
        stencil = stencil_for_target(tgt_mas[0], tgt_signz[0], tgt_tmp[0], tgt_de[0], tgt_vthrat[0])
    else:
        stencil = jax.vmap(stencil_for_target)(tgt_mas, tgt_signz, tgt_tmp, tgt_de, tgt_vthrat)

    out = {"coll_stencil": stencil}
    if params.coll_mom_conservation or params.coll_ene_conservation:
        out.update(_precompute_conservation(geometry, params, vpgr, mugr, bn))
    return out


def _precompute_conservation(geometry, params, vpgr, mugr, bn):
    """Precompute Xu-style momentum/energy conservation weights.

    Returns a dict with keys:
      coll_mom_factor: (nv, nmu, ns) — coefficient of Δp in the correction
      coll_ene_factor: (nv, nmu, ns) — coefficient of ΔE in the correction
      coll_vpar_weight: (nv, nmu, ns) — weight to compute Δp = ∫v_par C(f) d³v
      coll_vsq_weight: (nv, nmu, ns) — weight to compute ΔE = ∫v² C(f) d³v

    Kinetic case adds a leading (nsp,) axis to every array.
    """
    intvp = jnp.asarray(geometry["intvp"], dtype=jnp.float64)
    intmu = jnp.asarray(geometry["intmu"], dtype=jnp.float64)
    nv = int(vpgr.shape[0])
    nmu = int(mugr.shape[0])
    ns = int(bn.shape[0])

    vp = vpgr.reshape(nv, 1, 1)
    mu = mugr.reshape(1, nmu, 1)
    bn_b = bn.reshape(1, 1, ns)
    intvp_b = intvp.reshape(nv, 1, 1)
    intmu_b = intmu.reshape(1, nmu, 1)
    vsq = vp**2 + 2.0 * mu * bn_b
    d3v = intvp_b * intmu_b * bn_b

    def _for_species(tmp_val, de_val):
        t_rat = tmp_val
        fm_env = jnp.exp(-vsq / t_rat) / (jnp.sqrt(t_rat * jnp.pi) ** 3)
        fmax = de_val * fm_env
        part = jnp.sum(fmax * d3v, axis=(0, 1))
        ene = jnp.sum(vsq * fmax * d3v, axis=(0, 1))
        A = ene / jnp.maximum(part, _EPS)
        P = jnp.sum(vp**2 * fmax * d3v, axis=(0, 1))
        E = jnp.sum(vsq * (vsq - A.reshape(1, 1, ns)) * fmax * d3v, axis=(0, 1))
        mom_factor = vp * fmax / jnp.maximum(P, _EPS).reshape(1, 1, ns)
        ene_factor = (vsq - A.reshape(1, 1, ns)) * fmax / jnp.maximum(E, _EPS).reshape(1, 1, ns)
        vpar_w = vp * d3v
        vsq_w = vsq * d3v
        mom_on = jnp.where(params.coll_mom_conservation, 1.0, 0.0)
        ene_on = jnp.where(params.coll_ene_conservation, 1.0, 0.0)
        return mom_on * mom_factor, ene_on * ene_factor, vpar_w, vsq_w

    if params.adiabatic_electrons:
        tmp_val = jnp.asarray(params.tmp, dtype=jnp.float64)
        de_val = jnp.asarray(params.de, dtype=jnp.float64)
        mf, ef, vpw, vsw = _for_species(tmp_val, de_val)
        return {
            "coll_mom_factor": mf,
            "coll_ene_factor": ef,
            "coll_vpar_weight": vpw,
            "coll_vsq_weight": vsw,
        }
    else:
        tmp_arr = jnp.atleast_1d(jnp.asarray(params.tmp, dtype=jnp.float64))
        de_arr = jnp.atleast_1d(jnp.asarray(params.de, dtype=jnp.float64))
        mf, ef, vpw, vsw = jax.vmap(_for_species)(tmp_arr, de_arr)
        return {
            "coll_mom_factor": mf,
            "coll_ene_factor": ef,
            "coll_vpar_weight": vpw,
            "coll_vsq_weight": vsw,
        }


def conservation_correction(coll_rhs, mom_factor, ene_factor, vpar_w, vsq_w):
    """Add the Xu scalar conservation correction on top of the base collision RHS.

    coll_rhs: (nv, nmu, ns, nkx, nky). *_factor, *_weight: (nv, nmu, ns).
    Returns an additive correction of the same shape as coll_rhs.
    """
    dp = jnp.sum(vpar_w[:, :, :, None, None] * coll_rhs, axis=(0, 1))
    de = jnp.sum(vsq_w[:, :, :, None, None] * coll_rhs, axis=(0, 1))
    return -(
        dp[None, None, :, :, :] * mom_factor[:, :, :, None, None]
        + de[None, None, :, :, :] * ene_factor[:, :, :, None, None]
    )


def collision_rhs(df: jnp.ndarray, stencil: jnp.ndarray) -> jnp.ndarray:
    """Apply the 9-point collision stencil to a 5D df.

    df shape: (nv, nmu, ns, nkx, nky).
    stencil shape: (9, nv, nmu, ns). Out-of-grid neighbors contribute zero.
    """
    nv, nmu = df.shape[0], df.shape[1]
    iv = jnp.arange(nv)
    imu = jnp.arange(nmu)
    shifts = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
    out = jnp.zeros_like(df)
    for k, (di, dj) in enumerate(shifts):
        v_idx = jnp.clip(iv + di, 0, nv - 1)
        mu_idx = jnp.clip(imu + dj, 0, nmu - 1)
        v_valid = (iv + di >= 0) & (iv + di < nv)
        mu_valid = (imu + dj >= 0) & (imu + dj < nmu)
        valid = v_valid[:, None] & mu_valid[None, :]
        shifted = jnp.take(df, v_idx, axis=0)
        shifted = jnp.take(shifted, mu_idx, axis=1)
        shifted = jnp.where(valid[:, :, None, None, None], shifted, 0.0)
        out = out + stencil[k][:, :, :, None, None] * shifted
    return out
