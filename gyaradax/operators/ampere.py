"""Ampere's law: A_parallel solve and the coupled Poisson-B_parallel solve."""

from typing import Any, Dict, Mapping, Optional

import jax.numpy as jnp
from jax.scipy.special import i0e

from gyaradax.core.constants import EPS
from gyaradax.core.state import Precompute
from gyaradax.operators.poisson import _species_bessel_gamma, i1e, j1_hat


def ampere_gamma_num(
    geometry: Dict[str, jnp.ndarray], dgrid: Any = 1.0, bessel: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Numerical velocity-grid moment for the Ampere diagonal (Candy moment).

    GKW ampere_dia (linear_terms.f90:3756-3789):
        gamma_num = sum_{mu,vpar} 2*B*intmu*intvp * J0^2 * vpgr^2 * fmaxwl
    evaluated on the actual (vpar, mu) grid so the diagonal exactly cancels
    the discrete g2f correction (-2*Z*vthrat*vpgr*J0*F_M/T) in the
    mixed-variable Ampere solve.  The analytic Gamma_0 = I0(b)e^{-b} equals
    this only in the continuum limit; GKW notes the analytic form leads to
    "large errors at high beta and low kthrho" at default vpmax.

    geometry must carry per-species arrays (mas, signz, tmp, de, vthrat),
    e.g. the geom_sp dict built inside :func:`precompute_apar`.

    Returns gamma_num with shape (nsp, ns, nkx, nky).
    """
    if bessel is None:
        bessel, _ = _species_bessel_gamma(geometry)
    de = jnp.atleast_1d(jnp.asarray(geometry["de"], dtype=jnp.float64))
    nsp = de.shape[0]
    de_6d = de.reshape(nsp, 1, 1, 1, 1, 1)
    intvp = jnp.asarray(geometry["intvp"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    intmu = jnp.asarray(geometry["intmu"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, 1, -1, 1, 1)
    dgrid = jnp.asarray(dgrid, dtype=jnp.float64)
    # Maxwellian on the actual grid (GKW init.f90:1977-1983, tmp/tgrid = 1).
    fmaxwl = (de_6d / dgrid) * jnp.exp(-(vpgr**2 + 2.0 * bn * mugr)) / jnp.pi**1.5
    gamma_num = jnp.sum(
        2.0 * bn * intmu * intvp * bessel**2 * vpgr**2 * fmaxwl,
        axis=(1, 2),
    )
    return gamma_num


def precompute_apar(geometry: Dict[str, jnp.ndarray], params: Any = None):
    """
    Precompute static arrays for the Ampere A_parallel solve (kinetic species).

    From GKW linear_terms.f90 ampere_int + ampere_dia.

    Ampere's law (normalized):
        [k_perp^2 + beta * sum_sp(Z^2 n / m * gamma_num)] * A_par
        = beta * sum_sp(Z * vthrat * n * 2pi*B * integral(vpar * J0 * g * dvpar dmu))

    The diagonal uses the NUMERICAL grid moment gamma_num (see
    :func:`ampere_gamma_num`), not the analytic Gamma_0, matching GKW
    ampere_dia for exact cancellation against the discrete g2f source.

    Returns (apar_weight, apar_diag, kperp_sq) where:
        apar_weight: (nsp, nvpar, nmu, ns, nkx, nky) — Ampere numerator weight
        apar_diag: (ns, nkx, nky) — Ampere denominator (precomputed)
        kperp_sq: (1, 1, 1, ns, nkx, nky) — k_perp^2 for reuse
    """

    # Use species arrays from params when available (multi-species from input.dat),
    # falling back to geometry arrays (which may be single-species defaults).
    def _sp_arr(key, geom_key=None):
        if geom_key is None:
            geom_key = key
        if params is not None and hasattr(params, key):
            v = getattr(params, key)
            return jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float64))
        return jnp.atleast_1d(jnp.asarray(geometry[geom_key], dtype=jnp.float64))

    mas = _sp_arr("mas")
    signz = _sp_arr("signz")
    tmp = _sp_arr("tmp")
    de = _sp_arr("de")
    vthrat = _sp_arr("vthrat")
    nsp = mas.shape[0]

    geom_sp = dict(geometry)
    geom_sp["mas"] = mas
    geom_sp["signz"] = signz
    geom_sp["tmp"] = tmp
    geom_sp["de"] = de
    geom_sp["vthrat"] = vthrat
    bessel, _gamma_analytic = _species_bessel_gamma(geom_sp)

    mas_6d = mas.reshape(nsp, 1, 1, 1, 1, 1)
    signz_6d = signz.reshape(nsp, 1, 1, 1, 1, 1)
    de_6d = de.reshape(nsp, 1, 1, 1, 1, 1)
    vthrat_6d = vthrat.reshape(nsp, 1, 1, 1, 1, 1)

    intvp = jnp.asarray(geometry["intvp"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    intmu = jnp.asarray(geometry["intmu"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, 1, -1, 1, 1)

    beta = jnp.asarray(params.beta if params is not None else 0.0, dtype=jnp.float64)

    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64).reshape(1, 1, 1, 1, 1, -1)
    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64).reshape(1, 1, 1, 1, -1, 1)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)
    g0 = little_g[:, 0].reshape(1, 1, 1, -1, 1, 1)
    g1 = little_g[:, 1].reshape(1, 1, 1, -1, 1, 1)
    g2 = little_g[:, 2].reshape(1, 1, 1, -1, 1, 1)
    kperp_sq = krho**2 * g0 + 2 * krho * kxrh * g1 + kxrh**2 * g2

    # Ampere numerator (GKW ampere_int, linear_terms.f90:3246).
    # vthrat converts vpgr (v_ths units) to physical velocity in v_thi units.
    apar_weight = beta * signz_6d * de_6d * vthrat_6d * vpgr * bessel * bn * intvp * intmu
    apar_weight = jnp.where(jnp.abs(intvp) < EPS, 0.0, apar_weight)

    # Ampere denominator (GKW ampere_dia, linear_terms.f90:3760-3785): numerical
    # gamma_num from the actual velocity grid, NOT the analytic Gamma_0, so the
    # diagonal exactly cancels the discrete g2f source.  The kinetic 6D path in
    # precompute.py recomputes the identical moment from its sp arrays.
    dgrid = getattr(params, "dgrid", 1.0) if params is not None else 1.0
    gamma_num = ampere_gamma_num(geom_sp, dgrid=dgrid, bessel=bessel)  # (nsp, ns, nkx, nky)
    diag_per_sp = signz.reshape(nsp, 1, 1, 1) ** 2 * de.reshape(nsp, 1, 1, 1) * gamma_num / mas.reshape(nsp, 1, 1, 1)
    diag_em = beta * jnp.sum(diag_per_sp, axis=0)
    kperp_sq_3d = kperp_sq.reshape(kperp_sq.shape[-3], kperp_sq.shape[-2], kperp_sq.shape[-1])

    apar_diag = kperp_sq_3d + diag_em
    apar_diag = jnp.where(jnp.abs(apar_diag) < EPS, 1.0, apar_diag)
    return apar_weight, apar_diag, kperp_sq


def calculate_apar(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    params: Any = None,
    pre: Optional[Precompute] = None,
) -> jnp.ndarray:
    """
    Compute A_parallel from Ampere's law.

    df: legacy parameter name for the evolved mixed variable ``g``/``dg`` with
        shape (nsp, nvpar, nmu, ns, nkx, nky), not the physical distribution
        δf after the g2f transform.
    Returns: A_parallel with shape (ns, nkx, nky).

    GKW's mixed-variable Ampere solve uses the evolved ``g`` as the source and
    keeps the g2f correction in the diagonal.  Passing physical ``f`` here
    double-counts that correction and gives the wrong A_parallel.
    """
    dg = df
    if pre is not None and "apar_weight" in pre:
        apar_weight = pre["apar_weight"]
        apar_diag = pre["apar_diag"]
    else:
        apar_weight, apar_diag, _ = precompute_apar(geometry, params)
    apar_num = jnp.einsum("avmjkl,avmjkl->jkl", apar_weight, dg)
    return apar_num / apar_diag


def precompute_bpar(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    species_precompute: Precompute | Mapping[str, Any],
) -> Dict[str, jnp.ndarray]:
    """Precompute the kinetic multi-species B_parallel coupled solve arrays.

    This is a behavior-preserving extraction of the inline ``nlbpar`` block
    from :func:`gyaradax.operators.precompute.linear_precompute`.  It intentionally uses the
    already precomputed kinetic species arrays (not ``geom_tensors``) as the
    source of truth so shapes, dtypes, and multi-species broadcasting match the
    solver path exactly.

    Returns a dict with the exact solver precompute keys:
      ``phi_weight``, ``phi_diag``, ``bpar_weight``, ``bpar_chi_factor``.
    """

    nsp = jnp.asarray(params.signz, dtype=jnp.float64).reshape(-1).shape[0]
    beta = jnp.asarray(params.beta, dtype=jnp.float64)

    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    ky = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)

    signz_6 = jnp.asarray(params.signz, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
    tmp_6 = jnp.asarray(params.tmp, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
    vthrat_6 = jnp.asarray(params.vthrat, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
    mas_6 = jnp.asarray(params.mas, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
    de_6 = jnp.asarray(params.de, dtype=jnp.float64).reshape(nsp, 1, 1, 1, 1, 1)
    intmu_6 = jnp.asarray(geometry["intmu"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    intvp_6 = jnp.asarray(geometry["intvp"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    bn_6 = jnp.reshape(bn, (1, 1, 1, -1, 1, 1))
    mugr_6 = mugr.reshape(1, 1, -1, 1, 1, 1)

    kxrh_6 = jnp.reshape(kx, (1, 1, 1, 1, -1, 1))
    ky_6 = jnp.reshape(ky, (1, 1, 1, 1, 1, -1))
    krloc_sq = (
        ky_6**2 * little_g[:, 0].reshape(1, 1, 1, -1, 1, 1)
        + 2 * ky_6 * kxrh_6 * little_g[:, 1].reshape(1, 1, 1, -1, 1, 1)
        + kxrh_6**2 * little_g[:, 2].reshape(1, 1, 1, -1, 1, 1)
    )
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, EPS))
    krloc_is_zero = jnp.abs(krloc) < 1e-5

    sz_6 = jnp.where(jnp.abs(signz_6) < EPS, 1.0, signz_6)
    gamma_arg = 0.5 * (mas_6 * vthrat_6 * krloc / (sz_6 * bn_6)) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    gamma0 = i0e(gamma_arg)
    gamma1 = i1e(gamma_arg)
    gamma_diff = jnp.where(krloc_is_zero, 1.0, gamma0 - gamma1)

    mugr_bn = jnp.maximum(2.0 * mugr_6 / bn_6, EPS)
    bessel_arg = mas_6 * vthrat_6 * krloc * jnp.sqrt(mugr_bn) / sz_6
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    j1hat = jnp.where(krloc_is_zero, 0.5, j1_hat(bessel_arg))

    # coupling coefficients (summed over species)
    gamma0_for_fsp1 = jnp.where(krloc_is_zero, 1.0, gamma0)
    F_sp1 = jnp.sum(signz_6**2 * de_6 * (gamma0_for_fsp1 - 1.0) / tmp_6, axis=0, keepdims=True)
    F_sp2 = jnp.sum(signz_6 * beta * de_6 * gamma_diff / (2.0 * bn_6), axis=0, keepdims=True)
    B_sp1 = jnp.sum(signz_6 * de_6 * gamma_diff / bn_6, axis=0, keepdims=True)
    B_sp2 = jnp.sum(tmp_6 * de_6 * beta * gamma_diff / bn_6**2, axis=0, keepdims=True)

    cdiag = F_sp1 * (1.0 + B_sp2) - F_sp2 * B_sp1
    cdiag_3d = cdiag.reshape(cdiag.shape[-3], cdiag.shape[-2], cdiag.shape[-1])
    cdiag_3d = jnp.where(jnp.abs(cdiag_3d) < EPS, 1.0, cdiag_3d)

    I_sp1 = signz_6 * de_6 * bn_6 * species_precompute["bessel"] * intvp_6 * intmu_6
    I_sp2 = beta * bn_6 * tmp_6 * de_6 * intvp_6 * intmu_6 * mugr_6 * j1hat

    return {
        "phi_weight": I_sp1 * (1.0 + B_sp2) - I_sp2 * B_sp1,
        "phi_diag": cdiag_3d,
        "bpar_weight": I_sp2 * F_sp1 - I_sp1 * F_sp2,
        "bpar_chi_factor": 2.0 * mugr_6 * tmp_6 / signz_6 * j1hat,
    }
