from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import i0e, bessel_jn
from einops import rearrange
from typing import Dict, Tuple, Any

from gyaradax import _EPS


def j0(x):
    safe_x = jnp.where(jnp.abs(x) < 1e-10, 1.0, x)
    res = bessel_jn(safe_x, v=0)[0]
    return jnp.where(jnp.abs(x) < 1e-10, 1.0, res)


def j1_hat(x):
    """Modified J1 Bessel: 2*J_1(x)/x.

    Matches GKW ``mod_besselj1_gkw`` (functions.f90:119-142).
    GKW returns 0.5 when krloc < 1e-5 (zero-mode convention).
    For non-zero x the mathematical limit is 1.0, but we follow GKW exactly.
    The caller must pass ``krloc_is_zero`` to select the zero-mode value.
    """
    safe_x = jnp.where(jnp.abs(x) < 1e-30, 1.0, x)
    # bessel_jn(x, v=n) returns [J_0, J_1, ..., J_n]; index [1] for J_1
    j1_val = bessel_jn(safe_x, v=1)[1]
    return jnp.where(jnp.abs(x) < 1e-30, 1.0, 2.0 * j1_val / safe_x)


def i1e(b):
    """Gamma_1(b) = I_1(b)*exp(-b), the scaled modified Bessel of order 1.

    Matches GKW ``gamma1_gkw`` (functions.f90:42-56) via ``expbessi1``.
    """
    return jax.scipy.special.i1e(b)


def geom_tensors(geometry: Dict[str, jnp.ndarray], params: Any = None) -> Dict[str, jnp.ndarray]:
    """
    Expand geometry constants for broadcasting and compute Bessel terms.

    Single-species version. Species params are scalars reshaped to (1,1,1,1,1,1).
    """
    geom_ = {}
    geom_["krho"] = rearrange(geometry["krho"], "y -> 1 1 1 1 1 y")
    geom_["ints"] = rearrange(geometry["ints"], "s -> 1 1 1 s 1 1")
    geom_["intmu"] = rearrange(geometry["intmu"], "mu -> 1 1 mu 1 1 1")
    geom_["intvp"] = rearrange(geometry["intvp"], "par -> 1 par 1 1 1 1")
    geom_["vpgr"] = rearrange(geometry["vpgr"], "par -> 1 par 1 1 1 1")
    geom_["mugr"] = rearrange(geometry["mugr"], "mu -> 1 1 mu 1 1 1")
    geom_["bn"] = rearrange(geometry["bn"], "s -> 1 1 1 s 1 1")
    geom_["ffun"] = rearrange(geometry["ffun"], "s -> 1 1 1 s 1 1")
    geom_["efun"] = rearrange(geometry["efun"], "s -> 1 1 1 s 1 1")
    geom_["rfun"] = rearrange(geometry["rfun"], "s -> 1 1 1 s 1 1")
    geom_["bt_frac"] = rearrange(geometry["bt_frac"], "s -> 1 1 1 s 1 1")
    geom_["parseval"] = rearrange(geometry["parseval"], "y -> 1 1 1 1 1 y")

    for k in ["mas", "tmp", "de", "d2X", "signz", "signB"]:
        if params is not None and hasattr(params, k):
            val = getattr(params, k)
        else:
            val = geometry[k]
            if val.ndim > 0:
                val = val[0]
        geom_[k] = jnp.reshape(jnp.asarray(val, dtype=jnp.float64), (1, 1, 1, 1, 1, 1))

    if params is not None and hasattr(params, "vthrat"):
        vthrat = params.vthrat
    else:
        vthrat = geometry["vthrat"]
        if vthrat.ndim > 0:
            vthrat = vthrat[0]
    vthrat = jnp.reshape(jnp.asarray(vthrat, dtype=jnp.float64), (1, 1, 1, 1, 1, 1))

    kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 1 x 1")
    little_g = rearrange(geometry["little_g"], "s three -> three 1 1 1 s 1 1")

    krloc_sq = (
        geom_["krho"] ** 2 * little_g[0]
        + 2 * geom_["krho"] * kxrh * little_g[1]
        + kxrh**2 * little_g[2]
    )
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, _EPS))

    mugr_bn = jnp.maximum(2.0 * geom_["mugr"] / geom_["bn"], _EPS)

    sz = jnp.where(jnp.abs(geom_["signz"]) < _EPS, 1.0, geom_["signz"])
    bessel_arg = jnp.sqrt(mugr_bn) / sz
    bessel_arg = geom_["mas"] * vthrat * krloc * bessel_arg
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    geom_["bessel"] = j0(bessel_arg)

    gamma_arg = geom_["mas"] * vthrat * krloc
    gamma_arg = 0.5 * (gamma_arg / (sz * geom_["bn"])) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    geom_["gamma"] = i0e(gamma_arg)

    # zonal mode detection
    krho_flat = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    kxrh_flat = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    iyzero = jnp.argmin(jnp.abs(krho_flat))
    ixzero = jnp.argmin(jnp.abs(kxrh_flat))
    geom_["has_zonal"] = jnp.where(jnp.abs(krho_flat[iyzero]) < 1e-10, 1.0, 0.0)
    geom_["ixzero"] = ixzero
    geom_["iyzero"] = iyzero

    return geom_


@jax.jit
def _phi_adiabatic(geom: Dict[str, jnp.ndarray], df: jnp.ndarray) -> jnp.ndarray:
    """Adiabatic electron phi from single-species quasineutrality.

    df: (nvpar, nmu, ns, nkx, nky).
    Internal — use calculate_phi() as the public interface.
    """
    de = geom["de"]
    signz, tmp, bn = geom["signz"], geom["tmp"], geom["bn"]
    ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
    bessel, gamma = geom["bessel"], geom["gamma"]

    poisson_int = signz * de * intmu * intvp * bessel * bn
    poisson_int = jnp.where(jnp.abs(intvp) < _EPS, 0.0, poisson_int)

    cfen = 0.0
    diagz = signz * (gamma - 1.0) * jnp.exp(-cfen) / tmp
    denom = diagz - jnp.exp(-cfen) / tmp
    denom = jnp.where(jnp.abs(denom) < _EPS, 1.0, denom)
    matz = -ints / (signz * de * denom)
    has_zonal = geom["has_zonal"]
    ixzero, iyzero = geom["ixzero"], geom["iyzero"]

    # zero matz for all ky except the zonal mode
    ky_is_zonal = jnp.arange(matz.shape[-1]) == iyzero
    matz = matz * ky_is_zonal * has_zonal

    phi = poisson_int * df
    phi = jnp.sum(phi, axis=(1, 2), keepdims=True)

    y_mask = jnp.zeros_like(phi)
    y_mask = y_mask.at[..., iyzero].set(has_zonal)

    bufphi = matz * phi
    bufphi = jnp.sum(bufphi, axis=3, keepdims=True)

    maty_sum = jnp.sum(-matz * jnp.exp(-cfen), axis=3, keepdims=True)
    maty = tmp / (de * jnp.exp(-cfen)) + maty_sum / jnp.exp(-cfen)

    # at kx=0 (ixzero), set maty=1 to skip the correction
    x_is_zero = jnp.arange(phi.shape[-2]) == ixzero
    x_mask = jnp.broadcast_to(x_is_zero[None, None, None, None, :, None], phi.shape)
    maty_val = jnp.where(x_mask, 1.0 + 0j, maty)
    maty_val = jnp.where(jnp.abs(maty_val) < _EPS, 1.0, maty_val)
    maty_val = 1.0 / maty_val
    phi = phi + maty_val * bufphi * y_mask

    poisson_diag = jnp.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
    norm_mask = jnp.ones_like(phi)
    norm_mask = norm_mask.at[..., ixzero, iyzero].set(1.0 - has_zonal)

    pdiag = poisson_diag * norm_mask - signz * jnp.exp(-cfen) * de / tmp
    pdiag = jnp.where(jnp.abs(pdiag) < _EPS, -1.0, pdiag)

    phi = phi * (-1.0 / pdiag)
    return jnp.squeeze(phi, axis=(0, 1, 2))


def precompute_phi_adiabatic(geometry: Dict[str, jnp.ndarray], params: Any):
    """Precompute static arrays for the adiabatic phi solve."""
    geom = geom_tensors(geometry, params=params)

    # squeeze leading nsp=1 dimension to simplify to 5D: (v, mu, s, kx, ky)
    for k in list(geom.keys()):
        if isinstance(geom[k], jnp.ndarray) and geom[k].ndim > 0:
            geom[k] = jnp.squeeze(geom[k], axis=0)

    de = geom["de"]
    signz, tmp, bn = geom["signz"], geom["tmp"], geom["bn"]
    ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
    bessel, gamma = geom["bessel"], geom["gamma"]

    # weights for the first reduction: (nv, nmu, ns, nkx, nky)
    phi_weight = signz * de * intmu * intvp * bessel * bn
    phi_weight = jnp.where(jnp.abs(intvp) < 1e-9, 0.0, phi_weight)

    cfen = 0.0
    diagz = signz * (gamma - 1.0) * jnp.exp(-cfen) / tmp
    denom = diagz - jnp.exp(-cfen) / tmp
    denom = jnp.where(jnp.abs(denom) < 1e-15, 1.0, denom)
    matz = -ints / (signz * de * denom)

    # zonal mode detection
    krho_flat = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    kxrh_flat = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    iyzero = jnp.argmin(jnp.abs(krho_flat))
    ixzero = jnp.argmin(jnp.abs(kxrh_flat))
    has_zonal = jnp.where(jnp.abs(krho_flat[iyzero]) < 1e-10, 1.0, 0.0)

    # weight for the second reduction (zonal mode correction, sum over ns)
    # zero matz for all ky except the zonal mode
    ky_is_zonal = jnp.arange(matz.shape[-1]) == iyzero
    matz = matz * ky_is_zonal[None, None, None, None, :] * has_zonal
    phi_corr_weight = matz

    return phi_weight, phi_corr_weight, tmp, de, signz, gamma, ints, has_zonal, ixzero, iyzero


def calculate_phi_adiabatic(
    df: jnp.ndarray,
    phi_weight: jnp.ndarray,
    phi_corr_weight: jnp.ndarray,
    tmp: jnp.ndarray,
    de: jnp.ndarray,
    signz: jnp.ndarray,
    gamma: jnp.ndarray,
    ints: jnp.ndarray,
    has_zonal: float,
    ixzero: int,
    iyzero: int,
) -> jnp.ndarray:
    """Newly structured hot path for adiabatic phi solve."""
    # df: (nv, nmu, ns, nkx, nky)
    # phi_weight: (nv, nmu, ns, nkx, nky)
    phi_raw = jnp.sum(phi_weight * df, axis=(0, 1), keepdims=True)  # (1, 1, ns, nkx, nky)

    # zonal mode correction (sum over ns)
    # phi_corr_weight: (1, 1, ns, 1, 1)
    bufphi = jnp.sum(phi_corr_weight * phi_raw, axis=2, keepdims=True)  # (1, 1, 1, nkx, nky)

    cfen = 0.0
    exp_cfen = jnp.exp(-cfen)

    # factor 1: maty_val correction (applies only to ky=0)
    maty_sum = jnp.sum(-phi_corr_weight * exp_cfen, axis=2, keepdims=True)
    maty = tmp / (de * exp_cfen) + maty_sum / exp_cfen

    nkx, nky = phi_raw.shape[3], phi_raw.shape[4]

    # zonal (ky=iyzero) mask
    y_mask = jnp.zeros((1, 1, 1, 1, nky), dtype=phi_raw.dtype).at[..., iyzero].set(has_zonal)
    # kx=ixzero mask
    x_mask = jnp.zeros((1, 1, 1, nkx, 1), dtype=phi_raw.dtype).at[..., ixzero, :].set(1.0)

    maty_val = jnp.where(x_mask > 0, 1.0 + 0j, maty)
    maty_val = jnp.where(jnp.abs(maty_val) < 1e-15, 1.0, maty_val)

    phi_corr = phi_raw + (1.0 / maty_val) * bufphi * y_mask

    # factor 2: poisson_diag (pdiag)
    poisson_diag = exp_cfen * (signz**2) * de * (gamma - 1.0) / tmp
    # zonal mask (kx=ixzero, ky=iyzero)
    zonal_mask = x_mask * y_mask
    norm_mask = jnp.ones_like(zonal_mask) - zonal_mask

    pdiag = poisson_diag * norm_mask - signz * exp_cfen * de / tmp
    pdiag = jnp.where(jnp.abs(pdiag) < 1e-15, -1.0, pdiag)

    phi = phi_corr * (-1.0 / pdiag)
    # squeeze leading (1, 1) from keepdims but keep (ns, nkx, nky)
    return phi.reshape(phi.shape[2], phi.shape[3], phi.shape[4])


def _species_bessel_gamma(geometry):
    """Per-species Bessel J0 and Gamma_0 for multi-species phi solve."""
    mas = jnp.asarray(geometry["mas"], dtype=jnp.float64)
    signz = jnp.asarray(geometry["signz"], dtype=jnp.float64)
    vthrat = jnp.asarray(geometry["vthrat"], dtype=jnp.float64)
    nsp = mas.shape[0]

    mas_6d = mas.reshape(nsp, 1, 1, 1, 1, 1)
    signz_6d = signz.reshape(nsp, 1, 1, 1, 1, 1)
    vthrat_6d = vthrat.reshape(nsp, 1, 1, 1, 1, 1)
    sz = jnp.where(jnp.abs(signz_6d) < _EPS, 1.0, signz_6d)

    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64).reshape(1, 1, 1, 1, 1, -1)
    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64).reshape(1, 1, 1, 1, -1, 1)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, 1, -1, 1, 1)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)

    g0 = little_g[:, 0].reshape(1, 1, 1, -1, 1, 1)
    g1 = little_g[:, 1].reshape(1, 1, 1, -1, 1, 1)
    g2 = little_g[:, 2].reshape(1, 1, 1, -1, 1, 1)
    krloc_sq = krho**2 * g0 + 2 * krho * kxrh * g1 + kxrh**2 * g2
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, _EPS))

    mugr_bn = jnp.maximum(2.0 * mugr / jnp.maximum(bn, _EPS), _EPS)
    bessel_arg = mas_6d * vthrat_6d * krloc * jnp.sqrt(mugr_bn) / sz
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    bessel = j0(bessel_arg)

    gamma_arg = 0.5 * (mas_6d * vthrat_6d * krloc / (sz * bn)) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    gamma_arg_nommu = gamma_arg[:, :, 0:1, :, :, :]
    gamma = i0e(gamma_arg_nommu)

    return bessel, gamma


def precompute_phi_kinetic(geometry: Dict[str, jnp.ndarray]):
    """
    Precompute static arrays for the kinetic phi solve.

    returns (phi_weight, phi_diag) where:
        phi_weight: (nsp, nvpar, nmu, ns, nkx, nky) — poisson integral weight
        phi_diag: (ns, nkx, nky) — poisson diagonal (with zonal mode set to 1)
    """
    bessel, gamma = _species_bessel_gamma(geometry)

    mas = jnp.asarray(geometry["mas"], dtype=jnp.float64)
    signz = jnp.asarray(geometry["signz"], dtype=jnp.float64)
    tmp = jnp.asarray(geometry["tmp"], dtype=jnp.float64)
    de = jnp.asarray(geometry["de"], dtype=jnp.float64)
    nsp = mas.shape[0]

    signz_6d = signz.reshape(nsp, 1, 1, 1, 1, 1)
    de_6d = de.reshape(nsp, 1, 1, 1, 1, 1)
    tmp_6d = tmp.reshape(nsp, 1, 1, 1, 1, 1)

    intvp = jnp.asarray(geometry["intvp"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    intmu = jnp.asarray(geometry["intmu"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, 1, -1, 1, 1)

    # poisson integral weight: sum(weight * df) over (species, vpar, mu) gives phi numerator
    weight = signz_6d * de_6d * intmu * intvp * bessel * bn
    weight = jnp.where(jnp.abs(intvp) < _EPS, 0.0, weight)

    # poisson diagonal: sum over species of Z^2 * n * (Gamma0 - 1) / T
    diag_per_sp = signz_6d**2 * de_6d * (gamma - 1.0) / tmp_6d
    diag = jnp.sum(diag_per_sp, axis=0)
    # reshape to (ns, nkx, nky), dropping summed velocity axes
    diag = diag.reshape(diag.shape[-3], diag.shape[-2], diag.shape[-1])

    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    ixzero = jnp.argmin(jnp.abs(kxrh))
    iyzero = jnp.argmin(jnp.abs(krho))
    # only set zonal diagonal to 1 if a real ky=0 mode exists
    has_zonal = jnp.abs(krho[iyzero]) < 1e-10
    diag_with_zonal = diag.at[:, ixzero, iyzero].set(1.0)
    diag = jnp.where(has_zonal, diag_with_zonal, diag)
    diag = jnp.where(jnp.abs(diag) < _EPS, -1.0, diag)

    return weight, diag


def _phi_kinetic(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    phi_weight: jnp.ndarray = None,
    phi_diag: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Kinetic electron phi from multi-species quasineutrality.

    df: (nsp, nvpar, nmu, ns, nkx, nky).
    Internal — use calculate_phi() as the public interface.
    """
    if phi_weight is None or phi_diag is None:
        phi_weight, phi_diag = precompute_phi_kinetic(geometry)

    phi_num = jnp.einsum("avmjkl,avmjkl->jkl", phi_weight, df)
    return -phi_num / phi_diag


def calculate_phi(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    params: Any = None,
    pre: Dict = None,
) -> jnp.ndarray:
    """
    Compute electrostatic potential from quasineutrality.

    Unified interface for both adiabatic and kinetic electron models.
    Uses precomputed arrays from pre when available.

    Args:
        geometry: base geometry dict (or expanded geom_tensors for legacy calls).
        df: distribution function — (nvpar, nmu, ns, nkx, nky) for adiabatic,
            (nsp, nvpar, nmu, ns, nkx, nky) for kinetic.
        params: GKParams (used to determine adiabatic vs kinetic and species scalars).
        pre: precomputed arrays from linear_precompute (optional, for performance).
    """
    # legacy: calculate_phi(geom_tensors_dict, df) — detect by presence of "bessel"
    if "bessel" in geometry:
        return _phi_adiabatic(geometry, df)

    adiabatic = params.adiabatic_electrons if params is not None else (df.ndim == 5)
    if adiabatic:
        gt = pre["geom_tensors"] if pre is not None else geom_tensors(geometry, params=params)
        return _phi_adiabatic(gt, df)
    else:
        pw = pre.get("phi_weight") if pre is not None else None
        pd = pre.get("phi_diag") if pre is not None else None
        return _phi_kinetic(geometry, df, pw, pd)


# backward-compatible aliases
calculate_phi_kinetic = _phi_kinetic


# ── Ampere's law for A_parallel ──────────────────────────────────────────────


def precompute_apar(geometry: Dict[str, jnp.ndarray], params: Any = None):
    """
    Precompute static arrays for the Ampere A_parallel solve (kinetic species).

    From GKW linear_terms.f90 ampere_int + ampere_dia.

    Ampere's law (normalized):
        [k_perp^2 + beta * sum_sp(Z^2 n / m * Gamma_0)] * A_par
        = beta * sum_sp(Z * vthrat * n * 2pi*B * integral(vpar * J0 * g * dvpar dmu))

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

    # Build geometry with proper species arrays for Bessel computation
    geom_sp = dict(geometry)
    geom_sp["mas"] = mas
    geom_sp["signz"] = signz
    geom_sp["tmp"] = tmp
    geom_sp["de"] = de
    geom_sp["vthrat"] = vthrat
    bessel, gamma = _species_bessel_gamma(geom_sp)

    mas_6d = mas.reshape(nsp, 1, 1, 1, 1, 1)
    signz_6d = signz.reshape(nsp, 1, 1, 1, 1, 1)
    de_6d = de.reshape(nsp, 1, 1, 1, 1, 1)
    vthrat_6d = vthrat.reshape(nsp, 1, 1, 1, 1, 1)

    intvp = jnp.asarray(geometry["intvp"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    intmu = jnp.asarray(geometry["intmu"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1, 1)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, 1, -1, 1, 1)

    beta = jnp.asarray(params.beta if params is not None else 0.0, dtype=jnp.float64)

    # k_perp^2 (same computation as in _species_bessel_gamma)
    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64).reshape(1, 1, 1, 1, 1, -1)
    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64).reshape(1, 1, 1, 1, -1, 1)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)
    g0 = little_g[:, 0].reshape(1, 1, 1, -1, 1, 1)
    g1 = little_g[:, 1].reshape(1, 1, 1, -1, 1, 1)
    g2 = little_g[:, 2].reshape(1, 1, 1, -1, 1, 1)
    kperp_sq = krho**2 * g0 + 2 * krho * kxrh * g1 + kxrh**2 * g2

    # --- Ampere numerator weight ---
    # Matches GKW ampere_int (linear_terms.f90:3246):
    # elem = signz*de*veta*intvp*intmu*vthrat*bn*vpgr*J0
    # vthrat^1 converts vpgr (in v_ths units) to physical velocity in v_thi units.
    # GKW uses the same shared velocity grid for all species (velocitygrid.f90:197).
    apar_weight = beta * signz_6d * de_6d * vthrat_6d * vpgr * bessel * bn * intvp * intmu
    apar_weight = jnp.where(jnp.abs(intvp) < _EPS, 0.0, apar_weight)

    # --- Ampere denominator ---
    # Computed as analytical fallback here; solver.py overrides with
    # numerical gamma_num (using properly normalized fmaxwl) for GKW parity.
    diag_per_sp = signz_6d**2 * de_6d / mas_6d * gamma
    diag_em = beta * jnp.sum(diag_per_sp, axis=0)
    diag_em = diag_em.reshape(diag_em.shape[-3], diag_em.shape[-2], diag_em.shape[-1])
    kperp_sq_3d = kperp_sq.reshape(kperp_sq.shape[-3], kperp_sq.shape[-2], kperp_sq.shape[-1])

    apar_diag = kperp_sq_3d + diag_em
    apar_diag = jnp.where(jnp.abs(apar_diag) < _EPS, 1.0, apar_diag)

    return apar_weight, apar_diag, kperp_sq


def calculate_apar(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    params: Any = None,
    pre: Dict = None,
) -> jnp.ndarray:
    """
    Compute A_parallel from Ampere's law.

    df: (nsp, nvpar, nmu, ns, nkx, nky) — the physical distribution δf (not g).
    Returns: A_parallel with shape (ns, nkx, nky).

    Note: df must be the physical distribution (after g2f transform), not the
    mixed variable g. The caller is responsible for the g->f conversion.
    """
    if pre is not None and "apar_weight" in pre:
        apar_weight = pre["apar_weight"]
        apar_diag = pre["apar_diag"]
    else:
        apar_weight, apar_diag, _ = precompute_apar(geometry, params)

    # numerator: integral of weight * df over (species, vpar, mu)
    apar_num = jnp.einsum("avmjkl,avmjkl->jkl", apar_weight, df)
    return apar_num / apar_diag


def precompute_bpar(geometry, params):
    """Precompute the coupled Poisson-Bpar field solve arrays.

    When nlbpar=True, the Poisson and Bpar equations form a coupled 2x2
    system, decoupled via intermediate coefficients F_sp1, F_sp2, B_sp1,
    B_sp2 (GKW ``poisson_dia``, ``poisson_int``, ``ampere_bpar_int`` in
    linear_terms.f90:3112-3377, 3444-3525).

    Returns (coupled_phi_weight, bpar_weight, coupled_diag, j1hat_6d, gamma1_6d)
    where:
      coupled_phi_weight: (nsp, nvpar, nmu, ns, nkx, nky) -- replaces phi_weight
      bpar_weight: (nsp, nvpar, nmu, ns, nkx, nky) -- B_par numerator weight
      coupled_diag: (ns, nkx, nky) -- shared coupled diagonal
      j1hat_6d: (nsp, nvpar, nmu, ns, nkx, nky) -- for gyro-averaging B_par
      gamma1_6d: (nsp, 1, 1, ns, nkx, nky) -- Gamma_1 for diagnostics
    """

    geom_ = geom_tensors(geometry, params)
    beta = jnp.asarray(getattr(params, "beta", 0.0), dtype=jnp.float64)

    signz_6d = geom_["signz"]
    de_6d = geom_["de"]
    tmp_6d = geom_["tmp"]
    bn = geom_["bn"]
    mugr = geom_["mugr"]
    intvp = geom_["intvp"]
    intmu = geom_["intmu"]

    bessel_j0 = geom_["bessel"]  # J0(bessel_arg)
    gamma0 = geom_["gamma"]  # Gamma_0 = I_0(b)*exp(-b)

    # --- Gamma_1 and J1_hat ---
    kxrh = rearrange(jnp.asarray(geometry["kxrh"], dtype=jnp.float64), "x -> 1 1 1 1 x 1")
    little_g = rearrange(
        jnp.asarray(geometry["little_g"], dtype=jnp.float64), "s three -> three 1 1 1 s 1 1"
    )
    krloc_sq = (
        geom_["krho"] ** 2 * little_g[0]
        + 2 * geom_["krho"] * kxrh * little_g[1]
        + kxrh**2 * little_g[2]
    )
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, _EPS))

    # Gamma_1(b) = I_1(b)*exp(-b) where b = 0.5*(mas*vthrat*krloc/(signz*bn))^2
    sz = jnp.where(jnp.abs(signz_6d) < _EPS, 1.0, signz_6d)
    gamma_arg = 0.5 * (geom_["mas"] * geom_["vthrat"] * krloc / (sz * bn)) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    gamma1_6d = i1e(gamma_arg)

    # gamma_diff = (Gamma_0 - Gamma_1) * exp(-cfen)
    # Without rotation: cfen=0, so gamma_diff = Gamma_0 - Gamma_1
    # Zero-mode limit: gamma_diff = 1 (Gamma_0=1, Gamma_1=0)
    krloc_is_zero = jnp.abs(krloc) < 1e-5
    gamma_diff = jnp.where(krloc_is_zero, 1.0, gamma0 - gamma1_6d)

    # J1_hat: 2*J_1(bessel_arg) / bessel_arg
    mugr_bn = jnp.maximum(2.0 * mugr / bn, _EPS)
    bessel_arg = geom_["mas"] * geom_["vthrat"] * krloc * jnp.sqrt(mugr_bn) / sz
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    # GKW returns 0.5 for zero mode (krloc < 1e-5)
    j1hat_6d = jnp.where(krloc_is_zero, 0.5, j1_hat(bessel_arg))

    # --- Coupling coefficients (summed over species) ---
    # F_sp1 = sum_sp[ signz^2 * de * (Gamma_0 - 1) / tmp ]
    # F_sp2 = sum_sp[ signz * beta * de * gamma_diff / (2*bn) ]
    # B_sp1 = sum_sp[ signz * de * gamma_diff / bn ]
    # B_sp2 = sum_sp[ tmp * de * beta * gamma_diff / bn^2 ]
    # Note: GKW uses gamma*exp(-cfen) for F_sp1; without rotation gamma_0 already
    # includes exp(-cfen)=1. For the zero mode, gamma=1.
    gamma0_for_fsp1 = jnp.where(krloc_is_zero, 1.0, gamma0)
    F_sp1 = jnp.sum(signz_6d**2 * de_6d * (gamma0_for_fsp1 - 1.0) / tmp_6d, axis=0, keepdims=True)
    F_sp2 = jnp.sum(signz_6d * beta * de_6d * gamma_diff / (2.0 * bn), axis=0, keepdims=True)
    B_sp1 = jnp.sum(signz_6d * de_6d * gamma_diff / bn, axis=0, keepdims=True)
    B_sp2 = jnp.sum(tmp_6d * de_6d * beta * gamma_diff / bn**2, axis=0, keepdims=True)

    # --- Coupled diagonal ---
    # diagonal = F_sp1 * (1 + B_sp2) - F_sp2 * B_sp1
    coupled_diag = F_sp1 * (1.0 + B_sp2) - F_sp2 * B_sp1

    # Squeeze to (ns, nkx, nky) — remove species/vpar/mu dims
    coupled_diag = coupled_diag.reshape(
        coupled_diag.shape[-3], coupled_diag.shape[-2], coupled_diag.shape[-1]
    )
    coupled_diag = jnp.where(jnp.abs(coupled_diag) < _EPS, 1.0, coupled_diag)

    # --- Coupled phi weight (modified Poisson numerator) ---
    # I_sp1 = signz * de * bn * J0 * intvp * intmu   (standard Poisson weight)
    # I_sp2 = beta * bn * tmp * de * intvp * intmu * mugr * J1_hat
    # coupled_phi_weight = I_sp1 * (1 + B_sp2) - I_sp2 * B_sp1
    I_sp1 = signz_6d * de_6d * bn * bessel_j0 * intvp * intmu
    I_sp2 = beta * bn * tmp_6d * de_6d * intvp * intmu * mugr * j1hat_6d

    coupled_phi_weight = I_sp1 * (1.0 + B_sp2) - I_sp2 * B_sp1

    # --- Bpar weight ---
    # bpar_weight = I_sp2 * F_sp1 - I_sp1 * F_sp2
    bpar_weight = I_sp2 * F_sp1 - I_sp1 * F_sp2

    # Squeeze gamma1 to (nsp, 1, 1, ns, nkx, nky) for diagnostics
    gamma1_out = gamma1_6d[:, :1, :1, :, :, :]

    return coupled_phi_weight, bpar_weight, coupled_diag, j1hat_6d, gamma1_out


@partial(jax.jit, static_argnames=("reduce",))
def calculate_fluxes(
    geom: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    phi: jnp.ndarray,
    reduce: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-species fluxes. df: (nvpar, nmu, ns, nkx, nky).

    When reduce=True (default), returns flux-surface-averaged scalar
    (pflux, eflux, vflux) for backward compatibility.

    When reduce=False, returns per-mode flux fields of shape (nkx, nky):
    velocity-space and parallel integration are still applied, but the
    final sum over (kx, ky) is left to the caller. Useful for quasilinear
    weights Γ_lin(k_x, k_y) / |φ_lin(k_x, k_y)|² and for kx/ky spectra.
    """
    bn, bt_frac, parseval = geom["bn"], geom["bt_frac"], geom["parseval"]
    rfun, efun, d2X, signB = geom["rfun"], geom["efun"], geom["d2X"], geom["signB"]
    ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
    vpgr, mugr, krho = geom["vpgr"], geom["mugr"], geom["krho"]
    bessel = geom["bessel"]

    phi_expanded = rearrange(phi, "s x y -> 1 1 1 s x y")
    phi_gyro = bessel * phi_expanded

    dum = parseval * ints * (efun * krho) * df
    dum1 = dum * jnp.conj(phi_gyro)
    dum2 = dum1 * bn
    d3v = d2X * intmu * bn * intvp

    pflux = d3v * jnp.imag(dum1)
    eflux = d3v * (vpgr**2 * jnp.imag(dum1) + 2 * mugr * jnp.imag(dum2))
    vflux = d3v * (jnp.imag(dum1) * vpgr * rfun * bt_frac * signB)

    # flux-surface average (sum(ints) = 1 for nperiod=1)
    fsa = jnp.sum(ints)

    if reduce:
        return jnp.sum(pflux) / fsa, jnp.sum(eflux) / fsa, jnp.sum(vflux) / fsa

    # per-(kx, ky) flux fields: sum over everything except the last two axes.
    # the integrand can pick up an extra singleton leading axis from the
    # rearrange(phi, "s x y -> 1 1 1 s x y") above, so we compute the axis
    # tuple dynamically rather than hardcoding (0, 1, 2).
    axes = tuple(range(pflux.ndim - 2))
    return (
        jnp.sum(pflux, axis=axes) / fsa,
        jnp.sum(eflux, axis=axes) / fsa,
        jnp.sum(vflux, axis=axes) / fsa,
    )


def calculate_fluxes_kinetic(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    phi: jnp.ndarray,
    reduce: bool = True,
) -> jnp.ndarray:
    """
    Per-species fluxes for kinetic case.

    df: (nsp, nvpar, nmu, ns, nkx, nky).

    reduce=True  (default): returns (nsp, 3) — scalar [pflux, eflux, vflux] per species.
    reduce=False:           returns (nsp, 3, nkx, nky) — per-(kx, ky) flux fields.
    """
    nsp = df.shape[0]

    def _flux_single(isp):
        sp_geom = dict(geometry)
        for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
            if k in geometry and jnp.asarray(geometry[k]).ndim > 0:
                sp_geom[k] = jnp.asarray(geometry[k])[isp : isp + 1]
        gt = geom_tensors(sp_geom)
        pflux, eflux, vflux = calculate_fluxes(gt, df[isp], phi, reduce=reduce)
        return jnp.stack([pflux, eflux, vflux])  # (3,) or (3, nkx, nky)

    return jnp.stack([_flux_single(i) for i in range(nsp)])  # (nsp, 3) or (nsp, 3, nkx, nky)


def calculate_em_fluxes(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    apar: jnp.ndarray,
    params: Any = None,
    bpar: jnp.ndarray = None,
    pre: Dict = None,
    reduce: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Electromagnetic flux contributions from A_parallel and B_parallel.

    reduce=True  (default): returns scalar (em_pflux, em_eflux, em_vflux) for 5D df,
                            or (nsp, 3) for 6D df.
    reduce=False:           returns per-(kx, ky) flux fields — (nkx, nky) for 5D,
                            (nsp, 3, nkx, nky) for 6D.

    A_par flutter: coupling factor -2*vthrat*vpgr*conj(J0*apar)
    B_par flutter: coupling factor +2*mugr*tmp/signz*conj(J1hat*bpar)

    Adds the EM parts of the generalised-potential flux formula
    `f * conj(chi)` with `chi = phi - 2·vthrat·vpar·apar + ...` used by
    GKW's calc_fluxes_full_detail (diagnos_fluxes_vspace.F90:458-469).

    df: 5D or 6D distribution. apar, bpar: (ns, nkx, nky) or None.
    Returns: (em_pflux, em_eflux, em_vflux) — scalars for 5D,
    (nsp, 3) for 6D.
    """
    z = jnp.array(0.0, dtype=jnp.float64)
    if apar is None and bpar is None:
        return z, z, z

    ints = jnp.asarray(geometry["ints"], dtype=jnp.float64)
    fsa = jnp.sum(ints)

    if df.ndim == 5:
        parseval = jnp.asarray(geometry["parseval"], dtype=jnp.float64)
        intvp = jnp.asarray(geometry["intvp"], dtype=jnp.float64)
        intmu = jnp.asarray(geometry["intmu"], dtype=jnp.float64)
        vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
        mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
        bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)
        efun = jnp.asarray(geometry["efun"], dtype=jnp.float64)
        krho = jnp.asarray(geometry["krho"], dtype=jnp.float64)
        rfun = jnp.asarray(geometry["rfun"], dtype=jnp.float64)
        bt_frac = jnp.asarray(geometry["bt_frac"], dtype=jnp.float64)
        signB = jnp.asarray(geometry["signB"], dtype=jnp.float64)
        d2X = jnp.asarray(geometry.get("d2X", 1.0), dtype=jnp.float64)

        vthrat_val = float(getattr(params, "vthrat", 1.0)) if params else 1.0
        vpgr_b = vpgr.reshape(-1, 1, 1, 1, 1)
        mugr_b = mugr.reshape(1, -1, 1, 1, 1)
        bn_b = bn.reshape(1, 1, -1, 1, 1)
        ints_b = ints.reshape(1, 1, -1, 1, 1)
        parseval_b = parseval.reshape(1, 1, 1, 1, -1)
        efun_b = efun.reshape(1, 1, -1, 1, 1)
        krho_b = krho.reshape(1, 1, 1, 1, -1)
        intvp_b = intvp.reshape(-1, 1, 1, 1, 1)
        intmu_b = intmu.reshape(1, -1, 1, 1, 1)
        rfun_b = rfun.reshape(1, 1, -1, 1, 1)
        bt_frac_b = bt_frac.reshape(1, 1, -1, 1, 1)

        d3v = d2X * intmu_b * bn_b * intvp_b
        dum = parseval_b * ints_b * (efun_b * krho_b) * df

        # integrand shape (nvpar, nmu, ns, nkx, nky); axes (0,1,2) are velocity-space + s.
        # reduce=True sums everything (scalar); reduce=False keeps (kx, ky).
        sum_axes = (0, 1, 2, 3, 4) if reduce else (0, 1, 2)
        zero_shape = z if reduce else jnp.zeros((df.shape[-2], df.shape[-1]), dtype=jnp.float64)
        em_pflux, em_eflux, em_vflux = zero_shape, zero_shape, zero_shape
        if apar is not None:
            apar_b = apar[jnp.newaxis, jnp.newaxis, :, :, :]
            # matches GKW diagnos_fluxes_vspace.F90:464 (-2·vthrat·vpar in χ)
            dum_a = -2.0 * vthrat_val * vpgr_b * dum * jnp.conj(apar_b)
            em_pflux = em_pflux + jnp.sum(d3v * jnp.imag(dum_a), axis=sum_axes)
            em_eflux = em_eflux + jnp.sum(
                d3v * (vpgr_b**2 * jnp.imag(dum_a) + 2 * mugr_b * bn_b * jnp.imag(dum_a)),
                axis=sum_axes,
            )
            em_vflux = em_vflux + jnp.sum(
                d3v * (jnp.imag(dum_a) * vpgr_b * rfun_b * bt_frac_b * signB),
                axis=sum_axes,
            )
        return em_pflux / fsa, em_eflux / fsa, em_vflux / fsa

    elif df.ndim == 6:
        # kinetic: per-species em fluxes
        nsp = df.shape[0]
        results = []
        for isp in range(nsp):
            sp_geom = dict(geometry)
            vthrat_sp = float(jnp.asarray(geometry["vthrat"])[isp])
            for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
                if k in geometry and jnp.asarray(geometry[k]).ndim > 0:
                    sp_geom[k] = jnp.asarray(geometry[k])[isp : isp + 1]
            gt = geom_tensors(sp_geom)

            parseval = gt["parseval"]
            vpgr_6 = gt["vpgr"]
            mugr_6 = gt["mugr"]
            bn_6 = gt["bn"]
            ints_6 = gt["ints"]
            efun_6 = gt["efun"]
            krho_6 = gt["krho"]
            intvp_6 = gt["intvp"]
            intmu_6 = gt["intmu"]
            d2X_6 = gt["d2X"]
            bessel_6 = gt["bessel"]

            d3v = d2X_6 * intmu_6 * bn_6 * intvp_6
            dum = parseval * ints_6 * (efun_6 * krho_6) * df[isp]

            rfun_6 = gt["rfun"]
            bt_frac_6 = gt["bt_frac"]
            signB_6 = gt["signB"]

            # gt tensors are 6D (1, vpar, mu, s, kx, ky); df[isp] broadcasts to 6D too.
            # reduce=True: sum every axis -> scalar. reduce=False: keep (kx, ky) -> sum (0,1,2,3).
            sum_axes_6 = (0, 1, 2, 3, 4, 5) if reduce else (0, 1, 2, 3)
            zero_sp = z if reduce else jnp.zeros((df.shape[-2], df.shape[-1]), dtype=jnp.float64)
            sp_pf, sp_ef, sp_vf = zero_sp, zero_sp, zero_sp
            if apar is not None:
                apar_b = apar.reshape(1, 1, *apar.shape)
                apar_ga = bessel_6 * apar_b
                dum_a = -2.0 * vthrat_sp * vpgr_6 * dum * jnp.conj(apar_ga)
                sp_pf = sp_pf + jnp.sum(d3v * jnp.imag(dum_a), axis=sum_axes_6)
                sp_ef = sp_ef + jnp.sum(
                    d3v * (vpgr_6**2 * jnp.imag(dum_a) + 2 * mugr_6 * bn_6 * jnp.imag(dum_a)),
                    axis=sum_axes_6,
                )
                sp_vf = sp_vf + jnp.sum(
                    d3v * (jnp.imag(dum_a) * vpgr_6 * rfun_6 * bt_frac_6 * signB_6),
                    axis=sum_axes_6,
                )
            if bpar is not None and pre is not None and "bpar_chi_factor" in pre:
                bpar_b = bpar.reshape(1, 1, *bpar.shape)
                bpar_chi = pre["bpar_chi_factor"][isp]
                dum_b = dum * bpar_chi * jnp.conj(bpar_b)
                sp_pf = sp_pf + jnp.sum(d3v * jnp.imag(dum_b), axis=sum_axes_6)
                sp_ef = sp_ef + jnp.sum(
                    d3v * (vpgr_6**2 * jnp.imag(dum_b) + 2 * mugr_6 * bn_6 * jnp.imag(dum_b)),
                    axis=sum_axes_6,
                )
                sp_vf = sp_vf + jnp.sum(
                    d3v * (jnp.imag(dum_b) * vpgr_6 * rfun_6 * bt_frac_6 * signB_6),
                    axis=sum_axes_6,
                )
            results.append(jnp.stack([sp_pf / fsa, sp_ef / fsa, sp_vf / fsa]))
        return jnp.stack(results)  # (nsp, 3) or (nsp, 3, nkx, nky)
    else:
        return z, z


def get_integrals(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: Any = None,
    pre: Dict = None,
    adiabatic_electrons: bool = True,
    geom: Dict[str, jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Compute phi and fluxes from distribution function.

    Returns:
        (phi, fluxes) where fluxes is (pflux, eflux, vflux) for adiabatic
        or (nsp, 3) array for kinetic electrons.
    """
    if not adiabatic_electrons and df.ndim == 6:
        phi = calculate_phi(geometry, df, params=params, pre=pre)
        fluxes = calculate_fluxes_kinetic(geometry, df, phi)
    else:
        gt = geom or (
            pre["geom_tensors"] if pre is not None else geom_tensors(geometry, params=params)
        )
        phi = _phi_adiabatic(gt, df)
        fluxes = calculate_fluxes(gt, df, phi)

    return phi, fluxes
