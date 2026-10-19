"""Quasineutrality: Bessel helpers and the adiabatic / kinetic phi solve."""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import i0e, bessel_jn
from einops import rearrange

from gyaradax.core.constants import EPS
from gyaradax.core.state import Precompute


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
    """Expand geometry constants for broadcasting and compute Bessel terms.

    Single-species. Species params are scalars reshaped to (1,1,1,1,1,1).
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
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, EPS))

    mugr_bn = jnp.maximum(2.0 * geom_["mugr"] / geom_["bn"], EPS)

    sz = jnp.where(jnp.abs(geom_["signz"]) < EPS, 1.0, geom_["signz"])
    bessel_arg = jnp.sqrt(mugr_bn) / sz
    bessel_arg = geom_["mas"] * vthrat * krloc * bessel_arg
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    geom_["bessel"] = j0(bessel_arg)

    gamma_arg = geom_["mas"] * vthrat * krloc
    gamma_arg = 0.5 * (gamma_arg / (sz * geom_["bn"])) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    geom_["gamma"] = i0e(gamma_arg)

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
    poisson_int = jnp.where(jnp.abs(intvp) < EPS, 0.0, poisson_int)

    cfen = 0.0
    diagz = signz * (gamma - 1.0) * jnp.exp(-cfen) / tmp
    denom = diagz - jnp.exp(-cfen) / tmp
    denom = jnp.where(jnp.abs(denom) < EPS, 1.0, denom)
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
    maty_val = jnp.where(jnp.abs(maty_val) < EPS, 1.0, maty_val)
    maty_val = 1.0 / maty_val
    phi = phi + maty_val * bufphi * y_mask

    poisson_diag = jnp.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
    norm_mask = jnp.ones_like(phi)
    norm_mask = norm_mask.at[..., ixzero, iyzero].set(1.0 - has_zonal)

    pdiag = poisson_diag * norm_mask - signz * jnp.exp(-cfen) * de / tmp
    pdiag = jnp.where(jnp.abs(pdiag) < EPS, -1.0, pdiag)

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

    # zonal mode correction (sum over ns); zero matz for non-zonal ky
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
    """Hot path for adiabatic phi solve."""
    phi_raw = jnp.sum(phi_weight * df, axis=(0, 1), keepdims=True)
    bufphi = jnp.sum(phi_corr_weight * phi_raw, axis=2, keepdims=True)

    cfen = 0.0
    exp_cfen = jnp.exp(-cfen)

    # maty_val correction (ky=0 only)
    maty_sum = jnp.sum(-phi_corr_weight * exp_cfen, axis=2, keepdims=True)
    maty = tmp / (de * exp_cfen) + maty_sum / exp_cfen

    nkx, nky = phi_raw.shape[3], phi_raw.shape[4]
    y_mask = jnp.zeros((1, 1, 1, 1, nky), dtype=phi_raw.dtype).at[..., iyzero].set(has_zonal)
    x_mask = jnp.zeros((1, 1, 1, nkx, 1), dtype=phi_raw.dtype).at[..., ixzero, :].set(1.0)

    maty_val = jnp.where(x_mask > 0, 1.0 + 0j, maty)
    maty_val = jnp.where(jnp.abs(maty_val) < 1e-15, 1.0, maty_val)
    phi_corr = phi_raw + (1.0 / maty_val) * bufphi * y_mask

    poisson_diag = exp_cfen * (signz**2) * de * (gamma - 1.0) / tmp
    zonal_mask = x_mask * y_mask
    norm_mask = jnp.ones_like(zonal_mask) - zonal_mask

    pdiag = poisson_diag * norm_mask - signz * exp_cfen * de / tmp
    pdiag = jnp.where(jnp.abs(pdiag) < 1e-15, -1.0, pdiag)

    phi = phi_corr * (-1.0 / pdiag)
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
    sz = jnp.where(jnp.abs(signz_6d) < EPS, 1.0, signz_6d)

    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64).reshape(1, 1, 1, 1, 1, -1)
    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64).reshape(1, 1, 1, 1, -1, 1)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, 1, -1, 1, 1)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)

    g0 = little_g[:, 0].reshape(1, 1, 1, -1, 1, 1)
    g1 = little_g[:, 1].reshape(1, 1, 1, -1, 1, 1)
    g2 = little_g[:, 2].reshape(1, 1, 1, -1, 1, 1)
    krloc_sq = krho**2 * g0 + 2 * krho * kxrh * g1 + kxrh**2 * g2
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, EPS))

    mugr_bn = jnp.maximum(2.0 * mugr / jnp.maximum(bn, EPS), EPS)
    bessel_arg = mas_6d * vthrat_6d * krloc * jnp.sqrt(mugr_bn) / sz
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    bessel = j0(bessel_arg)

    gamma_arg = 0.5 * (mas_6d * vthrat_6d * krloc / (sz * bn)) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    gamma_arg_nommu = gamma_arg[:, :, 0:1, :, :, :]
    gamma = i0e(gamma_arg_nommu)

    return bessel, gamma


def precompute_phi_kinetic(geometry: Dict[str, jnp.ndarray]):
    """Precompute static arrays for the kinetic phi solve.

    Returns (phi_weight, phi_diag) where:
        phi_weight: (nsp, nvpar, nmu, ns, nkx, nky) — Poisson integral weight
        phi_diag: (ns, nkx, nky) — Poisson diagonal (zonal mode set to 1)
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

    weight = signz_6d * de_6d * intmu * intvp * bessel * bn
    weight = jnp.where(jnp.abs(intvp) < EPS, 0.0, weight)

    # Poisson diagonal: sum_sp Z^2*n*(Gamma_0 - 1)/T
    diag_per_sp = signz_6d**2 * de_6d * (gamma - 1.0) / tmp_6d
    diag = jnp.sum(diag_per_sp, axis=0)
    diag = diag.reshape(diag.shape[-3], diag.shape[-2], diag.shape[-1])

    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    ixzero = jnp.argmin(jnp.abs(kxrh))
    iyzero = jnp.argmin(jnp.abs(krho))
    has_zonal = jnp.abs(krho[iyzero]) < 1e-10
    diag_with_zonal = diag.at[:, ixzero, iyzero].set(1.0)
    diag = jnp.where(has_zonal, diag_with_zonal, diag)
    diag = jnp.where(jnp.abs(diag) < EPS, -1.0, diag)
    return weight, diag


def _velocity_moment(weight, df, keep_species):
    if weight.shape == df.shape:
        return jnp.einsum(f"avmjkl,avmjkl->{'ajkl' if keep_species else 'jkl'}", weight, df)
    return jnp.sum(weight * df, axis=(-5, -4) if keep_species else (-6, -5, -4))


def charge_density(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    phi_weight: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Per-species gyrocenter charge density c_s = Z_s n_s int J0 f_s d^3v.

    Its species sum is the numerator of the kinetic quasineutrality solve::

        sum_s Z_s n_s int J0 f_s d^3v = -sum_s Z_s^2 n_s (Gamma0_s - 1) / T_s * phi

    with the right-hand diagonal set to 1 at (kx, ky) = (0, 0).

    Args:
        geometry: geometry dict with per-species mas, signz, tmp, de, vthrat.
        df: (nsp, nvpar, nmu, ns, nkx, nky) distribution.
        phi_weight: Poisson weight Z_s n_s J0 B intvp intmu, dense shape
            (nsp, nvpar, nmu, ns, nkx, nky) or any shape broadcastable to it,
            e.g. (nsp, 1, nmu, ns, nkx, nky) on the uniform vpar grid. Defaults
            to ``precompute_phi_kinetic(geometry)[0]``; ``pre["phi_weight"]`` is
            the coupled phi-B_par weight when nlbpar is on.

    Returns:
        (nsp, ns, nkx, nky) charge density.
    """
    if phi_weight is None:
        phi_weight, _ = precompute_phi_kinetic(geometry)
    return _velocity_moment(phi_weight, df, keep_species=True)


def _phi_kinetic(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    phi_weight: Optional[jnp.ndarray] = None,
    phi_diag: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Kinetic electron phi from multi-species quasineutrality.

    phi = -sum_s charge_density_s / phi_diag, with the species sum taken inside
    the same contraction.

    df: (nsp, nvpar, nmu, ns, nkx, nky).
    Internal — use calculate_phi() as the public interface.
    """
    if phi_weight is None or phi_diag is None:
        phi_weight, phi_diag = precompute_phi_kinetic(geometry)

    return -_velocity_moment(phi_weight, df, keep_species=False) / phi_diag


def calculate_phi(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    params: Any = None,
    pre: Optional[Precompute] = None,
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
    # legacy entry: calculate_phi(geom_tensors_dict, df) detected via "bessel" key
    if "bessel" in geometry:
        return _phi_adiabatic(geometry, df)

    adiabatic = params.adiabatic_electrons if params is not None else (df.ndim == 5)
    if adiabatic:
        gt = pre["geom_tensors"] if pre is not None else geom_tensors(geometry, params=params)
        return _phi_adiabatic(gt, df)
    pw = pre.get("phi_weight") if pre is not None else None
    pd = pre.get("phi_diag") if pre is not None else None
    return _phi_kinetic(geometry, df, pw, pd)


calculate_phi_kinetic = _phi_kinetic
