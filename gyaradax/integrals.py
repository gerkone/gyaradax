import jax
import jax.numpy as jnp
from jax.scipy.special import i0, bessel_jn
from einops import rearrange
from typing import Dict, Tuple, Any


def j0(x):
    safe_x = jnp.where(jnp.abs(x) < 1e-10, 1.0, x)
    res = bessel_jn(safe_x, v=0)[0]
    return jnp.where(jnp.abs(x) < 1e-10, 1.0, res)


def geom_tensors(geometry: Dict[str, jnp.ndarray], params: Any = None) -> Dict[str, jnp.ndarray]:
    """Expand geometry constants for broadcasting and compute Bessel terms.

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
    krloc_sq = jnp.where(krloc_sq < 0, 0.0, krloc_sq)
    krloc = jnp.sqrt(krloc_sq)

    mugr_bn = 2.0 * geom_["mugr"] / geom_["bn"]
    mugr_bn = jnp.where(mugr_bn < 0, 0.0, mugr_bn)

    sz = jnp.where(jnp.abs(geom_["signz"]) < 1e-15, 1.0, geom_["signz"])
    bessel_arg = jnp.sqrt(mugr_bn) / sz
    bessel_arg = geom_["mas"] * vthrat * krloc * bessel_arg
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    geom_["bessel"] = j0(bessel_arg)

    gamma_arg = geom_["mas"] * vthrat * krloc
    gamma_arg = 0.5 * (gamma_arg / (sz * geom_["bn"])) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    geom_["gamma"] = i0(gamma_arg) * jnp.exp(-gamma_arg)

    # zonal mode detection: ky-index 0 is only the zonal mode if krho[0] ≈ 0
    krho_flat = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    iyzero = jnp.argmin(jnp.abs(krho_flat))
    geom_["has_zonal"] = jnp.where(jnp.abs(krho_flat[iyzero]) < 1e-10, 1.0, 0.0)

    return geom_


@jax.jit
def calculate_phi(geom: Dict[str, jnp.ndarray], df: jnp.ndarray) -> jnp.ndarray:
    """Adiabatic electron phi from single-species quasineutrality.

    df: (nvpar, nmu, ns, nkx, nky).
    """
    de = geom["de"]
    signz, tmp, bn = geom["signz"], geom["tmp"], geom["bn"]
    ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
    bessel, gamma = geom["bessel"], geom["gamma"]

    poisson_int = signz * de * intmu * intvp * bessel * bn
    poisson_int = jnp.where(jnp.abs(intvp) < 1e-9, 0.0, poisson_int)

    cfen = 0.0
    diagz = signz * (gamma - 1.0) * jnp.exp(-cfen) / tmp
    denom = diagz - jnp.exp(-cfen) / tmp
    denom = jnp.where(jnp.abs(denom) < 1e-15, 1.0, denom)
    matz = -ints / (signz * de * denom)
    # flux-surface averaging only applies to the zonal mode (ky=0)
    has_zonal = geom["has_zonal"]
    matz = matz.at[..., 1:].set(0.0)
    matz = matz * has_zonal

    phi = poisson_int * df
    phi = jnp.sum(phi, axis=(1, 2), keepdims=True)

    y_mask = jnp.zeros_like(phi)
    y_mask = y_mask.at[..., 0].set(has_zonal)

    bufphi = matz * phi
    bufphi = jnp.sum(bufphi, axis=3, keepdims=True)

    maty_sum = jnp.sum(-matz * jnp.exp(-cfen), axis=3, keepdims=True)
    maty = tmp / (de * jnp.exp(-cfen)) + maty_sum / jnp.exp(-cfen)

    x_mask = jnp.zeros_like(phi)
    x_mask = x_mask.at[..., 0, :].set(1.0)
    maty_val = jnp.where(x_mask > 0, 1.0 + 0j, maty)
    maty_val = jnp.where(jnp.abs(maty_val) < 1e-15, 1.0, maty_val)
    maty_val = 1.0 / maty_val
    phi = phi + maty_val * bufphi * y_mask

    poisson_diag = jnp.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
    # only zero phi at (kx=0, ky=0) when a real zonal mode exists
    norm_mask = jnp.ones_like(phi)
    norm_mask = norm_mask.at[..., 0, 0].set(1.0 - has_zonal)

    pdiag = poisson_diag * norm_mask - signz * jnp.exp(-cfen) * de / tmp
    pdiag = jnp.where(jnp.abs(pdiag) < 1e-15, -1.0, pdiag)

    phi = phi * (-1.0 / pdiag)
    # squeeze species + summed velocity axes, keep (ns, nkx, nky)
    return jnp.squeeze(phi, axis=(0, 1, 2))


def _species_bessel_gamma(geometry):
    """Per-species Bessel J0 and Gamma_0 for multi-species phi solve."""
    mas = jnp.asarray(geometry["mas"], dtype=jnp.float64)
    signz = jnp.asarray(geometry["signz"], dtype=jnp.float64)
    vthrat = jnp.asarray(geometry["vthrat"], dtype=jnp.float64)
    nsp = mas.shape[0]

    mas_6d = mas.reshape(nsp, 1, 1, 1, 1, 1)
    signz_6d = signz.reshape(nsp, 1, 1, 1, 1, 1)
    vthrat_6d = vthrat.reshape(nsp, 1, 1, 1, 1, 1)
    sz = jnp.where(jnp.abs(signz_6d) < 1e-15, 1.0, signz_6d)

    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64).reshape(1, 1, 1, 1, 1, -1)
    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64).reshape(1, 1, 1, 1, -1, 1)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, 1, -1, 1, 1)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1, 1)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)

    g0 = little_g[:, 0].reshape(1, 1, 1, -1, 1, 1)
    g1 = little_g[:, 1].reshape(1, 1, 1, -1, 1, 1)
    g2 = little_g[:, 2].reshape(1, 1, 1, -1, 1, 1)
    krloc_sq = krho**2 * g0 + 2 * krho * kxrh * g1 + kxrh**2 * g2
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, 0.0))

    mugr_bn = jnp.maximum(2.0 * mugr / jnp.maximum(bn, 1e-15), 0.0)
    bessel_arg = mas_6d * vthrat_6d * krloc * jnp.sqrt(mugr_bn) / sz
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    bessel = j0(bessel_arg)

    gamma_arg = 0.5 * (mas_6d * vthrat_6d * krloc / (sz * bn)) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    gamma_arg_nommu = gamma_arg[:, :, 0:1, :, :, :]
    gamma = i0(gamma_arg_nommu) * jnp.exp(-gamma_arg_nommu)

    return bessel, gamma


def precompute_phi_kinetic(geometry: Dict[str, jnp.ndarray]):
    """precompute static arrays for the kinetic phi solve.

    returns (phi_weight, phi_diag) where:
        phi_weight: (nsp, 1, nmu, ns, nkx, nky) — poisson integral weight
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
    weight = jnp.where(jnp.abs(intvp) < 1e-9, 0.0, weight)

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
    diag = jnp.where(jnp.abs(diag) < 1e-15, -1.0, diag)

    return weight, diag


def calculate_phi_kinetic(
    geometry: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    phi_weight: jnp.ndarray = None,
    phi_diag: jnp.ndarray = None,
) -> jnp.ndarray:
    """Kinetic electron phi from multi-species quasineutrality.

    df: (nsp, nvpar, nmu, ns, nkx, nky).
    If phi_weight and phi_diag are provided (from precompute_phi_kinetic),
    skips expensive bessel/gamma recomputation.
    """
    if phi_weight is None or phi_diag is None:
        phi_weight, phi_diag = precompute_phi_kinetic(geometry)

    phi_num = jnp.sum(phi_weight * df, axis=(0, 1, 2))
    return -phi_num / phi_diag


@jax.jit
def calculate_fluxes(
    geom: Dict[str, jnp.ndarray], df: jnp.ndarray, phi: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-species fluxes. df: (nvpar, nmu, ns, nkx, nky)."""
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
    d3v = ints * d2X * intmu * bn * intvp

    pflux = d3v * jnp.imag(dum1)
    eflux = d3v * (vpgr**2 * jnp.imag(dum1) + 2 * mugr * jnp.imag(dum2))
    vflux = d3v * (jnp.imag(dum1) * vpgr * rfun * bt_frac * signB)

    return jnp.sum(pflux), jnp.sum(eflux), jnp.sum(vflux)


def calculate_fluxes_kinetic(
    geometry: Dict[str, jnp.ndarray], df: jnp.ndarray, phi: jnp.ndarray
) -> jnp.ndarray:
    """Per-species fluxes for kinetic case.

    df: (nsp, nvpar, nmu, ns, nkx, nky).
    Returns: (nsp, 3) array of [pflux, eflux, vflux] per species.
    """
    nsp = df.shape[0]

    def _flux_single(isp):
        sp_geom = dict(geometry)
        for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
            if k in geometry and jnp.asarray(geometry[k]).ndim > 0:
                sp_geom[k] = jnp.asarray(geometry[k])[isp : isp + 1]
        gt = geom_tensors(sp_geom)
        pflux, eflux, vflux = calculate_fluxes(gt, df[isp], phi)
        return jnp.stack([pflux, eflux, vflux])

    return jnp.stack([_flux_single(i) for i in range(nsp)])


def get_integrals(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: Any = None,
    geom: Dict[str, jnp.ndarray] = None,
    adiabatic_electrons: bool = True,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Compute phi and fluxes from distribution function.

    For phi-only calls, use calculate_phi / calculate_phi_kinetic directly.

    Returns:
        (phi, fluxes) where fluxes is (pflux, eflux, vflux) for adiabatic
        or (nsp, 3) array for kinetic electrons.
    """
    if not adiabatic_electrons and df.ndim == 6:
        phi = calculate_phi_kinetic(geometry, df)
        fluxes = calculate_fluxes_kinetic(geometry, df, phi)  # (nsp, 3)
    else:
        if geom is None:
            geom = geom_tensors(geometry, params=params)
        phi = calculate_phi(geom, df)
        fluxes = calculate_fluxes(geom, df, phi)

    return phi, fluxes
