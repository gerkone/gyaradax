"""Flux-surface-averaged transport fluxes (electrostatic and magnetic flutter)."""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from gyaradax.core.state import Precompute
from gyaradax.operators.fields import _compute_fields, g_to_f
from gyaradax.operators.poisson import _phi_adiabatic, calculate_phi, geom_tensors


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
    """Per-species fluxes for the kinetic case.

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
    apar: Optional[jnp.ndarray],
    params: Any = None,
    bpar: Optional[jnp.ndarray] = None,
    pre: Optional[Precompute] = None,
    reduce: bool = True,
) -> Any:
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
        if bpar is not None:
            raise NotImplementedError(
                "calculate_em_fluxes: the 5D (adiabatic) branch implements the "
                "A_par flutter flux only; the B_par compressional flux is "
                "implemented in the 6D kinetic branch. Passing bpar here would "
                "silently drop it (REVIEW.md item 4)."
            )
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

        if pre is not None and "bessel" in pre:
            bessel = jnp.asarray(pre["bessel"], dtype=jnp.float64)
        else:
            bessel = jnp.asarray(geom_tensors(geometry, params=params)["bessel"], dtype=jnp.float64)

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
            # J0 gyro-average A_parallel (GKW get_averaged_apar), matching the
            # kinetic 6D branch and the generalized-potential chi convention.
            apar_ga = bessel * apar_b
            # matches GKW diagnos_fluxes_vspace.F90:464 (-2·vthrat·vpar in χ)
            # em-fix physics (J0-gyroaveraged apar) + per-mode reduce plumbing
            dum_a = -2.0 * vthrat_val * vpgr_b * dum * jnp.conj(apar_ga)
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
        nsp = df.shape[0]
        results = []
        for isp in range(nsp):
            sp_geom = dict(geometry)
            for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
                if params is not None and hasattr(params, k):
                    vals = jnp.asarray(getattr(params, k), dtype=jnp.float64).reshape(-1)
                elif k in geometry:
                    vals = jnp.asarray(geometry[k], dtype=jnp.float64).reshape(-1)
                else:
                    continue
                idx = isp if vals.shape[0] == nsp else 0
                sp_geom[k] = vals[idx : idx + 1]
            vthrat_sp = float(jnp.asarray(sp_geom["vthrat"])[0])
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
    pre: Optional[Precompute] = None,
    adiabatic_electrons: bool = True,
    geom: Optional[Dict[str, jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray, Any]:
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


def _compute_em_fluxes_arr(df, geometry, params, pre, fluxes_shape):
    """Compute magnetic-flutter EM fluxes as a numpy array shaped like the ES
    fluxes array. Returns zeros when nlapar is off, params/pre missing, or
    apar/bpar are unavailable.
    """
    zero = np.zeros(fluxes_shape, dtype=np.float64)
    if params is None or pre is None or not getattr(params, "nlapar", False):
        return zero
    _, apar, bpar = _compute_fields(df, geometry, params, pre)
    if apar is None:
        return zero
    df_f = g_to_f(df, apar, params, pre)
    res = calculate_em_fluxes(geometry, df_f, apar, params=params, bpar=bpar, pre=pre)
    if len(fluxes_shape) == 1:  # (3,)
        em_p, em_e, em_v = (np.asarray(r, dtype=np.float64) for r in res)
        out = np.zeros(fluxes_shape, dtype=np.float64)
        out[0] = float(em_p)
        out[1] = float(em_e)
        out[2] = float(em_v)
    else:  # 6D path returns (nsp, 3)
        arr = np.asarray(res, dtype=np.float64)
        out = np.zeros(fluxes_shape, dtype=np.float64)
        out[..., :3] = arr
    return out
