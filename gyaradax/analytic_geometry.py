"""analytic circular geometry computation (Lapillonne model).

computes all geometry arrays from input parameters alone,
eliminating the need for precomputed GKW geometry files.
formulas translated directly from gkw_ref/src/geom.f90.
"""

import os

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any


def _f64(x):
    return jnp.array(x, dtype=jnp.float64)


def _i32(x):
    return jnp.array(x, dtype=jnp.int32)


def _load_1d_array(path):
    """load a file and flatten to 1-d (GKW files may have nky rows)."""
    data = np.loadtxt(path)
    if data.ndim > 1:
        data = data[0]
    return data


# ---------------------------------------------------------------------------
# parallel grid and poloidal angle inversion
# ---------------------------------------------------------------------------

def _poloidal_angle(sgrid, eps, n_iter=10):
    """solve theta + eps*sin(theta) = 2*pi*s via fixed-point iteration."""
    theta = 2 * np.pi * sgrid.copy()
    for _ in range(n_iter):
        theta = 2 * np.pi * sgrid - eps * np.sin(theta)
    return theta


def _parallel_grid(ns, nperiod):
    """cell-centered parallel grid on [-sgrmax, sgrmax]."""
    sgrmax = nperiod - 0.5
    return np.array([-sgrmax + 2 * sgrmax * (i + 0.5) / ns for i in range(ns)])


def _parallel_weights(sgrid):
    if len(sgrid) < 2:
        return np.ones(1)
    return np.full(len(sgrid), sgrid[1] - sgrid[0])


# ---------------------------------------------------------------------------
# circular geometry: metric, B-field, derivatives (geom_circ in geom.f90)
# ---------------------------------------------------------------------------

def _dzetadeps(theta, q, shat, eps, signB, signJ):
    """metric coupling g_{psi,zeta}. geom.f90 lines 1492-1511."""
    ns = len(theta)
    dum2 = np.sqrt((1 - eps) / (1 + eps))

    # branch-tracked atan (sequential: element i depends on i-1)
    dzde = np.zeros(ns)
    dzde[0] = np.arctan(dum2 * np.tan(theta[0] / 2))
    for i in range(1, ns):
        dzde[i] = np.arctan(dum2 * np.tan(theta[i] / 2))
        while dzde[i] < dzde[i - 1]:
            dzde[i] += np.pi
    dzde -= np.pi * np.floor((dzde[0] - theta[0] / 2) / np.pi)

    # vectorised scaling (geom.f90 lines 1508-1510)
    t2 = np.tan(theta / 2)
    corr = eps / np.sqrt(1 - eps**2) * t2 / (1 + t2**2 + eps * (1 - t2**2))
    return signB * signJ / np.pi * q / eps * (shat * dzde - corr)


def _psi_theta_to_psi_s(f_psi, f_theta, theta, eps):
    """jacobian transform from (psi, theta) to (psi, s) coordinates."""
    R = 1 + eps * np.cos(theta)
    return f_psi - np.sin(theta) / R * f_theta, 2 * np.pi * f_theta / R


def _circular_geometry(theta, q, shat, eps, signB=1.0, signJ=1.0):
    """all geometry quantities for the Lapillonne circular model.

    returns intermediate arrays used by _calc_geom_tensors and compute_geometry.
    all derivatives are in (psi, s) coordinates after jacobian transform.
    """
    ns = len(theta)
    R = 1 + eps * np.cos(theta)

    dum = np.sqrt(1 + eps**2 / q**2 / (1 - eps**2))
    bn = dum / R
    bups = 1.0 / (2 * np.pi * q * np.sqrt(1 - eps**2))
    dpfdpsi = eps / (q * np.sqrt(1 - eps**2))

    dzde = _dzetadeps(theta, q, shat, eps, signB, signJ)

    # metric tensor (psi, zeta, s) -- geom.f90 lines 1558-1583
    metric = np.zeros((ns, 3, 3))
    metric[:, 0, 0] = 1.0
    metric[:, 0, 1] = metric[:, 1, 0] = dzde
    metric[:, 0, 2] = metric[:, 2, 0] = np.sin(theta) / (2 * np.pi)
    metric[:, 1, 1] = (
        (1 / (2 * np.pi * R)) ** 2 * (1 + (1 - eps**2) * (q / eps) ** 2)
        + dzde**2
    )
    metric[:, 1, 2] = metric[:, 2, 1] = (
        q * np.sqrt(1 - eps**2) / (2 * np.pi * eps) ** 2 * signB * signJ
        + dzde * np.sin(theta) / (2 * np.pi)
    )
    metric[:, 2, 2] = (
        (1 / (2 * np.pi)) ** 2
        * ((1 / eps + np.cos(theta)) ** 2 + np.sin(theta) ** 2)
    )

    # B-field derivatives: compute in (psi, theta) then transform
    dBdpsi_pt = bn * (
        -np.cos(theta) / R
        + eps * (1 - shat + eps**2 / (1 - eps**2))
        / (eps**2 + q**2 * (1 - eps**2))
    )
    dBds_pt = bn * eps * np.sin(theta) / R
    dBdpsi, dBds = _psi_theta_to_psi_s(dBdpsi_pt, dBds_pt, theta, eps)

    dRdpsi, dRds = _psi_theta_to_psi_s(
        np.cos(theta), -eps * np.sin(theta), theta, eps
    )
    dZdpsi, dZds = _psi_theta_to_psi_s(
        np.sin(theta), eps * np.cos(theta), theta, eps
    )

    return {
        "bn": bn, "dum": dum, "R": R,
        "ffun": bups / bn,
        "gfun": bups / bn * dBds / bn,
        "bt_frac": np.full(ns, 1 / dum),
        "bups": bups, "dpfdpsi": dpfdpsi,
        "metric": metric, "dzetadeps": dzde,
        "dBdpsi": dBdpsi, "dBds": dBds,
        "dRdpsi": dRdpsi, "dRds": dRds,
        "dZdpsi": dZdpsi, "dZds": dZds,
    }


# ---------------------------------------------------------------------------
# calc_geom_tensors (geom.f90 lines 3487-3634)
# ---------------------------------------------------------------------------

def _calc_geom_tensors(cg, signJ=1.0, signB=1.0):
    """E, D, H, I tensors from circular geometry quantities."""
    bn = cg["bn"]
    metric = cg["metric"]
    R = cg["R"]
    bups = cg["bups"]
    dBdpsi, dBds = cg["dBdpsi"], cg["dBds"]
    dRdpsi, dRds = cg["dRdpsi"], cg["dRds"]
    dZdpsi, dZds = cg["dZdpsi"], cg["dZds"]

    # E-tensor: antisymmetric cofactors of metric rows 0, 1
    # efun(j,k) = m(0,j)*m(1,k) - m(1,j)*m(0,k), then * signJ*pi*dpfdpsi/bn^2
    m0 = metric[:, 0, :]  # (ns, 3)
    m1 = metric[:, 1, :]
    efun = m0[:, :, None] * m1[:, None, :] - m1[:, :, None] * m0[:, None, :]
    efun *= signJ * np.pi * cg["dpfdpsi"] / bn[:, None, None] ** 2

    # columns 0 and 2 of E drive the drift tensors
    e0, e2 = efun[:, :, 0], efun[:, :, 2]  # (ns, 3) each

    # D-tensor: D_j = (-2*E_{j,0}*dBdpsi - 2*E_{j,2}*dBds) / bn
    dfun = (-2 * e0 * dBdpsi[:, None] - 2 * e2 * dBds[:, None]) / bn[:, None]

    # H-tensor (Coriolis) -- geom.f90 lines 3578-3594
    hfun = -signB * (
        metric[:, :, 0] * dZdpsi[:, None] + metric[:, :, 2] * dZds[:, None]
    )
    hfun[:, 2] += signB * bups**2 * dZds / bn**2
    hfun /= bn[:, None]

    # I-tensor (centrifugal) -- geom.f90 lines 3614-3621
    ifun = 2 * R[:, None] * (e0 * dRdpsi[:, None] + e2 * dRds[:, None])

    return efun, dfun, hfun, ifun


# ---------------------------------------------------------------------------
# velocity and wavenumber grids
# ---------------------------------------------------------------------------

def _build_velocity_grids(nvpar, nmu, vpar_max):
    dvp = 2 * vpar_max / nvpar
    vpgr = np.linspace(-vpar_max + dvp / 2, vpar_max - dvp / 2, nvpar)
    dvperp = vpar_max / nmu
    vperp = np.linspace(dvperp / 2, vpar_max - dvperp / 2, nmu)
    return vpgr, vperp**2 / 2, np.full(nvpar, dvp), 2 * np.pi * vperp * dvperp


def _build_wavevector_grids(nkx, nky, kxmax, krhomax):
    half = (nkx - 1) // 2
    dkx = kxmax / half if half > 0 else 0.0
    dky = krhomax / (nky - 1) if nky > 1 else krhomax
    return np.arange(-half, half + 1) * dkx, np.arange(nky) * dky


def _build_mode_label(nkx, nky, ikxspace):
    ml = np.zeros((nkx, nky), dtype=np.int32)
    label = 1
    for ix in range(nkx):
        ml[ix, 0] = label
        label += 1
    for iy in range(1, nky):
        for offset in range(ikxspace):
            lbl = label
            label += 1
            for ix in range(offset, nkx, ikxspace):
                ml[ix, iy] = lbl
    return ml


# ---------------------------------------------------------------------------
# public api
# ---------------------------------------------------------------------------

def compute_geometry(
    q: float,
    shat: float,
    eps: float,
    ns: int,
    nkx: int,
    nky: int,
    nvpar: int,
    nmu: int,
    vpar_max: float = 3.0,
    nperiod: int = 1,
    kxmax: float = 0.0,
    krhomax: float = 1.4,
    ikxspace: int = 5,
    signB: float = 1.0,
    Rref: float = 100.0,
) -> Dict[str, Any]:
    """compute full geometry dict from circular equilibrium parameters.

    returns a dict with the same keys and dtypes as load_geometry().
    """
    from gyaradax.geometry import (
        _build_mode_connectivity,
        _build_pos_par_grid_classes,
        _build_parallel_shift_maps,
    )

    signJ = 1.0
    sgrid = _parallel_grid(ns, nperiod)
    theta = _poloidal_angle(sgrid, eps)

    cg = _circular_geometry(theta, q, shat, eps, signB=signB, signJ=signJ)
    efun_3x3, dfun, hfun, ifun = _calc_geom_tensors(cg, signJ=signJ, signB=signB)

    bn, R = cg["bn"], cg["R"]
    little_g = np.stack([cg["metric"][:, 1, 1], cg["dzetadeps"], np.ones(ns)], axis=-1)

    g_zz_mid = (1 / (2 * np.pi * (1 + eps))) ** 2 * (
        1 + (1 - eps**2) * (q / eps) ** 2
    )
    kthnorm = np.sqrt(g_zz_mid)

    vpgr, mugr, intvp, intmu = _build_velocity_grids(nvpar, nmu, vpar_max)
    kxrh, krho_raw = _build_wavevector_grids(nkx, nky, kxmax, krhomax)
    krho = krho_raw / kthnorm

    ml = _build_mode_label(nkx, nky, ikxspace)
    ml_kxky, ixp, ixm, ixz, iyz = _build_mode_connectivity(ml, kxrh, krho)
    pos = _build_pos_par_grid_classes(ixp, ixm, ns)
    ss, ks, vs = _build_parallel_shift_maps(ixp, ixm, iyz, ns, max_shift=4)

    return {
        "kthnorm": _f64(kthnorm),
        "shat": _f64(shat),
        "q": _f64(q),
        "eps": _f64(eps),
        "kxrh": _f64(kxrh),
        "krho": _f64(krho),
        "parseval": _f64([1.0] + [float(nky)] * (nky - 1)),
        "intvp": _f64(intvp),
        "vpgr": _f64(vpgr),
        "vpgr_rms": _f64(np.sqrt(np.mean(vpgr**2))),
        "dvp": _f64(float(np.mean(np.diff(vpgr))) if len(vpgr) > 1 else 1.0),
        "intmu": _f64(intmu),
        "mugr": _f64(mugr),
        "mugr_rms": _f64(np.sqrt(np.mean(mugr**2))),
        "ints": _f64(_parallel_weights(sgrid)),
        "sgrid": _f64(sgrid),
        "sgr_dist": _f64(float(np.abs(sgrid[1] - sgrid[0])) if ns > 1 else 1.0),
        "bn": _f64(bn),
        "ffun": _f64(cg["ffun"]),
        "gfun": _f64(cg["gfun"]),
        "bt_frac": _f64(cg["bt_frac"]),
        "rfun": _f64(R),
        "little_g": _f64(little_g),
        "dfun": _f64(dfun),
        "hfun": _f64(hfun),
        "ifun": _f64(ifun),
        "efun": _f64(-efun_3x3[:, 0, 1]),
        "Rref": _f64(abs(Rref)),
        "signz": _f64([1.0]),
        "tmp": _f64([1.0]),
        "mas": _f64([1.0]),
        "de": _f64([1.0]),
        "vthrat": _f64([1.0]),
        "rlt": _f64([1.0]),
        "rln": _f64([1.0]),
        "d2X": _f64(1.0),
        "signB": _f64(signB),
        "mode_label": _i32(ml_kxky),
        "ixplus": _i32(ixp),
        "ixminus": _i32(ixm),
        "ixzero": _i32(ixz),
        "iyzero": _i32(iyz),
        "pos_par_grid_class": jnp.array(pos, dtype=jnp.int8),
        "s_shift": _i32(ss),
        "kx_shift": _i32(ks),
        "valid_shift": jnp.array(vs, dtype=jnp.bool_),
        "kxmax": _f64(float(np.max(np.abs(kxrh)))),
        "kymax": _f64(float(np.max(np.abs(krho)))),
    }


def compute_geometry_from_input(input_dat_path: str) -> Dict[str, Any]:
    """compute geometry from a GKW input.dat file.

    reads kxrh and vpgr.dat from the same directory if available,
    falls back to approximate formulas otherwise.
    """
    from gyaradax.geometry import parse_input_dat

    inp = parse_input_dat(input_dat_path)
    geom_sec = inp.get("geom", {})
    grid_sec = inp.get("gridsize", {})
    mode_sec = inp.get("mode", {})

    q = float(geom_sec.get("q", 1.0))
    shat = float(geom_sec.get("shat", 0.0))
    eps = float(geom_sec.get("eps", 0.0))
    nkx = int(grid_sec.get("nx", 1))
    nky = int(grid_sec.get("nmod", 1))
    krhomax = float(mode_sec.get("krhomax", 1.4))
    ikxspace = int(mode_sec.get("ikxspace", 5))

    data_dir = os.path.dirname(input_dat_path)

    kxrh_path = os.path.join(data_dir, "kxrh")
    if os.path.exists(kxrh_path):
        kxmax = float(_load_1d_array(kxrh_path)[-1])
    else:
        dky = krhomax / (nky - 1) if nky > 1 else krhomax
        kxmax = 2 * np.pi * abs(shat) * dky / ikxspace * (nkx - 1) / 2

    vpgr_path = os.path.join(data_dir, "vpgr.dat")
    if os.path.exists(vpgr_path):
        vpgr_data = _load_1d_array(vpgr_path)
        vpar_max = float(vpgr_data[-1] + np.mean(np.diff(vpgr_data)) / 2)
    else:
        vpar_max = 3.0

    return compute_geometry(
        q=q, shat=shat, eps=eps,
        ns=int(grid_sec.get("n_s_grid", 16)),
        nkx=nkx, nky=nky,
        nvpar=int(grid_sec.get("n_vpar_grid", 32)),
        nmu=int(grid_sec.get("n_mu_grid", 8)),
        vpar_max=vpar_max,
        nperiod=int(grid_sec.get("nperiod", 1)),
        kxmax=kxmax, krhomax=krhomax, ikxspace=ikxspace,
    )
