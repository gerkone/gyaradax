"""Analytic circular geometry computation (Lapillonne model).

Computes all geometry arrays from equilibrium parameters alone, eliminating
the need for precomputed GKW geometry files. Formulas are translated directly
from ``gkw_ref/src/geom.f90`` (``geom_circ`` lines 1444-1616 and
``calc_geom_tensors`` lines 3487-3634).

The public entry points are :func:`compute_geometry` (from scalar parameters)
and :func:`compute_geometry_from_input` (from a GKW ``input.dat`` file).
Both return a dict identical to the one produced by
:func:`gyaradax.geometry.load_geometry`.
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
    """Load a GKW grid file and flatten to 1-d (files may have nky rows)."""
    data = np.loadtxt(path)
    if data.ndim > 1:
        data = data[0]
    return data


def _poloidal_angle(sgrid, eps, geom_type="circ", n_iter=10):
    """Map field-line coordinate s to poloidal angle theta.

    For circ: invert theta + eps*sin(theta) = 2*pi*s (fixed-point, 10 iters).
    For s-alpha: simple linear mapping theta = 2*pi*s.
    """
    if geom_type == "s-alpha":
        return 2 * np.pi * sgrid.copy()
    theta = 2 * np.pi * sgrid.copy()
    for _ in range(n_iter):
        theta = 2 * np.pi * sgrid - eps * np.sin(theta)
    return theta


def _parallel_grid(ns, nperiod):
    """Cell-centered uniform parallel grid on [-sgrmax, sgrmax]."""
    sgrmax = nperiod - 0.5
    return np.array([-sgrmax + 2 * sgrmax * (i + 0.5) / ns for i in range(ns)])


def _parallel_weights(sgrid):
    """Uniform integration weights (cell width)."""
    if len(sgrid) < 2:
        return np.ones(1)
    return np.full(len(sgrid), sgrid[1] - sgrid[0])


def _dzetadeps(theta, q, shat, eps, signB, signJ):
    """Metric coupling g_{psi,zeta} = d(zeta)/d(eps) at fixed s.

    Uses branch-tracked atan with monotonicity correction, then scales
    by the shear and finite-epsilon correction terms.
    Translated from geom.f90 lines 1492-1511.

    Note: the finite-eps correction introduces ~0.1% model-level error
    that propagates into all zeta-direction tensors (D_zeta, H_zeta,
    I_zeta). The radial (eps) components are unaffected.
    """
    ns = len(theta)
    dum2 = np.sqrt((1 - eps) / (1 + eps))

    dzde = np.zeros(ns)
    dzde[0] = np.arctan(dum2 * np.tan(theta[0] / 2))
    for i in range(1, ns):
        dzde[i] = np.arctan(dum2 * np.tan(theta[i] / 2))
        while dzde[i] < dzde[i - 1]:
            dzde[i] += np.pi
    dzde -= np.pi * np.floor((dzde[0] - theta[0] / 2) / np.pi)

    t2 = np.tan(theta / 2)
    corr = eps / np.sqrt(1 - eps**2) * t2 / (1 + t2**2 + eps * (1 - t2**2))
    return signB * signJ / np.pi * q / eps * (shat * dzde - corr)


def _psi_theta_to_psi_s(f_psi, f_theta, theta, eps):
    """Jacobian transform from (psi, theta) to (psi, s) coordinates.

    Returns (f_psi_s, f_s) where:
        f_psi_s = f_psi - sin(theta)/R * f_theta
        f_s     = 2*pi/R * f_theta
    with R = 1 + eps*cos(theta).
    """
    R = 1 + eps * np.cos(theta)
    return f_psi - np.sin(theta) / R * f_theta, 2 * np.pi * f_theta / R


def _circular_geometry(theta, q, shat, eps, signB=1.0, signJ=1.0, geom_type="circ"):
    """Compute all geometry quantities for circular/s-alpha models.

    For circ: full Lapillonne model with delta correction and nonlinear theta.
    For s-alpha: simplified B = 1/(1+eps*cos(theta)), delta=1.

    Translated from ``geom_circ`` / ``geom_s_alpha`` in geom.f90.
    """
    ns = len(theta)
    R = 1 + eps * np.cos(theta)

    if geom_type == "s-alpha":
        dum = 1.0
    else:
        dum = np.sqrt(1 + eps**2 / q**2 / (1 - eps**2))
    bn = dum / R
    bups = 1.0 / (2 * np.pi * q * np.sqrt(1 - eps**2))
    dpfdpsi = eps / (q * np.sqrt(1 - eps**2))

    dzde = _dzetadeps(theta, q, shat, eps, signB, signJ)

    metric = np.zeros((ns, 3, 3))
    metric[:, 0, 0] = 1.0
    metric[:, 0, 1] = metric[:, 1, 0] = dzde
    metric[:, 0, 2] = metric[:, 2, 0] = np.sin(theta) / (2 * np.pi)
    metric[:, 1, 1] = (1 / (2 * np.pi * R)) ** 2 * (1 + (1 - eps**2) * (q / eps) ** 2) + dzde**2
    metric[:, 1, 2] = metric[:, 2, 1] = q * np.sqrt(1 - eps**2) / (
        2 * np.pi * eps
    ) ** 2 * signB * signJ + dzde * np.sin(theta) / (2 * np.pi)
    metric[:, 2, 2] = (1 / (2 * np.pi)) ** 2 * ((1 / eps + np.cos(theta)) ** 2 + np.sin(theta) ** 2)

    dBdpsi_pt = bn * (
        -np.cos(theta) / R
        + eps * (1 - shat + eps**2 / (1 - eps**2)) / (eps**2 + q**2 * (1 - eps**2))
    )
    dBds_pt = bn * eps * np.sin(theta) / R
    dBdpsi, dBds = _psi_theta_to_psi_s(dBdpsi_pt, dBds_pt, theta, eps)

    dRdpsi, dRds = _psi_theta_to_psi_s(np.cos(theta), -eps * np.sin(theta), theta, eps)
    dZdpsi, dZds = _psi_theta_to_psi_s(np.sin(theta), eps * np.cos(theta), theta, eps)

    return {
        "bn": bn,
        "dum": dum,
        "R": R,
        "ffun": bups / bn,
        "gfun": bups / bn * dBds / bn,
        "bt_frac": np.full(ns, 1 / dum),
        "bups": bups,
        "dpfdpsi": dpfdpsi,
        "metric": metric,
        "dzetadeps": dzde,
        "dBdpsi": dBdpsi,
        "dBds": dBds,
        "dRdpsi": dRdpsi,
        "dRds": dRds,
        "dZdpsi": dZdpsi,
        "dZds": dZds,
    }


def _calc_geom_tensors(cg, signJ=1.0, signB=1.0):
    """Compute E, D, H, I drift tensors from the circular geometry.

    Translated from ``calc_geom_tensors`` in geom.f90 lines 3487-3634.

    E-tensor (ExB): antisymmetric cofactors of metric rows 0/1,
        scaled by pi * dpfdpsi / bn^2.
    D-tensor (curvature + grad-B drift):
        D_j = (-2*E_{j,psi}*dB/dpsi - 2*E_{j,s}*dB/ds) / B
    H-tensor (Coriolis drift): from dZ/dpsi, dZ/ds with metric coupling
        and a finite-eps correction on the s-component.
    I-tensor (centrifugal drift): 2*R*(E . grad(R)).
    """
    bn = cg["bn"]
    metric = cg["metric"]
    R = cg["R"]
    bups = cg["bups"]
    dBdpsi, dBds = cg["dBdpsi"], cg["dBds"]
    dRdpsi, dRds = cg["dRdpsi"], cg["dRds"]
    dZdpsi, dZds = cg["dZdpsi"], cg["dZds"]

    m0 = metric[:, 0, :]
    m1 = metric[:, 1, :]
    efun = m0[:, :, None] * m1[:, None, :] - m1[:, :, None] * m0[:, None, :]
    efun *= signJ * np.pi * cg["dpfdpsi"] / bn[:, None, None] ** 2

    e0, e2 = efun[:, :, 0], efun[:, :, 2]

    dfun = (-2 * e0 * dBdpsi[:, None] - 2 * e2 * dBds[:, None]) / bn[:, None]

    hfun = -signB * (metric[:, :, 0] * dZdpsi[:, None] + metric[:, :, 2] * dZds[:, None])
    hfun[:, 2] += signB * bups**2 * dZds / bn**2
    hfun /= bn[:, None]

    ifun = 2 * R[:, None] * (e0 * dRdpsi[:, None] + e2 * dRds[:, None])

    return efun, dfun, hfun, ifun


def _build_velocity_grids(nvpar, nmu, vpar_max):
    """Uniform v_par grid and uniform-in-v_perp mu grid (GKW convention)."""
    dvp = 2 * vpar_max / nvpar
    vpgr = np.linspace(-vpar_max + dvp / 2, vpar_max - dvp / 2, nvpar)
    dvperp = vpar_max / nmu
    vperp = np.linspace(dvperp / 2, vpar_max - dvperp / 2, nmu)
    return vpgr, vperp**2 / 2, np.full(nvpar, dvp), 2 * np.pi * vperp * dvperp


def _build_wavevector_grids(nkx, nky, kxmax, krhomax):
    """Centered kx grid and uniform ky grid.

    for nky=1 the single mode is placed at krhomax (not zero),
    matching GKW single-mode eigenvalue convention.
    """
    half = (nkx - 1) // 2
    dkx = kxmax / half if half > 0 else 0.0
    if nky == 1:
        return np.arange(-half, half + 1) * dkx, np.array([krhomax])
    dky = krhomax / (nky - 1)
    return np.arange(-half, half + 1) * dkx, np.arange(nky) * dky


def _build_mode_label(nkx, nky, ikxspace):
    """Mode-label array for open parallel boundary connectivity.

    For ky=0 each kx is its own mode (periodic). For ky>0 modes are
    grouped into chains spaced ikxspace apart in kx-index.
    """
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


def _build_mode_connectivity(mode_label, kxrh, krho):
    """
    Build spectral parallel-boundary connectivity from mode labels.

    Returns:
      mode_label_kxky: int32[nkx, nky]
      ixplus: int32[nkx, nky], -1 means open boundary (no connection)
      ixminus: int32[nkx, nky], -1 means open boundary (no connection)
      ixzero: int32 scalar, index of kx=0 mode
      iyzero: int32 scalar, index of ky=0 mode
    """
    mode_label = np.asarray(mode_label, dtype=np.int32)
    nkx = int(kxrh.shape[0])
    nky = int(krho.shape[0])

    if mode_label.shape == (nkx, nky):
        mode_label_kxky = mode_label
    elif mode_label.shape == (nky, nkx):
        mode_label_kxky = mode_label.T
    else:
        raise ValueError(
            f"mode_label shape {mode_label.shape} incompatible with nkx/nky=({nkx},{nky})"
        )

    ixzero = int(np.argmin(np.abs(kxrh)))
    iyzero = int(np.argmin(np.abs(krho)))

    ixplus = -np.ones((nkx, nky), dtype=np.int32)
    ixminus = -np.ones((nkx, nky), dtype=np.int32)

    for iy in range(nky):
        # ky=0 mode is always periodic in spectral mode_box runs.
        if iy == iyzero:
            ix = np.arange(nkx, dtype=np.int32)
            ixplus[:, iy] = ix
            ixminus[:, iy] = ix
            continue

        labels = mode_label_kxky[:, iy]
        for lbl in np.unique(labels):
            chain = np.where(labels == lbl)[0].astype(np.int32)
            if chain.size <= 1:
                continue
            chain = np.sort(chain)
            ixplus[chain[:-1], iy] = chain[1:]
            ixminus[chain[1:], iy] = chain[:-1]

    return mode_label_kxky, ixplus, ixminus, ixzero, iyzero


def _build_pos_par_grid_classes(ixplus, ixminus, ns):
    """
    Build pos_par_grid class values (-2,-1,0,1,2) for open parallel boundaries.
    Shape: [ns, nkx, nky]
    """
    pos = np.zeros((ns,) + ixplus.shape, dtype=np.int8)
    left_open = ixminus < 0
    right_open = ixplus < 0

    if ns >= 1:
        pos[0, left_open] = -2
        pos[ns - 1, right_open] = 2
    if ns >= 2:
        pos[1, left_open] = -1
        pos[ns - 2, right_open] = 1

    return pos


def _build_parallel_shift_maps(ixplus, ixminus, iyzero, ns, max_shift=4):
    """
    Precompute parallel shift connectivity maps for s-stencil application.

    Returns arrays with shape [2*max_shift+1, ns, nkx, nky]:
      s_shift   : target s-index
      kx_shift  : target kx-index
      valid     : whether shifted point is in-grid (open boundary aware)
    """
    nkx, nky = ixplus.shape
    nshifts = 2 * max_shift + 1

    s_shift = np.zeros((nshifts, ns, nkx, nky), dtype=np.int32)
    kx_shift = np.zeros((nshifts, ns, nkx, nky), dtype=np.int32)
    valid = np.zeros((nshifts, ns, nkx, nky), dtype=np.bool_)

    for shift_idx, delta_s in enumerate(range(-max_shift, max_shift + 1)):
        for s in range(ns):
            for kx in range(nkx):
                for ky in range(nky):
                    tgt_s = s + delta_s
                    tgt_kx = kx
                    ok = True

                    if tgt_s < 0:
                        if ky == iyzero:
                            tgt_s += ns
                        else:
                            kx_conn = ixminus[kx, ky]
                            if kx_conn >= 0:
                                tgt_kx = kx_conn
                                tgt_s += ns
                            else:
                                ok = False
                    elif tgt_s >= ns:
                        if ky == iyzero:
                            tgt_s -= ns
                        else:
                            kx_conn = ixplus[kx, ky]
                            if kx_conn >= 0:
                                tgt_kx = kx_conn
                                tgt_s -= ns
                            else:
                                ok = False

                    if ok and 0 <= tgt_s < ns:
                        s_shift[shift_idx, s, kx, ky] = tgt_s
                        kx_shift[shift_idx, s, kx, ky] = tgt_kx
                        valid[shift_idx, s, kx, ky] = True

    return s_shift, kx_shift, valid


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
    geom_type: str = "circ",
) -> Dict[str, Any]:
    """Compute geometry dict from equilibrium parameters.

    geom_type='circ': full Lapillonne circular model (nonlinear theta, delta).
    geom_type='s-alpha': simplified model (theta=2*pi*s, B=1/(1+eps*cos(theta))).
    """
    assert geom_type in ["circ", "s-alpha"], "Only circular geometries supported."

    signJ = 1.0
    sgrid = _parallel_grid(ns, nperiod)
    theta = _poloidal_angle(sgrid, eps, geom_type=geom_type)

    cg = _circular_geometry(theta, q, shat, eps, signB=signB, signJ=signJ, geom_type=geom_type)
    efun_3x3, dfun, hfun, ifun = _calc_geom_tensors(cg, signJ=signJ, signB=signB)

    bn, R = cg["bn"], cg["R"]
    little_g = np.stack([cg["metric"][:, 1, 1], cg["dzetadeps"], np.ones(ns)], axis=-1)

    g_zz_mid = (1 / (2 * np.pi * (1 + eps))) ** 2 * (1 + (1 - eps**2) * (q / eps) ** 2)
    kthnorm = np.sqrt(g_zz_mid)

    vpgr, mugr, intvp, intmu = _build_velocity_grids(nvpar, nmu, vpar_max)
    kxrh, krho_raw = _build_wavevector_grids(nkx, nky, kxmax, krhomax)
    krho = krho_raw / kthnorm

    # use actual grid length (may differ from input nkx for even nx)
    nkx_actual = len(kxrh)
    ml = _build_mode_label(nkx_actual, nky, ikxspace)
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
    """Compute geometry from a GKW ``input.dat`` file.

    Reads ``kxrh`` and ``vpgr.dat`` from the same directory if available;
    falls back to approximate formulas otherwise.
    """
    from gyaradax.utils import parse_input_dat

    inp = parse_input_dat(input_dat_path)
    geom_sec = inp.get("geom", {})
    grid_sec = inp.get("gridsize", {})
    mode_sec = inp.get("mode", {})

    q = float(geom_sec.get("q", 1.0))
    shat = float(geom_sec.get("shat", 0.0))
    eps = float(geom_sec.get("eps", 0.0))
    nkx = int(grid_sec.get("nx", 1))
    nky = int(grid_sec.get("nmod", 1))
    ikxspace = int(mode_sec.get("ikxspace", 5))
    geom_type = str(geom_sec.get("geom_type", "s-alpha")).strip("'\"").lower()

    # for single-mode (non-mode_box) cases, use kthrho as the wavenumber
    mode_box = mode_sec.get("mode_box", False)
    if not mode_box and "kthrho" in mode_sec and nky == 1:
        krhomax = float(mode_sec["kthrho"])
    else:
        krhomax = float(mode_sec.get("krhomax", 1.4))

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
        q=q,
        shat=shat,
        eps=eps,
        ns=int(grid_sec.get("n_s_grid", 16)),
        nkx=nkx,
        nky=nky,
        nvpar=int(grid_sec.get("n_vpar_grid", 32)),
        nmu=int(grid_sec.get("n_mu_grid", 8)),
        vpar_max=vpar_max,
        nperiod=int(grid_sec.get("nperiod", 1)),
        kxmax=kxmax,
        krhomax=krhomax,
        ikxspace=ikxspace,
        geom_type=geom_type,
    )


def geometry_from_geom_dat_and_input(input_dat_path: str) -> Dict[str, Any]:
    """build geometry from a ``geom.dat`` file + grids from ``input.dat``.

    use this for geometry types not supported by the analytic circular model
    (e.g. slab_periodic) when reference/geom.dat is available.
    """
    from gyaradax.utils import parse_input_dat, load_geom_dat_file

    data_dir = os.path.dirname(input_dat_path)
    ref_dir = os.path.join(data_dir, "reference")
    geom_dat_path = os.path.join(ref_dir, "geom.dat")
    if not os.path.exists(geom_dat_path):
        raise FileNotFoundError(f"geom.dat not found at {geom_dat_path}")

    gd = load_geom_dat_file(geom_dat_path)
    inp = parse_input_dat(input_dat_path)
    geom_sec = inp.get("geom", {})
    grid_sec = inp.get("gridsize", {})
    mode_sec = inp.get("mode", {})

    q = float(geom_sec.get("q", 1.0))
    shat = float(geom_sec.get("shat", 0.0))
    eps = float(geom_sec.get("eps", 0.0))
    nkx = int(grid_sec.get("nx", 1))
    nky = int(grid_sec.get("nmod", 1))
    nvpar = int(grid_sec.get("n_vpar_grid", 32))
    nmu = int(grid_sec.get("n_mu_grid", 8))
    ns = int(grid_sec.get("n_s_grid", 16))
    nperiod = int(grid_sec.get("nperiod", 1))
    krhomax = float(mode_sec.get("krhomax", 1.4))
    ikxspace = int(mode_sec.get("ikxspace", 5))
    signB = 1.0
    Rref = 100.0

    # grids from input.dat params
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
        vpar_max = float(grid_sec.get("vpmax", 3.0))

    vpgr, mugr, intvp, intmu = _build_velocity_grids(nvpar, nmu, vpar_max)
    kxrh, krho_raw = _build_wavevector_grids(nkx, nky, kxmax, krhomax)
    kthnorm = float(np.asarray(gd.get("kthnorm", 1.0)).reshape(-1)[0])
    krho = krho_raw / kthnorm

    sgrid = _parallel_grid(ns, nperiod)

    nkx_actual = len(kxrh)
    ml = _build_mode_label(nkx_actual, nky, ikxspace)
    ml_kxky, ixp, ixm, ixz, iyz = _build_mode_connectivity(ml, kxrh, krho)
    pos = _build_pos_par_grid_classes(ixp, ixm, ns)
    ss, ks, vs = _build_parallel_shift_maps(ixp, ixm, iyz, ns, max_shift=4)

    # geometry tensors from geom.dat
    bn = np.asarray(gd.get("bn", np.ones(ns)))
    ffun = np.asarray(gd.get("F", np.ones(ns)))
    gfun = np.asarray(gd.get("G", np.zeros(ns)))
    bt_frac = np.asarray(gd.get("Bt_frac", np.ones(ns)))
    rfun = np.asarray(gd.get("R", np.ones(ns)))
    efun = np.asarray(gd.get("E_eps_zeta", np.zeros(ns)))

    # metric components
    g_zz = np.asarray(gd.get("g_zeta_zeta", np.ones(ns)))
    g_ez = np.asarray(gd.get("g_eps_zeta", np.zeros(ns)))
    little_g = np.stack([g_zz, g_ez, np.ones(ns)], axis=-1)

    # drift tensors
    d_eps = np.asarray(gd.get("D_eps", np.zeros(ns)))
    d_zeta = np.asarray(gd.get("D_zeta", np.zeros(ns)))
    d_s = np.asarray(gd.get("D_s", np.zeros(ns)))
    dfun = np.stack([d_eps, d_zeta, d_s], axis=-1)

    h_eps = np.asarray(gd.get("H_eps", np.zeros(ns)))
    h_zeta = np.asarray(gd.get("H_zeta", np.zeros(ns)))
    h_s = np.asarray(gd.get("H_s", np.zeros(ns)))
    hfun = np.stack([h_eps, h_zeta, h_s], axis=-1)

    i_eps = np.asarray(gd.get("I_eps", np.zeros(ns)))
    i_zeta = np.asarray(gd.get("I_zeta", np.zeros(ns)))
    i_s = np.asarray(gd.get("I_s", np.zeros(ns)))
    ifun = np.stack([i_eps, i_zeta, i_s], axis=-1)

    Rref_val = abs(float(np.asarray(gd.get("Rref", Rref)).reshape(-1)[0]))

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
        "ffun": _f64(ffun),
        "gfun": _f64(gfun),
        "bt_frac": _f64(bt_frac),
        "rfun": _f64(rfun),
        "little_g": _f64(little_g),
        "dfun": _f64(dfun),
        "hfun": _f64(hfun),
        "ifun": _f64(ifun),
        "efun": _f64(-efun),
        "Rref": _f64(Rref_val),
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
