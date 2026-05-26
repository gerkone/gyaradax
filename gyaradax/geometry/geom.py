"""Model-independent geometry helpers and public entry points.

Holds the parallel / velocity / wavevector grids, the open-boundary
topology maps, the E/D/H/I/J/K tensor builder, and the dispatchers that
pick the geometry model (circ, s-alpha, miller) and assemble the dict
consumed by the solver.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Mapping, cast

from gyaradax.geometry.lapillonne import _circular_geometry, _poloidal_angle


def _f64(x):
    return jnp.array(x, dtype=jnp.float64)


def _i32(x):
    return jnp.array(x, dtype=jnp.int32)


def _load_1d_array(path):
    """Load a GKW grid file and flatten to 1-d (files may have nky rows)."""
    data = np.loadtxt(path)
    if data.ndim > 1:
        data = data[0]
    return np.atleast_1d(data)


def _parallel_grid(ns, nperiod):
    """Cell-centered uniform parallel grid on [-sgrmax, sgrmax]."""
    sgrmax = nperiod - 0.5
    return jnp.array([-sgrmax + 2 * sgrmax * (i + 0.5) / ns for i in range(ns)])


def _parallel_weights(sgrid):
    """Uniform integration weights (cell width)."""
    if len(sgrid) < 2:
        return jnp.ones(1)
    return jnp.full(len(sgrid), sgrid[1] - sgrid[0])


def _calc_geom_tensors(cg, signJ=1.0, signB=1.0):
    """E, D, H, I, J, K drift tensors from the geometry dict.

    Port of calc_geom_tensors (geom.f90:3487-3634). jfun = R²-R0² (centrifugal
    trapping); kfun = 2*R*dR/dpsi - lfun, lfun = 2*R0*dR0/dpsi. R0 defaults to
    the magnetic-axis location if cg provides it, else R[ns//2].
    """
    bn = cg["bn"]
    metric = cg["metric"]
    R = cg["R"]
    bups = cg["bups"]
    finite_epsilon = cg.get("finite_epsilon", True)
    dBdpsi, dBds = cg["dBdpsi"], cg["dBds"]
    dRdpsi, dRds = cg["dRdpsi"], cg["dRds"]
    dZdpsi, dZds = cg["dZdpsi"], cg["dZds"]

    m0 = metric[:, 0, :]
    m1 = metric[:, 1, :]
    efun = m0[:, :, None] * m1[:, None, :] - m1[:, :, None] * m0[:, None, :]
    efun = efun * (signJ * jnp.pi * cg["dpfdpsi"])
    if finite_epsilon:
        efun = efun / bn[:, None, None] ** 2

    e0, e2 = efun[:, :, 0], efun[:, :, 2]

    dfun = -2 * e0 * dBdpsi[:, None] - 2 * e2 * dBds[:, None]
    if finite_epsilon:
        dfun = dfun / bn[:, None]

    hfun = -signB * (metric[:, :, 0] * dZdpsi[:, None] + metric[:, :, 2] * dZds[:, None])
    if finite_epsilon:
        hfun = hfun.at[:, 2].add(signB * bups**2 * dZds / bn**2)
    hfun = hfun / bn[:, None]

    ifun = 2 * R[:, None] * (e0 * dRdpsi[:, None] + e2 * dRds[:, None])

    # centrifugal tensors J, K (geom.f90:3599-3625)
    R0 = cg.get("R0", R[len(R) // 2])
    dR0dpsi = cg.get("dR0dpsi", dRdpsi[len(R) // 2])
    lfun = 2.0 * R0 * dR0dpsi
    jfun = R**2 - R0**2
    kfun = 2.0 * R * dRdpsi - lfun
    return efun, dfun, hfun, ifun, jfun, kfun


def _build_velocity_grids(nvpar, nmu, vpar_max):
    """Uniform v_par grid and uniform-in-v_perp mu grid (GKW convention)."""
    dvp = 2 * vpar_max / nvpar
    vpgr = jnp.linspace(-vpar_max + dvp / 2, vpar_max - dvp / 2, nvpar)
    dvperp = vpar_max / nmu
    vperp = jnp.linspace(dvperp / 2, vpar_max - dvperp / 2, nmu)
    return vpgr, vperp**2 / 2, jnp.full(nvpar, dvp), 2 * jnp.pi * vperp * dvperp


def _build_wavevector_grids(
    nkx, nky, kxmax, krhomax, q=1.0, shat=0.0, eps=0.1, ikxspace=5, kthnorm=1.0
):
    """Centered kx grid and uniform ky grid.

    For nky=1 the single mode sits at krhomax. For nky>1 the kx spacing
    follows the shear connectivity: kxspace = |q*shat*krho[1]/(eps*ikxspace)|
    (GKW mode.f90:698).
    """
    if nky == 1:
        half = (nkx - 1) // 2
        dkx = kxmax / half if half > 0 else 0.0
        return jnp.arange(-half, half + 1) * dkx, jnp.array([krhomax])

    dky = krhomax / (nky - 1)
    krho_norm = jnp.arange(nky) * dky / kthnorm

    half = (nkx - 1) // 2
    if half > 0 and abs(shat) > 1e-10 and eps > 1e-10:
        kxspace = abs(q * shat * krho_norm[1] / (eps * ikxspace))
    elif half > 0:
        kxspace = kxmax / half
    else:
        kxspace = 0.0

    kxrh = jnp.arange(-half, half + 1) * kxspace
    return kxrh, jnp.arange(nky) * dky


def _build_mode_label(nkx, nky, ikxspace):
    """Mode-label array for open parallel boundary connectivity.

    ky=0: each kx is its own mode (periodic). ky>0: modes grouped into
    chains spaced ikxspace apart in kx-index.
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
    """Build spectral parallel-boundary connectivity from mode labels.

    Returns (mode_label_kxky, ixplus, ixminus, ixzero, iyzero, iyzero_bc).
    `ixplus`/`ixminus` use -1 to mark open boundaries. `iyzero_bc` is -1
    when ky=0 is absent so the periodic zonal treatment never applies to
    a non-zonal mode.
    """
    mode_label = np.atleast_1d(np.asarray(mode_label, dtype=np.int32))
    kxrh_np = np.atleast_1d(np.asarray(kxrh))
    krho_np = np.atleast_1d(np.asarray(krho))
    nkx = int(kxrh_np.shape[0])
    nky = int(krho_np.shape[0])

    if mode_label.shape == (nkx, nky):
        mode_label_kxky = mode_label
    elif mode_label.shape == (nky, nkx):
        mode_label_kxky = mode_label.T
    elif mode_label.size == nkx * nky:
        mode_label_kxky = mode_label.reshape(nkx, nky)
    else:
        raise ValueError(
            f"mode_label shape {mode_label.shape} incompatible with nkx/nky=({nkx},{nky})"
        )

    ixzero = int(np.argmin(np.abs(kxrh_np)))
    iyzero = int(np.argmin(np.abs(krho_np)))
    ky_is_truly_zonal = np.abs(krho_np[iyzero]) < 1e-10

    ixplus = -np.ones((nkx, nky), dtype=np.int32)
    ixminus = -np.ones((nkx, nky), dtype=np.int32)

    for iy in range(nky):
        if iy == iyzero and ky_is_truly_zonal:
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

    iyzero_bc = iyzero if ky_is_truly_zonal else -1
    return mode_label_kxky, ixplus, ixminus, ixzero, iyzero, iyzero_bc


def _build_pos_par_grid_classes(ixplus, ixminus, ns):
    """Position class (-2,-1,0,1,2) for open parallel boundary handling.

    Shape: [ns, nkx, nky].
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
    """Precompute parallel shift connectivity maps for s-stencil application.

    Returns arrays with shape [2*max_shift+1, ns, nkx, nky]: s_shift, kx_shift,
    valid. `valid` is False on out-of-grid shifts (open boundary).
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
    **miller_params,
) -> Dict[str, Any]:
    """Compute the geometry dict from equilibrium parameters.

    Continuous geometry is JAX (differentiable w.r.t. q, shat, eps and the
    Miller shape parameters). Discrete topology (mode labels, connectivity)
    uses numpy.

    geom_type:
      'circ'    full Lapillonne circular model
      's-alpha' simplified theta=2*pi*s, B=1/(1+eps*cos(theta))
      'miller'  Miller parametrisation with elongation/triangularity/
                squareness (see gyaradax.geometry.miller for the shape
                parameters, passed through **miller_params).
    """
    assert geom_type in ("circ", "s-alpha", "miller"), f"unknown geom_type: {geom_type}"

    signJ = 1.0
    sgrid = _parallel_grid(ns, nperiod)

    cg: dict[str, Any]
    if geom_type == "miller":
        from gyaradax.geometry.miller import _miller_geometry

        cg = cast(
            dict[str, Any],
            _miller_geometry(
                sgrid=sgrid,
                q=q,
                shat=shat,
                eps=eps,
                nperiod=nperiod,
                signB=signB,
                signJ=signJ,
                **miller_params,
            ),
        )
    else:
        theta = _poloidal_angle(sgrid, eps, geom_type=geom_type)
        cg = cast(
            dict[str, Any],
            _circular_geometry(theta, q, shat, eps, signB=signB, signJ=signJ, geom_type=geom_type),
        )

    efun_3x3, dfun, hfun, ifun, jfun, kfun = _calc_geom_tensors(cg, signJ=signJ, signB=signB)

    bn, R = cg["bn"], cg["R"]
    little_g = jnp.stack([cg["metric"][:, 1, 1], cg["dzetadeps"], jnp.ones(ns)], axis=-1)

    g_zz_mid: Any
    if geom_type == "s-alpha":
        g_zz_mid = (q / (2 * jnp.pi * eps)) ** 2
    elif geom_type == "circ":
        g_zz_mid = (1 / (2 * jnp.pi * (1 + eps))) ** 2 * (1 + (1 - eps**2) * (q / eps) ** 2)
    else:  # miller
        g_zz_mid = cg.get("g_zz_mid", cg["metric"][ns // 2, 1, 1])
    kthnorm = jnp.sqrt(g_zz_mid)

    vpgr, mugr, intvp, intmu = _build_velocity_grids(nvpar, nmu, vpar_max)
    kxrh, krho_raw = _build_wavevector_grids(
        nkx,
        nky,
        kxmax,
        krhomax,
        q=q,
        shat=shat,
        eps=eps,
        ikxspace=ikxspace,
        kthnorm=kthnorm,
    )
    krho = krho_raw / kthnorm

    # discrete topology (numpy, not differentiable)
    kxrh_np = np.asarray(jax.lax.stop_gradient(kxrh))
    krho_np = np.asarray(jax.lax.stop_gradient(krho))
    nkx_actual = len(kxrh_np)
    ml = _build_mode_label(nkx_actual, nky, ikxspace)
    ml_kxky, ixp, ixm, ixz, iyz, iyz_bc = _build_mode_connectivity(ml, kxrh_np, krho_np)
    pos = _build_pos_par_grid_classes(ixp, ixm, ns)
    ss, ks, vs = _build_parallel_shift_maps(ixp, ixm, iyz_bc, ns, max_shift=4)

    return {
        "kthnorm": _f64(kthnorm),
        "shat": _f64(shat),
        "q": _f64(q),
        "eps": _f64(eps),
        "kxrh": _f64(kxrh),
        "krho": _f64(krho),
        "parseval": _f64(jnp.where(jnp.abs(krho) < 1e-12, 1.0, 2.0)),
        "intvp": _f64(intvp),
        "vpgr": _f64(vpgr),
        "vpgr_rms": _f64(jnp.sqrt(jnp.mean(vpgr**2))),
        "dvp": _f64(jnp.mean(jnp.diff(vpgr)) if len(vpgr) > 1 else 1.0),
        "intmu": _f64(intmu),
        "mugr": _f64(mugr),
        "mugr_rms": _f64(jnp.sqrt(jnp.mean(mugr**2))),
        "ints": _f64(_parallel_weights(sgrid)),
        "sgrid": _f64(sgrid),
        "sgr_dist": _f64(jnp.abs(sgrid[1] - sgrid[0]) if ns > 1 else 1.0),
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
        "efun_3x3": _f64(efun_3x3),
        "jfun": _f64(jfun),
        "kfun": _f64(kfun),
        "R0": _f64(cg.get("R0", cg["R"][ns // 2])),
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
        "kxmax": _f64(jnp.max(jnp.abs(kxrh))),
        "kymax": _f64(jnp.max(jnp.abs(krho))),
    }


def compute_geometry_from_input(input_dat_path: str) -> Dict[str, Any]:
    """Compute geometry from a GKW input.dat file.

    Reads kxrh and vpgr.dat from the same directory if available; falls
    back to approximate formulas otherwise.
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

    # Miller shape parameters (only used when geom_type='miller'); the alias
    # map handles GKW's case-insensitive namelist vs. the JAX entry-point.
    miller_params: dict[str, Any] = {}
    if geom_type == "miller":
        alias = {"zmil": "Zmil", "drmil": "dRmil", "dzmil": "dZmil"}
        for k in (
            "kappa",
            "delta",
            "square",
            "zmil",
            "drmil",
            "dzmil",
            "skappa",
            "sdelta",
            "ssquare",
        ):
            if k in geom_sec:
                miller_params[alias.get(k, k)] = float(geom_sec[k])

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
        **miller_params,
    )


def geometry_from_geom_dat_and_input(input_dat_path: str) -> Dict[str, Any]:
    """Build geometry from a GKW geom.dat + grids from input.dat.

    Use this for geometry types not supported by the analytic models
    (e.g. slab_periodic) when reference/geom.dat is available.
    """
    from gyaradax.utils import parse_input_dat, load_geom_dat_file

    data_dir = os.path.dirname(input_dat_path)
    ref_dir = os.path.join(data_dir, "reference")
    geom_dat_path = os.path.join(ref_dir, "geom.dat")
    if not os.path.exists(geom_dat_path):
        raise FileNotFoundError(f"geom.dat not found at {geom_dat_path}")

    gd: Mapping[str, Any] = load_geom_dat_file(geom_dat_path)
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
    ml_kxky, ixp, ixm, ixz, iyz, iyz_bc = _build_mode_connectivity(ml, kxrh, krho)
    pos = _build_pos_par_grid_classes(ixp, ixm, ns)
    ss, ks, vs = _build_parallel_shift_maps(ixp, ixm, iyz_bc, ns, max_shift=4)

    bn = np.asarray(gd.get("bn", np.ones(ns)))
    ffun = np.asarray(gd.get("F", np.ones(ns)))
    gfun = np.asarray(gd.get("G", np.zeros(ns)))
    bt_frac = np.asarray(gd.get("Bt_frac", np.ones(ns)))
    rfun = np.asarray(gd.get("R", np.ones(ns)))
    efun = np.asarray(gd.get("E_eps_zeta", np.zeros(ns)))

    g_zz = np.asarray(gd.get("g_zeta_zeta", np.ones(ns)))
    g_ez = np.asarray(gd.get("g_eps_zeta", np.zeros(ns)))
    little_g = np.stack([g_zz, g_ez, np.ones(ns)], axis=-1)

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
        "parseval": _f64(jnp.where(jnp.abs(krho) < 1e-12, 1.0, 2.0)),
        "intvp": _f64(intvp),
        "vpgr": _f64(vpgr),
        "vpgr_rms": _f64(jnp.sqrt(jnp.mean(vpgr**2))),
        "dvp": _f64(jnp.mean(jnp.diff(vpgr)) if len(vpgr) > 1 else 1.0),
        "intmu": _f64(intmu),
        "mugr": _f64(mugr),
        "mugr_rms": _f64(jnp.sqrt(jnp.mean(mugr**2))),
        "ints": _f64(_parallel_weights(sgrid)),
        "sgrid": _f64(sgrid),
        "sgr_dist": _f64(jnp.abs(sgrid[1] - sgrid[0]) if ns > 1 else 1.0),
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
        "kxmax": _f64(jnp.max(jnp.abs(kxrh))),
        "kymax": _f64(jnp.max(jnp.abs(krho))),
    }
