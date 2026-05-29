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

from gyaradax.geometry.assembly import assemble_geometry_dict
from gyaradax.geometry.circular import register_circular_geometry_models
from gyaradax.geometry.grids import (
    _build_mode_label,
    _build_velocity_grids,
    _build_wavevector_grids,
    _parallel_grid,
    _parallel_weights,
)
from gyaradax.geometry.miller import register_miller_geometry_model
from gyaradax.geometry.registry import ContinuousGeometryModel, get_geometry_model
from gyaradax.geometry.spec import (
    GeometrySpec,
    geometry_spec_from_compute_kwargs,
    geometry_spec_from_config,
)
from gyaradax.geometry.tensors import _calc_geom_tensors
from gyaradax.geometry.topology import (
    _build_mode_connectivity,
    _build_parallel_shift_maps,
    _build_pos_par_grid_classes,
)


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
    **miller_params: Any,
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
    spec = geometry_spec_from_compute_kwargs(
        q=q,
        shat=shat,
        eps=eps,
        ns=ns,
        nkx=nkx,
        nky=nky,
        nvpar=nvpar,
        nmu=nmu,
        vpar_max=vpar_max,
        nperiod=nperiod,
        kxmax=kxmax,
        krhomax=krhomax,
        ikxspace=ikxspace,
        signB=signB,
        Rref=Rref,
        geom_type=geom_type,
        **miller_params,
    )
    return create_geometry(spec)


def _compute_geometry_impl(spec: GeometrySpec) -> Dict[str, Any]:
    geom_type = spec.model
    try:
        model = cast(ContinuousGeometryModel, get_geometry_model(geom_type))
    except KeyError as exc:
        raise AssertionError(f"unknown geom_type: {geom_type}") from exc
    q = spec.q
    shat = spec.shat
    eps = spec.eps
    ns = spec.ns
    nkx = spec.nkx
    nky = spec.nky
    nvpar = spec.nvpar
    nmu = spec.nmu
    vpar_max = spec.vpar_max
    nperiod = spec.nperiod
    kxmax = spec.kxmax
    krhomax = spec.krhomax
    ikxspace = spec.ikxspace
    signB = spec.signB
    miller_params = dict(spec.model_params)

    signJ = 1.0
    sgrid = _parallel_grid(ns, nperiod)

    cg = model.continuous_geometry(
        sgrid=sgrid,
        q=q,
        shat=shat,
        eps=eps,
        nperiod=nperiod,
        signB=signB,
        signJ=signJ,
        model_params=miller_params,
    )

    efun_3x3, dfun, hfun, ifun, jfun, kfun = _calc_geom_tensors(cg, signJ=signJ, signB=signB)

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

    return assemble_geometry_dict(
        spec=spec,
        cg=cg,
        tensors=(efun_3x3, dfun, hfun, ifun, jfun, kfun),
        sgrid=sgrid,
        kthnorm=kthnorm,
        kxrh=kxrh,
        krho=krho,
        vpgr=vpgr,
        mugr=mugr,
        intvp=intvp,
        intmu=intmu,
        mode_label=ml_kxky,
        ixplus=ixp,
        ixminus=ixm,
        ixzero=ixz,
        iyzero=iyz,
        pos_par_grid_class=pos,
        s_shift=ss,
        kx_shift=ks,
        valid_shift=vs,
    )


register_circular_geometry_models(_compute_geometry_impl)
register_miller_geometry_model(_compute_geometry_impl)


def create_geometry(spec: GeometrySpec) -> Dict[str, Any]:
    """Create analytic geometry arrays from a normalized spec via the registry."""
    try:
        model = get_geometry_model(spec.model)
    except KeyError as exc:
        raise AssertionError(f"unknown geom_type: {spec.model}") from exc
    return model.compute(spec)


def geometry_spec_from_input_dat(input_dat_path: str) -> GeometrySpec:
    """Build a ``GeometrySpec`` from a GKW input.dat file.

    Preserves the historical ``compute_geometry_from_input`` defaults,
    including absent ``geom_type`` -> ``s-alpha``.
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

    return geometry_spec_from_compute_kwargs(
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


def compute_geometry_from_config(cfg: Any) -> Dict[str, Any]:
    """Compute analytic geometry from a YAML/OmegaConf-style config.

    Missing ``geometry.geometry_model`` preserves the historical config-wrapper
    default of circular geometry.
    """
    return create_geometry(geometry_spec_from_config(cfg))


def compute_geometry_from_input(input_dat_path: str) -> Dict[str, Any]:
    """Compute geometry from a GKW input.dat file.

    Reads kxrh and vpgr.dat from the same directory if available; falls
    back to approximate formulas otherwise.
    """
    return create_geometry(geometry_spec_from_input_dat(input_dat_path))


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
