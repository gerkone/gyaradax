"""Final analytic geometry dictionary assembly helpers.

This module contains model-independent assembly of the solver-facing geometry
``dict`` from a normalized ``GeometrySpec``, continuous model geometry, tensor
outputs, grid arrays, and topology arrays.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from gyaradax.geometry.grids import _parallel_weights
from gyaradax.geometry.spec import GeometrySpec


def _f64(x: Any) -> jnp.ndarray:
    return jnp.array(x, dtype=jnp.float64)


def _i32(x: Any) -> jnp.ndarray:
    return jnp.array(x, dtype=jnp.int32)


def assemble_geometry_dict(
    *,
    spec: GeometrySpec,
    cg: dict[str, Any],
    tensors: tuple[Any, Any, Any, Any, Any, Any],
    sgrid: Any,
    kthnorm: Any,
    kxrh: Any,
    krho: Any,
    vpgr: Any,
    mugr: Any,
    intvp: Any,
    intmu: Any,
    mode_label: Any,
    ixplus: Any,
    ixminus: Any,
    ixzero: Any,
    iyzero: Any,
    pos_par_grid_class: Any,
    s_shift: Any,
    kx_shift: Any,
    valid_shift: Any,
) -> dict[str, Any]:
    """Assemble the solver-facing analytic geometry dictionary.

    This is a behavior-preserving extraction of the final return block from
    ``gyaradax.geometry.geom._compute_geometry_impl``.  Model-specific formulas
    remain in the continuous geometry providers; this helper only packages
    already-computed arrays and metadata into the historical dict contract.
    """
    efun_3x3, dfun, hfun, ifun, jfun, kfun = tensors
    ns = spec.ns
    signB = spec.signB
    Rref = spec.Rref

    bn, R = cg["bn"], cg["R"]
    little_g = jnp.stack([cg["metric"][:, 1, 1], cg["dzetadeps"], jnp.ones(ns)], axis=-1)

    return {
        "kthnorm": _f64(kthnorm),
        "shat": _f64(spec.shat),
        "q": _f64(spec.q),
        "eps": _f64(spec.eps),
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
        "mode_label": _i32(mode_label),
        "ixplus": _i32(ixplus),
        "ixminus": _i32(ixminus),
        "ixzero": _i32(ixzero),
        "iyzero": _i32(iyzero),
        "pos_par_grid_class": jnp.array(pos_par_grid_class, dtype=jnp.int8),
        "s_shift": _i32(s_shift),
        "kx_shift": _i32(kx_shift),
        "valid_shift": jnp.array(valid_shift, dtype=jnp.bool_),
        "kxmax": _f64(jnp.max(jnp.abs(kxrh))),
        "kymax": _f64(jnp.max(jnp.abs(krho))),
    }
