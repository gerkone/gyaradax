"""Model-independent geometry tensor assembly helpers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def _calc_geom_tensors(cg: dict[str, Any], signJ: float = 1.0, signB: float = 1.0):
    """E, D, H, I, J, K drift tensors from the geometry dict.

    Port of calc_geom_tensors (geom.f90:3487-3634). jfun = R²-R0² (centrifugal
    trapping); kfun = 2*R*dR/dpsi - lfun, lfun = 2*R0*dR/dpsi. R0 defaults to
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
