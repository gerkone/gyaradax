"""Circular (Lapillonne) and s-alpha geometry.

Port of ``geom_circ`` (geom.f90:1444-1616) and ``geom_s_alpha``
(geom.f90:1110-1200). Both produce the same dict shape as
``_miller_geometry`` so the downstream tensor builder is unchanged.
"""

import jax.numpy as jnp


def _poloidal_angle(sgrid, eps, geom_type="circ", n_iter=10):
    """Map field-line coordinate s to poloidal angle theta.

    circ: invert theta + eps*sin(theta) = 2*pi*s (fixed-point, 10 iters).
    s-alpha: linear mapping theta = 2*pi*s.
    """
    if geom_type == "s-alpha":
        return 2 * jnp.pi * sgrid
    theta = 2 * jnp.pi * sgrid
    for _ in range(n_iter):
        theta = 2 * jnp.pi * sgrid - eps * jnp.sin(theta)
    return theta


def _dzetadeps(theta, q, shat, eps, signB, signJ):
    """Metric coupling g_{psi,zeta} = d(zeta)/d(eps) at fixed s.

    Branch-tracked atan with monotonicity correction
    (geom.f90:1492-1511). Carries ~0.1% model-level error into
    zeta-direction tensors — inherent to the Lapillonne approximation.
    """
    raw = jnp.arctan(jnp.sqrt((1 - eps) / (1 + eps)) * jnp.tan(theta / 2))
    # cumulative pi jumps detect branch-cut crossings of atan
    jumps = jnp.where(jnp.diff(raw) < 0, jnp.pi, 0.0)
    dzde = raw + jnp.concatenate([jnp.zeros(1), jnp.cumsum(jumps)])
    dzde = dzde - jnp.pi * jnp.floor((dzde[0] - theta[0] / 2) / jnp.pi)

    t2 = jnp.tan(theta / 2)
    corr = eps / jnp.sqrt(1 - eps**2) * t2 / (1 + t2**2 + eps * (1 - t2**2))
    return signB * signJ / jnp.pi * q / eps * (shat * dzde - corr)


def _psi_theta_to_psi_s(f_psi, f_theta, theta, eps):
    """Jacobian transform from (psi, theta) to (psi, s).

    f_psi_s = f_psi - sin(theta)/R * f_theta
    f_s     = 2*pi/R * f_theta
    where R = 1 + eps*cos(theta).
    """
    R = 1 + eps * jnp.cos(theta)
    return f_psi - jnp.sin(theta) / R * f_theta, 2 * jnp.pi * f_theta / R


def _circular_geometry(theta, q, shat, eps, signB=1.0, signJ=1.0, geom_type="circ"):
    """Geometry quantities for circular / s-alpha models.

    circ: Lapillonne model (finite_epsilon=True, nonlinear theta).
    s-alpha: simplified B = 1/(1 + eps·cos(theta)), finite_epsilon=False.
    """
    ns = len(theta)
    finite_epsilon = geom_type != "s-alpha"
    R = 1 + eps * jnp.cos(theta)
    dzde = _dzetadeps(theta, q, shat, eps, signB, signJ)

    if geom_type == "s-alpha":
        dum = 1.0
        bups = signJ / (2 * jnp.pi * q)
        dpfdpsi = eps / q
    else:
        dum = jnp.sqrt(1 + eps**2 / q**2 / (1 - eps**2))
        bups = 1.0 / (2 * jnp.pi * q * jnp.sqrt(1 - eps**2))
        dpfdpsi = eps / (q * jnp.sqrt(1 - eps**2))
    bn = dum / R

    # F tensor: bups, or bups/bn when finite_epsilon (geom.f90:3507-3508)
    ffun = bups / bn if finite_epsilon else jnp.full(ns, bups)

    metric = jnp.zeros((ns, 3, 3))
    metric = metric.at[:, 0, 0].set(1.0)

    if geom_type == "s-alpha":
        # s-alpha metric (geom.f90:1375-1392)
        sgrid = theta / (2 * jnp.pi)
        cross_01 = q * shat * sgrid / eps * signB * signJ
        metric = metric.at[:, 0, 1].set(cross_01)
        metric = metric.at[:, 1, 0].set(cross_01)
        metric = metric.at[:, 1, 1].set(
            (q / (2 * jnp.pi * eps)) ** 2 * (1 + (2 * jnp.pi * shat * sgrid) ** 2)
        )
        cross_12 = q / (2 * jnp.pi * eps) ** 2 * signB * signJ
        metric = metric.at[:, 1, 2].set(cross_12)
        metric = metric.at[:, 2, 1].set(cross_12)
        metric = metric.at[:, 2, 2].set(1.0 / (2 * jnp.pi * eps) ** 2)

        dBdpsi = -jnp.cos(theta)
        dBds = 2 * jnp.pi * eps * jnp.sin(theta)
        dRdpsi = jnp.cos(theta)
        dRds = -2 * jnp.pi * eps * jnp.sin(theta)
        dZdpsi = jnp.sin(theta)
        dZds = 2 * jnp.pi * eps * jnp.cos(theta)
    else:
        # circular metric (geom.f90:1543-1579)
        metric = metric.at[:, 0, 1].set(dzde)
        metric = metric.at[:, 1, 0].set(dzde)
        sin_2pi = jnp.sin(theta) / (2 * jnp.pi)
        metric = metric.at[:, 0, 2].set(sin_2pi)
        metric = metric.at[:, 2, 0].set(sin_2pi)
        metric = metric.at[:, 1, 1].set(
            (1 / (2 * jnp.pi * R)) ** 2 * (1 + (1 - eps**2) * (q / eps) ** 2) + dzde**2
        )
        cross_12 = (
            q * jnp.sqrt(1 - eps**2) / (2 * jnp.pi * eps) ** 2 * signB * signJ + dzde * sin_2pi
        )
        metric = metric.at[:, 1, 2].set(cross_12)
        metric = metric.at[:, 2, 1].set(cross_12)
        metric = metric.at[:, 2, 2].set(
            (1 / (2 * jnp.pi)) ** 2 * ((1 / eps + jnp.cos(theta)) ** 2 + jnp.sin(theta) ** 2)
        )

        # circular field derivatives (geom.f90:1525-1541)
        dBdpsi_pt = bn * (
            -jnp.cos(theta) / R
            + eps * (1 - shat + eps**2 / (1 - eps**2)) / (eps**2 + q**2 * (1 - eps**2))
        )
        dBds_pt = bn * eps * jnp.sin(theta) / R
        dBdpsi, dBds = _psi_theta_to_psi_s(dBdpsi_pt, dBds_pt, theta, eps)
        dRdpsi, dRds = _psi_theta_to_psi_s(jnp.cos(theta), -eps * jnp.sin(theta), theta, eps)
        dZdpsi, dZds = _psi_theta_to_psi_s(jnp.sin(theta), eps * jnp.cos(theta), theta, eps)

    gfun = ffun * dBds / bn if finite_epsilon else ffun * bn * dBds

    return {
        "bn": bn,
        "R": R,
        "ffun": ffun,
        "gfun": gfun,
        "bt_frac": jnp.full(ns, 1 / dum),
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
        "finite_epsilon": finite_epsilon,
    }
