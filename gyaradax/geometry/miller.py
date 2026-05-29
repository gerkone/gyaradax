"""Miller geometry (port of GKW geom_miller, geom.f90:1623-2567).

Flux surface (Miller et al. PoP 5, 973, 1998):

    R(theta, r) = Rmil + r * cos(theta + arcsin(delta) * sin(theta))
    Z(theta, r) = Zmil + r * kappa * sin(theta + square * sin(2*theta))

with r = eps * Rmil, kappa = elongation, delta = triangularity, square =
squareness, Zmil = elevation. Radial derivatives use the GKW scaled
forms skappa, sdelta, ssquare, dRmil, dZmil.

``_miller_geometry`` returns a dict with the same fields as
``_circular_geometry`` so the downstream tensor builder
``_calc_geom_tensors`` consumes it unchanged.
"""

from collections.abc import Callable
from typing import Any, Mapping

import jax.numpy as jnp

from gyaradax.geometry.registry import register_geometry_model
from gyaradax.geometry.spec import GeometrySpec


class MillerGeometryModel:
    """Registry adapter for the existing Miller analytic geometry path."""

    name = "miller"

    def __init__(self, compute_impl: Callable[[GeometrySpec], dict[str, Any]]) -> None:
        self._compute_impl = compute_impl

    def compute(self, spec: GeometrySpec) -> dict[str, Any]:
        return self._compute_impl(spec)

    def continuous_geometry(
        self,
        *,
        sgrid: Any,
        q: float,
        shat: float,
        eps: float,
        nperiod: int,
        signB: float,
        signJ: float,
        model_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the model-specific continuous Miller geometry dict."""
        return _miller_geometry(
            sgrid=sgrid,
            q=q,
            shat=shat,
            eps=eps,
            nperiod=nperiod,
            signB=signB,
            signJ=signJ,
            **dict(model_params),
        )


def register_miller_geometry_model(
    compute_impl: Callable[[GeometrySpec], dict[str, Any]],
) -> None:
    """Register the Miller geometry model with the shared registry."""
    register_geometry_model(MillerGeometryModel(compute_impl))


def _interpquad(x_fine, y_fine, x_out):
    """3-point quadratic Lagrange interpolation (GKW interpquad).

    For each x_out[i] picks the nearest x_fine[j] and interpolates using
    the triplet (j-1, j, j+1). Shape follows x_out.
    """
    n = x_fine.shape[0]
    diffs = jnp.abs(x_out[:, None] - x_fine[None, :])
    j = jnp.clip(jnp.argmin(diffs, axis=1), 1, n - 2)
    xm, x0, xp = x_fine[j - 1], x_fine[j], x_fine[j + 1]
    ym, y0, yp = y_fine[j - 1], y_fine[j], y_fine[j + 1]
    wm = (x_out - x0) * (x_out - xp) / ((xm - x0) * (xm - xp))
    w0 = (x_out - xp) * (x_out - xm) / ((x0 - xp) * (x0 - xm))
    wp = (x_out - xm) * (x_out - x0) / ((xp - xm) * (xp - x0))
    return wm * ym + w0 * y0 + wp * yp


def _simpson_segments(x, y):
    """Simpson 1/3 per-segment integrals with midpoint interpolation."""
    x_mid = 0.5 * (x[:-1] + x[1:])
    y_mid = _interpquad(x, y, x_mid)
    dx = x[1:] - x[:-1]
    return dx * (y[:-1] + 4.0 * y_mid + y[1:]) / 6.0


def _simpson_cumulative(x, y):
    """Cumulative ∫_{s(x)=0}^x y dx (Simpson 1/3, midpoint-interp segments)."""
    seg = _simpson_segments(x, y)
    ixzero = jnp.argmin(jnp.abs(x))
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    return cum - cum[ixzero]


def _simpson_total(x, y):
    """Total ∫_{x[0]}^{x[-1]} y dx (Simpson 1/3, midpoint-interp segments)."""
    return jnp.sum(_simpson_segments(x, y))


def _miller_surface(theta, eps, kappa, delta, square, Zmil, dRmil, dZmil, skappa, sdelta, ssquare):
    """R, Z and the analytic first/second derivatives at each theta.

    Returns a dict with rfun, z_fs, dRdpsi, dZdpsi, dRdth, dZdth, d2Rdth,
    d2Zdth, d2Rdpsidth, d2Zdpsidth — all shape-of-theta.
    """
    asd = jnp.arcsin(delta)
    x1 = theta + asd * jnp.sin(theta)
    x2 = 1.0 + asd * jnp.cos(theta)
    x3 = 1.0 + 2.0 * square * jnp.cos(2.0 * theta)
    x4 = theta + square * jnp.sin(2.0 * theta)

    rfun = 1.0 + eps * jnp.cos(x1)
    z_fs = Zmil + kappa * eps * jnp.sin(x4)

    dRdpsi = dRmil + jnp.cos(x1) - sdelta * jnp.sin(theta) * jnp.sin(x1)
    dZdpsi = (
        dZmil
        + kappa * jnp.sin(x4) * (1.0 + skappa)
        + kappa * ssquare * jnp.sin(2.0 * theta) * jnp.cos(x4)
    )
    dRdth = -eps * x2 * jnp.sin(x1)
    dZdth = kappa * eps * x3 * jnp.cos(x4)

    d2Rdth = eps * asd * jnp.sin(theta) * jnp.sin(x1) - eps * x2**2 * jnp.cos(x1)
    d2Zdth = -4.0 * kappa * eps * square * jnp.sin(2.0 * theta) * jnp.cos(
        x4
    ) - kappa * eps * x3**2 * jnp.sin(x4)
    d2Rdpsidth = (
        -x2 * jnp.sin(x1)
        - sdelta * jnp.cos(theta) * jnp.sin(x1)
        - sdelta * x2 * jnp.sin(theta) * jnp.cos(x1)
    )
    d2Zdpsidth = (
        kappa * (1.0 + skappa) * x3 * jnp.cos(x4)
        + 2.0 * kappa * ssquare * jnp.cos(2.0 * theta) * jnp.cos(x4)
        - kappa * ssquare * jnp.sin(2.0 * theta) * x3 * jnp.sin(x4)
    )

    return dict(
        rfun=rfun,
        z_fs=z_fs,
        dRdpsi=dRdpsi,
        dZdpsi=dZdpsi,
        dRdth=dRdth,
        dZdth=dZdth,
        d2Rdth=d2Rdth,
        d2Zdth=d2Zdth,
        d2Rdpsidth=d2Rdpsidth,
        d2Zdpsidth=d2Zdpsidth,
    )


def _miller_geometry(
    sgrid,
    q,
    shat,
    eps,
    nperiod,
    signB=1.0,
    signJ=1.0,
    *,
    kappa=1.0,
    delta=0.0,
    square=0.0,
    Zmil=0.0,
    dRmil=0.0,
    dZmil=0.0,
    skappa=0.0,
    sdelta=0.0,
    ssquare=0.0,
    gradp=0.0,
    gradp_type="alpha",
):
    """Miller geometry quantities interpolated onto the solver's s-grid.

    Builds everything on a refined theta grid (N = 501·(2·nperiod+1))
    using Simpson flux-surface integrals to assemble s(theta), the
    Jacobian and the full contravariant metric, then quadratically
    interpolates onto sgrid.
    """
    # fine theta grid: 501 points per poloidal turn on the extended domain
    nperiod_ext = nperiod + 1
    span = 2 * nperiod_ext - 1
    N = 501 * span
    theta = jnp.linspace(-span * jnp.pi, span * jnp.pi, N)

    s = _miller_surface(
        theta, eps, kappa, delta, square, Zmil, dRmil, dZmil, skappa, sdelta, ssquare
    )
    R_, Z_ = s["rfun"], s["z_fs"]
    dRp, dZp = s["dRdpsi"], s["dZdpsi"]
    dRt, dZt = s["dRdth"], s["dZdth"]
    d2Rt, d2Zt = s["d2Rdth"], s["d2Zdth"]
    d2Rpt, d2Zpt = s["d2Rdpsidth"], s["d2Zdpsidth"]

    # covariant metric (psi, theta)
    Jpsi = R_ * (dRp * dZt - dRt * dZp)
    gpp = dRp**2 + dZp**2
    gtt = dRt**2 + dZt**2
    gpt = dRp * dRt + dZp * dZt
    # contravariant metric g^{i j} = cof / Jpsi²
    g_pp = gtt * R_**2 / Jpsi**2
    g_tt = gpp * R_**2 / Jpsi**2
    g_pt = -gpt * R_**2 / Jpsi**2

    # Mercier-Luc: arc length, (cos u, sin u), |grad psi|, radius of curvature
    dldt = jnp.sqrt(gtt)
    cosu = dZt / dldt
    sinu = -dRt / dldt
    drhodpsi = cosu * dRp + sinu * dZp
    dldpsi = cosu * dZp - sinu * dRp
    rc = gtt**1.5 / (dRt * d2Zt - dZt * d2Rt)
    abs_gradpsi = jnp.sqrt(gtt) / (dRp * dZt - dRt * dZp)

    # volume integrals (integ3, integ4, integ10 in GKW)
    ds_arc = jnp.sqrt(dRt**2 + dZt**2)
    i3 = jnp.abs(_simpson_total(theta, Z_ * dRt)) / span
    i_rb = _simpson_total(theta, R_ * ds_arc) / span
    i_ds = _simpson_total(theta, ds_arc) / span
    i4 = i_rb / i_ds

    i5 = _simpson_total(theta, dRp * ds_arc) / span
    i6 = _simpson_total(theta, R_ * (d2Rpt * dRt + d2Zpt * dZt) / ds_arc) / span
    i7 = _simpson_total(theta, dZp * dRt + Z_ * d2Rpt) / span
    i11 = _simpson_total(theta, (d2Rpt * dRt + d2Zpt * dZt) / ds_arc) / span

    # cumulative ∫Jpsi -> s(theta); ∫Jpsi/R² -> dpfdpsi
    integ_Jpsi = _simpson_cumulative(theta, Jpsi)
    integ_JR2_tot = _simpson_total(theta, Jpsi / R_**2)
    total_Jpsi = integ_Jpsi[-1] - integ_Jpsi[0]
    s_of_theta = integ_Jpsi * span / total_Jpsi

    F = 1.0
    dpfdpsi = F * integ_JR2_tot / (span * 2.0 * jnp.pi * q)

    # volume + radial derivative -> pressure-gradient coupling grdp
    vol = 2.0 * jnp.pi * i3 * i4
    dvoldpsi = 2.0 * jnp.pi * ((i5 + i6) / i_ds - i11 * i4 / i_ds) * i3 + 2.0 * jnp.pi * i4 * i7
    if gradp_type == "alpha":
        grdp = (
            gradp
            * (4.0 * jnp.pi**2)
            * dpfdpsi
            * jnp.sqrt(2.0 * jnp.pi**2)
            / (dvoldpsi * jnp.sqrt(vol))
        )
    elif gradp_type == "beta_prime_input":
        grdp = gradp / (2.0 * dpfdpsi)
    elif gradp_type == "pprime":
        grdp = gradp
    else:
        raise NotImplementedError(f"gradp_type='{gradp_type}' not supported")

    # magnetic field components
    bups = signJ * span * dpfdpsi / total_Jpsi
    Bt = F / R_
    Bp = dpfdpsi * abs_gradpsi / R_
    bn = jnp.sqrt(Bp**2 + Bt**2)
    pf1 = R_ * Bp

    # d|grad psi|/dtheta (needed for dpf1/dl)
    dabs_gradpsi_dth = (d2Rt * dRt + d2Zt * dZt) * R_ / (Jpsi * dldt) - dldt * R_**2 * (
        d2Rpt * dZt + dRp * d2Zt - d2Rt * dZp - dRt * d2Zpt
    ) / Jpsi**2
    dpf1dl = dpfdpsi * dabs_gradpsi_dth / dldt

    # F' and grad(zeta) via three integrals (integ8, integ9, integ1 in GKW)
    f1 = (
        dldt
        * signB
        * signJ
        * F
        * (1.0 / (rc * R_ * Bp * jnp.pi) - cosu / (R_**2 * Bp * jnp.pi))
        / (R_**2 * Bp)
    )
    f2 = dldt * signB * signJ * (F / (R_**2 * Bp**2) + 1.0 / F) / (2.0 * jnp.pi * R_**2 * Bp)
    f3 = dldt * signB * signJ * F / (R_**2 * Bp**3 * 2.0 * jnp.pi)

    cum_f1 = _simpson_cumulative(theta, f1)
    cum_f2 = _simpson_cumulative(theta, f2)
    cum_f3 = _simpson_cumulative(theta, f3)
    Fprime = (
        signB * signJ * span * q * shat / (eps * dpfdpsi)
        - (cum_f1[-1] - cum_f1[0])
        - (cum_f3[-1] - cum_f3[0]) * grdp
    ) / ((cum_f2[-1] - cum_f2[0]) * F)

    dzetadth = signB * signJ * F * dldt / (R_ * pf1 * 2.0 * jnp.pi)
    zeta1 = pf1 * (cum_f1 + cum_f2 * Fprime * F + cum_f3 * grdp)
    pf2 = 0.5 * (Bp * (cosu - R_ / rc) - R_**2 * grdp - F * Fprime)

    # s-direction derivatives: dsdth, dsdpsi via two more integrals
    gA = signJ * dldt * R_ / pf1
    gB = signJ * dldt * (cosu + R_ / rc - 2.0 * R_ * pf2 / pf1) / pf1**2
    cum_gA = _simpson_cumulative(theta, gA)
    cum_gB = _simpson_cumulative(theta, gB)
    total_gA = cum_gA[-1] - cum_gA[0]
    dsdth = signJ * R_ * dldt * span / (pf1 * total_gA)
    dsdpf = cum_gB * span / total_gA - (cum_gB[-1] - cum_gB[0]) * cum_gA * span / total_gA**2
    dzetadpsi = dzetadth * dldpsi / dldt + zeta1 * drhodpsi
    dsdpsi = dsdpf * dpfdpsi + dsdth * dldpsi / dldt

    # (psi, zeta, s) contravariant metric, upper-triangular entries
    m11 = g_pp
    m12 = dzetadpsi * g_pp + dzetadth * g_pt
    m13 = dsdth * g_pt + dsdpsi * g_pp
    m22 = (
        dzetadpsi**2 * g_pp
        + dzetadth**2 * g_tt
        + 1.0 / (R_**2 * 4.0 * jnp.pi**2)
        + 2.0 * dzetadpsi * dzetadth * g_pt
    )
    m23 = (
        dzetadpsi * dsdpsi * g_pp
        + dzetadth * dsdth * g_tt
        + (dsdpsi * dzetadth + dsdth * dzetadpsi) * g_pt
    )
    m33 = dsdpsi**2 * g_pp + dsdth**2 * g_tt + 2.0 * dsdpsi * dsdth * g_pt

    # B, R, Z derivatives along s (theta -> s coordinate change)
    dBdl = (
        0.5
        * (2.0 * pf1 * dpf1dl + 2.0 * sinu * pf1**2 / R_ - 2.0 * F**2 * dRt / (dldt * R_))
        / (bn * R_**2)
    )
    dBds_fine = dBdl * dldt / dsdth
    dBdrho = (
        0.5
        * (
            -2.0 * F**2 * cosu / R_**3
            + 2.0 * F * Fprime * Bp / R_
            - 2.0 * Bp**2 * cosu / R_
            + 4.0 * Bp * pf2 / R_
        )
        / bn
    )
    dBdpsi_fine = drhodpsi * dBdrho + dldpsi * dBdl - dBds_fine * dsdpsi

    dRds_fine = dRt / dsdth
    dZds_fine = dZt / dsdth
    dRdpsi_fine = dRp - dRds_fine * dsdpsi
    dZdpsi_fine = dZp - dZds_fine * dsdpsi

    def interp(y):
        return _interpquad(s_of_theta, y, sgrid)

    itheta0 = jnp.argmin(jnp.abs(theta))  # LFS (theta=0) for R0
    bn_out = interp(bn)
    R_out = interp(R_)
    dBds = interp(dBds_fine)
    dRds = interp(dRds_fine)
    dZds = interp(dZds_fine)
    dBdpsi = interp(dBdpsi_fine)
    dRdpsi = interp(dRdpsi_fine)
    dZdpsi = interp(dZdpsi_fine)
    metric = jnp.stack(
        [
            jnp.stack([interp(m11), interp(m12), interp(m13)], axis=-1),
            jnp.stack([interp(m12), interp(m22), interp(m23)], axis=-1),
            jnp.stack([interp(m13), interp(m23), interp(m33)], axis=-1),
        ],
        axis=1,
    )

    # finite-epsilon F, G (matches _circular_geometry finite_epsilon=True path)
    ffun = bups / bn_out
    gfun = ffun * dBds / bn_out

    return {
        "bn": bn_out,
        "R": R_out,
        "ffun": ffun,
        "gfun": gfun,
        "bt_frac": interp(Bt / bn),
        "bups": bups,
        "dpfdpsi": dpfdpsi,
        "metric": metric,
        "dzetadeps": metric[:, 0, 1],
        "dBdpsi": dBdpsi,
        "dBds": dBds,
        "dRdpsi": dRdpsi,
        "dRds": dRds,
        "dZdpsi": dZdpsi,
        "dZds": dZds,
        "finite_epsilon": True,
        "R0": R_[itheta0],
        "dR0dpsi": s["dRdpsi"][itheta0],
        "g_zz_mid": metric[len(sgrid) // 2, 1, 1],
    }
