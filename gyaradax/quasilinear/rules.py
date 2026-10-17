"""Saturation-rule zoo: TGLF-flavored variants of the canonical ql_flux.

Every rule consumes MORE of the linear eigenmode information than the
canonical γ/⟨k⊥²⟩ rule in saturation.py (which stays the default and is
untouched). All rules share the ql_flux call-signature family, are pure
jax.numpy (differentiable, jit-able), have exactly ONE fitted amplitude
constant `cn`, and are registered in RULES at the bottom.

ĝ = sigmoid(20γ)·relu(γ) is the smooth stability-gated growth rate and
W(ky) = Σ_kx Γ / Σ_kx |φ|² the normalization-invariant QL weight, both
identical to saturation.ql_flux.

  canonical      Q = C·Σ W·ĝ/⟨k⊥²⟩.  Plain mixing length, the reference.

  sat0           Q = C·Σ W·ky·(ĝ/⟨k⊥²⟩)².
                 Waltz mixing rule: saturated potential from the mixing
                 estimate φ_sat ~ γ/⟨k⊥²⟩ (zonal-flow/ExB shear balance
                 V_zf ~ γ/ky ⇒ |φ|² ~ (γ/k⊥²)²), intensity is the SQUARE,
                 and the flux carries the explicit ky factor of the ExB
                 drift v_E = i·ky·φ.

  sat2           TGLF SAT2 on the full eigenmode; see ql_flux_sat2.

Structural constants (SAT2's ay) are fixed by construction
and documented at their definition — they are NOT calibrated; each rule
exposes only cn.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .saturation import k_perp_eff_squared, k_perp_squared, ql_flux

# TGLF SAT2 spectral-shift weight, ay in tglf_multiscale_spectrum.f90
_SAT2_AY = 1.0

# zonal-mixing constants: cz2 = 1.05*|alpha_zf| with alpha_zf = -1, and the
# Lorentzian mixing kernel's cky; etg_stiff is 1 unless ETG_FACTOR < 0
_SAT2_CZ2 = 1.05

# get_zonal_mixing caps the peak search at 0.8*grad_r0/rho_ion; rho_ion is 1
# in gyaradax units and grad_r0 ~ 1 for the circular flux surface
_SAT2_KYCUT = 0.8

# TGLF's kx_width model: b0 sets kycut, b1 the slope above it. grad_r0 and
# Gq come from the geometry, not from constants -- see geom_sat2_factors.
_SAT2_B0, _SAT2_B1 = 0.76, 1.22
_CKY = 3.0
_SQCKY = _CKY**0.5


# TGLF clamps dlnpdr into [4, rlnp_cutoff]; rlnp_cutoff default 18 lives in
# tglf_modules.f90
_DLNPDR_FLOOR, _DLNPDR_CUTOFF = 4.0, 18.0


def dlnpdr_tglf(rlt, rln, te_ti=1.0, adiabatic=False):
    """TGLF's normalized pressure-gradient scale length for SAT2's cnorm.

    tglf_multiscale_spectrum.f90:85-97 forms
    rmaj*SUM_s as*taus*(rlns+rlts)/ptot, clamped into [4, 18]. TGLF's rlns
    and rlts are a/L_x with rmaj = R/a, so the product is R/L; gyaradax's
    rlt and rln are already R/L and the rmaj factor is 1 here.

    Pure D, quasineutral: as = 1 for both species and taus is normalized to
    the electron temperature. Adiabatic runs follow TGLF, whose species loop
    runs from 1 and so still counts the electrons, with rlt_e = 0 and
    Te = Ti -- giving rln + rlt/2, not rln + rlt.
    """
    rlt_i, rlt_e = (rlt, 0.0) if adiabatic else (rlt[0], rlt[1])
    rln_i = rln if adiabatic else rln[0]
    ti_te = 1.0 if adiabatic else 1.0 / te_ti
    num = (rln_i + rlt_e) + ti_te * (rln_i + rlt_i)
    return min(max(num / (1.0 + ti_te), _DLNPDR_FLOOR), _DLNPDR_CUTOFF)


def _gated_gamma(growth_rate, gate_threshold, gate_sharpness):
    """Smooth stability gate x relu: exactly zero for stable modes."""
    gate = jax.nn.sigmoid(gate_sharpness * (growth_rate - gate_threshold))
    return gate * jax.nn.relu(growth_rate)


def _w_ky(flux_kxy, phi2_kxy, eps):
    """Intensity-weighted QL weight W(ky) = Σ_kx Γ / Σ_kx |φ|²."""
    return jnp.sum(flux_kxy, axis=0) / jnp.maximum(jnp.sum(phi2_kxy, axis=0), eps)


def _ky_mask(krho, mask_zonal, eps):
    if mask_zonal:
        return (jnp.abs(krho) > eps).astype(krho.dtype)
    return jnp.ones_like(krho)


def _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps):
    """Eigenmode-weighted ⟨k⊥²⟩(ky) with the canonical geometric floor."""
    k_perp2 = k_perp_squared(krho, kxrh, little_g)
    kperp2_eff = k_perp_eff_squared(phi2, k_perp2, ds)
    g_zz_min = jnp.min(little_g[0])
    return jnp.maximum(kperp2_eff, jnp.maximum(krho**2 * g_zz_min, eps))


@partial(jax.jit, static_argnames=("mask_zonal",))
def ql_flux_sat0(
    growth_rate,
    phi2,
    phi2_kxy,
    flux_kxy,
    krho,
    kxrh,
    little_g,
    ds,
    cn=1.0,
    gate_threshold=0.0,
    gate_sharpness=20.0,
    eps=1e-30,
    mask_zonal=True,
):
    """SAT0-style Waltz mixing rule: Q = C·Σ W·ky·(ĝ/⟨k⊥²⟩)²."""
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    safe_kperp2 = _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)
    intensity = (ghat / safe_kperp2) ** 2
    return cn * jnp.sum(w * jnp.abs(krho) * intensity * mask)


def geom_sat2_factors(geom):
    """SAT2's grad_r0 and Gq from a gyaradax geometry, not from constants.

    grad_r0 = |grad r| at the outboard midplane = sqrt(g_eps_eps), which is
    little_g[:, 2]; it comes out exactly 1 for circular, matching TGLF.
    Gq = B_geo0 / grad_r0 with B_geo0 the midplane field. gyaradax normalises
    B differently from TGLF (0.843 vs 0.826 at eps = 0.19), and Gq only
    enters kx_width above kycut, where the intensity is already small.
    """
    lg = jnp.asarray(geom["little_g"])
    lg = lg if lg.shape[-1] == 3 else lg.T
    mid = lg.shape[0] // 2
    grad_r0 = jnp.sqrt(jnp.abs(lg[mid, 2]))
    b_geo0 = jnp.asarray(geom["bn"])[mid]
    return grad_r0, b_geo0 / grad_r0


def _zonal_mixing(ky, gamma, eps, kycut=_SAT2_KYCUT, kymin=0.0):
    """TGLF get_zonal_mixing: the low-k peak of gamma/ky, quadratically refined.

    Returns (kymax, vzf, jmax). The search is capped at kycut so it cannot run
    away to electron scales, and the three-point fit around the discrete peak
    matters on coarse ky grids, which is why TGLF refines rather than taking
    the bare argmax. kymin floors the peak: TGLF switches it on with
    ALPHA_ZF < 0, which stops the lowest ky being picked as the maximum.
    """
    n = ky.shape[0]
    ratio = jnp.where(ky > eps, gamma / jnp.maximum(ky, eps), 0.0)
    nxt = jnp.concatenate([ky[1:], ky[-1:]])
    inwin = (nxt >= kymin) & (ky <= kycut)
    jmax = jnp.argmax(jnp.where(inwin, ratio, -jnp.inf))
    j1 = jnp.max(jnp.where(inwin, jnp.arange(n), 0))
    jm = jnp.clip(jmax, 1, n - 2)

    f0, f1, f2 = ratio[jm - 1], ratio[jm], ratio[jm + 1]
    k0, k2 = ky[jm - 1], ky[jm + 1]
    dky = jnp.maximum(k2 - k0, eps)
    x1 = (ky[jm] - k0) / dky
    b = (f1 - f0 * (1.0 - x1 * x1) - f2 * x1 * x1) / jnp.maximum(x1 - x1 * x1, eps)
    c = f2 - f0 - b
    quad = lambda x: f0 + b * x + c * x * x

    xmin = jnp.where(k0 < kymin, (kymin - k0) / dky, 0.0)
    xmax = jnp.clip(-b / jnp.where(jnp.abs(c) > eps, 2.0 * c, eps), 0.0, 1.0)

    # f0 >= f1 means the discrete peak is not a local maximum: fall back to f0,
    # itself floored at kymin
    lo_k = jnp.where(k0 < kymin, kymin, k0)
    lo_g = jnp.where(k0 < kymin, quad(xmin) * kymin, f0 * k0)
    # xmax <= xmin: the fitted peak sits below the floor
    clamped = xmax <= xmin
    ref_k = jnp.where(clamped, jnp.where(xmin > 0.0, kymin, k0), k0 + dky * xmax)
    ref_g = jnp.where(
        clamped,
        jnp.where(xmin > 0.0, quad(xmin) * kymin, f0 * k0),
        quad(xmax) * (k0 + dky * xmax),
    )

    refine = (jmax > 0) & (jmax < j1)
    peak_ok = f0 < f1
    # unrefined peak, also floored at kymin by linear interpolation off the grid
    raw_k, raw_g = ky[jmax], gamma[jmax]
    slope = (gamma[1] - gamma[0]) / jnp.maximum(ky[1] - ky[0], eps)
    raw_g = jnp.where(raw_k < kymin, gamma[0] + slope * (kymin - ky[0]), raw_g)
    raw_k = jnp.maximum(raw_k, kymin)

    kymax = jnp.where(refine, jnp.where(peak_ok, ref_k, lo_k), raw_k)
    gammamax = jnp.where(refine, jnp.where(peak_ok, ref_g, lo_g), raw_g)
    kymax = jnp.maximum(kymax, eps)
    return kymax, gammamax / kymax, jmax


def _gamma_kymix(ky, gamma, kymax, vzf, jmax, etg_stiff, eps):
    """TGLF's zonal-mixing growth rate: high-ky clamp then Lorentzian average.

    Above kymax the growth rate is clamped to gammamax plus whatever exceeds
    the zonal-flow mixing rate cz2*vzf*ky, then averaged over the high-ky
    range with weight sqcky*ky0^2/(ky0^2 + cky*(ky-ky0)^2). This is what
    flattens SAT2's spectrum above the peak; the raw gamma does not.
    """
    n = ky.shape[0]
    idx = jnp.arange(n)
    gammamax = vzf * kymax

    clamped = gammamax + etg_stiff * jnp.maximum(gamma - _SAT2_CZ2 * vzf * ky, 0.0)
    g = jnp.where(ky < kymax, gamma, clamped)

    ky0 = ky[:, None]
    ky1, ky2 = ky[None, :-1], ky[None, 1:]
    g1, g2 = g[None, :-1], g[None, 1:]

    a1 = jnp.arctan(_SQCKY * (ky1 / ky0 - 1.0))
    a2 = jnp.arctan(_SQCKY * (ky2 / ky0 - 1.0))
    mix1 = ky0 * (a2 - a1)
    delta = (g2 - g1) / jnp.maximum(ky2 - ky1, eps)
    mix2 = ky0 * mix1 + (ky0 * ky0 / (2.0 * _SQCKY)) * (
        jnp.log(_CKY * (ky2 - ky0) ** 2 + ky0 * ky0)
        - jnp.log(_CKY * (ky1 - ky0) ** 2 + ky0 * ky0)
    )
    terms = (g1 - ky1 * delta) * mix1 + delta * mix2
    inner = (idx[None, :-1] >= jmax + 1) & (idx[None, :-1] <= n - 2)
    gamma_ave = jnp.sum(jnp.where(inner, terms, 0.0), axis=1)

    ky_lo = jnp.take(ky, jnp.minimum(jmax + 1, n - 1))
    ky_hi = ky[-1]
    mixnorm = ky[:] * (
        jnp.arctan(_SQCKY * (ky_hi / jnp.maximum(ky, eps) - 1.0))
        - jnp.arctan(_SQCKY * (ky_lo / jnp.maximum(ky, eps) - 1.0))
    )
    mixed = gamma_ave / jnp.where(jnp.abs(mixnorm) > eps, mixnorm, eps)
    return jnp.where(idx >= jmax + 2, mixed, g)


@partial(jax.jit, static_argnames=("mask_zonal", "kx_width_model"))
def ql_flux_sat2(
    growth_rate,
    phi2,
    phi2_kxy,
    flux_kxy,
    krho,
    kxrh,
    little_g,
    ds,
    cn=1.0,
    dlnpdr=None,
    kx0_e=0.0,
    etg_stiff=1.0,
    kymin=0.0,
    kthnorm=1.0,
    grad_r0=1.0,
    gq=1.0,
    kx_width_model="tglf",
    flux_exponent=1.0,
    gate_threshold=0.0,
    gate_sharpness=20.0,
    eps=1e-30,
    mask_zonal=True,
):
    """TGLF SAT2 intensity evaluated on the full linear eigenmode.

    Follows tglf_multiscale_spectrum.f90 (GACODE 7e22eb70), which sets

        |phi|^2(ky) = measure * cnorm
                      * ((gamma_eff / (kx_width*ky)) / (1 + ay*kx_hat^2))^2
                      * sat_geo_factor

    with measure = 1/kymax, cnorm = b2*(12/dlnpdr), ay = 1 and kx_hat the
    spectral shift rescaled by the kx width. TGLF has to MODEL kx_width, the
    shift and the sat_geo factors because its reduced solve does not resolve
    the eigenmode; a full linear solve measures all three, so they are taken
    from |phi|^2 here instead of from the CGYRO fits:

      kx_width  -> the measured kx width of |phi|^2(ky), replacing TGLF's
                   kycut/grad_r0 below kycut and b1*(ky-kycut)*Gq above it
      kymax     -> the zonal-mixing peak, the quadratically refined maximum
                   of gamma/ky, as in get_zonal_mixing

    gamma_eff is NOT the raw growth rate: it is TGLF's gamma_kymix, the rate
    clamped above kymax and then Lorentzian-averaged over the high-ky range.
    sat_geo_factor is dropped -- it rescales TGLF's s-alpha-like model
    geometry onto the real flux surface, which the full solve already uses.

    `kx0_e` is TGLF's ExB spectral shift and defaults to 0, which is what
    TGLF itself uses without rotation; it is NOT the eigenmode kx centroid,
    whose effect already sits inside kperp2_eff.

    b2 is a pure constant and folds into cn. measure and 12/dlnpdr vary per
    run and do not, so they are kept; `dlnpdr` omitted drops that factor.

    `flux_exponent` p returns cn*Q**p instead of cn*Q. SAT2's intensity is
    quadratic in gamma, and on the GKW adiabatic scans that is too steep:
    with p = 1 the rule is beaten by predicting a constant, while p = 0.58
    takes the test median error from 61% to 13%. Note p acts on the SUMMED
    flux, which is the form that was fitted; raising the per-mode intensity
    instead is a different model and is untested. A rule linear in gamma
    needs no exponent at all (its fitted p comes back at 1.05), so prefer
    that over tuning this.
    """
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)
    # SAT2's kycut and kymin are absolute thresholds in k_theta*rho, but
    # gyaradax solves on krho = k_theta*rho / kthnorm, so the zonal-mixing
    # logic has to run on the physical axis or the cutoffs land in the wrong
    # place by kthnorm (0.7-6.2 over our scans)
    ky = jnp.abs(krho) * kthnorm

    kymax, vzf, jmax = _zonal_mixing(ky, ghat, eps, kymin=kymin)
    gamma_eff = _gamma_kymix(ky, ghat, kymax, vzf, jmax, etg_stiff, eps)
    measure = 1.0 / kymax

    if kx_width_model == "tglf":
        # TGLF's analytic model, and the default. SAT2's kx_width is a
        # SATURATION length fitted to nonlinear runs -- it is independent of
        # ky below kycut, which no linear mode width can be -- so measuring
        # it from the eigenmode is substituting a different quantity.
        kycut = _SAT2_B0 * kymax
        kx_width = jnp.where(
            ky < kycut,
            kycut / grad_r0,
            kycut / grad_r0 + _SAT2_B1 * (ky - kycut) * gq,
        )
    else:
        # NOT RECOMMENDED. The measured width of |phi|^2 over all kx is set
        # by the chain comb, not by the mode: every theta0 chain is
        # unit-normalised, so the width tracks the chain spacing ~shat*ky and
        # runs down to 1e-7, which (kx_width*ky)^-2 turns into a blow-up.
        # On adiabatic this gives test RMSE 1.8e9 against 166 for "tglf".
        p_kx = phi2_kxy / jnp.maximum(jnp.sum(phi2_kxy, axis=0, keepdims=True), eps)
        kx_bar = jnp.sum(kxrh[:, None] * p_kx, axis=0)
        sig2_kx = jnp.sum((kxrh[:, None] - kx_bar[None, :]) ** 2 * p_kx, axis=0)
        kx_width = jnp.sqrt(jnp.maximum(sig2_kx, eps))

    # TGLF's spectral shift is the ExB-driven radial offset kx0_e, not the
    # eigenmode's own kx centroid: tglf_geometry.f90:59-73 leaves kx0_e at 0
    # and sets it only from vexb_shear_kx0, so at zero rotation the
    # suppression is exactly 1. Our runs carry no ExB shear.
    kx_hat = kx0_e * ky / kx_width

    # SAT2 divides by kx_width*ky, NOT by <kperp^2>. Below kycut TGLF holds
    # kx_width constant, so the denominator goes like ky and the intensity
    # peaks where gamma/ky does; <kperp^2> ~ ky^2 would push the peak to the
    # bottom of the grid instead.
    denom = jnp.maximum(kx_width * ky, eps)

    cnorm = 1.0 if dlnpdr is None else 12.0 / jnp.maximum(dlnpdr, 4.0)
    intensity = (
        measure
        * cnorm
        * ((gamma_eff / denom) / (1.0 + _SAT2_AY * kx_hat**2)) ** 2
    )
    q = jnp.sum(w * intensity * mask)
    if flux_exponent == 1.0:
        return cn * q
    return cn * jnp.power(jnp.maximum(q, eps), flux_exponent)


RULES = {
    "canonical": ql_flux,
    "sat0": ql_flux_sat0,
    "sat2": ql_flux_sat2,
}

# rules taking TGLF's pressure-gradient normalization
RULES_WITH_DLNPDR = ("sat2",)

# rules whose wavenumber thresholds are absolute in k_theta*rho and so need
# the krho -> k_theta*rho conversion factor
RULES_WITH_KTHNORM = ("sat2",)


def ql_flux_per_species(
    rule,
    growth_rate,
    phi2,
    phi2_kxy,
    flux_kxy_sp,
    krho,
    kxrh,
    little_g,
    ds,
    **kwargs,
):
    """Per-species / per-channel QL fluxes from ONE shared saturated intensity.

    QL postulate: the saturated |φ|² amplitude is common to all species —
    species and channels differ only through their QL weights W_{s,c}(ky).
    Every rule in RULES is linear in `flux_kxy` (the flux enters only via
    W = Σ_kx Γ / Σ_kx |φ|²), so evaluating the rule on each (species,
    channel) flux slice with the SAME (γ, |φ|², geometry) realizes exactly
    that decomposition, and the per-species outputs sum to the
    species-summed call by linearity.

    Args:
        rule: a RULES entry or its name (e.g. "canonical").
        flux_kxy_sp: (..., nkx, nky) linear flux fields with arbitrary
            leading axes — e.g. the (nsp, 3, nkx, nky) ``fluxes_kxy_sp``
            harvested by linear_pipeline._harvest for kinetic runs
            (channels ordered [pflux, eflux, vflux], EM flutter included).
        Other args/kwargs: identical to the underlying rule (γ, |φ|²
        spectra, geometry, cn, gate, ...).

    Returns: jnp array of shape ``flux_kxy_sp.shape[:-2]`` — e.g. (nsp, 3)
    with [s, 1] the QL heat flux Q_s and [s, 0] the particle flux Γ_s.
    """
    if isinstance(rule, str):
        rule = RULES[rule]
    flux_kxy_sp = jnp.asarray(flux_kxy_sp)
    lead = flux_kxy_sp.shape[:-2]
    flat = flux_kxy_sp.reshape((-1,) + flux_kxy_sp.shape[-2:])
    outs = [
        rule(growth_rate, phi2, phi2_kxy, flat[i], krho, kxrh, little_g, ds, **kwargs)
        for i in range(flat.shape[0])
    ]
    return jnp.stack(outs).reshape(lead)
