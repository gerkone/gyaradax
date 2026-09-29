"""Saturation-rule zoo: TGLF-SAT-flavored variants of the canonical ql_flux.

Every rule consumes MORE of the linear eigenmode information than the
canonical γ/⟨k⊥²⟩ rule in saturation.py (which stays the default and is
untouched). All rules share the ql_flux call-signature family, are pure
jax.numpy (differentiable, jit-able), have exactly ONE fitted amplitude
constant `cn`, and are registered in RULES at the bottom.

Chosen functional forms (one line each; ĝ = sigmoid(20γ)·relu(γ) is the
smooth stability-gated growth rate and W(ky) = Σ_kx Γ / Σ_kx |φ|² the
normalization-invariant QL weight, both identical to saturation.ql_flux):

  sat0_waltz     Q = C·Σ W·ky·(ĝ/⟨k⊥²⟩)²
                 Waltz mixing rule: saturated potential from the mixing
                 estimate φ_sat ~ γ/⟨k⊥²⟩ (zonal-flow/ExB shear balance
                 V_zf ~ γ/ky ⇒ |φ|² ~ (γ/k⊥²)²), intensity is the SQUARE,
                 and the flux carries the explicit ky factor of the ExB
                 drift v_E = i·ky·φ.

  sat1_zonal     Q = C·Σ W·ĝ/⟨k⊥²⟩ · D_zf(ky),
                 D_zf = ĝ²/(ĝ² + α²·ω_zf²),  ω_zf = ky·V_zf,
                 V_zf² = Σ_kx kx²·|φ_zf(kx)|² / Σ_{kx,ky} |φ|².
                 Zonal-flow mixing: the canonical intensity is quenched by
                 the ratio of a zonal ExB-shear mixing rate ω_zf (built from
                 the RELATIVE zonal content of the harvested spectrum, so it
                 is amplitude-invariant) to the drive γ — a Diamond-style
                 γ_eff = γ²/(γ² + ω_zf²) predator-prey quench. α = 1 is a
                 structural (non-fitted) constant. If the linear end-state
                 retains no ky=0 amplitude, D_zf → 1 and sat1 → canonical.

  sat2_spectral  Q = C·Σ W·ĝ/k⊥²_eff(ky) · G_∥(ky),
                 k⊥²_eff = ky²·⟨g_zz⟩_φ + 2ky·k̄x·⟨g_ez⟩_φ + (k̄x² + σ_kx²)·⟨g_ee⟩_φ,
                 G_∥ = 1/√(1 + σ_s²/ℓ0²),  ℓ0 = 1/√12.
                 Spectral-shift + mode-width: k̄x(ky) and σ_kx(ky) are the
                 |φ|²(kx)-weighted centroid and width (TGLF SAT2 spectral
                 shift), the metric coefficients are |φ|²(s)-weighted, and
                 the variance-based parallel width σ_s of |φ|²(s) penalizes
                 parallel-extended (slab-like) modes; ℓ0 is the σ_s of a
                 uniform distribution on the unit field line, so G_∥ ∈
                 (1/√2, 1] with G_∥ = 1 for a perfectly localized mode.

  sat3_regime    Q = C·Σ W·ky·Y(ω)·(ĝ/⟨k⊥²⟩)²,
                 Y = y_TEM + (1 − y_TEM)·sigmoid(20·ω),  y_TEM = 0.5.
                 Simplified SAT3: saturated-potential intensity (γ/⟨k⊥²⟩)²
                 with an ITG/TEM regime split on the sign of the linear mode
                 frequency. GKW convention (gkw_ref/doc/manual/
                 diagnostics.tex:31): ω > 0 propagates in the ion
                 diamagnetic direction (ITG-like), ω < 0 in the electron
                 direction (TEM/ETG-like); the electron branch gets the
                 fixed multiplier y_TEM = 0.5 mimicking SAT3's lower
                 saturated potential for electron-direction modes. With
                 frequency=None (not harvested) the split is skipped
                 (Y ≡ 1) and sat3 reduces exactly to sat0.

  qualikiz       Q = C·Σ W·(ĝ/⟨k⊥²⟩)·S(ky),  S = 1/(1 + (ky/k0)³), k0 = 0.3.
                 QuaLiKiz-style: the γ-linear mixing-length intensity of the
                 canonical rule damped by an IMPOSED k_θ^-3 high-k envelope
                 — QuaLiKiz prescribes one universal saturated spectral
                 shape instead of letting the per-mode linear intensity set
                 it. k0 is the ITG spectral peak k_θρ_s, a structural
                 constant like α, ℓ0 and y_TEM, not a fitted one.

Structural constants (α, ℓ0, y_TEM, sharpness) are fixed by construction
and documented above — they are NOT calibrated; each rule exposes only cn.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .saturation import k_perp_eff_squared, k_perp_squared, ql_flux


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


@partial(jax.jit, static_argnames=("mask_zonal",))
def ql_flux_sat1(
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
    alpha_zf=1.0,
):
    """SAT1-style zonal-flow mixing: canonical intensity x D_zf quench."""
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    safe_kperp2 = _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)

    # amplitude-invariant zonal shear content: kx²-weighted ky=0 over total intensity
    zonal_sel = (jnp.abs(krho) <= eps).astype(phi2_kxy.dtype)
    p_tot = jnp.maximum(jnp.sum(phi2_kxy), eps)
    vzf2 = jnp.sum(phi2_kxy * (kxrh**2)[:, None] * zonal_sel[None, :]) / p_tot
    omega_zf = krho * jnp.sqrt(vzf2)
    d_zf = ghat**2 / (ghat**2 + alpha_zf**2 * omega_zf**2 + eps)

    return cn * jnp.sum(w * ghat / safe_kperp2 * d_zf * mask)


@partial(jax.jit, static_argnames=("mask_zonal",))
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
    gate_threshold=0.0,
    gate_sharpness=20.0,
    eps=1e-30,
    mask_zonal=True,
):
    """SAT2-style spectral shift + mode width in the effective k⊥²."""
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)

    # kx centroid and width of |φ|²(kx) per ky (TGLF SAT2 spectral shift)
    p_kx = phi2_kxy / jnp.maximum(jnp.sum(phi2_kxy, axis=0, keepdims=True), eps)
    kx_bar = jnp.sum(kxrh[:, None] * p_kx, axis=0)
    sig2_kx = jnp.sum((kxrh[:, None] - kx_bar[None, :]) ** 2 * p_kx, axis=0)

    # |φ|²(s)-weighted metrics + parallel width; s is the index-based coordinate i·ds
    phi2_s = jnp.sum(phi2, axis=1)
    q_s = phi2_s / jnp.maximum(jnp.sum(phi2_s, axis=0, keepdims=True), eps)
    ns = phi2.shape[0]
    s_coord = (jnp.arange(ns, dtype=phi2.dtype) - 0.5 * (ns - 1)) * ds
    s_bar = jnp.sum(s_coord[:, None] * q_s, axis=0)
    sig2_s = jnp.sum((s_coord[:, None] - s_bar[None, :]) ** 2 * q_s, axis=0)
    gzz_bar = jnp.sum(little_g[0][:, None] * q_s, axis=0)
    gez_bar = jnp.sum(little_g[1][:, None] * q_s, axis=0)
    gee_bar = jnp.sum(little_g[2][:, None] * q_s, axis=0)

    kperp2_eff = (
        krho**2 * gzz_bar
        + 2.0 * krho * kx_bar * gez_bar
        + (kx_bar**2 + sig2_kx) * gee_bar
    )
    g_zz_min = jnp.min(little_g[0])
    safe_kperp2 = jnp.maximum(kperp2_eff, jnp.maximum(krho**2 * g_zz_min, eps))

    # ℓ0 = 1/√12: σ_s of a uniform |φ|²(s) on the unit field line
    ell0_sq = 1.0 / 12.0
    g_par = 1.0 / jnp.sqrt(1.0 + sig2_s / ell0_sq)

    return cn * jnp.sum(w * ghat / safe_kperp2 * g_par * mask)


@partial(jax.jit, static_argnames=("mask_zonal",))
def ql_flux_sat3(
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
    frequency=None,
    y_tem=0.5,
    freq_sharpness=20.0,
):
    """Simplified SAT3: (ĝ/⟨k⊥²⟩)² intensity with an ITG/TEM regime split.

    frequency: (nky,) linear mode frequency in GKW convention (ω > 0 = ion
    diamagnetic / ITG-like), or None to skip the split (Y ≡ 1, = sat0).
    """
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    safe_kperp2 = _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)

    if frequency is None:
        y_regime = jnp.ones_like(krho)
    else:
        y_regime = y_tem + (1.0 - y_tem) * jax.nn.sigmoid(freq_sharpness * frequency)

    intensity = (ghat / safe_kperp2) ** 2
    return cn * jnp.sum(w * jnp.abs(krho) * y_regime * intensity * mask)


@partial(jax.jit, static_argnames=("mask_zonal",))
def ql_flux_qualikiz(
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
    ky_peak=0.3,
    spectral_exponent=3.0,
):
    """QuaLiKiz-style rule: Q = C·Σ W·(ĝ/⟨k⊥²⟩)·S(ky), S = 1/(1 + (ky/k0)^p)."""
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    safe_kperp2 = _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)
    intensity = ghat / safe_kperp2
    envelope = 1.0 / (1.0 + (jnp.abs(krho) / ky_peak) ** spectral_exponent)
    return cn * jnp.sum(w * intensity * envelope * mask)


RULES = {
    "canonical": ql_flux,
    "sat0_waltz": ql_flux_sat0,
    "sat1_zonal": ql_flux_sat1,
    "sat2_spectral": ql_flux_sat2,
    "sat3_regime": ql_flux_sat3,
    "qualikiz": ql_flux_qualikiz,
}

# rules whose signature accepts the optional per-ky mode frequency
RULES_WITH_FREQUENCY = ("sat3_regime",)


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
        spectra, geometry, cn, gate, frequency for sat3, ...).

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
