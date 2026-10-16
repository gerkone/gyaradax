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

  qualikiz       Q = C·Σ W·(ĝ/⟨k⊥²⟩)·S(ky),  S = 1/(1 + (ky/k0)³), k0 = 0.3.
                 QuaLiKiz-style: the γ-linear mixing-length intensity of the
                 canonical rule damped by an IMPOSED k_θ^-3 high-k envelope
                 — QuaLiKiz prescribes one universal saturated spectral
                 shape instead of letting the per-mode linear intensity set
                 it. k0 is the ITG spectral peak k_θρ_s, a structural
                 constant, not a fitted one.

Structural constants (k0, sharpness, SAT2's ay) are fixed by construction
and documented at their definition — they are NOT calibrated; each rule
exposes only cn.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .saturation import k_perp_eff_squared, k_perp_squared, ql_flux

# TGLF SAT2 spectral-shift weight, ay in tglf_multiscale_spectrum.f90
_SAT2_AY = 1.0


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

      kx_width  -> kperp2_eff(ky), the |phi|^2-weighted <k_perp^2> including
                   the kx centroid and width, which already carries the real
                   metric and so subsumes sat_geo_factor
      kx_hat    -> kx centroid in units of the kx width, the shift TGLF fits
      kymax     -> the zonal-mixing peak, argmax of gamma/ky, as in
                   get_zonal_mixing

    b2 is a pure constant and folds into cn. measure and 12/dlnpdr vary per
    run and do not, so they are kept; `dlnpdr` omitted drops that factor.
    """
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)
    ky = jnp.abs(krho)

    # zonal-mixing peak: the ky maximising gamma/ky (get_zonal_mixing)
    ratio = jnp.where(ky > eps, ghat / jnp.maximum(ky, eps), 0.0)
    kymax = jnp.maximum(ky[jnp.argmax(ratio)], eps)
    measure = 1.0 / kymax

    # kx centroid and width of |phi|^2(kx) per ky: TGLF's spectral shift and
    # kx_width, measured rather than fitted
    p_kx = phi2_kxy / jnp.maximum(jnp.sum(phi2_kxy, axis=0, keepdims=True), eps)
    kx_bar = jnp.sum(kxrh[:, None] * p_kx, axis=0)
    sig2_kx = jnp.sum((kxrh[:, None] - kx_bar[None, :]) ** 2 * p_kx, axis=0)
    kx_hat = kx_bar / jnp.sqrt(jnp.maximum(sig2_kx, eps))

    kperp2 = _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps)

    cnorm = 1.0 if dlnpdr is None else 12.0 / jnp.maximum(dlnpdr, 4.0)
    intensity = (
        measure
        * cnorm
        * ((ghat / kperp2) / (1.0 + _SAT2_AY * kx_hat**2)) ** 2
    )
    return cn * jnp.sum(w * intensity * mask)


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
    "sat0": ql_flux_sat0,
    "sat2": ql_flux_sat2,
    "qualikiz": ql_flux_qualikiz,
}

# rules whose signature accepts the optional per-ky mode frequency
RULES_WITH_FREQUENCY = ()


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
