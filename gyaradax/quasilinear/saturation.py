"""JAX-native QL saturation rule for gyaradax linear outputs.

Single canonical form (TGLF-flavored mixing length, γ-linear amplitude):

    Q_QL = C_n · Σ_{kx, ky>0}  W(kx, ky) · γ(ky) / <k_⊥²>(ky)

with
    W(kx, ky)    = Γ_lin(kx, ky) / |φ_lin(kx, ky)|²    quasilinear weight
                   (normalization-invariant ratio — robust to gyaradax's
                    per-ky renormalization of linear modes)
    <k_⊥²>(ky)   = ∫ds Σ_kx k_⊥² |φ|² / ∫ds Σ_kx |φ|²   eigenmode-weighted
    floor on <k_⊥²>: krho² · min(g_zz) keeps the rule well-conditioned for
                                      pathologically broad eigenmodes
    γ(ky)        stability-gated: sigmoid(20·γ) · γ for smooth differentiability

The γ-linear form was selected on a 50-sim sandbox: best Spearman on this
dataset (+0.95) and the simplest form that beats the torch-port baseline.
Per-channel and parametric-C_n extensions live in calibration.py.
"""

from functools import partial

import jax
import jax.numpy as jnp


def k_perp_squared(krho, kxrh, little_g):
    """k_⊥²(s, kx, ky) from geometry metric tensors.

    krho: (nky,), kxrh: (nkx,), little_g: (3, ns) = [g_zz, g_ez, g_ee].
    Returns (ns, nkx, nky).
    """
    g_zz = little_g[0][:, None, None]
    g_ez = little_g[1][:, None, None]
    g_ee = little_g[2][:, None, None]
    krho_t = krho[None, None, :]
    kxrh_t = kxrh[None, :, None]
    return krho_t**2 * g_zz + 2.0 * krho_t * kxrh_t * g_ez + kxrh_t**2 * g_ee


def k_perp_eff_squared(phi2, k_perp2, ds):
    """Eigenmode-weighted <k_⊥²>(ky): ∫ds Σ_kx k_⊥²|φ|² / ∫ds Σ_kx |φ|².

    phi2: (ns, nkx, nky), k_perp2: (ns, nkx, nky), ds: scalar.
    Returns (nky,).
    """
    norm = jnp.sum(phi2, axis=(0, 1)) * ds
    weighted = jnp.sum(k_perp2 * phi2, axis=(0, 1)) * ds
    return weighted / jnp.maximum(norm, 1e-30)


@partial(jax.jit, static_argnames=("mask_zonal",))
def ql_flux(
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
    """Compute Q_QL from a gyaradax linear-run end-state.

    Args:
        growth_rate : (nky,)            converged γ per ky.
        phi2        : (ns, nkx, nky)    |φ|² of the eigenmode (for <k_⊥²>).
        phi2_kxy    : (nkx, nky)        |φ|² integrated over s with `ints` weights.
        flux_kxy    : (nkx, nky)        per-mode linear flux (energy by default)
                                        from calculate_fluxes(..., reduce=False).
        krho        : (nky,)
        kxrh        : (nkx,)
        little_g    : (3, ns)           [g_zz, g_ez, g_ee].
        ds          : scalar            mean(ints).
        cn          : scalar amplitude calibration.
        gate_*      : smooth sigmoid on γ > gate_threshold.
        mask_zonal  : suppress ky=0 contribution.

    Returns: scalar Q_QL.
    """
    k_perp2 = k_perp_squared(krho, kxrh, little_g)
    kperp2_eff = k_perp_eff_squared(phi2, k_perp2, ds)

    # geometric floor keeps γ/<k_⊥²> bounded for broad eigenmodes
    g_zz_min = jnp.min(little_g[0])
    safe_kperp2 = jnp.maximum(kperp2_eff, jnp.maximum(krho**2 * g_zz_min, eps))

    # smooth stability gate, γ-linear amplitude
    gate = jax.nn.sigmoid(gate_sharpness * (growth_rate - gate_threshold))
    sat_amp = gate * growth_rate / safe_kperp2

    # normalization-invariant QL weight per (kx, ky)
    w_kxy = flux_kxy / jnp.maximum(phi2_kxy, eps)

    ky_mask = (
        (jnp.arange(krho.shape[0]) > 0).astype(krho.dtype) if mask_zonal else jnp.ones_like(krho)
    )
    return cn * jnp.sum(w_kxy * sat_amp[None, :] * ky_mask[None, :])


def ql_flux_diagnostics(
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
    """Same as ql_flux but returns intermediate quantities for inspection."""
    k_perp2 = k_perp_squared(krho, kxrh, little_g)
    kperp2_eff = k_perp_eff_squared(phi2, k_perp2, ds)
    g_zz_min = jnp.min(little_g[0])
    safe_kperp2 = jnp.maximum(kperp2_eff, jnp.maximum(krho**2 * g_zz_min, eps))
    gate = jax.nn.sigmoid(gate_sharpness * (growth_rate - gate_threshold))
    sat_amp = gate * growth_rate / safe_kperp2
    w_kxy = flux_kxy / jnp.maximum(phi2_kxy, eps)
    ky_mask = (
        (jnp.arange(krho.shape[0]) > 0).astype(krho.dtype) if mask_zonal else jnp.ones_like(krho)
    )
    contrib_kxy = w_kxy * sat_amp[None, :] * ky_mask[None, :]
    return {
        "Q_QL": cn * jnp.sum(contrib_kxy),
        "sat_amp": sat_amp,
        "w_kxy": w_kxy,
        "k_perp2_eff": kperp2_eff,
        "gate": gate,
        "contrib_kxy": contrib_kxy,
    }
