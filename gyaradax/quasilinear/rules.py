"""Saturation rules: the canonical ql_flux and its sat0 / sat2 variants.

The variants use more of the linear eigenmode than the canonical γ/⟨k⊥²⟩ rule (ql_flux, the
default). All rules share the ql_flux call signature, are pure jax.numpy (differentiable,
jit-able), have one fitted amplitude `cn`, and are registered in RULES at the bottom.

ĝ = sigmoid(20γ)·relu(γ) is the smooth stability-gated growth rate and
W(ky) = Σ_kx Γ / Σ_kx |φ|² the normalization-invariant QL weight, both
identical to ql_flux.

  canonical      Q = C·Σ W·ĝ/⟨k⊥²⟩.  Plain mixing length, the reference.

  sat0           Q = C·Σ W·ky·(ĝ/⟨k⊥²⟩)².
                 Waltz mixing rule: saturated potential from the mixing
                 estimate φ_sat ~ γ/⟨k⊥²⟩ (zonal-flow/ExB shear balance
                 V_zf ~ γ/ky ⇒ |φ|² ~ (γ/k⊥²)²), intensity is the SQUARE,
                 and the flux carries the explicit ky factor of the ExB
                 drift v_E = i·ky·φ.

  sat2           SAT2 on the full eigenmode; see ql_flux_sat2.

Structural constants (SAT2's ay) are fixed by construction
and documented at their definition — they are NOT calibrated; each rule
exposes only cn.
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


@partial(jax.jit, static_argnames=("mask_zonal", "kx_ratio_sum"))
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
    kx_ratio_sum=False,
    kperp2_eff=None,
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
    if kperp2_eff is None:
        # averages all theta0 chains equally; pass kperp2_eff for the kx = 0 chain alone
        kperp2_eff = k_perp_eff_squared(phi2, k_perp_squared(krho, kxrh, little_g), ds)

    # geometric floor keeps γ/<k_⊥²> bounded for broad eigenmodes
    g_zz_min = jnp.min(little_g[0])
    safe_kperp2 = jnp.maximum(kperp2_eff, jnp.maximum(krho**2 * g_zz_min, eps))

    # smooth stability gate; relu keeps stable modes at exactly zero
    gate = jax.nn.sigmoid(gate_sharpness * (growth_rate - gate_threshold))
    sat_amp = gate * jax.nn.relu(growth_rate) / safe_kperp2

    # W(ky) = sum_kx flux / sum_kx |phi|^2; kx_ratio_sum sums per-kx ratios instead
    if kx_ratio_sum:
        w_kxy = flux_kxy / jnp.maximum(phi2_kxy, eps)
    else:
        w_kxy = (jnp.sum(flux_kxy, axis=0, keepdims=True)
                 / jnp.maximum(jnp.sum(phi2_kxy, axis=0, keepdims=True), eps))

    # mask the zonal mode by wavenumber, so grids without a ky = 0 column keep every mode
    ky_mask = (
        (jnp.abs(krho) > eps).astype(krho.dtype) if mask_zonal else jnp.ones_like(krho)
    )
    return cn * jnp.sum(w_kxy * sat_amp[None, :] * ky_mask[None, :])


# SAT2 spectral-shift weight ay
_SAT2_AY = 1.0

# zonal-mixing cz2 = 1.05 |alpha_zf| with alpha_zf = -1
_SAT2_CZ2 = 1.05

# floor of the gamma/ky peak search, so the lowest ky is not taken as the peak
_SAT2_KYMIN = 0.173 * 2.0**0.5

# cap of the gamma/ky peak search, 0.8 grad_r0 / rho_ion with rho_ion = 1
_SAT2_KYCUT = 0.8

# kx width model: b0 sets kycut, b1 the slope above it (grad_r0, gq from the geometry)
_SAT2_B0, _SAT2_B1 = 0.76, 1.22
_CKY = 3.0
_SQCKY = _CKY**0.5


# dlnpdr clamp [4, rlnp_cutoff]
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
    # jnp.clip: the arguments are tracers under the plugin's vmap
    return jnp.clip(num / (1.0 + ti_te), _DLNPDR_FLOOR, _DLNPDR_CUTOFF)


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


def _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps, kperp2_eff=None):
    """Eigenmode-weighted ⟨k⊥²⟩(ky) with the canonical geometric floor.

    `kperp2_eff` supplied skips the internal all-chain average, which is
    contaminated by the chain comb (it scales with ikxspace).
    """
    if kperp2_eff is None:
        kperp2_eff = k_perp_eff_squared(
            phi2, k_perp_squared(krho, kxrh, little_g), ds)
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
    kperp2_eff=None,
):
    """SAT0-style Waltz mixing rule: Q = C·Σ W·ky·(ĝ/⟨k⊥²⟩)²."""
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    safe_kperp2 = _safe_kperp2_eff(phi2, krho, kxrh, little_g, ds, eps,
                                   kperp2_eff=kperp2_eff)
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

    def quad(x):
        return f0 + b * x + c * x * x

    xmin = jnp.where(k0 < kymin, (kymin - k0) / dky, 0.0)
    xmax = jnp.clip(-b / jnp.where(jnp.abs(c) > eps, 2.0 * c, eps), 0.0, 1.0)

    # f0 >= f1: the discrete peak is not a local maximum, fall back to f0 floored at kymin
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
    kymin=None,
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
    """SAT2 flux Q = cn * sum_ky W(ky) I(ky)^p on the full linear eigenmode.

    I(ky) = (1/kymax) (12/dlnpdr) ((gamma_eff / (kx_width ky)) / (1 + ay kx_hat^2))^2, with kymax
    the quadratically refined peak of gamma/ky (floored at `kymin`, default 0.173 sqrt(2) grad_r0),
    gamma_eff the growth rate clamped above kymax and Lorentzian-averaged over high ky, and
    kx_width from SAT2's model (`kx_width_model="tglf"`) or measured from |phi|^2. b2 folds into cn
    and sat_geo is left out. `flux_exponent` p acts on each mode's intensity (1 is SAT2's own form;
    fitted values live in `calibrations`); `dlnpdr=None` drops 12/dlnpdr; `kthnorm` converts krho
    to k_theta rho.
    """
    ghat = _gated_gamma(growth_rate, gate_threshold, gate_sharpness)
    w = _w_ky(flux_kxy, phi2_kxy, eps)
    mask = _ky_mask(krho, mask_zonal, eps)
    # the thresholds are in k_theta rho; gyaradax's krho is k_theta rho / kthnorm
    ky = jnp.abs(krho) * kthnorm

    ky_floor = _SAT2_KYMIN * grad_r0 if kymin is None else kymin
    kymax, vzf, jmax = _zonal_mixing(ky, ghat, eps, kymin=ky_floor)
    gamma_eff = _gamma_kymix(ky, ghat, kymax, vzf, jmax, etg_stiff, eps)
    measure = 1.0 / kymax

    if kx_width_model == "tglf":
        kycut = _SAT2_B0 * kymax
        kx_width = jnp.where(
            ky < kycut,
            kycut / grad_r0,
            kycut / grad_r0 + _SAT2_B1 * (ky - kycut) * gq,
        )
    else:
        # measured width of |phi|^2 over all kx: follows the chain spacing, not the mode
        p_kx = phi2_kxy / jnp.maximum(jnp.sum(phi2_kxy, axis=0, keepdims=True), eps)
        kx_bar = jnp.sum(kxrh[:, None] * p_kx, axis=0)
        sig2_kx = jnp.sum((kxrh[:, None] - kx_bar[None, :]) ** 2 * p_kx, axis=0)
        kx_width = jnp.sqrt(jnp.maximum(sig2_kx, eps))

    # ExB spectral shift; kx0_e = 0 without rotation
    kx_hat = kx0_e * ky / kx_width

    denom = jnp.maximum(kx_width * ky, eps)

    cnorm = 1.0 if dlnpdr is None else 12.0 / jnp.maximum(dlnpdr, 4.0)
    intensity = (
        measure
        * cnorm
        * ((gamma_eff / denom) / (1.0 + _SAT2_AY * kx_hat**2)) ** 2
    )
    # exponent per mode keeps the rule linear in the flux; the where keeps stable modes at zero
    shaped = jnp.where(intensity > 0.0,
                       jnp.power(jnp.maximum(intensity, eps), flux_exponent), 0.0)
    return cn * jnp.sum(w * shaped * mask)


RULES = {
    "canonical": ql_flux,
    "sat0": ql_flux_sat0,
    "sat2": ql_flux_sat2,
}

# rules taking the pressure-gradient normalization dlnpdr
RULES_WITH_DLNPDR = ("sat2",)

# rules whose thresholds are in k_theta rho, so need kthnorm
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
