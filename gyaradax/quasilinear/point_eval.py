"""Evaluate the QL model at one operating point (params + geometry).

jit/vmap-safe, the path the TORAX gyaradax-ql model takes: `initial_df`, `linear_early_stop`,
`ql_at_point`; `nl_at_point` for the nonlinear ground truth.
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from gyaradax.diagnostics.fluxes import calculate_fluxes
from gyaradax.operators.poisson import geom_tensors
from gyaradax.operators.precompute import linear_precompute
from gyaradax.solver.rk4 import default_state, gksolve

from .rules import (
    RULES,
    RULES_WITH_DLNPDR,
    RULES_WITH_KTHNORM,
    geom_sat2_factors,
    dlnpdr_tglf,
    ql_flux_per_species,
)

DF_SEED_AMPLITUDE = 1e-3


def initial_df(
    nvpar, nmu, ns, nkx, nky, amplitude=DF_SEED_AMPLITUDE, nsp=0
) -> jnp.ndarray:
    """Initial df: cosine in s, every non-zonal ky seeded (bypasses init_f's .item()).

    nsp > 0 prepends a species axis for the kinetic 6D grid.
    """
    s_grid = (jnp.arange(ns) + 0.5) / ns - 0.5
    seed_s = amplitude * (jnp.cos(2 * jnp.pi * s_grid) + 1.0)
    df = jnp.zeros((nvpar, nmu, ns, nkx, nky), dtype=jnp.complex128)
    seed = jnp.broadcast_to(
        seed_s[None, None, :, None, None] / (nkx * (nky - 1)),
        (nvpar, nmu, ns, nkx, nky),
    )
    ky_mask = jnp.arange(nky) > 0
    df = df + seed * ky_mask[None, None, None, None, :]
    if nsp:
        df = jnp.broadcast_to(df[None], (nsp,) + df.shape)
    return df


def linear_early_stop(
    df,
    geom,
    params,
    sim_state,
    pre,
    *,
    block=100,
    max_steps=2000,
    min_steps=200,
    atol=1e-4,
    rtol=1e-3,
    patience=2,
    return_blocks=False,
):
    """Chunked linear gksolve, exiting once per-ky growth rates converge.

    jit/vmap-safe (lax.while_loop; under vmap it runs until all batch
    elements converge). Patience guards against slowly decaying transients
    that move little between adjacent blocks. `return_blocks` additionally
    returns the traced number of blocks actually run.
    """
    block = int(block)
    max_blocks = max(int(max_steps) // block, 1)
    min_blocks = max(int(min_steps) // block, 1)

    df1, (phi1, _flx), sim1 = gksolve(df, geom, params, sim_state, n_steps=block, pre=pre)
    g0 = sim1.last_growth_rate

    def cond(state):
        i, _df, _phi, _sim, _prev, streak = state
        return jnp.logical_and(i < max_blocks, streak < patience)

    def body(state):
        i, df_c, _phi_c, sim_c, prev_g, streak = state
        df_n, (phi_n, _f), sim_n = gksolve(df_c, geom, params, sim_c, n_steps=block, pre=pre)
        new_g = sim_n.last_growth_rate
        ok = jnp.logical_and(
            i + 1 >= min_blocks,
            jnp.max(jnp.abs(new_g - prev_g)) <= atol + rtol * jnp.max(jnp.abs(new_g)),
        )
        return (i + 1, df_n, phi_n, sim_n, new_g, jnp.where(ok, streak + 1, 0))

    init = (jnp.asarray(1), df1, phi1, sim1, g0, jnp.asarray(0))
    n_blocks, df_f, phi_f, sim_f, _g, _s = jax.lax.while_loop(cond, body, init)
    if return_blocks:
        return df_f, phi_f, sim_f, n_blocks
    return df_f, phi_f, sim_f


def ql_at_point(
    params,
    geom,
    grid_shape: Tuple[int, int, int, int, int],
    *,
    cn,
    rule="canonical",
    rule_options=None,
    n_steps_linear=2000,
    early_stop=True,
    early_stop_opts: Dict[str, Any] | None = None,
    qi_clip=1e3,
    qi_tiny=1e-8,
    return_diagnostics=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear solve + saturation rule at one point -> (qi, qe, pfe) in GKW GB units.

    `rule` names an entry of quasilinear.rules.RULES; each has its own cn.
    `rule_options` is forwarded to it, so rule-specific knobs need no
    plumbing through the callers.

    Electromagnetic (params.nlapar) runs convert the mixed g to physical f
    for diagnostics and add the A_par flutter flux into the QL weight,
    matching the calibration X stage.

    Kinetic runs take qe and pfe from ql_flux_per_species, which evaluates
    the rule per (species, channel) against one shared saturated intensity.
    Adiabatic runs have no electron channel, so qe = qi and pfe = 0.

    `return_diagnostics` appends a dict of traced per-call solver diagnostics
    (blocks and steps actually run, converged growth rates).
    """
    spec, n_blocks, n_steps, gamma, flux_kxy_sp = _linear_harvest(
        params, geom, grid_shape, n_steps_linear=n_steps_linear,
        early_stop=early_stop, early_stop_opts=early_stop_opts,
    )
    extra = ({"dlnpdr": _dlnpdr_from_params(params)}
             if rule in RULES_WITH_DLNPDR else {})
    if rule in RULES_WITH_KTHNORM:
        extra["kthnorm"] = jnp.asarray(geom.get("kthnorm", 1.0), dtype=jnp.float64)
        extra["grad_r0"], extra["gq"] = geom_sat2_factors(geom)
    # caller-supplied knobs win, so a rule's own options need no plumbing here
    extra.update(dict(rule_options or {}))

    def _clean(q, lo=0.0):
        q = jnp.where(jnp.isfinite(q), jnp.clip(q, lo, qi_clip), 0.0)
        # zero below noise floor so linear transients don't feed TORAX gradients
        return jnp.where(jnp.abs(q) < qi_tiny, 0.0, q)

    if flux_kxy_sp is not None:
        # one shared saturated intensity, species and channels differ only
        # through their QL weights
        rest = {k: v for k, v in spec.items() if k != "flux_kxy"}
        f = ql_flux_per_species(rule, flux_kxy_sp=flux_kxy_sp, cn=cn, **rest, **extra)
        q_i, q_e, pfe = _clean(f[0, 1]), _clean(f[1, 1]), _clean(f[1, 0], lo=-qi_clip)
    else:
        q_i = _clean(RULES[rule](cn=cn, **spec, **extra))
        q_e, pfe = q_i, jnp.asarray(0.0)
    if return_diagnostics:
        diagnostics = {
            "n_blocks": n_blocks,
            "n_steps": n_steps,
            "gamma_max": jnp.max(gamma),
            "cn": jnp.asarray(cn),
        }
        return q_i, q_e, pfe, diagnostics
    return q_i, q_e, pfe


def _species_geom(geom, params):
    """geom with the per-species arrays from params, as calculate_fluxes_kinetic wants."""
    out = dict(geom)
    for key in ("mas", "signz", "tmp", "de", "vthrat", "rlt", "rln"):
        if hasattr(params, key):
            v = getattr(params, key)
            if v is not None:
                out[key] = jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float64))
    return out


def _linear_harvest(
    params,
    geom,
    grid_shape,
    *,
    n_steps_linear,
    early_stop,
    early_stop_opts,
):
    """Linear solve -> the (gamma, spectra, geometry) kwargs every rule consumes."""
    nv, nmu, ns, nkx, nky = grid_shape
    kinetic = not bool(getattr(params, "adiabatic_electrons", True))
    nsp = jnp.asarray(params.mas).size if kinetic else 0
    df = initial_df(nv, nmu, ns, nkx, nky, nsp=nsp)
    sim_state = default_state(nky=nky)
    pre = linear_precompute(geom, params)
    opts = dict(early_stop_opts or {})
    if early_stop:
        df_final, phi, sim_final, n_blocks = linear_early_stop(
            df, geom, params, sim_state, pre,
            max_steps=n_steps_linear, return_blocks=True, **opts,
        )
        n_steps = n_blocks * int(opts.get("block", 100))
    else:
        df_final, (phi, _fluxes), sim_final = gksolve(
            df, geom, params, sim_state, n_steps=n_steps_linear, pre=pre
        )
        n_blocks, n_steps = jnp.asarray(1), jnp.asarray(n_steps_linear)

    gt = geom_tensors(geom)
    diag_df = df_final
    if params.nlapar:
        from gyaradax.operators.fields import _compute_fields, g_to_f
        from gyaradax.diagnostics.fluxes import calculate_em_fluxes

        _phi2, apar, _bpar = _compute_fields(df_final, geom, params, pre)
        diag_df = g_to_f(df_final, apar, params, pre)
    flux_kxy_sp = None
    if kinetic:
        from gyaradax.integrals import calculate_fluxes_kinetic

        flux_kxy_sp = calculate_fluxes_kinetic(
            _species_geom(geom, params), diag_df, phi, reduce=False
        )
        eflux_kxy = flux_kxy_sp[0, 1]
    else:
        _p, eflux_kxy, _v = calculate_fluxes(gt, diag_df, phi, reduce=False)
    if params.nlapar:
        _ep, em_eflux_kxy, _ev = calculate_em_fluxes(
            geom, diag_df, apar, params=params, bpar=None, pre=pre, reduce=False
        )
        eflux_kxy = eflux_kxy + em_eflux_kxy
        if kinetic:
            flux_kxy_sp = flux_kxy_sp.at[0, 1].set(eflux_kxy)

    ints = jnp.asarray(geom["ints"])
    phi2 = jnp.abs(phi) ** 2
    lg = jnp.asarray(geom["little_g"])
    spec = dict(
        growth_rate=sim_final.last_growth_rate,
        phi2=phi2,
        phi2_kxy=jnp.sum(phi2 * ints[:, None, None], axis=0),
        flux_kxy=eflux_kxy,
        krho=jnp.asarray(geom["krho"], dtype=jnp.float64),
        kxrh=jnp.asarray(geom["kxrh"], dtype=jnp.float64),
        little_g=lg.T if lg.shape[0] != 3 else lg,
        ds=jnp.mean(ints),
    )
    return spec, n_blocks, n_steps, sim_final.last_growth_rate, flux_kxy_sp


def _dlnpdr_from_params(params):
    """TGLF dlnpdr for the rules that want it; see rules.dlnpdr_tglf."""
    # traced under the TORAX plugin's vmap over radii, so everything here
    # stays in jnp: no np.asarray, no float(), no branching on a value.
    # adiabatic_electrons and the array shapes are static, so they may.
    adiabatic = bool(params.adiabatic_electrons)
    tmp = jnp.atleast_1d(jnp.asarray(params.tmp, dtype=jnp.float64))
    te_ti = 1.0 if adiabatic or tmp.shape[0] < 2 else tmp[1] / tmp[0]
    rlt = jnp.atleast_1d(jnp.asarray(params.rlt, dtype=jnp.float64))
    rln = jnp.atleast_1d(jnp.asarray(params.rln, dtype=jnp.float64))
    if adiabatic:
        rlt, rln = rlt[0], rln[0]
    return dlnpdr_tglf(rlt, rln, te_ti=te_ti, adiabatic=adiabatic)


def nl_at_point(
    params,
    geom,
    grid_shape: Tuple[int, int, int, int, int],
    *,
    n_steps=30000,
    tail_blocks=12,
    block=500,
    qi_clip=1e5,
    df_init=None,
    return_df=False,
):
    """Full nonlinear ground truth at one point -> tail-averaged (qi, qe, pfe) in GKW GB units.

    Pure-JAX (backend='jax' required for vmap/jit): burn-in to saturation,
    then a lax.scan over `tail_blocks` blocks collecting fluxes to average.
    Orders of magnitude slower than ql_at_point — offline validation and
    ground-truth generation only, never inside a stiff transport loop.

    `df_init` warm-starts from an already-saturated turbulent state (GENE-TANGO
    style), which skips the linear-growth transient: pair it with a smaller
    `n_steps`. `return_df` additionally returns the final distribution so the
    next outer iteration can resume from it.
    """
    import dataclasses

    params = dataclasses.replace(
        params, non_linear=True, disable_per_ky_norm=False, adaptive_dt=True
    )
    nv, nmu, ns, nkx, nky = grid_shape
    warm = df_init is not None
    df = jnp.asarray(df_init) if warm else initial_df(nv, nmu, ns, nkx, nky)
    sim_state = default_state(nky=nky)
    pre = linear_precompute(geom, params)
    gt = geom_tensors(geom)

    burn = int(n_steps) - int(tail_blocks) * int(block)
    burn = max(burn, 0 if warm else int(block))
    if burn > 0:
        df, (phi, _f), sim_state = gksolve(df, geom, params, sim_state, n_steps=burn, pre=pre)

    def body(carry, _):
        df_c, sim_c = carry
        df_n, (phi_n, _fl), sim_n = gksolve(df_c, geom, params, sim_c, n_steps=block, pre=pre)
        _pf, ef, _vf = calculate_fluxes(gt, df_n, phi_n, reduce=True)
        return (df_n, sim_n), ef

    (df_f, _sim), ef_blocks = jax.lax.scan(body, (df, sim_state), None, length=int(tail_blocks))
    q_i = jnp.mean(ef_blocks)
    q_i = jnp.where(jnp.isfinite(q_i), jnp.clip(q_i, 0.0, qi_clip), 0.0)
    if return_df:
        return q_i, q_i, jnp.asarray(0.0), df_f
    return q_i, q_i, jnp.asarray(0.0)
