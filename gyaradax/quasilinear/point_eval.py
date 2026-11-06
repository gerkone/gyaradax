"""Evaluate the QL model at one operating point (params + geometry).

jit/vmap-safe, the path the TORAX ql-gyx model takes: `initial_df`, `linear_early_stop`,
`ql_at_point`; `nl_at_point` for the nonlinear ground truth.
"""

import dataclasses
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from gyaradax.diagnostics.fluxes import calculate_fluxes
from gyaradax.operators.poisson import geom_tensors
from gyaradax.operators.precompute import linear_precompute
from gyaradax.solver.cfl import estimate_linear_timestep
from gyaradax.solver.rk4 import default_state, gksolve

from .rules import (
    RULES,
    RULES_WITH_DLNPDR,
    RULES_WITH_KTHNORM,
    RULES_WITH_KX_CHAIN,
    geom_sat2_factors,
    dlnpdr_tglf,
    kx_chain0,
)

DF_SEED_AMPLITUDE = 1e-3


def initial_df(
    nvpar, nmu, ns, nkx, nky, amplitude=DF_SEED_AMPLITUDE, nsp=0
) -> jnp.ndarray:
    """Initial distribution: cosine in s, every non-zonal ky seeded, without ``init_f``'s host calls.

    Args:
        nvpar: Parallel-velocity points.
        nmu: Magnetic-moment points.
        ns: Parallel grid points.
        nkx: Radial modes.
        nky: Binormal modes.
        amplitude: Seed amplitude.
        nsp: Species count; above 0 adds a leading species axis.

    Returns:
        The distribution, complex128.
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
    """Linear ``gksolve`` in blocks, stopped once every ky growth rate has converged.

    Safe under ``jit`` and ``vmap`` (a ``lax.while_loop``; under ``vmap`` it runs until
    every element has converged).

    Args:
        df: Initial distribution.
        geom: Geometry dict.
        params: Run parameters.
        sim_state: Initial GKState.
        pre: Precomputed coefficients.
        block: Steps per block.
        max_steps: Most steps; may be traced.
        min_steps: Fewest steps.
        atol: Absolute tolerance on the change of the growth rates between blocks.
        rtol: Tolerance relative to the largest growth rate.
        patience: Consecutive converged blocks needed to stop.
        return_blocks: Also return the number of blocks run.

    Returns:
        ``(df, phi, state)``, plus the block count with ``return_blocks``.
    """
    block = int(block)
    max_blocks = jnp.maximum(jnp.asarray(max_steps) // block, 1)
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
    cn=1.0,
    rule="canonical",
    rule_options=None,
    n_steps_linear=2000,
    early_stop=True,
    early_stop_opts: Dict[str, Any] | None = None,
    qi_clip=1e3,
    qi_tiny=1e-8,
    return_diagnostics=False,
    channel_params=None,
    df0=None,
    return_df=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear solve and saturation rule at one point, in GKW gyro-Bohm units.

    Electromagnetic runs convert g to f for the diagnostics and add the A_par
    flutter flux to the QL weight. The QL weights are taken on the kx = 0 chain. Kinetic
    runs evaluate the rule per flux channel (ion heat, electron heat, electron particle)
    against one shared saturated intensity, each channel with its own calibration.
    Adiabatic runs have no electron channel, so ``qe = qi`` and ``pfe = 0``.

    Args:
        params: Run parameters.
        geom: Geometry dict.
        grid_shape: ``(nvpar, nmu, ns, nkx, nky)``.
        cn: Amplitude of channels without their own ``cn``.
        rule: Name of an entry of ``quasilinear.rules.RULES``.
        rule_options: Keyword arguments passed to the rule.
        channel_params: Rule keyword arguments per channel (``"Qi"``, ``"Qe"``, ``"Gamma"``), e.g. ``cn`` and
            ``flux_exponent`` from :func:`~gyaradax.quasilinear.calibration.channel_parameters`.
        n_steps_linear: Most linear steps.
        early_stop: Stop when the growth rates have converged (:func:`linear_early_stop`).
        early_stop_opts: Options of :func:`linear_early_stop`.
        qi_clip: Upper clip of the fluxes.
        qi_tiny: Fluxes below this are set to zero.
        return_diagnostics: Also return a dict of solver diagnostics (blocks and
            steps run, growth rates).
        df0: Starting distribution (e.g. the previous call's); the seed of
            :func:`initial_df` when None.
        return_df: Also return the final distribution.

    Returns:
        ``(qi, qe, pfe)``, then the diagnostics dict and the final distribution when requested.
    """
    spec, n_blocks, n_steps, gamma, flux_kxy_sp, final = _linear_harvest(
        params, geom, grid_shape, n_steps_linear=n_steps_linear,
        early_stop=early_stop, early_stop_opts=early_stop_opts, df0=df0,
    )
    extra = ({"dlnpdr": _dlnpdr_from_params(params)}
             if rule in RULES_WITH_DLNPDR else {})
    if rule in RULES_WITH_KTHNORM:
        extra["kthnorm"] = jnp.asarray(geom.get("kthnorm", 1.0), dtype=jnp.float64)
        extra["grad_r0"], extra["gq"] = geom_sat2_factors(geom)
    chain = kx_chain0(jnp.asarray(geom["mode_label"]), spec["kxrh"])
    if rule in RULES_WITH_KX_CHAIN:
        extra["kx_chain"] = chain
    # caller-supplied knobs win, so a rule's own options need no plumbing here
    extra.update(dict(rule_options or {}))
    channel_params = dict(channel_params or {})
    rest = {k: v for k, v in spec.items() if k not in ("flux_kxy", "phi2_kxy")}
    phi2_chain = spec["phi2_kxy"] * chain

    def _clean(q, lo=0.0):
        q = jnp.clip(q, lo, qi_clip)
        # zero below noise floor so linear transients don't feed TORAX gradients
        return jnp.where(jnp.abs(q) < qi_tiny, 0.0, q)

    def channel(name, flux_kxy):
        # one shared saturated intensity; channels differ in their weights (on the theta0 = 0 chain) and calibration
        kw = {"cn": cn, **extra, **channel_params.get(name, {})}
        return RULES[rule](flux_kxy=flux_kxy * chain, phi2_kxy=phi2_chain, **rest, **kw)

    if flux_kxy_sp is not None:
        q_i, q_e = _clean(channel("Qi", flux_kxy_sp[0, 1])), _clean(channel("Qe", flux_kxy_sp[1, 1]))
        pfe = _clean(channel("Gamma", flux_kxy_sp[1, 0]), lo=-qi_clip)
    else:
        q_i = _clean(channel("Qi", spec["flux_kxy"]))
        q_e, pfe = q_i, jnp.asarray(0.0)
    out = (q_i, q_e, pfe)
    if return_diagnostics:
        out += ({"n_blocks": n_blocks, "n_steps": n_steps, "gamma_max": jnp.max(gamma)},)
    if return_df:
        out += (final[0],)
    return out


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
    df0=None,
):
    """Linear solve -> the (gamma, spectra, geometry) kwargs every rule consumes."""
    nv, nmu, ns, nkx, nky = grid_shape
    kinetic = not bool(getattr(params, "adiabatic_electrons", True))
    nsp = jnp.asarray(params.mas).size if kinetic else 0
    df = initial_df(nv, nmu, ns, nkx, nky, nsp=nsp) if df0 is None else jnp.asarray(df0, jnp.complex128)
    sim_state = default_state(nky=nky)
    pre = linear_precompute(geom, params)
    if kinetic:
        # electron streaming sets the limit: the input dt is a cap, as in the stored linear runs
        params = dataclasses.replace(params, dt=jnp.minimum(params.dt, estimate_linear_timestep(pre, params)))
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
        from gyaradax.diagnostics.fluxes import calculate_fluxes_kinetic

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
    return spec, n_blocks, n_steps, sim_final.last_growth_rate, flux_kxy_sp, (df_final, phi, sim_final, pre, params)


def _dlnpdr_from_params(params):
    """TGLF dlnpdr for the rules that want it; see rules.dlnpdr_tglf."""
    # traced under the plugin's vmap: jnp only, branching only on static fields
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
    """Nonlinear ground truth at one point: tail-averaged ``(qi, qe, pfe)`` in GKW gyro-Bohm units.

    With adiabatic electrons ``qe = qi`` and ``pfe = 0``. A burn-in to
    saturation, then ``tail_blocks`` blocks whose fluxes are averaged. Use
    ``backend='jax'`` in ``params`` to ``jit`` or ``vmap`` it. Much slower than
    :func:`ql_at_point`, meant for validation and ground-truth data.

    Args:
        params: Run parameters.
        geom: Geometry dict.
        grid_shape: ``(nvpar, nmu, ns, nkx, nky)``.
        n_steps: Total steps.
        tail_blocks: Blocks averaged at the end.
        block: Steps per block.
        qi_clip: Upper clip of the fluxes.
        df_init: Saturated state to start from, which skips the linear phase.
        return_df: Also return the final distribution.

    Returns:
        ``(qi, qe, pfe)``, plus the distribution with ``return_df``.
    """

    params = dataclasses.replace(
        params, non_linear=True, disable_per_ky_norm=False, adaptive_dt=True
    )
    nv, nmu, ns, nkx, nky = grid_shape
    kinetic = not bool(getattr(params, "adiabatic_electrons", True))
    nsp = jnp.asarray(params.mas).size if kinetic else 0
    warm = df_init is not None
    df = jnp.asarray(df_init) if warm else initial_df(nv, nmu, ns, nkx, nky, nsp=nsp)
    sim_state = default_state(nky=nky)
    pre = linear_precompute(geom, params)
    gt = geom_tensors(geom)
    sp_geom = _species_geom(geom, params) if kinetic else None

    def fluxes(df_n, phi_n):
        if kinetic:
            from gyaradax.diagnostics.fluxes import calculate_fluxes_kinetic

            f = calculate_fluxes_kinetic(sp_geom, df_n, phi_n)
            return jnp.stack([f[0, 1], f[1, 1], f[1, 0]])
        _pf, ef, _vf = calculate_fluxes(gt, df_n, phi_n, reduce=True)
        return jnp.stack([ef, ef, jnp.zeros_like(ef)])

    burn = int(n_steps) - int(tail_blocks) * int(block)
    burn = max(burn, 0 if warm else int(block))
    if burn > 0:
        df, (phi, _f), sim_state = gksolve(df, geom, params, sim_state, n_steps=burn, pre=pre)

    def body(carry, _):
        df_c, sim_c = carry
        df_n, (phi_n, _fl), sim_n = gksolve(df_c, geom, params, sim_c, n_steps=block, pre=pre)
        return (df_n, sim_n), fluxes(df_n, phi_n)

    (df_f, _sim), blocks = jax.lax.scan(body, (df, sim_state), None, length=int(tail_blocks))
    q_i, q_e, pfe = jnp.mean(blocks, axis=0)
    out = (jnp.clip(q_i, 0.0, qi_clip), jnp.clip(q_e, 0.0, qi_clip), jnp.clip(pfe, -qi_clip, qi_clip))
    return (*out, df_f) if return_df else out
