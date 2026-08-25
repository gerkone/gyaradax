"""Evaluate the QL model at one operating point (params + geometry).

This is the engine behind both the TORAX `gyaradax-ql` transport model
(jit/vmap-safe path: `initial_df`, `linear_early_stop`, `ql_at_point`) and
`scripts/run.py --ql-linear` campaign harvesting (host path:
`harvest_linear_from_config`, CFL-pinned dt + convergence loop).
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from gyaradax.integrals import calculate_fluxes, geom_tensors
from gyaradax.solver import default_state, gksolve, linear_precompute

from .saturation import ql_flux

DF_SEED_AMPLITUDE = 1e-3


def initial_df(nvpar, nmu, ns, nkx, nky, amplitude=DF_SEED_AMPLITUDE) -> jnp.ndarray:
    """Initial df: cosine in s, every non-zonal ky seeded (bypasses init_f's .item())."""
    s_grid = (jnp.arange(ns) + 0.5) / ns - 0.5
    seed_s = amplitude * (jnp.cos(2 * jnp.pi * s_grid) + 1.0)
    df = jnp.zeros((nvpar, nmu, ns, nkx, nky), dtype=jnp.complex128)
    seed = jnp.broadcast_to(
        seed_s[None, None, :, None, None] / (nkx * (nky - 1)),
        (nvpar, nmu, ns, nkx, nky),
    )
    ky_mask = jnp.arange(nky) > 0
    return df + seed * ky_mask[None, None, None, None, :]


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
    n_steps_linear=2000,
    early_stop=True,
    early_stop_opts: Dict[str, Any] | None = None,
    qi_clip=1e3,
    qi_tiny=1e-8,
    return_diagnostics=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Linear solve + canonical saturation rule at one point -> (qi, qe, pfe) in GKW GB units.

    Electromagnetic (params.nlapar) runs convert the mixed g to physical f
    for diagnostics and add the A_par flutter flux into the QL weight,
    matching the calibration X stage. qe = qi and pfe = 0 remain
    placeholders until the kinetic-electron calibration lands.

    `return_diagnostics` appends a dict of traced per-call solver diagnostics
    (blocks and steps actually run, converged growth rates).
    """
    nv, nmu, ns, nkx, nky = grid_shape
    df = initial_df(nv, nmu, ns, nkx, nky)
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
        from gyaradax.fields import _compute_fields, g_to_f
        from gyaradax.integrals import calculate_em_fluxes

        _phi2, apar, _bpar = _compute_fields(df_final, geom, params, pre)
        diag_df = g_to_f(df_final, apar, params, pre)
    _p, eflux_kxy, _v = calculate_fluxes(gt, diag_df, phi, reduce=False)
    if params.nlapar:
        _ep, em_eflux_kxy, _ev = calculate_em_fluxes(
            geom, diag_df, apar, params=params, bpar=None, pre=pre, reduce=False
        )
        eflux_kxy = eflux_kxy + em_eflux_kxy

    ints = jnp.asarray(geom["ints"])
    phi2 = jnp.abs(phi) ** 2
    lg = jnp.asarray(geom["little_g"])
    q_i = ql_flux(
        growth_rate=sim_final.last_growth_rate,
        phi2=phi2,
        phi2_kxy=jnp.sum(phi2 * ints[:, None, None], axis=0),
        flux_kxy=eflux_kxy,
        krho=jnp.asarray(geom["krho"], dtype=jnp.float64),
        kxrh=jnp.asarray(geom["kxrh"], dtype=jnp.float64),
        little_g=lg.T if lg.shape[0] != 3 else lg,
        ds=jnp.mean(ints),
        cn=cn,
    )
    q_i = jnp.where(jnp.isfinite(q_i), jnp.clip(q_i, 0.0, qi_clip), 0.0)
    # zero below noise floor so linear-transient residues don't feed TORAX gradients
    q_i = jnp.where(jnp.abs(q_i) < qi_tiny, 0.0, q_i)
    if return_diagnostics:
        diagnostics = {
            "n_blocks": n_blocks,
            "n_steps": n_steps,
            "gamma_max": jnp.max(sim_final.last_growth_rate),
            "cn": jnp.asarray(cn),
        }
        return q_i, q_i, jnp.asarray(0.0), diagnostics
    return q_i, q_i, jnp.asarray(0.0)


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


def harvest_linear_from_config(
    config_path,
    *,
    block=200,
    max_steps=6000,
    atol=1e-4,
    rtol=1e-3,
    patience=2,
    dump_df=False,
    converge=True,
    disable_per_ky_norm=True,
) -> Dict[str, np.ndarray]:
    """Run one linear point from a YAML config to gamma convergence, return QL harvest arrays.

    Host-side loop for campaign workers. dt is pinned to the von Neumann
    linear CFL estimate: adaptive dt is NL-gated in gksolve and the
    kinetic-electron field CFL is tighter than typical config dt.
    """
    import dataclasses

    from gyaradax.cfl import estimate_linear_timestep
    from gyaradax.simulate import gk_from_config

    from .linear_pipeline import _harvest

    df, geom, params, state, pre = gk_from_config(
        config_path,
        non_linear=False,
        disable_per_ky_norm=disable_per_ky_norm,
        adaptive_dt=False,
        backend="jax",
    )
    dt_cfl = float(estimate_linear_timestep(pre, params))
    if np.isfinite(dt_cfl) and dt_cfl < params.dt:
        params = dataclasses.replace(params, dt=dt_cfl)

    prev, steps, phi, streak = None, 0, None, 0
    ghist, shist = [], []  # per-block growth rate g(ky) and ky power spectrum
    while steps < max_steps:
        df, (phi, _f), state = gksolve(df, geom, params, state, n_steps=block, pre=pre)
        jax.block_until_ready(df)
        steps += block
        g = np.asarray(state.last_growth_rate)
        ghist.append(g)
        ph = np.asarray(phi)
        # normalized ky power spectrum; unnormalized would overflow f64
        sp = np.sum(np.abs(ph) ** 2, axis=tuple(range(ph.ndim - 1)))
        tot = sp.sum()
        shist.append(sp / tot if np.isfinite(tot) and tot > 0 else sp)
        if prev is not None and np.all(np.isfinite(g)):
            ok = np.max(np.abs(g - prev)) <= atol + rtol * np.max(np.abs(g))
            streak = streak + 1 if ok else 0
        prev = g
        # converge=False -> fixed-length run (exactly max_steps), no early stop
        if converge and streak >= patience:
            break

    arrays = {k: np.asarray(v) for k, v in dict(_harvest(geom, df, phi, state, params=params, pre=pre)).items()}
    arrays["gamma"] = np.asarray(prev)
    arrays["n_steps"] = np.asarray(steps)
    arrays["dt"] = np.asarray(float(params.dt))
    # per-block histories, sampled every `block` steps
    arrays["gamma_history"] = np.asarray(ghist)
    arrays["ky_spectrum_history"] = np.asarray(shist)
    arrays["history_block"] = np.asarray(int(block))
    # streak hits patience only on convergence; always False when converge=False
    arrays["converged"] = np.asarray(bool(streak >= patience))
    # rescale the converged eigenmode to a seed amplitude, keeping shape and phase
    if dump_df:
        dfm = np.asarray(df)
        peak = float(np.abs(dfm).max())
        amp = float(getattr(params, "amp_init", 1e-4)) or 1e-4
        arrays["df"] = (dfm / peak * amp) if peak > 0 and np.isfinite(peak) else dfm
    return arrays
