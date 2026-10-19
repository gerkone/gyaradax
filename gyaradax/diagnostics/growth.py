"""Linear growth rates of a config, optionally at another parallel resolution.

A linear run of the config (non_linear off) whose per-ky growth rates are averaged once they
have settled. Run at a finer ns it gives a ground truth to check the production resolution and
dissipation against (``gyaradax growth``, ``gyaradax run --linear-check``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

# exit code of a run whose linear check deviates beyond the tolerance
EXIT_LINEAR_CHECK = 3
# settings a linear growth rate does not depend on
_IGNORED_SECTIONS = ("run", "dataset", "sharding")
_IGNORED_SOLVER_KEYS = (
    "n_steps",
    "t_end",
    "dump_dt",
    "dump_interval",
    "block_size",
    "non_linear",
    "adaptive_dt",
    "cfl_safety",
    "fac_dtim_nl",
    "backend",
    "mixed_precision",
    "use_z2z",
)


def adjusted_config(cfg: Any, ns: int | None = None, sets: Mapping[str, Any] | None = None) -> Any:
    """Copy of ``cfg`` run linearly, with dotted-key overrides and another ``grid.ns`` applied."""
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key, value in (sets or {}).items():
        OmegaConf.update(out, key, value, force_add=True)
    if ns is not None and int(ns) != int(out.grid.ns):
        old = int(out.grid.ns)
        out.grid.ns = int(ns)
        sgr = OmegaConf.select(out, "geometry.sgr_dist")
        if sgr is not None:
            # keeps the field-line length: nperiod turns over ns points
            out.geometry.sgr_dist = float(sgr) * old / int(ns)
    OmegaConf.update(out, "solver.non_linear", False, force_add=True)
    return out


def config_key(cfg: Any) -> str:
    """Hash of what a linear growth rate depends on (not the run's name, length, backend or layout)."""
    data = OmegaConf.to_container(cfg, resolve=True)
    for key in _IGNORED_SECTIONS:
        data.pop(key, None)
    for key in _IGNORED_SOLVER_KEYS:
        (data.get("solver") or {}).pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


def growth_rates(
    cfg: Any,
    ns: int | None = None,
    sets: Mapping[str, Any] | None = None,
    *,
    backend: str | None = None,
    t_end: float = 40.0,
    t_min: float = 15.0,
    settle: float = 6.0,
    tol: float = 0.005,
    ky_max: float = 1.0,
    gamma_min: float = 0.02,
    block: int = 250,
    log: Callable[[str], None] | None = None,
    frequencies: bool = False,
    keep_mode: bool = False,
    mesh: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Per-ky growth rates of ``cfg`` run linearly, averaged over the last ``settle`` time units.

    The time step is capped at the linear stability estimate (as GKW does for every explicit
    run). The run stops once the unstable ky (kθρ <= ``ky_max``, γ > ``gamma_min``) vary by less
    than ``tol`` over ``settle`` time units after ``t_min``, or at ``t_end``. ``frequencies``
    adds the real frequencies (``omega``, ``omega_std``) from a few short blocks after that;
    ``keep_mode`` returns the final df and phi under ``mode`` and their fluxes per ky under
    ``mode_fluxes`` (:func:`mode_fluxes`). ``mesh`` (species, vpar, mu) shards the run over that
    many visible GPUs, as for a nonlinear run; the default is one device.
    """
    import jax.numpy as jnp

    from gyaradax.core.params import gkparams_from_config
    from gyaradax.geometry import compute_geometry_from_config
    from gyaradax.operators.precompute import linear_precompute
    from gyaradax.solver.cfl import estimate_linear_timestep
    from gyaradax.solver.driver import gk_init, gk_run

    run_cfg = adjusted_config(cfg, ns, sets)
    sp, vp, mu = (int(m) for m in mesh) if mesh is not None else (1, 1, 1)
    overrides: dict[str, Any] = {"n_gpus_sp": sp, "n_gpus_vp": vp, "n_gpus_mu": mu}
    if backend:
        overrides["backend"] = backend
    params = gkparams_from_config(run_cfg, **overrides)
    n_species = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
    geometry = compute_geometry_from_config(run_cfg)
    df, geometry, state = gk_init(geometry, params, n_species=n_species)
    pre = linear_precompute(geometry, params)
    dt_input = float(params.dt)
    dt_lin = float(estimate_linear_timestep(pre, params=params))
    params = replace(params, dt=min(dt_input, dt_lin))
    kthnorm = float(np.asarray(geometry.get("kthnorm", 1.0)).reshape(-1)[0])
    ky = np.asarray(geometry["krho"], dtype=float) * kthnorm
    in_range = (ky > 0) & (ky <= ky_max)

    times: list[float] = []
    rates: list[np.ndarray] = []
    converged = False
    t = float(state.time)
    phi = None
    while t < t_end:
        df, phi, _, state = gk_run(df, geometry, params, state, block, pre=pre)
        t = float(state.time)
        g = np.asarray(state.last_growth_rate, dtype=float)
        if not np.all(np.isfinite(g)):
            raise FloatingPointError(f"non-finite growth rate at t={t:.2f} (ns {run_cfg.grid.ns})")
        times.append(t)
        rates.append(g)
        if log is not None:
            log(f"t {t:6.2f}  max γ {g[in_range].max() if in_range.any() else float('nan'):+.4f}")
        if t >= t_min:
            late = np.asarray(rates)[np.asarray(times) >= t - settle]
            unstable = in_range & (late.mean(0) > gamma_min)
            sel = unstable if unstable.any() else in_range
            if sel.any() and late[:, sel].std(0).max() < tol:
                converged = True
                break

    tt, gg = np.asarray(times), np.asarray(rates)
    late = gg[tt >= tt[-1] - settle]
    mean, std = late.mean(0), late.std(0)
    result = {
        "ns": int(run_cfg.grid.ns),
        "ky": ky.tolist(),
        "ky_convention": "kthrho: geometry krho * kthnorm (not krho / kthnorm)",
        "kthnorm": kthnorm,
        "kxrh": np.asarray(geometry["kxrh"], dtype=float).tolist(),
        "gamma": mean.tolist(),
        "gamma_std": std.tolist(),
        "t": t,
        "converged": converged,
        "dt": float(params.dt),
        "dt_input": dt_input,
        "dt_linear_estimate": dt_lin,
        "disp_par": float(params.disp_par),
        "sets": dict(sets or {}),
        "config_key": config_key(run_cfg),
    }
    result["peak"] = peak(result, ky_max=ky_max)
    if frequencies:
        omega, omega_std, df, phi, state = real_frequency(df, phi, geometry, params, state, pre)
        result["omega"], result["omega_std"] = omega.tolist(), omega_std.tolist()
    if keep_mode:
        result["mode_fluxes"] = {
            k: np.asarray(v).tolist() for k, v in mode_fluxes(df, geometry, params, pre).items()
        }
        result["mode"] = {"df": np.asarray(df), "phi": np.asarray(phi), "time": float(state.time)}
    return result


def mode_fluxes(df: Any, geometry: Mapping[str, Any], params: Any, pre: Any) -> dict[str, Any]:
    """Fluxes of a linear eigenmode per ky and its quasilinear weights.

    The flux integrals of the solver diagnostics (``quasilinear.linear_pipeline._harvest``: the
    electrostatic fluxes of the physical f plus the A_par / B_par flutter parts), kept per
    (species, channel, ky) with channels (particle, heat, momentum) as in ``flux_labels``.
    Returns ``flux_kxy`` (nsp, 3, nkx, nky) and ``phi2_kxy`` (nkx, nky); ``flux`` (nsp, 3), the sum
    over ky; ``flux_ky`` (nsp, 3, nky), summed over kx;
    ``phi2_ky`` (nky,), the flux-surface average of |phi|^2 summed over kx; the same over the kx
    chain through kx = 0 alone (``flux_ky_chain0``, ``phi2_ky_chain0``), the mode whose growth
    rate and frequency are reported; and the weights ``weight_ky`` = ``flux_ky_chain0 /
    phi2_ky_chain0`` and ``weight_ky_allkx`` = ``flux_ky / phi2_ky`` (nan where |phi|^2 is 0).
    The linear run normalizes every kx chain on its own, so only per-chain ratios are physical.
    """
    from types import SimpleNamespace

    from gyaradax.operators.fields import _compute_fields, g_to_f
    from gyaradax.quasilinear.linear_pipeline import _harvest

    phi, apar, bpar = _compute_fields(df, geometry, params, pre)
    f = g_to_f(df, apar, params, pre) if params.nlapar else df
    out = _harvest(
        geometry,
        f,
        phi,
        SimpleNamespace(last_growth_rate=None),
        params=params,
        apar=apar if params.nlapar else None,
        bpar=bpar if params.nlbpar else None,
        pre=pre,
    )
    if "fluxes_kxy_sp" in out:
        kxy = np.asarray(out["fluxes_kxy_sp"], np.float64)
    else:
        channels = ("pflux", "eflux", "vflux")
        kxy = np.stack([np.asarray(out[f"{c}_kxy"], np.float64) for c in channels])[None]
    phi2 = np.asarray(out["phi2_kxy"], np.float64)
    mode_label = np.asarray(geometry["mode_label"])
    chain0 = mode_label == mode_label[int(np.asarray(geometry["ixzero"])), :]
    res = {
        "flux_kxy": kxy,
        "phi2_kxy": phi2,
        "flux": kxy.sum(axis=(-2, -1)),
        "flux_ky": kxy.sum(axis=-2),
        "phi2_ky": phi2.sum(axis=0),
        "flux_ky_chain0": (kxy * chain0).sum(axis=-2),
        "phi2_ky_chain0": (phi2 * chain0).sum(axis=0),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, suffix in (("weight_ky", "_chain0"), ("weight_ky_allkx", "")):
            p2 = res["phi2_ky" + suffix]
            res[name] = np.where(p2 > 0, res["flux_ky" + suffix] / p2, np.nan)
    return res


def _chain_phase(phi: Any, geometry: Mapping[str, Any]) -> np.ndarray:
    # sum of phi over s and the kx chain through kx = 0, per ky (GKW diagnos_growth_freq)
    mode_label = np.asarray(geometry["mode_label"])
    chain = mode_label == mode_label[int(np.asarray(geometry["ixzero"])), :]
    return np.sum(np.asarray(phi) * chain[None], axis=(0, 1))


def real_frequency(
    df: Any,
    phi: Any,
    geometry: Mapping[str, Any],
    params: Any,
    state: Any,
    pre: Any,
    blocks: int = 4,
    steps: int = 10,
) -> tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    """Real frequency per ky from the phase advance of phi over ``blocks`` runs of ``steps`` steps.

    ω = d arg(Σ phi)/dt, the sum over s and the kx chain through kx = 0, with GKW's sign. The
    blocks are short so that |ω| dt stays below π. Returns the mean and spread over the blocks,
    and the state after them.
    """
    from gyaradax.solver.driver import gk_run

    samples = []
    a, ta = _chain_phase(phi, geometry), float(state.time)
    for _ in range(blocks):
        df, phi, _, state = gk_run(df, geometry, params, state, steps, pre=pre)
        b, tb = _chain_phase(phi, geometry), float(state.time)
        samples.append(np.angle(np.conj(a) * b) / (tb - ta))
        a, ta = b, tb
    w = np.asarray(samples)
    return w.mean(0), w.std(0), df, phi, state


def peak(result: Mapping[str, Any], ky_max: float = 1.0) -> dict[str, float] | None:
    ky, g = np.asarray(result["ky"]), np.asarray(result["gamma"])
    sel = (ky > 0) & (ky <= ky_max)
    if not sel.any():
        return None
    i = int(np.flatnonzero(sel)[np.argmax(g[sel])])
    return {"ky": float(ky[i]), "gamma": float(g[i])}


def compare(
    result: Mapping[str, Any],
    reference: Mapping[str, Any],
    ky_max: float = 1.0,
    gamma_min: float = 0.02,
) -> dict[str, Any]:
    """γ of ``result`` at the dominant ky of ``reference``, and its relative deviation."""
    ky, ref_ky = np.asarray(result["ky"]), np.asarray(reference["ky"])
    if ky.shape != ref_ky.shape or not np.allclose(ky, ref_ky):
        raise ValueError("growth rates on different ky grids")
    top = peak(reference, ky_max)
    if top is None or top["gamma"] <= gamma_min:
        return {"ky": None, "gamma": None, "gamma_ref": top["gamma"] if top else None, "rel": None}
    i = int(np.argmin(np.abs(ky - top["ky"])))
    g = float(np.asarray(result["gamma"])[i])
    return {
        "ky": top["ky"],
        "gamma": g,
        "gamma_ref": top["gamma"],
        "rel": (g - top["gamma"]) / abs(top["gamma"]),
    }


def check(
    cfg: Any,
    reference_ns: Sequence[int],
    *,
    backend: str | None = None,
    log: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Growth rates at the config's own resolution and at ``reference_ns``, compared to the finest."""
    own = int(cfg.grid.ns)
    runs = [growth_rates(cfg, None, backend=backend, log=log, **kwargs)]
    for ns in sorted({int(n) for n in reference_ns} - {own}):
        runs.append(growth_rates(cfg, ns, backend=backend, log=log, **kwargs))
    return assemble(runs, own)


def assemble(runs: Sequence[Mapping[str, Any]], own_ns: int) -> dict[str, Any]:
    """Report of several resolutions of one config; the finest is the reference."""
    reference = max(runs, key=lambda r: r["ns"])
    return {
        "config_key": config_key_of(runs, own_ns),
        "ns": int(own_ns),
        "reference_ns": reference["ns"],
        "runs": list(runs),
        "comparison": {str(r["ns"]): compare(r, reference) for r in runs},
    }


def config_key_of(runs: Sequence[Mapping[str, Any]], own_ns: int) -> str | None:
    own = [r for r in runs if r["ns"] == int(own_ns)]
    return own[0]["config_key"] if own else None


def format_check(report: Mapping[str, Any]) -> str:
    """One line per resolution: peak γ, γ at the reference's dominant ky and its deviation."""
    lines = [f"{'ns':>4}  {'t':>6}  {'peak γ':>8}  {'at kθρ':>7}  {'γ(ky ref)':>9}  {'vs ref':>7}"]
    for r in report["runs"]:
        top = r["peak"] or {"gamma": float("nan"), "ky": float("nan")}
        c = report["comparison"][str(r["ns"])]
        g = "-" if c["gamma"] is None else f"{c['gamma']:+.4f}"
        rel = "-" if c["rel"] is None else f"{c['rel']:+.1%}"
        mark = "" if r["converged"] else " (not settled)"
        lines.append(
            f"{r['ns']:>4}  {r['t']:6.1f}  {top['gamma']:+8.4f}  {top['ky']:7.3f}  {g:>9}  {rel:>7}{mark}"
        )
    return "\n".join(lines)


def summary(report: Mapping[str, Any]) -> str:
    """Short form for status lines: γ at the reference's dominant ky, own vs reference."""
    own = report["comparison"].get(str(report["ns"])) or {}
    ref = report["reference_ns"]
    if own.get("rel") is None:
        return f"linear: no unstable ky at ns {ref}"
    return f"γ ns{report['ns']} {own['gamma']:.3f} / ns{ref} {own['gamma_ref']:.3f} ({own['rel']:+.0%})"
