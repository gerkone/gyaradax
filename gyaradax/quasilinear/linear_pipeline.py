"""Linear gyaradax pipeline for the QL calibration.

For each iteration_N pair (Lin + NL):
  1. parse input.dat -> GKParams
  2. load geometry from geom.dat
  3. (optional) warm-start df from FDS, or fresh gk_init
  4. run gyaradax linear solver for n_steps (or skip with linear_from_fds)
  5. harvest γ, |φ|², per-(kx, ky) flux fields, geometry
  6. NL target Y comes from iteration_N/fluxes.dat (last 240 rows)

No GKW diagnostic files are required for QL inputs — only input.dat,
geom.dat, and (optionally) FDS. The flux target Y is loaded by construction
from NL fluxes.dat; we do not compute NL fluxes ourselves.
"""

import dataclasses
import os
import time

import jax.numpy as jnp
import numpy as np


def _setup_params(gkw_dir, disable_per_ky_norm):
    from gyaradax.params import gkparams_from_input_and_geometry
    from gyaradax.utils import load_geometry

    geometry = load_geometry(gkw_dir)
    params = gkparams_from_input_and_geometry(os.path.join(gkw_dir, "input.dat"), geometry)
    if disable_per_ky_norm and not params.disable_per_ky_norm:
        params = dataclasses.replace(params, disable_per_ky_norm=True)
    return geometry, params


def _harvest(geometry, df, phi, state, params=None, apar=None, bpar=None, pre=None):
    """Compute QL inputs from a linear-sim end-state.

    Handles adiabatic (5D df) and kinetic (6D df) transparently, and adds
    A_∥ / B_∥ contributions when `apar` / `bpar` are provided.
    """
    from gyaradax.integrals import (
        calculate_em_fluxes,
        calculate_fluxes,
        calculate_fluxes_kinetic,
        geom_tensors,
    )

    gt = geom_tensors(geometry)
    is_kinetic = df.ndim == 6

    if is_kinetic:
        # (nsp, 3, nkx, nky); sum species into a single (nkx, nky) per channel
        fluxes_sp = calculate_fluxes_kinetic(gt, df, phi, reduce=False)
        pflux_kxy = jnp.sum(fluxes_sp[:, 0], axis=0)
        eflux_kxy = jnp.sum(fluxes_sp[:, 1], axis=0)
        vflux_kxy = jnp.sum(fluxes_sp[:, 2], axis=0)
    else:
        fluxes_sp = None
        pflux_kxy, eflux_kxy, vflux_kxy = calculate_fluxes(gt, df, phi, reduce=False)

    # em flutter contributions; kinetic returns a stacked (nsp, 3, nkx, nky) array
    if (apar is not None) or (bpar is not None):
        em = calculate_em_fluxes(gt, df, apar, params=params, bpar=bpar, pre=pre, reduce=False)
        if is_kinetic:
            em_pflux_kxy = jnp.sum(em[:, 0], axis=0)
            em_eflux_kxy = jnp.sum(em[:, 1], axis=0)
            em_vflux_kxy = jnp.sum(em[:, 2], axis=0)
        else:
            em_pflux_kxy, em_eflux_kxy, em_vflux_kxy = em
        pflux_kxy = pflux_kxy + em_pflux_kxy
        eflux_kxy = eflux_kxy + em_eflux_kxy
        vflux_kxy = vflux_kxy + em_vflux_kxy
    else:
        zk = jnp.zeros_like(pflux_kxy)
        em_pflux_kxy = em_eflux_kxy = em_vflux_kxy = zk

    # FSA |φ|²(kx, ky) using gyaradax integration weights
    ints = jnp.asarray(geometry["ints"])
    ds = float(jnp.mean(ints))
    phi2 = jnp.abs(phi) ** 2
    phi2_kxy = jnp.sum(phi2 * ints[:, None, None], axis=0)

    lg = jnp.asarray(geometry["little_g"])
    little_g = lg.T if lg.shape[0] != 3 else lg
    krho = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)

    out = {
        "growth_rate": state.last_growth_rate,
        "phi": phi,
        "phi2": phi2,
        "phi2_kxy": phi2_kxy,
        "pflux_kxy": pflux_kxy,
        "eflux_kxy": eflux_kxy,
        "vflux_kxy": vflux_kxy,
        "em_pflux_kxy": em_pflux_kxy,
        "em_eflux_kxy": em_eflux_kxy,
        "em_vflux_kxy": em_vflux_kxy,
        "krho": krho,
        "kxrh": kxrh,
        "little_g": little_g,
        "ds": ds,
        "is_kinetic": is_kinetic,
    }
    if fluxes_sp is not None:
        out["fluxes_kxy_sp"] = fluxes_sp
    if apar is not None:
        out["apar"] = apar
    if bpar is not None:
        out["bpar"] = bpar
    return out


def linear_from_fds(gkw_dir):
    """Compute QL inputs directly from the FDS file — no time stepping.

    Detects adiabatic / kinetic from input.dat and ES / EM from `params.nlapar`,
    `params.nlbpar`. γ(ky) comes from `growth.dat`. ~1–2 s per ES adiabatic sim.
    """
    from gyaradax.solver import _compute_fields, linear_precompute
    from gyaradax.utils import load_gkw_k_dump

    from .data import _robust_loadtxt

    geometry, params = _setup_params(gkw_dir, disable_per_ky_norm=True)
    n_species = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])

    res = tuple(int(len(geometry[k])) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
    df_raw = load_gkw_k_dump(os.path.join(gkw_dir, "FDS"), res, n_species=n_species)
    df = jnp.asarray(df_raw, dtype=jnp.complex128)

    pre = linear_precompute(geometry, params)
    phi, apar, bpar = _compute_fields(df, geometry, params, pre)

    # converged γ from growth.dat last row
    growth = _robust_loadtxt(os.path.join(gkw_dir, "growth.dat"))
    gamma = growth[-1, :] if growth.ndim > 1 else growth
    nky = int(len(geometry["krho"]))
    if gamma.shape[0] == nky + 1:
        gamma = gamma[1:]
    gamma = jnp.asarray(gamma)

    class _S:
        last_growth_rate = gamma

    em = params.nlapar or params.nlbpar
    out = _harvest(
        geometry,
        df,
        phi,
        _S(),
        params=params,
        apar=apar if (em and params.nlapar) else None,
        bpar=bpar if (em and params.nlbpar) else None,
        pre=pre,
    )
    out["df"] = df
    out["params"] = params
    return out


def linear_run(gkw_dir, n_steps=1000, warm_start_from_fds=False, disable_per_ky_norm=True):
    """Run gyaradax linear solver on a GKW directory.

    Args:
        gkw_dir: path to an iteration_N_Lin directory.
        n_steps: number of linear steps.
        warm_start_from_fds: initialize df from FDS instead of gk_init.
        disable_per_ky_norm: GKParams flag to keep cross-ky phi amplitudes
            physically meaningful in the solver.

    Returns: dict with γ, |φ|², per-(kx, ky) Γ/Q/Π linear flux fields,
    |φ|²(kx, ky), krho, kxrh, little_g, ds.
    """
    from gyaradax.simulate import gk_init
    from gyaradax.solver import gksolve, linear_precompute
    from gyaradax.utils import load_gkw_k_dump

    geometry, params = _setup_params(gkw_dir, disable_per_ky_norm)

    if warm_start_from_fds and os.path.exists(os.path.join(gkw_dir, "FDS")):
        res = tuple(int(len(geometry[k])) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = jnp.asarray(
            load_gkw_k_dump(os.path.join(gkw_dir, "FDS"), res, n_species=1), dtype=jnp.complex128
        )
        _, _, state = gk_init(geometry, params, n_species=1)
    else:
        df, geometry, state = gk_init(geometry, params, n_species=1)

    pre = linear_precompute(geometry, params)
    df_final, (phi, _), state_final = gksolve(df, geometry, params, state, n_steps=n_steps, pre=pre)

    out = _harvest(geometry, df_final, phi, state_final)
    out["df"] = df_final
    return out


def linear_run_blocked(
    gkw_dir,
    block_size=200,
    max_blocks=20,
    gamma_tol=1e-3,
    warm_start_from_fds=False,
    disable_per_ky_norm=True,
    verbose=False,
):
    """Run linear gyaradax in fixed-size blocks until γ at the physical peak stops moving.

    Compile cost is paid once for the (block_size, shape) signature; subsequent
    blocks reuse the compiled code. Convergence is judged on γ at the dominant
    ky (argmax of γ), not the global max, so a spurious tail mode does not keep
    the loop running.
    """
    from gyaradax.simulate import gk_init
    from gyaradax.solver import gksolve, linear_precompute
    from gyaradax.utils import load_gkw_k_dump

    geometry, params = _setup_params(gkw_dir, disable_per_ky_norm)

    if warm_start_from_fds and os.path.exists(os.path.join(gkw_dir, "FDS")):
        res = tuple(int(len(geometry[k])) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = jnp.asarray(
            load_gkw_k_dump(os.path.join(gkw_dir, "FDS"), res, n_species=1), dtype=jnp.complex128
        )
        _, _, state = gk_init(geometry, params, n_species=1)
    else:
        df, geometry, state = gk_init(geometry, params, n_species=1)

    pre = linear_precompute(geometry, params)
    gamma_prev = jnp.zeros_like(state.last_growth_rate)
    history = []
    phi = None

    for b in range(max_blocks):
        df, (phi, _), state = gksolve(df, geometry, params, state, n_steps=block_size, pre=pre)
        df.block_until_ready()
        gamma = state.last_growth_rate
        peak_idx = int(jnp.argmax(gamma))
        delta_peak = float(jnp.abs(gamma[peak_idx] - gamma_prev[peak_idx])) / (
            float(jnp.abs(gamma_prev[peak_idx])) + 1e-6
        )
        history.append(float(gamma[peak_idx]))
        if verbose:
            print(
                f"  block {b + 1}: γ@peak={float(gamma[peak_idx]):+.4f} (ky={peak_idx})  "
                f"γ_max={float(jnp.max(gamma)):+.4f}  Δpeak={delta_peak:.2e}"
            )
        if b > 0 and delta_peak < gamma_tol:
            break
        gamma_prev = gamma

    out = _harvest(geometry, df, phi, state)
    out.update({"n_blocks": b + 1, "gamma_history": history, "df": df})
    return out


def harvest(
    triples,
    ql_flux_fn,
    loader=linear_from_fds,
    max_workers=8,
    n_average=240,
    label="",
    every=50,
):
    """Parallel I/O harvest of (X_QL, Y_NL, features) from (lin, nl, name) triples.

    Most of `linear_from_fds`'s cost is FDS read + geom load + linear_precompute,
    all I/O- or Python-bound. Threading parallelizes these; the JAX compute
    serializes on the GPU but is fast enough not to bottleneck. ~2.7–10× wall
    speedup vs serial depending on disk cache state.

    Args:
        triples: list of (lin_dir, nl_dir, name).
        ql_flux_fn: callable(arr_dict) -> float, e.g. lambda arr: float(ql_flux(...)).
        loader: callable(lin_dir) -> arr_dict. Default `linear_from_fds`; pass
            `lambda d: linear_run(d, n_steps=1000)` to use gyaradax time-stepping
            instead of the FDS file.
        max_workers: pool size. 8 is a reasonable default.
        n_average: tail-rows averaged for Y_NL from fluxes.dat.
        label: printed in progress messages.
        every: log progress every `every` completed sims (0 to silence).

    Returns: (X, Y, F, names, n_skipped) in input order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .data import load_nonlinear_target, parse_input_dat, physics_features

    def one(triple):
        lin, nl, name = triple
        try:
            arr = loader(lin)
            X = ql_flux_fn(arr)
            Y = load_nonlinear_target(nl, n_average=n_average)
            F = physics_features(parse_input_dat(f"{lin}/input.dat"))
            return name, (X, Y, F)
        except Exception:
            return name, None

    results_by_name = {}
    t0 = time.perf_counter()
    done = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(one, t) for t in triples]
        for fut in as_completed(futures):
            name, payload = fut.result()
            results_by_name[name] = payload
            done += 1
            if payload is None:
                skipped += 1
            if every and done % every == 0:
                rate = done / (time.perf_counter() - t0)
                print(
                    f"  [{label}] {done}/{len(triples)}  {time.perf_counter() - t0:.0f}s "
                    f"({rate:.1f} sim/s)  skipped={skipped}",
                    flush=True,
                )

    Xs, Ys, Fs, names = [], [], [], []
    for _, _, name in triples:
        r = results_by_name.get(name)
        if r is None:
            continue
        Xs.append(r[0])
        Ys.append(r[1])
        Fs.append(r[2])
        names.append(name)
    return np.asarray(Xs), np.asarray(Ys), np.asarray(Fs), names, skipped


def root_mse(y_true, y_pred):
    """RMSE in linear space."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def root_mse_log(y_true, y_pred, eps=1e-12):
    """RMSE in log space."""
    yt = np.maximum(np.asarray(y_true), eps)
    yp = np.maximum(np.asarray(y_pred), eps)
    return float(np.sqrt(np.mean((np.log(yt) - np.log(yp)) ** 2)))
