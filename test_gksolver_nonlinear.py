import os
import re

import jax
import jax.numpy as jnp
import numpy as np

from gksolver import GKParams, GKState, default_state, gkparams_from_input_dat, gksolve_with_state
from jax_geometry import load_geometry
from jax_integrals import get_integrals
from utils import load_gkw_k_dump

# Ensure fp64 everywhere.
jax.config.update("jax_enable_x64", True)


NONLIN_DIR = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13"


@jax.jit
def _step_jitted(prev_df, geom, params, state):
    return gksolve_with_state(prev_df, geom, params, state)


def _grid_resolution(geom):
    return (
        len(geom["intvp"]),
        len(geom["intmu"]),
        len(geom["ints"]),
        len(geom["kxrh"]),
        len(geom["krho"]),
    )


def _read_dump_time(dat_path: str) -> float:
    with open(dat_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"TIME\s*=\s*([0-9eE+\-.]+)", text)
    if m is None:
        raise ValueError(f"TIME entry not found in {dat_path}")
    return float(m.group(1))


def _selected_ky_representatives(iyzero: int, nky: int):
    candidates = [1, nky // 2, nky - 1]
    out = []
    for ky in candidates:
        ky = int(np.clip(ky, 0, nky - 1))
        if ky == iyzero:
            continue
        if ky not in out:
            out.append(ky)
    if not out:
        out = [int((iyzero + 1) % nky)]
    return out


def _subset_mask_from_mode_chains(mode_label, ixzero: int, ky_list):
    nkx, nky = mode_label.shape
    mask = np.zeros((nkx, nky), dtype=bool)
    labels = []
    for ky in ky_list:
        lbl = int(mode_label[ixzero, ky])
        labels.append(lbl)
        mask[:, ky] = mode_label[:, ky] == lbl
    return mask, np.asarray(labels, dtype=np.int32)


def _rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-30) -> float:
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref) + eps
    return float(num / den)


def test_nonlinear_switch_off_matches_linear_for_same_step():
    geom = load_geometry(NONLIN_DIR)
    shape = _grid_resolution(geom)
    key_r, key_i = jax.random.split(jax.random.PRNGKey(7))
    prev_df = (
        jax.random.normal(key_r, shape, dtype=jnp.float64)
        + 1j * jax.random.normal(key_i, shape, dtype=jnp.float64)
    ) * 1.0e-4

    params_linear = GKParams(non_linear=False)
    params_switch_off = GKParams(non_linear=True, enable_term_iii=False)
    state = default_state()

    out_linear = _step_jitted(prev_df, geom, params_linear, state)
    out_switch = _step_jitted(prev_df, geom, params_switch_off, state)

    df_l, (phi_l, flux_l), st_l = out_linear
    df_s, (phi_s, flux_s), st_s = out_switch

    np.testing.assert_allclose(np.asarray(df_l), np.asarray(df_s), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(phi_l), np.asarray(phi_s), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(flux_l[0]), np.asarray(flux_s[0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(flux_l[1]), np.asarray(flux_s[1]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(flux_l[2]), np.asarray(flux_s[2]), rtol=0.0, atol=0.0)
    assert int(np.asarray(st_l.step)) == int(np.asarray(st_s.step))


def test_nonlinear_term_zero_input_invariant_and_finite():
    geom = load_geometry(NONLIN_DIR)
    prev_df = jnp.zeros(_grid_resolution(geom), dtype=jnp.complex128)
    params = GKParams(non_linear=True, enable_term_iii=True)
    state = default_state()

    next_df, (phi, fluxes), next_state = _step_jitted(prev_df, geom, params, state)
    pflux, eflux, vflux = fluxes

    assert jnp.allclose(next_df, 0.0)
    assert jnp.allclose(phi, 0.0)
    assert jnp.allclose(pflux, 0.0)
    assert jnp.allclose(eflux, 0.0)
    assert jnp.allclose(vflux, 0.0)
    assert jnp.isfinite(next_df).all()
    assert jnp.isfinite(phi).all()
    assert jnp.isfinite(next_state.time)


def test_nonlinear_term_scales_quadratically_against_linear_part():
    geom = load_geometry(NONLIN_DIR)
    shape = _grid_resolution(geom)
    key_r, key_i = jax.random.split(jax.random.PRNGKey(123))
    base_df = (
        jax.random.normal(key_r, shape, dtype=jnp.float64)
        + 1j * jax.random.normal(key_i, shape, dtype=jnp.float64)
    )

    state = default_state()
    params_nl = GKParams(dt=0.01, non_linear=True, enable_term_iii=True, naverage=10_000)
    params_lin = GKParams(dt=0.01, non_linear=False, enable_term_iii=False, naverage=10_000)

    amp1 = 1.0e-5
    amp2 = 2.0e-5
    df1 = amp1 * base_df
    df2 = amp2 * base_df

    nl1, _, _ = _step_jitted(df1, geom, params_nl, state)
    li1, _, _ = _step_jitted(df1, geom, params_lin, state)
    nl2, _, _ = _step_jitted(df2, geom, params_nl, state)
    li2, _, _ = _step_jitted(df2, geom, params_lin, state)

    d1 = np.asarray(nl1 - li1)
    d2 = np.asarray(nl2 - li2)
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    ratio = n2 / max(n1, 1.0e-30)
    # Term III is quadratic in amplitude: (2a)^2 / (a^2) = 4.
    assert 3.0 <= ratio <= 5.0


def test_iteration13_checkpoint_smoke_window_120_steps_subset_and_diagnostics():
    geom = load_geometry(NONLIN_DIR)
    res = _grid_resolution(geom)
    start_name = "100"
    end_name = "101"

    start_df = load_gkw_k_dump(f"{NONLIN_DIR}/{start_name}", res)
    end_df_ref = load_gkw_k_dump(f"{NONLIN_DIR}/{end_name}", res)

    params = gkparams_from_input_dat(
        f"{NONLIN_DIR}/input.dat",
        non_linear=True,
        enable_term_iii=True,
    )
    state = GKState(
        time=jnp.array(_read_dump_time(f"{NONLIN_DIR}/{start_name}.dat"), dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.array(1.0, dtype=jnp.float64),
        window_start_amp=jnp.array(1.0, dtype=jnp.float64),
        last_growth_rate=jnp.array(0.0, dtype=jnp.float64),
    )

    def _scan_step(carry, _):
        df, st = carry
        next_df, _, next_st = gksolve_with_state(df, geom, params, st)
        return (next_df, next_st), None

    run_120 = jax.jit(lambda df0, st0: jax.lax.scan(_scan_step, (df0, st0), None, length=120)[0])
    pred_df, pred_state = run_120(start_df, state)
    assert int(np.asarray(pred_state.step)) == 120
    assert np.isfinite(float(np.asarray(pred_state.time)))

    # Subset checkpoint error on connected kx chains containing kx=0 for representative ky.
    mode_label = np.asarray(geom["mode_label"], dtype=np.int32)
    ixzero = int(np.asarray(geom["ixzero"]))
    iyzero = int(np.asarray(geom["iyzero"]))
    ky_sel = _selected_ky_representatives(iyzero, mode_label.shape[1])
    subset_mask_2d, subset_labels = _subset_mask_from_mode_chains(mode_label, ixzero, ky_sel)
    subset_mask_5d = subset_mask_2d[None, None, None, :, :]

    pred_sub = np.asarray(pred_df) * subset_mask_5d
    ref_sub = np.asarray(end_df_ref) * subset_mask_5d
    subset_rel_l2 = _rel_l2(pred_sub, ref_sub)
    assert subset_rel_l2 <= 1.0e-1

    # Selected-label growth comparison over this 1.2-time-unit window.
    phi_start, _ = get_integrals(start_df, geom)
    phi_end_pred, flux_pred = get_integrals(pred_df, geom)
    pflux, eflux, vflux = flux_pred
    assert np.isfinite(float(np.asarray(pflux)))
    assert np.isfinite(float(np.asarray(eflux)))
    assert np.isfinite(float(np.asarray(vflux)))

    ds = float(np.asarray(geom["ints"])[0])
    amp_start = np.sqrt(ds * np.sum(np.abs(np.asarray(phi_start)) ** 2, axis=0))
    amp_end = np.sqrt(ds * np.sum(np.abs(np.asarray(phi_end_pred)) ** 2, axis=0))
    growth_pred = np.log(
        np.maximum(amp_end[ixzero, ky_sel], 1.0e-30) / np.maximum(amp_start[ixzero, ky_sel], 1.0e-30)
    ) / 1.2

    growth_all = np.loadtxt(f"{NONLIN_DIR}/growth_rates_all_modes")
    times = np.loadtxt(f"{NONLIN_DIR}/time.dat")
    t_start = _read_dump_time(f"{NONLIN_DIR}/{start_name}.dat")
    t_end = _read_dump_time(f"{NONLIN_DIR}/{end_name}.dat")
    i_start = int(np.argmin(np.abs(times - t_start)))
    i_end = int(np.argmin(np.abs(times - t_end)))
    i_lo = min(i_start + 1, i_end)
    i_hi = max(i_start + 1, i_end)
    cols = subset_labels - 1
    growth_ref = np.mean(growth_all[i_lo : i_hi + 1, cols], axis=0)
    # For near-marginal modes, pure relative error is numerically unstable.
    growth_rel_err = np.max(np.abs(growth_pred - growth_ref) / np.maximum(np.abs(growth_ref), 2.0e-2))
    assert growth_rel_err <= 1.5e-1

    # Diagnostics trend consistency at the end window time.
    flux_ref = np.loadtxt(f"{NONLIN_DIR}/fluxes.dat")
    eflux_ref = float(flux_ref[i_end, 1])
    eflux_pred = float(np.asarray(eflux))
    eflux_rel = abs(eflux_pred - eflux_ref) / max(abs(eflux_ref), 1.0e-12)
    assert eflux_rel <= 2.0e-1
