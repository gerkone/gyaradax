import re
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyaradax.solver import gksolve
from gyaradax.params import GKParams, GKState, default_state, gkparams_from_input_dat
from gyaradax.utils import load_gkw_k_dump


@jax.jit
def _step_jitted(prev_df, geom, params, state):
    return gksolve(prev_df, geom, params, state, n_steps=1)


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
    return out if out else [(iyzero + 1) % nky]


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
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))


@pytest.mark.parametrize("start_name, end_name, steps", [("100", "101", 120)])
def test_iteration_parity(
    nonlin_dir, nonlin_geom, nonlin_shape, start_name, end_name, steps
):
    """verify trajectory parity against GKW reference dumps."""
    start_df = load_gkw_k_dump(f"{nonlin_dir}/{start_name}", nonlin_shape)
    end_df_ref = load_gkw_k_dump(f"{nonlin_dir}/{end_name}", nonlin_shape)

    params = gkparams_from_input_dat(f"{nonlin_dir}/input.dat", non_linear=True)
    nky = len(nonlin_geom["krho"])
    state = GKState(
        time=jnp.array(
            _read_dump_time(f"{nonlin_dir}/{start_name}.dat"), dtype=jnp.float64
        ),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

    pred_df, _, _ = jax.jit(gksolve, static_argnums=(4,))(
        start_df, nonlin_geom, params, state, steps
    )

    # validate subset of modes for parity
    mode_label = np.asarray(nonlin_geom["mode_label"], dtype=np.int32)
    ixzero, iyzero = int(nonlin_geom["ixzero"]), int(nonlin_geom["iyzero"])
    ky_sel = _selected_ky_representatives(iyzero, mode_label.shape[1])
    subset_mask_2d, _ = _subset_mask_from_mode_chains(mode_label, ixzero, ky_sel)

    pred_sub = np.asarray(pred_df) * subset_mask_2d[None, None, None, :, :]
    ref_sub = np.asarray(end_df_ref) * subset_mask_2d[None, None, None, :, :]

    # relaxed tolerance for multi-scenario parity check
    assert _rel_l2(pred_sub, ref_sub) <= 1.0e-3


def test_nonlinear_scaling(nonlin_geom, nonlin_shape):
    """verify quadratic scaling of the nonlinear term iii."""
    key = jax.random.PRNGKey(42)
    df_rand = jax.random.normal(key, nonlin_shape, dtype=jnp.float64) + 0j

    params_nl = GKParams(dt=0.01, non_linear=True, enable_term_iii=True)
    params_lin = GKParams(dt=0.01, non_linear=False)
    state = default_state(nky=len(nonlin_geom["krho"]))

    def get_nl_part(amp):
        df = amp * df_rand
        next_nl, _, _ = _step_jitted(df, nonlin_geom, params_nl, state)
        next_lin, _, _ = _step_jitted(df, nonlin_geom, params_lin, state)
        return next_nl - next_lin

    diff1 = get_nl_part(1e-5)
    diff2 = get_nl_part(2e-5)
    ratio = np.linalg.norm(diff2) / np.linalg.norm(diff1)
    assert 3.9 <= ratio <= 4.1
