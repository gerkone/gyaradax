import os
import jax
import jax.numpy as jnp
import numpy as np

from gyaradax.diag import term_iii_fft_pack_roundtrip, term_iii_rhs
from gyaradax.geometry import load_runtime_params
from gyaradax.solver import (
    init_f,
    gksolve,
    default_state,
    linear_precompute,
    estimate_nl_timestep,
)
from gyaradax.params import gkparams_from_input_dat
from gyaradax.utils import load_gkw_k_dump
from gyaradax.integrals import calculate_phi, geom_tensors


def _rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-30) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))


def test_init_f_trajectory_parity(nonlin_dir, nonlin_geom, nonlin_shape):
    """
    Verify that init_f exactly matches GKW's internal initial conditions.
    Since GKW does not output t=0 data (K00), we prove parity by initializing
    from scratch in JAX and integrating forward to match the K01 dump.
    """
    # GKW uses a default amp_init of 1e-4 if not specified
    from gyaradax.geometry import parse_input_dat

    inp = parse_input_dat(f"{nonlin_dir}/input.dat")
    amp_init = inp.get("spcgeneral", {}).get("amp_init")
    if amp_init is None:
        amp_init = inp.get("components", {}).get("amp_init", 1.0e-4)
    amp_init = float(amp_init)

    finit = inp.get("spcgeneral", {}).get("finit")
    if finit is None:
        finit = inp.get("components", {}).get("finit", "cosine2")

    # We ensure we don't normalize at t=0, matching GKW's behavior when phi=0.
    df_init = init_f(
        nonlin_geom,
        finit=finit,
        amp_init_real=amp_init,
        normalize_per_toroidal_mode=False,
    )

    params = gkparams_from_input_dat(f"{nonlin_dir}/input.dat", non_linear=True)
    nky = len(nonlin_geom["krho"])

    # 120 steps reaches K01 in iteration_13
    state = default_state(nky=nky)

    pred_df, _, _ = jax.jit(gksolve, static_argnames="n_steps")(
        df_init, nonlin_geom, params, state, 120
    )

    # We compare against K01, which is the dump at t=1.2 (120 steps)
    ref_df = load_gkw_k_dump(f"{nonlin_dir}/K01", nonlin_shape)

    # Check that error is extremely low (accounting for integrator drift over 120 steps)
    error = _rel_l2(np.array(pred_df), np.array(ref_df))
    assert error < 1e-2


def test_runtime_params_types_and_values(nonlin_dir):
    """verify that runtime parameters are parsed with correct types."""
    runtime = load_runtime_params(os.path.join(nonlin_dir, "input.dat"))

    assert isinstance(runtime["dtim"], float)
    assert isinstance(runtime["naverage"], int)
    assert isinstance(runtime["non_linear"], bool)
    assert isinstance(runtime["method"], str)


def test_term_iii_fft_roundtrip(nonlin_geom, nonlin_shape):
    """verify pseudospectral fft roundtrip preserves physical modes."""
    key = jax.random.PRNGKey(123)
    nkx, nky = nonlin_shape[3], nonlin_shape[4]
    spec_kxky = jax.random.normal(
        key, (nkx, nky), dtype=jnp.float64
    ) + 1j * jax.random.normal(key, (nkx, nky), dtype=jnp.float64)

    # zero out ky=0 to avoid parity issues at the DC component for the roundtrip identity
    spec_kxky = spec_kxky.at[:, 0].set(0.0)

    # roundtrip through dealiased grids
    repacked = term_iii_fft_pack_roundtrip(spec_kxky, nonlin_geom)

    assert repacked.shape == spec_kxky.shape
    # modes should be preserved (modulo floating point error)
    # we use a slightly more relaxed tolerance for the full complex roundtrip
    np.testing.assert_allclose(
        np.asarray(repacked), np.asarray(spec_kxky), rtol=1e-10, atol=1e-10
    )


def test_term_iii_rhs_shapes(nonlin_geom, nonlin_shape):
    """verify nonlinear term iii output shape."""
    df = jnp.zeros(nonlin_shape, dtype=jnp.complex128)
    rhs_nl = term_iii_rhs(df, nonlin_geom)
    assert rhs_nl.shape == nonlin_shape


def test_cfl_timestep_estimate(nonlin_dir, nonlin_geom, nonlin_shape):
    """verify that the cfl estimate produces reasonable timesteps.

    the estimate should be finite, positive, and comparable to the reference dt
    for a typical nonlinear state.
    """
    df = load_gkw_k_dump(f"{nonlin_dir}/100", nonlin_shape)
    params = gkparams_from_input_dat(f"{nonlin_dir}/input.dat", non_linear=True)
    pre = linear_precompute(nonlin_geom, params)

    phi = calculate_phi(geom_tensors(nonlin_geom, params=params), df)
    bessel = pre["bessel"]

    dt_est = estimate_nl_timestep(
        phi, pre, bessel, dt_input=float(params.dt), safety_factor=0.95
    )

    dt_est_val = float(dt_est)
    assert np.isfinite(dt_est_val), "cfl estimate should be finite"
    assert dt_est_val > 0, "cfl estimate should be positive"
    assert dt_est_val <= float(params.dt), "cfl estimate should not exceed dt_input"
    # for a turbulent state, cfl should be within an order of magnitude of dt
    assert (
        dt_est_val > float(params.dt) * 0.01
    ), f"cfl estimate {dt_est_val:.3e} is unreasonably small vs dt={params.dt}"


def test_cfl_zero_phi_returns_dt_input(nonlin_geom, nonlin_shape):
    """with zero potential, cfl estimate should return dt_input (no constraint)."""
    params = gkparams_from_input_dat(
        "/restricteddata/ukaea/gyrokinetics/raw/iteration_13/input.dat",
        non_linear=True,
    )
    pre = linear_precompute(nonlin_geom, params)
    phi_zero = jnp.zeros(nonlin_shape[2:], dtype=jnp.complex128)
    bessel = pre["bessel"]

    dt_est = estimate_nl_timestep(
        phi_zero, pre, bessel, dt_input=float(params.dt), safety_factor=0.95
    )
    assert float(dt_est) == float(params.dt), "zero phi should give dt_input"
