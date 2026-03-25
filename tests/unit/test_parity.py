from conftest import rel_l2, read_dump_time, read_dump_dtim
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyaradax.diag import term_iii_fft_pack_roundtrip, term_iii_rhs
from gyaradax.utils import load_runtime_params
from gyaradax.solver import (
    init_f,
    gksolve,
    GKState,
    default_state,
    linear_precompute,
    estimate_nl_timestep,
    estimate_linear_timestep,
    estimate_timestep,
)
from gyaradax.params import gkparams_from_input_dat
from gyaradax.utils import load_gkw_k_dump
from gyaradax.integrals import calculate_phi, calculate_phi_kinetic, geom_tensors


def test_init_f_trajectory_parity(nonlin_dir, nonlin_geom, nonlin_shape):
    """
    Verify that init_f exactly matches GKW's internal initial conditions.
    Since GKW does not output t=0 data (K00), we prove parity by initializing
    from scratch in JAX and integrating forward to match the K01 dump.
    """
    # GKW uses a default amp_init of 1e-4 if not specified
    from gyaradax.utils import parse_input_dat

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

    pred_df, _, _ = gksolve(df_init, nonlin_geom, params, state, 120)

    # We compare against K01, which is the dump at t=1.2 (120 steps)
    ref_df = load_gkw_k_dump(f"{nonlin_dir}/K01", nonlin_shape)

    # Check that error is extremely low (accounting for integrator drift over 120 steps)
    error = rel_l2(np.array(pred_df), np.array(ref_df))
    assert error < 1e-2


def test_init_f_kinetic_parity(kinetic_dir, kinetic_geom, kinetic_shape):
    """
    Verify kinetic gk_init: correct amp_init, species-array injection, and
    short forward integration with flux diagnostics.

    Checks:
    1. amp_init is parsed from input.dat and applied (not the default 1e-4).
    2. geometry carries per-species arrays after gk_init.
    3. A short gksolve (20 steps, fixed dt) runs without error and
       produces finite df, phi, and per-species fluxes.
    """
    from gyaradax import gk_init, gksolve
    from gyaradax.utils import parse_input_dat

    n_species = 2
    input_path = os.path.join(kinetic_dir, "input.dat")

    params = gkparams_from_input_dat(
        input_path,
        non_linear=True,
        adiabatic_electrons=False,
    )

    # amp_init must be parsed from input.dat (kinetic cases use 0.001)
    inp = parse_input_dat(input_path)
    expected_amp = float(
        inp.get("spcgeneral", {}).get(
            "amp_init", inp.get("components", {}).get("amp_init", 1e-4)
        )
    )
    assert params.amp_init == pytest.approx(expected_amp), (
        f"params.amp_init={params.amp_init} != input.dat amp_init={expected_amp}"
    )

    # gk_init should use the parsed amp_init and return 6D df
    df_init, geom_out, state = gk_init(kinetic_geom, params, n_species=n_species)
    assert df_init.ndim == 6
    assert df_init.shape[0] == n_species

    # init amplitude must reflect params.amp_init (max of cosine2 = 2*amp)
    expected_max = params.amp_init * 2.0
    actual_max = float(jnp.max(jnp.abs(df_init)))
    assert actual_max == pytest.approx(expected_max, rel=0.02), (
        f"init max|df|={actual_max:.4e} != 2*amp_init={expected_max:.4e}"
    )

    # geometry must carry per-species arrays
    for k in ("mas", "signz", "de", "tmp", "vthrat", "rlt", "rln"):
        assert jnp.asarray(geom_out[k]).shape[0] == n_species, (
            f"geometry[{k}] should have {n_species} species"
        )

    # short forward integration with fixed dt (safe for CFL)
    import dataclasses
    safe_params = dataclasses.replace(params, dt=0.002, adaptive_dt=False)
    pre = linear_precompute(geom_out, safe_params)
    pred_df, (phi, fluxes), final_state = gksolve(
        df_init, geom_out, safe_params, state, n_steps=20, pre=pre
    )

    assert jnp.all(jnp.isfinite(pred_df)), (
        f"df has {int(jnp.sum(~jnp.isfinite(pred_df)))} non-finite values"
    )
    assert jnp.all(jnp.isfinite(phi)), "phi should be finite"
    fluxes_arr = jnp.asarray(fluxes)
    assert fluxes_arr.shape == (n_species, 3)
    assert jnp.all(jnp.isfinite(fluxes_arr)), "fluxes should be finite"


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
    spec_kxky = jax.random.normal(key, (nkx, nky), dtype=jnp.float64) + 1j * jax.random.normal(
        key, (nkx, nky), dtype=jnp.float64
    )

    # zero out ky=0 to avoid parity issues at the DC component for the roundtrip identity
    spec_kxky = spec_kxky.at[:, 0].set(0.0)

    # roundtrip through dealiased grids
    repacked = term_iii_fft_pack_roundtrip(spec_kxky, nonlin_geom)

    assert repacked.shape == spec_kxky.shape
    # modes should be preserved (modulo floating point error)
    # we use a slightly more relaxed tolerance for the full complex roundtrip
    np.testing.assert_allclose(np.asarray(repacked), np.asarray(spec_kxky), rtol=1e-10, atol=1e-10)


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

    dt_est = estimate_nl_timestep(phi, pre, bessel, dt_input=float(params.dt), safety_factor=0.95)

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


# ── Linear CFL and combined estimate tests ──────────────────────────────────


def test_linear_cfl_kinetic_restricts_dt(kinetic_geom, kinetic_shape):
    """Linear CFL for kinetic electrons (vthrat~60) must be much smaller than input dt.

    With vthrat=60.6, parallel streaming speed is ~60x the ion reference.
    The linear CFL should give dt ~ sgr_dist / (vthrat * max|vpar|) ~ 0.002,
    far below the input.dat dt=0.004.
    """
    params = gkparams_from_input_dat(
        (
            os.path.join(
                os.path.dirname(kinetic_geom["_source_dir"]),
                "input.dat",
            )
            if "_source_dir" in kinetic_geom
            else "/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/v3_kiteration_991_half_rlt/input.dat"
        ),
        non_linear=True,
        adiabatic_electrons=False,
    )
    pre = linear_precompute(kinetic_geom, params)

    dt_lin = float(estimate_linear_timestep(pre, safety_factor=0.5))

    assert np.isfinite(dt_lin), "linear CFL estimate should be finite"
    assert dt_lin > 0, "linear CFL estimate should be positive"
    # Key assertion: linear CFL must be below 0.004 (the input dt)
    assert dt_lin < 0.004, (
        f"linear CFL {dt_lin:.6f} should be below input dt=0.004 "
        f"for kinetic electrons with vthrat~60"
    )
    # Should be in the right ballpark (~0.001-0.003)
    assert dt_lin > 1e-4, f"linear CFL {dt_lin:.6e} is unreasonably small"


def test_linear_cfl_adiabatic_is_loose(nonlin_geom, nonlin_shape):
    """For adiabatic ions (vthrat=1), linear CFL should not restrict dt much."""
    params = gkparams_from_input_dat(
        "/restricteddata/ukaea/gyrokinetics/raw/iteration_13/input.dat",
        non_linear=True,
    )
    pre = linear_precompute(nonlin_geom, params)

    dt_lin = float(estimate_linear_timestep(pre, safety_factor=0.5))

    assert np.isfinite(dt_lin), "linear CFL estimate should be finite"
    assert dt_lin > 0, "linear CFL estimate should be positive"
    # For adiabatic (vthrat=1), linear CFL should be much larger than dt
    assert dt_lin > float(params.dt), (
        f"linear CFL {dt_lin:.6f} should not restrict adiabatic runs "
        f"where vthrat=1 and dt={params.dt}"
    )


def test_combined_cfl_tighter_than_nl_alone(kinetic_dir, kinetic_geom, kinetic_shape):
    """Combined estimate must be <= nonlinear-only estimate for kinetic cases."""
    from gyaradax.utils import K_files

    ks = K_files(kinetic_dir)
    if len(ks) == 0:
        pytest.skip(f"no K files in {kinetic_dir}")

    df = load_gkw_k_dump(os.path.join(kinetic_dir, ks[0]), kinetic_shape, n_species=2)
    params = gkparams_from_input_dat(
        os.path.join(kinetic_dir, "input.dat"),
        non_linear=True,
        adiabatic_electrons=False,
    )
    pre = linear_precompute(kinetic_geom, params)
    phi = calculate_phi_kinetic(kinetic_geom, df)

    # ion Bessel for CFL (drop species dim)
    bessel = pre["bessel"][0]
    dt_input = float(params.dt)

    dt_nl = float(estimate_nl_timestep(phi, pre, bessel, dt_input, 0.95))
    dt_combined = float(estimate_timestep(phi, pre, bessel, dt_input, 0.95))

    assert dt_combined <= dt_nl + 1e-15, (
        f"combined CFL {dt_combined:.6e} should not exceed " f"nonlinear-only CFL {dt_nl:.6e}"
    )
    # For kinetic, linear CFL should actually be the binding constraint
    dt_lin = float(estimate_linear_timestep(pre, safety_factor=0.5))
    assert dt_combined <= dt_lin + 1e-15, (
        f"combined CFL {dt_combined:.6e} should be at most " f"linear CFL {dt_lin:.6e}"
    )


def test_combined_cfl_matches_gkw_adaptive_dt(kinetic_dir, kinetic_geom, kinetic_shape):
    """Combined CFL estimate should be in the same ballpark as GKW's adaptive dtim.

    GKW runs CFL adaptation and records the actual dtim in dump metadata.
    Our combined estimate should produce a dt of similar magnitude.
    """
    from gyaradax.utils import K_files

    ks = K_files(kinetic_dir)
    if len(ks) == 0:
        pytest.skip(f"no K files in {kinetic_dir}")

    dat_path = os.path.join(kinetic_dir, f"{ks[0]}.dat")
    if not os.path.exists(dat_path):
        pytest.skip(f"metadata {dat_path} not found")

    gkw_dtim = read_dump_dtim(dat_path)

    df = load_gkw_k_dump(os.path.join(kinetic_dir, ks[0]), kinetic_shape, n_species=2)
    params = gkparams_from_input_dat(
        os.path.join(kinetic_dir, "input.dat"),
        non_linear=True,
        adiabatic_electrons=False,
    )
    pre = linear_precompute(kinetic_geom, params)
    phi = calculate_phi_kinetic(kinetic_geom, df)
    bessel = pre["bessel"][0]

    dt_combined = float(estimate_timestep(phi, pre, bessel, float(params.dt), 0.95))

    # Our estimate should be within a factor of 3 of GKW's adaptive dtim
    ratio = dt_combined / gkw_dtim
    assert 0.3 < ratio < 3.0, (
        f"combined CFL {dt_combined:.6e} vs GKW dtim {gkw_dtim:.6e} "
        f"(ratio {ratio:.2f}) — should be within factor 3"
    )


def test_adaptive_dt_kinetic_no_nan(kinetic_dir, kinetic_geom, kinetic_shape):
    """Adaptive CFL must prevent NaN divergence for kinetic electron runs.

    Without CFL adaptation, dt=0.004 causes NaN within ~15 steps.
    With adaptive CFL, the simulation must remain finite.
    """
    from gyaradax.utils import K_files

    ks = K_files(kinetic_dir)
    if len(ks) == 0:
        pytest.skip(f"no K files in {kinetic_dir}")

    n_species = 2
    df = load_gkw_k_dump(os.path.join(kinetic_dir, ks[0]), kinetic_shape, n_species=n_species)

    dat_path = os.path.join(kinetic_dir, f"{ks[0]}.dat")
    t_start = read_dump_time(dat_path) if os.path.exists(dat_path) else 0.0

    # Use input.dat dt (0.004) which is too large without CFL
    params = gkparams_from_input_dat(
        os.path.join(kinetic_dir, "input.dat"),
        non_linear=True,
        adiabatic_electrons=False,
        adaptive_dt=True,
    )
    nky = len(kinetic_geom["krho"])
    state = GKState(
        time=jnp.array(t_start, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

    # 20 steps is enough to trigger NaN without CFL adaptation
    pred_df, (phi, _), _ = gksolve(df, kinetic_geom, params, state, n_steps=20)

    assert jnp.all(jnp.isfinite(pred_df)), (
        f"adaptive CFL must prevent NaN — "
        f"got {int(jnp.sum(~jnp.isfinite(pred_df)))} non-finite values"
    )
    assert jnp.all(jnp.isfinite(phi)), "phi must remain finite with adaptive CFL"
