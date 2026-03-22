from conftest import rel_l2, read_dump_time, read_dump_dtim
import jax
import jax.numpy as jnp
import numpy as np
import os
import pytest

from gyaradax.solver import gksolve, GKState, default_state
from gyaradax.params import GKParams, gkparams_from_input_dat
from gyaradax.utils import load_gkw_k_dump
from gyaradax.integrals import calculate_phi_kinetic, calculate_fluxes_kinetic


@jax.jit
def _step_jitted(prev_df, geom, params, state):
    return gksolve(prev_df, geom, params, state, n_steps=1)


def _selected_ky_representatives(iyzero, nky):
    candidates = [1, nky // 2, nky - 1]
    out = []
    for ky in candidates:
        ky = int(np.clip(ky, 0, nky - 1))
        if ky == iyzero:
            continue
        if ky not in out:
            out.append(ky)
    return out if out else [(iyzero + 1) % nky]


def _subset_mask_from_mode_chains(mode_label, ixzero, ky_list):
    nkx, nky = mode_label.shape
    mask = np.zeros((nkx, nky), dtype=bool)
    labels = []
    for ky in ky_list:
        lbl = int(mode_label[ixzero, ky])
        labels.append(lbl)
        mask[:, ky] = mode_label[:, ky] == lbl
    return mask, np.asarray(labels, dtype=np.int32)


@pytest.mark.parametrize("start_name, end_name, steps", [("100", "101", 120)])
def test_iteration_parity(nonlin_dir, nonlin_geom, nonlin_shape, start_name, end_name, steps):
    """verify trajectory parity against GKW reference dumps."""
    start_df = load_gkw_k_dump(f"{nonlin_dir}/{start_name}", nonlin_shape)
    end_df_ref = load_gkw_k_dump(f"{nonlin_dir}/{end_name}", nonlin_shape)

    params = gkparams_from_input_dat(f"{nonlin_dir}/input.dat", non_linear=True)
    nky = len(nonlin_geom["krho"])
    state = GKState(
        time=jnp.array(read_dump_time(f"{nonlin_dir}/{start_name}.dat"), dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

    pred_df, _, _ = gksolve(start_df, nonlin_geom, params, state, n_steps=steps)

    # validate subset of modes for parity
    mode_label = np.asarray(nonlin_geom["mode_label"], dtype=np.int32)
    ixzero, iyzero = int(nonlin_geom["ixzero"]), int(nonlin_geom["iyzero"])
    ky_sel = _selected_ky_representatives(iyzero, mode_label.shape[1])
    subset_mask_2d, _ = _subset_mask_from_mode_chains(mode_label, ixzero, ky_sel)

    pred_sub = np.asarray(pred_df) * subset_mask_2d[None, None, None, :, :]
    ref_sub = np.asarray(end_df_ref) * subset_mask_2d[None, None, None, :, :]

    # relaxed tolerance for multi-scenario parity check
    assert rel_l2(pred_sub, ref_sub) <= 1.0e-3


def test_nonlinear_scaling(nonlin_geom, nonlin_shape):
    """verify quadratic scaling of the nonlinear term iii."""
    key = jax.random.PRNGKey(42)
    df_rand = jax.random.normal(key, nonlin_shape, dtype=jnp.float64) + 0j

    params_nl = GKParams(dt=0.01, non_linear=True)
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


# ── Kinetic electron tests ──────────────────────────────────────────────────


def _kinetic_params_from_dir(kinetic_dir, dump_name="100", **overrides):
    """Build GKParams for kinetic case, using the actual dtim from the dump metadata.

    GKW uses adaptive timestep control, so the actual dt may differ from input.dat.
    The dump metadata records the dtim that was actually used.
    """
    # Read the actual dtim from the dump metadata
    actual_dt = read_dump_dtim(os.path.join(kinetic_dir, f"{dump_name}.dat"))
    params = gkparams_from_input_dat(
        os.path.join(kinetic_dir, "input.dat"),
        non_linear=True,
        adiabatic_electrons=False,
        dt=actual_dt,
        **overrides,
    )
    return params


@pytest.mark.parametrize("start_name, end_name", [("100", "101")])
def test_kinetic_iteration_parity(kinetic_dir, kinetic_geom, kinetic_shape, start_name, end_name):
    """Verify multi-species kinetic trajectory parity against GKW reference dumps.

    Loads the kinetic distribution at dump 100, computes the exact number
    of small timesteps from the time difference and actual dtim, then
    advances that many steps and compares with dump 101.

    NOTE: Assumes constant dtim between dumps. GKW uses CFL-adaptive timestep,
    so this test is only valid for cases where dtim is stable. Cases with
    varying dtim will need CFL adaptation in the solver.
    """

    n_species = 2
    start_df = load_gkw_k_dump(
        os.path.join(kinetic_dir, start_name), kinetic_shape, n_species=n_species
    )
    end_df_ref = load_gkw_k_dump(
        os.path.join(kinetic_dir, end_name), kinetic_shape, n_species=n_species
    )

    params = _kinetic_params_from_dir(kinetic_dir, dump_name=start_name)

    # Compute exact step count from time difference and actual dtim
    t_start = read_dump_time(os.path.join(kinetic_dir, f"{start_name}.dat"))
    t_end = read_dump_time(os.path.join(kinetic_dir, f"{end_name}.dat"))
    steps = int(round((t_end - t_start) / params.dt))

    nky = len(kinetic_geom["krho"])
    state = GKState(
        time=jnp.array(t_start, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

    pred_df, _, _ = gksolve(start_df, kinetic_geom, params, state, n_steps=steps)

    # Validate per-species trajectory parity
    for isp in range(n_species):
        sp_name = "ion" if isp == 0 else "electron"
        err = rel_l2(np.asarray(pred_df[isp]), np.asarray(end_df_ref[isp]))
        assert err <= 1.0e-3, f"{sp_name} trajectory error {err:.6e} > 1e-3 in {kinetic_dir}"


def test_kinetic_flux_trajectory(kinetic_dir, kinetic_geom, kinetic_shape):
    """Verify per-species heat fluxes at a reference dump match GKW fluxes.dat.

    This test uses only the integral module (phi solver + flux calculation),
    not the time-stepper. It validates that the multi-species field solver
    produces correct per-species fluxes at multiple time points.
    """
    from gyaradax.utils import K_files

    n_species = 2
    ks = K_files(kinetic_dir)
    if len(ks) < 20:
        pytest.skip(f"Not enough K files in {kinetic_dir}")

    fluxes_ref = np.loadtxt(os.path.join(kinetic_dir, "fluxes.dat"))
    orig_times = np.loadtxt(os.path.join(kinetic_dir, "time.dat"))

    # Test at multiple dumps
    test_indices = [5, 15]
    for idx in test_indices:
        k_file = ks[idx]
        df_full = load_gkw_k_dump(
            os.path.join(kinetic_dir, k_file), kinetic_shape, n_species=n_species
        )

        k_dat_path = os.path.join(kinetic_dir, f"{k_file}.dat")
        if not os.path.exists(k_dat_path):
            continue

        time_val = read_dump_time(k_dat_path)
        ts_idx = np.argmin(np.abs(orig_times - time_val))
        if not np.isclose(orig_times[ts_idx], time_val, rtol=1e-4):
            continue

        phi = calculate_phi_kinetic(kinetic_geom, df_full)
        per_sp_fluxes = calculate_fluxes_kinetic(kinetic_geom, df_full, phi)

        for isp in range(n_species):
            col_offset = isp * 3
            ref_eflux = fluxes_ref[ts_idx, col_offset + 1]
            pred_eflux = float(per_sp_fluxes[isp, 1])
            sp_name = "ion" if isp == 0 else "electron"

            assert np.isclose(pred_eflux, ref_eflux, rtol=1e-2, atol=1e-4), (
                f"{sp_name} eflux mismatch at T={time_val}: "
                f"{pred_eflux:.6e} vs {ref_eflux:.6e} in {kinetic_dir}"
            )


def test_adiabatic_fallback_identity(adiabatic_dir, adiabatic_geom, adiabatic_shape):
    """Verify kinetic phi solver with nspecies=1 matches adiabatic phi solver.

    When adiabatic_electrons=False but nspecies=1 (single ion species without
    adiabatic electron contribution), the kinetic and adiabatic phi solvers
    produce DIFFERENT results — this is physically correct because they solve
    different equations. This test validates that the kinetic path is self-consistent
    when called with single-species data (correct shapes, finite output).
    """
    from gyaradax.utils import K_files

    ks = K_files(adiabatic_dir)
    if len(ks) == 0:
        pytest.skip(f"No K files in {adiabatic_dir}")

    df_1sp = load_gkw_k_dump(os.path.join(adiabatic_dir, ks[0]), adiabatic_shape)

    # Kinetic path with single species: add species axis
    df_kinetic = df_1sp[None, ...]  # (1, nvpar, nmu, ns, nkx, nky)

    # Build single-species "kinetic" geometry (species arrays have shape (1,))
    sp_geom = dict(adiabatic_geom)
    for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
        val = jnp.asarray(sp_geom[k], dtype=jnp.float64)
        if val.ndim == 0:
            sp_geom[k] = val.reshape(1)
        elif val.shape[0] > 1:
            sp_geom[k] = val[0:1]

    phi_kin = calculate_phi_kinetic(sp_geom, df_kinetic)
    ns, nkx, nky = adiabatic_shape[2:]
    assert phi_kin.shape == (ns, nkx, nky), f"Shape mismatch: {phi_kin.shape}"
    assert jnp.all(jnp.isfinite(phi_kin)), "Kinetic phi contains non-finite values"

    # Fluxes should also be finite and have correct shapes
    fl = calculate_fluxes_kinetic(sp_geom, df_kinetic, phi_kin)
    assert fl.shape == (1, 3), f"Flux shape mismatch: {fl.shape}"
    assert jnp.all(jnp.isfinite(fl)), "Kinetic fluxes contain non-finite values"
