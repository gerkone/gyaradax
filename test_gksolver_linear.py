import os
import numpy as np
import jax
import jax.numpy as jnp

from gksolver import (
    GKParams,
    default_state,
    gksolve,
    gksolve_with_state,
    init_df_cosine2,
    project_all_modes_to_kx0,
)
from jax_geometry import load_geometry, parse_input_dat
from jax_integrals import get_integrals
from utils import poten_files

# Ensure fp64 everywhere.
jax.config.update("jax_enable_x64", True)


LIN_DIR = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13_Lin"


def _build_zero_df(geom):
    shape = (
        len(geom["intvp"]),
        len(geom["intmu"]),
        len(geom["ints"]),
        len(geom["kxrh"]),
        len(geom["krho"]),
    )
    return jnp.zeros(shape, dtype=jnp.complex128)


def test_geometry_has_connectivity_and_active_linear_keys():
    geom = load_geometry(LIN_DIR)
    ns = len(geom["ints"])
    nkx = len(geom["kxrh"])
    nky = len(geom["krho"])

    assert "gfun" in geom
    assert "dfun" in geom
    assert "mode_label" in geom
    assert "ixplus" in geom
    assert "ixminus" in geom
    assert "ixzero" in geom
    assert "iyzero" in geom
    assert "pos_par_grid_class" in geom
    assert "s_shift" in geom
    assert "kx_shift" in geom
    assert "valid_shift" in geom

    assert geom["gfun"].shape == (ns,)
    assert geom["dfun"].shape == (ns, 3)
    assert geom["mode_label"].shape == (nkx, nky)
    assert geom["ixplus"].shape == (nkx, nky)
    assert geom["ixminus"].shape == (nkx, nky)
    assert geom["pos_par_grid_class"].shape == (ns, nkx, nky)
    assert geom["s_shift"].shape == (9, ns, nkx, nky)
    assert geom["kx_shift"].shape == (9, ns, nkx, nky)
    assert geom["valid_shift"].shape == (9, ns, nkx, nky)

    ixzero = int(geom["ixzero"])
    iyzero = int(geom["iyzero"])
    assert ixzero == int(np.argmin(np.abs(np.asarray(geom["kxrh"]))))
    assert iyzero == int(np.argmin(np.abs(np.asarray(geom["krho"]))))

    # ky=0 mode is periodic over parallel boundaries.
    ix = np.arange(nkx, dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(geom["ixplus"])[:, iyzero], ix)
    np.testing.assert_array_equal(np.asarray(geom["ixminus"])[:, iyzero], ix)

    # For a non-zonal ky, open boundaries map to pos_par_grid edge classes.
    non_zonal_ky = 1 if iyzero == 0 else 0
    ixplus = np.asarray(geom["ixplus"])[:, non_zonal_ky]
    ixminus = np.asarray(geom["ixminus"])[:, non_zonal_ky]
    pos = np.asarray(geom["pos_par_grid_class"])[:, :, non_zonal_ky]

    left_open = ixminus < 0
    right_open = ixplus < 0
    assert np.all(pos[0, left_open] == -2)
    assert np.all(pos[1, left_open] == -1)
    assert np.all(pos[-1, right_open] == 2)
    assert np.all(pos[-2, right_open] == 1)


def test_init_df_cosine2_contract_and_zonal_suppression():
    geom = load_geometry(LIN_DIR)
    df = init_df_cosine2(
        geom,
        amp_init_real=1.0e-4,
        amp_init_imag=0.0,
        normalize_per_toroidal_mode=False,
    )

    expected_shape = (
        len(geom["intvp"]),
        len(geom["intmu"]),
        len(geom["ints"]),
        len(geom["kxrh"]),
        len(geom["krho"]),
    )
    assert df.shape == expected_shape
    assert df.dtype == jnp.complex128

    iyzero = int(geom["iyzero"])
    assert jnp.allclose(df[..., iyzero], 0.0)

    non_zonal_ky = 1 if iyzero == 0 else 0
    sgrid = np.asarray(geom["sgrid"])
    expected_profile = 1.0e-4 * (np.cos(2.0 * np.pi * sgrid) + 1.0)
    sample = np.asarray(df[0, 0, :, 0, non_zonal_ky]).real
    np.testing.assert_allclose(sample, expected_profile, rtol=0.0, atol=0.0)


def test_init_df_cosine2_startup_normalization_per_mode():
    geom = load_geometry(LIN_DIR)
    raw_df = init_df_cosine2(
        geom,
        amp_init_real=1.0e-4,
        amp_init_imag=0.0,
        normalize_per_toroidal_mode=False,
    )
    norm_df = init_df_cosine2(
        geom,
        amp_init_real=1.0e-4,
        amp_init_imag=0.0,
        normalize_per_toroidal_mode=True,
    )

    raw_phi, _ = get_integrals(raw_df, geom)
    norm_phi, _ = get_integrals(norm_df, geom)

    ds = float(np.asarray(geom["ints"])[0])
    amp_raw = np.sqrt(ds * np.sum(np.abs(np.asarray(raw_phi)) ** 2, axis=(0, 1)))
    amp_norm = np.sqrt(ds * np.sum(np.abs(np.asarray(norm_phi)) ** 2, axis=(0, 1)))

    iyzero = int(geom["iyzero"])
    assert amp_raw[iyzero] == 0.0
    assert amp_norm[iyzero] == 0.0

    non_zonal = np.arange(len(geom["krho"])) != iyzero
    np.testing.assert_allclose(amp_norm[non_zonal], 1.0, rtol=1.0e-11, atol=1.0e-11)


def test_gksolve_shape_dtype_and_contract():
    geom = load_geometry(LIN_DIR)
    prev_df = _build_zero_df(geom)

    params = GKParams(dt=0.01, naverage=40)
    state = default_state()

    next_df, (phi, fluxes) = gksolve(prev_df, geom, params, state)
    pflux, eflux, vflux = fluxes

    assert next_df.shape == prev_df.shape
    assert next_df.dtype == jnp.complex128
    assert phi.shape == (len(geom["ints"]), len(geom["kxrh"]), len(geom["krho"]))
    assert phi.dtype == jnp.complex128
    assert pflux.shape == ()
    assert eflux.shape == ()
    assert vflux.shape == ()
    assert pflux.dtype == jnp.float64
    assert eflux.dtype == jnp.float64
    assert vflux.dtype == jnp.float64


def test_gksolve_with_state_is_jittable():
    geom = load_geometry(LIN_DIR)
    prev_df = _build_zero_df(geom)
    params = GKParams(dt=0.01, naverage=40)
    state = default_state()

    jitted = jax.jit(gksolve_with_state)
    next_df, (phi, fluxes), next_state = jitted(prev_df, geom, params, state)

    assert next_df.shape == prev_df.shape
    assert phi.shape == (len(geom["ints"]), len(geom["kxrh"]), len(geom["krho"]))
    assert jnp.isfinite(next_state.time)
    assert next_state.step == 1
    assert all(jnp.isfinite(val) for val in fluxes)


def test_gksolve_zero_input_invariance():
    geom = load_geometry(LIN_DIR)
    prev_df = _build_zero_df(geom)
    params = GKParams(dt=0.01, naverage=40)
    state = default_state()

    next_df, (phi, fluxes), next_state = gksolve_with_state(prev_df, geom, params, state)

    pflux, eflux, vflux = fluxes
    assert jnp.allclose(next_df, 0.0)
    assert jnp.allclose(phi, 0.0)
    assert jnp.allclose(pflux, 0.0)
    assert jnp.allclose(eflux, 0.0)
    assert jnp.allclose(vflux, 0.0)
    assert next_state.step == 1
    assert jnp.isfinite(next_state.accumulated_norm_factor)


def test_gksolve_is_deterministic_and_finite():
    geom = load_geometry(LIN_DIR)
    shape = (
        len(geom["intvp"]),
        len(geom["intmu"]),
        len(geom["ints"]),
        len(geom["kxrh"]),
        len(geom["krho"]),
    )
    key_r, key_i = jax.random.split(jax.random.PRNGKey(42))
    prev_df = (
        jax.random.normal(key_r, shape, dtype=jnp.float64)
        + 1j * jax.random.normal(key_i, shape, dtype=jnp.float64)
    ) * 1.0e-4

    params = GKParams(dt=0.01, naverage=40)
    state = default_state()

    out1 = gksolve_with_state(prev_df, geom, params, state)
    out2 = gksolve_with_state(prev_df, geom, params, state)

    next_df1, (phi1, fluxes1), state1 = out1
    next_df2, (phi2, fluxes2), state2 = out2

    assert jnp.allclose(next_df1, next_df2)
    assert jnp.allclose(phi1, phi2)
    assert all(jnp.allclose(a, b) for a, b in zip(fluxes1, fluxes2))
    assert jnp.isfinite(next_df1).all()
    assert jnp.isfinite(phi1).all()
    assert all(jnp.isfinite(val) for val in fluxes1)
    assert jnp.isfinite(state1.time)
    assert jnp.isfinite(state2.time)


def test_gksolve_normalizes_only_at_naverage_boundaries():
    geom = load_geometry(LIN_DIR)
    params = GKParams(dt=0.01, naverage=4, disp_par=1.0, disp_vp=0.2, disp_x=0.1, disp_y=0.1)
    state = default_state()

    df = init_df_cosine2(geom, normalize_per_toroidal_mode=True)

    for _ in range(3):
        df, _, state = gksolve_with_state(df, geom, params, state)

    phi3, _ = get_integrals(df, geom)
    ds = float(np.asarray(geom["ints"])[0])
    amp3 = np.sqrt(ds * np.sum(np.abs(np.asarray(phi3)) ** 2, axis=(0, 1)))
    iyzero = int(geom["iyzero"])
    non_zonal = np.arange(len(geom["krho"])) != iyzero

    assert int(np.asarray(state.step)) == 3
    # No normalization applied yet.
    np.testing.assert_allclose(float(np.asarray(state.accumulated_norm_factor)), 1.0, rtol=0.0, atol=0.0)
    assert np.max(np.abs(amp3[non_zonal] - 1.0)) > 1.0e-8

    df, _, state = gksolve_with_state(df, geom, params, state)
    phi4, _ = get_integrals(df, geom)
    amp4 = np.sqrt(ds * np.sum(np.abs(np.asarray(phi4)) ** 2, axis=(0, 1)))

    assert int(np.asarray(state.step)) == 4
    np.testing.assert_allclose(amp4[non_zonal], 1.0, rtol=1.0e-10, atol=1.0e-10)
    assert np.isfinite(float(np.asarray(state.accumulated_norm_factor)))


def test_growth_rates_all_modes_maps_exactly_to_growth_dat_kx0():
    growth = np.loadtxt(f"{LIN_DIR}/growth.dat")
    growth_all = np.loadtxt(f"{LIN_DIR}/growth_rates_all_modes")
    mode_label = np.loadtxt(f"{LIN_DIR}/mode_label")
    kxrh = np.loadtxt(f"{LIN_DIR}/kxrh")

    projected = np.asarray(project_all_modes_to_kx0(growth_all, mode_label, kxrh))
    assert projected.shape == growth.shape
    np.testing.assert_allclose(projected, growth, rtol=0.0, atol=0.0)


def test_frequencies_all_modes_maps_exactly_to_frequencies_dat_kx0():
    freq = np.loadtxt(f"{LIN_DIR}/frequencies.dat")
    freq_all = np.loadtxt(f"{LIN_DIR}/frequencies_all_modes")
    mode_label = np.loadtxt(f"{LIN_DIR}/mode_label")
    kxrh = np.loadtxt(f"{LIN_DIR}/kxrh")

    projected = np.asarray(project_all_modes_to_kx0(freq_all, mode_label, kxrh))
    assert projected.shape == freq.shape
    np.testing.assert_allclose(projected, freq, rtol=0.0, atol=0.0)


def test_lin_dataset_uses_diagnostics_not_k_dumps():
    files = os.listdir(LIN_DIR)
    k_like = [f for f in files if f.startswith("K") and not f.endswith(".dat")]
    numeric = [f for f in files if f.isdigit()]
    assert len(k_like) == 0
    assert len(numeric) == 0

    poten, timestep_slices = poten_files(LIN_DIR)
    assert len(poten) > 0
    assert len(timestep_slices) == len(poten)
    assert np.all(np.diff(timestep_slices) > 0)

    inp = parse_input_dat(f"{LIN_DIR}/input.dat")
    ntime = int(inp["control"]["ntime"])
    ndump_ts = int(inp["control"]["ndump_ts"])
    expected = ntime // ndump_ts
    assert len(poten) == expected
