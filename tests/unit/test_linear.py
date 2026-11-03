import numpy as np
import jax
import jax.numpy as jnp
import pytest
from typing import cast
from dataclasses import replace

from conftest import (  # type: ignore[import-not-found]
    ALL_BACKENDS,
    PHASES,
    read_dump_dtim,
    read_dump_time,
    reference_pair,
    rel_l2,
)

from gyaradax.solver.rk4 import gksolve, init_f, default_state
from gyaradax.operators.precompute import linear_precompute
from gyaradax.core.params import GKParams, gkparams_from_input_dat
from gyaradax.diagnostics.diag import project_all_modes_to_kx0
from gyaradax.io.gkw import load_gkw_k_dump, poten_files
from gyaradax.operators.poisson import (
    geom_tensors,
    _phi_adiabatic,
    precompute_phi_adiabatic,
    calculate_phi_adiabatic,
)


@jax.jit
def _step_jitted(prev_df, geom, params, state, pre):
    return gksolve(prev_df, geom, params, state, n_steps=1, pre=pre)


def test_phi_adiabatic_no_zonal_bug():
    """Confirms Fix: calculate_phi_adiabatic works correctly when has_zonal=0."""
    nkx, nky, ns, nvpar, nmu = 1, 1, 4, 4, 3

    # ky does not include 0
    krho = jnp.array([0.426], dtype=jnp.float64)
    kxrh = jnp.array([0.0], dtype=jnp.float64)

    geometry = {
        "krho": krho,
        "kxrh": kxrh,
        "ints": jnp.ones(ns),
        "intmu": jnp.ones(nmu),
        "intvp": jnp.ones(nvpar),
        "vpgr": jnp.linspace(-3, 3, nvpar),
        "mugr": jnp.linspace(0, 1, nmu),
        "bn": jnp.ones(ns),
        "ffun": jnp.ones(ns),
        "efun": jnp.ones(ns),
        "rfun": jnp.ones(ns),
        "bt_frac": jnp.ones(ns),
        "parseval": jnp.ones(nky),
        "little_g": jnp.zeros((ns, 3)).at[:, 0].set(1.0),
        "mas": 1.0,
        "tmp": 1.0,
        "de": 1.0,
        "signz": 1.0,
        "signB": 1.0,
        "vthrat": 1.0,
        "iyzero": 0,
        "ixzero": 0,
    }

    params = GKParams(adiabatic_electrons=True)
    typed_geometry = cast(dict[str, jnp.ndarray], geometry)
    gt = geom_tensors(typed_geometry, params=params)

    # has_zonal should be 0.0 because krho[0] != 0
    assert gt["has_zonal"] == 0.0

    key = jax.random.PRNGKey(42)
    df = jax.random.normal(key, (nvpar, nmu, ns, nkx, nky)).astype(jnp.complex128)

    # Reference
    phi_ref = _phi_adiabatic(gt, df)

    # Optimized path
    pw, pcw, tmp, de, signz, gamma, ints, has_zonal, ixz, iyz = precompute_phi_adiabatic(
        typed_geometry, params
    )
    phi_new = calculate_phi_adiabatic(
        df,
        pw,
        pcw,
        tmp,
        de,
        signz,
        gamma,
        ints,
        cast(float, has_zonal),
        cast(int, ixz),
        cast(int, iyz),
    )

    diff = float(jnp.linalg.norm(phi_new - phi_ref))
    assert diff < 1e-12, f"calculate_phi_adiabatic deviates from _phi_adiabatic; diff={diff:.3e}"


def test_geometry_has_connectivity_and_active_linear_keys(lin_geom):
    geom = lin_geom
    ns = len(geom["ints"])
    nkx = len(geom["kxrh"])
    nky = len(geom["krho"])

    expected_keys = [
        "gfun",
        "dfun",
        "mode_label",
        "ixplus",
        "ixminus",
        "ixzero",
        "iyzero",
        "pos_par_grid_class",
        "s_shift",
        "kx_shift",
        "valid_shift",
    ]
    for key in expected_keys:
        assert key in geom

    assert geom["gfun"].shape == (ns,)
    assert geom["dfun"].shape == (ns, 3)
    assert geom["mode_label"].shape == (nkx, nky)

    ixzero = int(geom["ixzero"])
    iyzero = int(geom["iyzero"])
    assert ixzero == int(np.argmin(np.abs(np.asarray(geom["kxrh"]))))
    assert iyzero == int(np.argmin(np.abs(np.asarray(geom["krho"]))))


@pytest.mark.parametrize("normalize", [True, False])
def test_init_f_contract(lin_geom, lin_shape, normalize):
    df = init_f(
        lin_geom,
        amp_init_real=1.0e-4,
        normalize_per_toroidal_mode=normalize,
    )
    assert df.shape == lin_shape
    assert df.dtype == jnp.complex128


@pytest.mark.parametrize("backend, use_z2z, mixed_precision", ALL_BACKENDS)
def test_gksolve_contract(lin_geom, lin_shape, backend, use_z2z, mixed_precision):
    prev_df = jnp.zeros(lin_shape, dtype=jnp.complex128)
    params = GKParams(
        dt=0.01, naverage=40, backend=backend, use_z2z=use_z2z, mixed_precision=mixed_precision
    )
    state = default_state(nky=len(lin_geom["krho"]))

    next_df, (phi, fluxes), _ = gksolve(prev_df, lin_geom, params, state, n_steps=1)
    pflux, eflux, vflux = fluxes

    assert next_df.shape == lin_shape
    assert phi.shape == (lin_shape[2], lin_shape[3], lin_shape[4])
    assert all(isinstance(f, jnp.ndarray) and f.shape == () for f in [pflux, eflux, vflux])


@pytest.mark.parametrize("backend, use_z2z, mixed_precision", ALL_BACKENDS)
def test_gksolve_zero_input_invariance(lin_geom, lin_shape, backend, use_z2z, mixed_precision):
    prev_df = jnp.zeros(lin_shape, dtype=jnp.complex128)
    params = GKParams(
        dt=0.01, naverage=40, backend=backend, use_z2z=use_z2z, mixed_precision=mixed_precision
    )
    state = default_state(nky=len(lin_geom["krho"]))

    pre = linear_precompute(lin_geom, params)
    next_df, (phi, fluxes), next_state = _step_jitted(prev_df, lin_geom, params, state, pre)

    assert jnp.allclose(next_df, 0.0)
    assert jnp.allclose(phi, 0.0)
    assert all(jnp.allclose(f, 0.0) for f in fluxes)
    assert next_state.step == 1


def test_growth_rates_mapping(lin_dir):
    growth = np.loadtxt(f"{lin_dir}/growth.dat")
    growth_all = np.loadtxt(f"{lin_dir}/growth_rates_all_modes")
    mode_label = np.loadtxt(f"{lin_dir}/mode_label")
    kxrh = np.loadtxt(f"{lin_dir}/kxrh")

    projected = np.asarray(
        project_all_modes_to_kx0(
            cast(jnp.ndarray, growth_all),
            cast(jnp.ndarray, mode_label),
            cast(jnp.ndarray, kxrh),
        )
    )
    assert projected.shape == growth.shape
    np.testing.assert_allclose(projected, growth, rtol=1e-10, atol=1e-10)


def test_poten_files(tmp_path):
    for name in ("Poten00000018", "Poten00000003", "Poten00000009", "K01", "input.dat"):
        (tmp_path / name).write_text("")
    names, slices = poten_files(str(tmp_path))
    assert names == ["Poten00000003", "Poten00000009", "Poten00000018"]
    np.testing.assert_array_equal(slices, [2, 8, 17])


def _chain_normalised(df, mode_label):
    """df with every connected kx chain scaled to unit norm, as GKW normalises linear runs."""
    df = np.asarray(df)
    out = np.zeros_like(df)
    for label in np.unique(mode_label):
        mask = mode_label == label
        norm = np.linalg.norm(df[..., mask])
        if norm > 0:
            out[..., mask] = df[..., mask] / norm
    return out


@pytest.mark.parametrize("backend, use_z2z, mixed_precision", ALL_BACKENDS)
@pytest.mark.parametrize("phase", PHASES)
def test_linear_trajectory_parity(lin_dir, lin_geom, lin_shape, backend, use_z2z, mixed_precision, phase):
    """A linear run from one GKW dump reproduces the next one, chain by chain."""
    start, end = reference_pair(lin_dir, phase)
    params = gkparams_from_input_dat(f"{lin_dir}/input.dat", non_linear=False)
    dt = read_dump_dtim(f"{lin_dir}/{start}.dat")
    params = replace(params, dt=dt, backend=backend, use_z2z=use_z2z, mixed_precision=mixed_precision)
    t_start = read_dump_time(f"{lin_dir}/{start}.dat")
    steps = int(round((read_dump_time(f"{lin_dir}/{end}.dat") - t_start) / dt))

    start_df = load_gkw_k_dump(f"{lin_dir}/{start}", lin_shape)
    end_df = load_gkw_k_dump(f"{lin_dir}/{end}", lin_shape)
    state = default_state(nky=len(lin_geom["krho"]))
    pred_df, _, _ = gksolve(start_df, lin_geom, params, state, n_steps=steps, pre=linear_precompute(lin_geom, params))

    mode_label = np.asarray(lin_geom["mode_label"], dtype=np.int32)
    pred = _chain_normalised(pred_df, mode_label)
    ref = _chain_normalised(end_df, mode_label)
    assert rel_l2(pred, ref) <= 1.0e-3
