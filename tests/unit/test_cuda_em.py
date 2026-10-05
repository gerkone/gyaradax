"""CUDA backend parity against the JAX backend for electromagnetic (A_par / B_par) runs.

Cases are built from the repo YAML configs on a reduced grid with analytic geometry,
so no GKW reference data is needed.
"""

import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import HAS_CUDA, rel_l2  # type: ignore[import-not-found]

from gyaradax.backends import create_ops
from gyaradax.fields import _compute_fields, g_to_f
from gyaradax.geometry import compute_geometry_from_config
from gyaradax.params import gkparams_from_config, load_config
from gyaradax.precompute import linear_precompute
from gyaradax.simulate import gk_init
from gyaradax.solver import gksolve

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SMALL_GRID = dict(nvpar=16, nmu=4, ns=16, nkx=21, nky=5)
# 9 x 5 modes -> 16 x 16 dealiased planes (power-of-two FFT path)
POW2_GRID = dict(nvpar=16, nmu=4, ns=16, nkx=9, nky=5)
# ns * nky = 1280 > 1024 -> ky-tiled linear kernel
TILED_GRID = dict(nvpar=8, nmu=2, ns=32, nkx=11, nky=40)

CASES = {
    "kinetic_es": ("nl_em_waltz_b01.yaml", dict(nlapar=False, beta=0.0)),
    "kinetic_apar": ("nl_em_waltz_b01.yaml", dict()),
    "kinetic_apar_bpar": ("nl_em_waltz_b01.yaml", dict(nlbpar=True)),
    "kinetic_bpar": ("nl_em_waltz_b01.yaml", dict(nlapar=False, nlbpar=True)),
    "kinetic_apar_dpc": ("nl_em_waltz_b01.yaml", dict(disp_par_conserve=1)),
    "kinetic_apar_dpc3": ("nl_em_waltz_b01.yaml", dict(disp_par_conserve=3)),
    "adiabatic_apar": ("iteration_13.yaml", dict(non_linear=True, nlapar=True, beta=0.002)),
    "adiabatic_collisions": ("iteration_13.yaml", dict(collisions=True, coll_freq=0.05)),
    "adiabatic_collisions_cons": (
        "iteration_13.yaml",
        dict(
            collisions=True, coll_freq=0.05, coll_mom_conservation=True, coll_ene_conservation=True
        ),
    ),
}


def _setup(name, mixed_precision=False, grid=SMALL_GRID, **extra):
    cfg_name, overrides = CASES[name]
    overrides = {**overrides, **extra}
    cfg = load_config(os.path.join(REPO_ROOT, "configs", cfg_name))
    for key, value in grid.items():
        cfg.grid[key] = value
    params = gkparams_from_config(cfg, mixed_precision=mixed_precision)
    params = replace(params, **overrides)
    geometry = compute_geometry_from_config(cfg)
    nsp = 1 if params.adiabatic_electrons else int(np.asarray(params.mas).shape[0])
    df, geometry, state = gk_init(geometry, params, n_species=nsp)
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    noise = jax.random.normal(k1, df.shape) + 1j * jax.random.normal(k2, df.shape)
    df = (df + 1e-4 * noise).astype(jnp.complex128)
    pre = linear_precompute(geometry, params)
    return df, geometry, params, state, pre


def _fields(df, geometry, params, pre):
    return jax.jit(lambda d: _compute_fields(d, geometry, params, pre))(df)


@pytest.mark.parametrize("name", list(CASES))
def test_linear_rhs_from_g_matches_jax(name):
    df, geometry, params, _, pre = _setup(name)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    out = {}
    for backend in ("jax", "cuda"):
        ops = create_ops(pre, backend=backend, mixed_precision=False)
        out[backend] = jax.jit(
            lambda d, p, a, b: ops.linear_rhs_from_g(d, p, geometry, params, pre, apar=a, bpar=b)
        )(df, phi, apar, bpar)
    assert rel_l2(out["cuda"], out["jax"]) < 1e-12


@pytest.mark.parametrize(
    "name", ["kinetic_es", "kinetic_apar_bpar", "kinetic_apar_dpc", "adiabatic_apar"]
)
def test_linear_rhs_ky_tiled_matches_jax(name):
    df, geometry, params, _, pre = _setup(name, grid=TILED_GRID)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    out = {}
    for backend in ("jax", "cuda"):
        ops = create_ops(pre, backend=backend, mixed_precision=False)
        out[backend] = jax.jit(
            lambda d, p, a, b: ops.linear_rhs_from_g(d, p, geometry, params, pre, apar=a, bpar=b)
        )(df, phi, apar, bpar)
    assert rel_l2(out["cuda"], out["jax"]) < 1e-12


@pytest.mark.parametrize(
    "name, off",
    [
        ("kinetic_apar_dpc", dict(disp_par_conserve=0)),
        ("kinetic_apar_dpc3", dict(disp_par_conserve=0)),
        ("adiabatic_collisions", dict(collisions=False)),
        ("adiabatic_collisions_cons", dict(collisions=False)),
    ],
)
def test_linear_rhs_optional_term_matches_jax(name, off):
    df, geometry, params, _, pre = _setup(name)
    params_off = replace(params, **off)
    pre_off = linear_precompute(geometry, params_off)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    term = {}
    for backend in ("jax", "cuda"):
        rhs = []
        for prm, prc in ((params, pre), (params_off, pre_off)):
            ops = create_ops(prc, backend=backend, mixed_precision=False)
            rhs.append(
                jax.jit(
                    lambda d, p, a, b: ops.linear_rhs_from_g(
                        d, p, geometry, prm, prc, apar=a, bpar=b
                    )
                )(df, phi, apar, bpar)
            )
        term[backend] = rhs[0] - rhs[1]
    assert float(jnp.linalg.norm(term["jax"])) > 0.0
    assert rel_l2(term["cuda"], term["jax"]) < 1e-9


@pytest.mark.parametrize("name", ["kinetic_apar", "kinetic_apar_bpar", "adiabatic_apar"])
def test_linear_rhs_physical_f_with_apar_matches_jax(name):
    df, geometry, params, _, pre = _setup(name)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    f = g_to_f(df, apar, params, pre)
    out = {}
    for backend in ("jax", "cuda"):
        ops = create_ops(pre, backend=backend, mixed_precision=False)
        out[backend] = jax.jit(
            lambda d, p, a, b: ops.linear_rhs(d, p, geometry, params, pre, apar=a, bpar=b)
        )(f, phi, apar, bpar)
    assert rel_l2(out["cuda"], out["jax"]) < 1e-12


@pytest.mark.parametrize("mixed_precision, tol", [(False, 1e-12), (True, 1e-5)])
@pytest.mark.parametrize(
    "name", ["kinetic_es", "kinetic_apar", "kinetic_apar_bpar", "kinetic_bpar", "adiabatic_apar"]
)
def test_nonlinear_term_matches_jax(name, mixed_precision, tol):
    df, geometry, params, _, pre = _setup(name, mixed_precision=mixed_precision)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    out = {}
    for backend in ("jax", "cuda"):
        ops = create_ops(pre, backend=backend, mixed_precision=mixed_precision)
        out[backend] = jax.jit(
            lambda d, p, a, b: ops.nonlinear_term_iii(d, p, geometry, apar=a, bpar=b)
        )(df, phi, apar, bpar)
    assert float(jnp.linalg.norm(out["jax"])) > 0.0
    assert rel_l2(out["cuda"], out["jax"]) < tol


@pytest.mark.parametrize("mixed_precision, tol", [(False, 1e-12), (True, 1e-5)])
def test_nonlinear_term_explicit_chi_correction_matches_jax(mixed_precision, tol):
    df, geometry, params, _, pre = _setup("kinetic_apar_bpar", mixed_precision=mixed_precision)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    chi = (
        pre["apar_chi_factor"] * apar[None, None, None]
        + pre["bpar_chi_factor"] * bpar[None, None, None]
    )
    out = {}
    for backend in ("jax", "cuda"):
        ops = create_ops(pre, backend=backend, mixed_precision=mixed_precision)
        out[backend] = jax.jit(
            lambda d, p, c: ops.nonlinear_term_iii(d, p, geometry, chi_correction=c)
        )(df, phi, chi)
    assert rel_l2(out["cuda"], out["jax"]) < tol


@pytest.mark.parametrize("mixed_precision, tol", [(False, 1e-12), (True, 1e-5)])
@pytest.mark.parametrize("name", ["kinetic_es", "kinetic_apar"])
def test_nonlinear_term_power_of_two_planes(name, mixed_precision, tol):
    df, geometry, params, _, pre = _setup(name, mixed_precision=mixed_precision, grid=POW2_GRID)
    assert (pre["nl_mrad"], pre["nl_mphi"]) == (16, 16)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    out = {}
    for backend in ("jax", "cuda"):
        ops = create_ops(pre, backend=backend, mixed_precision=mixed_precision)
        out[backend] = jax.jit(
            lambda d, p, a, b: ops.nonlinear_term_iii(d, p, geometry, apar=a, bpar=b)
        )(df, phi, apar, bpar)
    assert rel_l2(out["cuda"], out["jax"]) < tol


def test_nonlinear_term_keeps_zero_mode_when_not_excluded():
    df, geometry, params, _, pre = _setup("kinetic_apar")
    phi, apar, bpar = _fields(df, geometry, params, pre)
    out = {}
    for backend in ("jax", "cuda"):
        ops = create_ops(pre, backend=backend, mixed_precision=False)
        out[backend] = jax.jit(
            lambda d, p, a: ops.nonlinear_term_iii(d, p, geometry, apar=a, exclude_zero_mode=False)
        )(df, phi, apar)
    zero_mode = out["jax"][..., pre["ixzero"], pre["iyzero"]]
    assert float(jnp.max(jnp.abs(zero_mode))) > 0.0
    assert rel_l2(out["cuda"], out["jax"]) < 1e-12


@pytest.mark.parametrize("name", ["kinetic_apar", "kinetic_apar_bpar", "adiabatic_apar"])
def test_gksolve_em_trajectory_matches_jax(name):
    df, geometry, params, state, pre = _setup(name)
    params = replace(params, adaptive_dt=True)
    final = {}
    for backend in ("jax", "cuda"):
        p = replace(params, backend=backend)
        df_out, (phi, fluxes), st, dt_info = gksolve(
            df, geometry, p, state, n_steps=10, pre=pre, return_dt_info=True
        )
        final[backend] = (df_out, phi, st, dt_info["dt_used"])
    assert np.allclose(final["cuda"][3], final["jax"][3], rtol=1e-12, atol=0.0)
    assert rel_l2(final["cuda"][0], final["jax"][0]) < 1e-10
    assert rel_l2(final["cuda"][1], final["jax"][1]) < 1e-10


def _has_bracket_v6():
    from gyaradax.backends import _cuda

    _cuda._register_ffi()
    return _cuda._has_bracket_v6


def _bracket(monkeypatch, mode, df, phi, apar, geometry, pre, mixed_precision):
    monkeypatch.setenv("GYARADAX_BRACKET", mode)
    ops = create_ops(pre, backend="cuda", mixed_precision=mixed_precision)
    return jax.jit(lambda d, p, a: ops.nonlinear_term_iii(d, p, geometry, apar=a))(df, phi, apar)


@pytest.mark.skipif(not HAS_CUDA or not _has_bracket_v6(), reason="v6 bracket not built")
@pytest.mark.parametrize("mixed_precision, tol", [(False, 1e-13), (True, 1e-5)])
@pytest.mark.parametrize("name", ["kinetic_es", "kinetic_apar_bpar"])
def test_bracket_v6_matches_v5(monkeypatch, name, mixed_precision, tol):
    df, geometry, params, _, pre = _setup(name, mixed_precision=mixed_precision)
    phi, apar, _ = _fields(df, geometry, params, pre)
    v5 = _bracket(monkeypatch, "v5", df, phi, apar, geometry, pre, mixed_precision)
    v6 = _bracket(monkeypatch, "auto", df, phi, apar, geometry, pre, mixed_precision)
    assert rel_l2(v6, v5) < tol
    if mixed_precision:
        # distinct FFT pipelines, so the env switch must really select v5
        assert not np.array_equal(np.asarray(v6), np.asarray(v5))


@pytest.mark.skipif(not HAS_CUDA or not _has_bracket_v6(), reason="v6 bracket not built")
def test_bracket_v6_runs_v5_for_unsupported_planes(monkeypatch):
    # nky = 23 -> 70-point dealiased ky planes, not instantiated by the v6 row kernels
    grid = dict(SMALL_GRID, nky=23)
    df, geometry, params, _, pre = _setup("kinetic_apar", mixed_precision=True, grid=grid)
    assert pre["nl_mphi"] == 70
    phi, apar, _ = _fields(df, geometry, params, pre)
    v5 = _bracket(monkeypatch, "v5", df, phi, apar, geometry, pre, True)
    auto = _bracket(monkeypatch, "auto", df, phi, apar, geometry, pre, True)
    assert np.array_equal(np.asarray(auto), np.asarray(v5))
