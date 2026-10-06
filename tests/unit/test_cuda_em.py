"""CUDA backend parity against the JAX backend for electromagnetic (A_par / B_par) runs.

Cases are built from the repo YAML configs on a reduced grid with analytic geometry,
so no GKW reference data is needed.
"""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import HAS_CUDA, noisy_case, rel_l2  # type: ignore[import-not-found]

from gyaradax.backends import create_ops
from gyaradax.fields import _compute_fields, g_to_f
from gyaradax.precompute import linear_precompute
from gyaradax.solver import gksolve

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

SMALL_GRID = dict(nvpar=16, nmu=4, ns=16, nkx=21, nky=5)
# 9 x 5 modes -> 16 x 16 dealiased planes (power-of-two FFT path)
POW2_GRID = dict(nvpar=16, nmu=4, ns=16, nkx=9, nky=5)
# ns * nky = 1280 > 1024 -> ky-tiled linear kernel
TILED_GRID = dict(nvpar=8, nmu=2, ns=32, nkx=11, nky=40)
# nky = 23 -> 70-point dealiased ky planes, not instantiated by the v6 row kernels
NKY23_GRID = dict(SMALL_GRID, nky=23)
GRIDS = {"small": SMALL_GRID, "pow2": POW2_GRID, "tiled": TILED_GRID, "nky23": NKY23_GRID}

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


def _setup(name, mixed_precision=False, grid="small", **extra):
    cfg_name, overrides = CASES[name]
    df, geometry, params, state = noisy_case(
        cfg_name, GRIDS[grid], {**overrides, **extra}, seed=3, mixed_precision=mixed_precision
    )
    return df, geometry, params, state, linear_precompute(geometry, params)


def _fields(df, geometry, params, pre):
    return jax.jit(lambda d: _compute_fields(d, geometry, params, pre))(df)


def _per_backend(pre, mixed_precision, call, *arrays):
    """``call(ops, *arrays)`` jitted over ``arrays`` with the JAX and with the CUDA backend."""
    return {
        backend: jax.jit(
            partial(call, create_ops(pre, backend=backend, mixed_precision=mixed_precision))
        )(*arrays)
        for backend in ("jax", "cuda")
    }


@pytest.mark.parametrize(
    "name, grid",
    [(name, "small") for name in CASES]
    + [
        (name, "tiled")
        for name in ("kinetic_es", "kinetic_apar_bpar", "kinetic_apar_dpc", "adiabatic_apar")
    ],
)
def test_linear_rhs_from_g_matches_jax(name, grid):
    df, geometry, params, _, pre = _setup(name, grid=grid)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    out = _per_backend(
        pre,
        False,
        lambda ops, d, p, a, b: ops.linear_rhs_from_g(d, p, geometry, params, pre, apar=a, bpar=b),
        df, phi, apar, bpar,
    )
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
    rhs = [
        _per_backend(
            prc,
            False,
            lambda ops, d, p, a, b, prm=prm, prc=prc: ops.linear_rhs_from_g(
                d, p, geometry, prm, prc, apar=a, bpar=b
            ),
            df, phi, apar, bpar,
        )
        for prm, prc in ((params, pre), (params_off, pre_off))
    ]
    term = {b: rhs[0][b] - rhs[1][b] for b in rhs[0]}
    assert float(jnp.linalg.norm(term["jax"])) > 0.0
    assert rel_l2(term["cuda"], term["jax"]) < 1e-9


@pytest.mark.parametrize("name", ["kinetic_apar", "kinetic_apar_bpar", "adiabatic_apar"])
def test_linear_rhs_physical_f_with_apar_matches_jax(name):
    df, geometry, params, _, pre = _setup(name)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    f = g_to_f(df, apar, params, pre)
    out = _per_backend(
        pre,
        False,
        lambda ops, d, p, a, b: ops.linear_rhs(d, p, geometry, params, pre, apar=a, bpar=b),
        f, phi, apar, bpar,
    )
    assert rel_l2(out["cuda"], out["jax"]) < 1e-12


@pytest.mark.parametrize("mixed_precision, tol", [(False, 1e-12), (True, 1e-5)])
@pytest.mark.parametrize(
    "name, grid",
    [
        (name, "small")
        for name in ("kinetic_es", "kinetic_apar", "kinetic_apar_bpar", "kinetic_bpar", "adiabatic_apar")
    ]
    + [("kinetic_es", "pow2"), ("kinetic_apar", "pow2")],
)
def test_nonlinear_term_matches_jax(name, grid, mixed_precision, tol):
    df, geometry, params, _, pre = _setup(name, mixed_precision=mixed_precision, grid=grid)
    phi, apar, bpar = _fields(df, geometry, params, pre)
    out = _per_backend(
        pre,
        mixed_precision,
        lambda ops, d, p, a, b: ops.nonlinear_term_iii(d, p, geometry, apar=a, bpar=b),
        df, phi, apar, bpar,
    )
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
    out = _per_backend(
        pre,
        mixed_precision,
        lambda ops, d, p, c: ops.nonlinear_term_iii(d, p, geometry, chi_correction=c),
        df, phi, chi,
    )
    assert rel_l2(out["cuda"], out["jax"]) < tol


def test_nonlinear_term_keeps_zero_mode_when_not_excluded():
    df, geometry, params, _, pre = _setup("kinetic_apar")
    phi, apar, _ = _fields(df, geometry, params, pre)
    out = _per_backend(
        pre,
        False,
        lambda ops, d, p, a: ops.nonlinear_term_iii(d, p, geometry, apar=a, exclude_zero_mode=False),
        df, phi, apar,
    )
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
    df, geometry, params, _, pre = _setup("kinetic_apar", mixed_precision=True, grid="nky23")
    assert pre["nl_mphi"] == 70
    phi, apar, _ = _fields(df, geometry, params, pre)
    v5 = _bracket(monkeypatch, "v5", df, phi, apar, geometry, pre, True)
    auto = _bracket(monkeypatch, "auto", df, phi, apar, geometry, pre, True)
    assert np.array_equal(np.asarray(auto), np.asarray(v5))
