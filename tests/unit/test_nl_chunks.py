"""The looped (species / vpar-chunk) JAX bracket must agree bitwise with the batched one."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from conftest import noisy_case  # type: ignore[import-not-found]

import gyaradax.backends._jax as jax_backend
from gyaradax.backends import create_ops
from gyaradax.fields import _compute_fields
from gyaradax.precompute import linear_precompute
from gyaradax.solver import gksolve

GRID = dict(nvpar=8, nmu=2, ns=8, nkx=11, nky=5)

CASES = {
    "kinetic_es": ("nl_em_waltz_b01.yaml", dict(nlapar=False, beta=0.0)),
    "kinetic_apar_bpar": ("nl_em_waltz_b01.yaml", dict(nlbpar=True)),
    "kinetic_bpar": ("nl_em_waltz_b01.yaml", dict(nlapar=False, nlbpar=True)),
    "adiabatic_apar": ("iteration_13.yaml", dict(non_linear=True, nlapar=True, beta=0.002)),
}


def _setup(name, mixed_precision):
    cfg_name, overrides = CASES[name]
    df, geometry, params, state = noisy_case(
        cfg_name, GRID, overrides, seed=5, backend="jax", mixed_precision=mixed_precision
    )
    return df, geometry, params, state, linear_precompute(geometry, params)


@pytest.mark.parametrize("use_z2z", [False, True])
@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("name", list(CASES))
def test_chunked_bracket_is_bitwise_batched(monkeypatch, name, mixed_precision, use_z2z):
    df, geometry, params, _, pre = _setup(name, mixed_precision)
    phi, apar, bpar = _compute_fields(df, geometry, params, pre)
    out = []
    for budget in (jax_backend._NL_CHUNK_BYTES, 1):
        monkeypatch.setattr(jax_backend, "_NL_CHUNK_BYTES", budget)
        ops = create_ops(pre, backend="jax", use_z2z=use_z2z, mixed_precision=mixed_precision)
        nl = jax.jit(lambda d, p, a, b: ops.nonlinear_term_iii(d, p, geometry, apar=a, bpar=b))
        out.append(np.asarray(nl(df, phi, apar, bpar)))
    assert np.linalg.norm(out[0]) > 0.0
    np.testing.assert_array_equal(out[1], out[0])


@pytest.mark.parametrize("name", ["kinetic_apar_bpar", "adiabatic_apar"])
def test_chunked_bracket_trajectory_is_bitwise(monkeypatch, name):
    df, geometry, params, state, pre = _setup(name, mixed_precision=True)
    params = replace(params, adaptive_dt=True)
    out = []
    for budget in (jax_backend._NL_CHUNK_BYTES, 1):
        monkeypatch.setattr(jax_backend, "_NL_CHUNK_BYTES", budget)
        final_df, _, _, dt_info = gksolve(
            df, geometry, params, state, n_steps=5, pre=pre, return_dt_info=True
        )
        out.append((np.asarray(final_df), np.asarray(dt_info["dt_used"])))
    np.testing.assert_array_equal(out[1][0], out[0][0])
    np.testing.assert_array_equal(out[1][1], out[0][1])
