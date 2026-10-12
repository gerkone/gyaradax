"""Arakawa scheme for terms I + IV (GKW ltrapping_arakawa): bracket properties and limits."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import noisy_case, rel_l2  # type: ignore[import-not-found]

from gyaradax.backends import create_ops
from gyaradax.solver.cfl import estimate_linear_timestep
from gyaradax.operators.precompute import linear_precompute

GRID = dict(nvpar=16, nmu=4, ns=16, nkx=11, nky=4)


def _case(ara, grid=GRID, **overrides):
    df, geometry, params, _ = noisy_case(
        "iteration_13.yaml", grid, dict(ltrapping_arakawa=ara, **overrides), seed=5
    )
    return df, geometry, params, linear_precompute(geometry, params)


def _terms(df, params, pre):
    phi = jnp.zeros(df.shape[-3:], dtype=jnp.complex128)
    return create_ops(pre, backend="jax")._linear_rhs_terms(df, phi, params, pre)


def test_bracket_is_antisymmetric_on_periodic_zonal_modes():
    df, geometry, params, pre = _case(True, disp_par=0.0, disp_vp=0.0)
    zonal = jnp.zeros_like(df).at[..., 0].set(df[..., 0])
    term = _terms(zonal, params, pre)["I_IV_arakawa"]
    # each row carries vthrat * F(s); without it the bracket is an antisymmetric matrix
    row = params.vthrat * jnp.reshape(geometry["ffun"], (1, 1, -1, 1, 1))
    energy = jnp.sum(jnp.conj(zonal) * term / row)
    scale = jnp.sum(jnp.abs(zonal) * jnp.abs(term / row))
    assert float(jnp.linalg.norm(term)) > 0.0
    assert abs(float(jnp.real(energy))) < 1e-13 * float(scale)


def test_bracket_converges_to_streaming_plus_trapping():
    errs = []
    for ns in (16, 32):
        df, geometry, params, pre = _case(True, dict(GRID, ns=ns, nvpar=32), disp_par=0.0, disp_vp=0.0)
        sep_params = replace(params, ltrapping_arakawa=False)
        vp = jnp.reshape(geometry["vpgr"], (-1, 1, 1))
        s = jnp.reshape(geometry["sgrid"], (1, 1, -1))
        f = jnp.zeros_like(df).at[..., df.shape[-2] // 2, 0].set(
            jnp.broadcast_to(jnp.exp(-(vp**2)) * jnp.cos(2 * jnp.pi * s), df.shape[:3])
        )
        ta = _terms(f, params, pre)
        ts = _terms(f, sep_params, linear_precompute(geometry, sep_params))
        errs.append(rel_l2(ta["I_IV_arakawa"], ts["I_par_streaming_plus_diss"] + ts["IV_trapping"]))
    assert errs[0] < 2e-2
    assert errs[1] < errs[0] / 4


def test_arakawa_replaces_streaming_and_trapping_only():
    df, geometry, params, pre = _case(True)
    sep = replace(params, ltrapping_arakawa=False)
    ta = _terms(df, params, pre)
    ts = _terms(df, sep, linear_precompute(geometry, sep))
    assert float(jnp.max(jnp.abs(ta["IV_trapping"]))) == 0.0
    for key in ts:
        if key not in ("I_par_streaming_plus_diss", "IV_trapping", "I_IV_arakawa"):
            np.testing.assert_array_equal(np.asarray(ta[key]), np.asarray(ts[key]))
    # the parallel table keeps the dissipation only
    nodisp = replace(sep, disp_par=0.0)
    streaming = _terms(df, nodisp, linear_precompute(geometry, nodisp))["I_par_streaming_plus_diss"]
    assert rel_l2(ta["I_par_streaming_plus_diss"], ts["I_par_streaming_plus_diss"] - streaming) < 1e-12


@pytest.mark.parametrize("ara", [False, True])
def test_timestep_estimate_is_of_the_streaming_order(ara):
    _, _, params, pre = _case(ara)
    _, _, _, pre_s = _case(False)
    ratio = float(estimate_linear_timestep(pre, params=params)) / float(
        estimate_linear_timestep(pre_s, params=params)
    )
    assert 0.3 < ratio < 3.0
