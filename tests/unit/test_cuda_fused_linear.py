"""fused_linear (CUDA) against the default CUDA path: linear RHS and short trajectories."""

import jax
import numpy as np
import pytest

from conftest import HAS_CUDA, noisy_case  # type: ignore[import-not-found]

from gyaradax.backends import create_ops
from gyaradax.operators.precompute import linear_precompute
from gyaradax.solver.driver import gk_run

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

GRID = dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32)

CASES = {
    "adiabatic": ("iteration_13.yaml", dict(non_linear=False)),
    "adiabatic_nl": ("iteration_13.yaml", dict(non_linear=True)),
    "kinetic": ("kinetic.yaml", dict(non_linear=False, adaptive_dt=False)),
    "kinetic_collisions": (
        "kinetic.yaml",
        dict(non_linear=False, adaptive_dt=False, collisions=True, coll_freq=1e-3),
    ),
    "kinetic_collisions_cons": (
        "kinetic.yaml",
        dict(
            non_linear=False, adaptive_dt=False, collisions=True, coll_freq=1e-3,
            coll_mom_conservation=True, coll_ene_conservation=True,
        ),
    ),
}


def _run(name, fused):
    cfg_name, overrides = CASES[name]
    df, geometry, params, state = noisy_case(
        cfg_name, GRID, {**overrides, "fused_linear": fused, "backend": "cuda"},
        seed=7, mixed_precision=False,
    )
    pre = linear_precompute(geometry, params)
    ops = create_ops(pre, backend="cuda", mixed_precision=False)
    phi = df[(0,) * (df.ndim - 3)] * 0.1
    rhs = jax.jit(lambda d: ops.linear_rhs(d, phi, geometry, params, pre))(df)
    out, _, _, _ = jax.jit(lambda d, s: gk_run(d, geometry, params, s, 20, pre=pre))(df, state)
    return np.asarray(rhs), np.asarray(out)


@pytest.mark.parametrize("name", list(CASES))
def test_fused_linear_matches_default(name):
    (rhs_ref, out_ref), (rhs, out) = _run(name, False), _run(name, True)
    assert not np.array_equal(rhs, rhs_ref), "fused path did not run"
    assert np.max(np.abs(rhs - rhs_ref)) / np.max(np.abs(rhs_ref)) < 1e-14
    assert np.max(np.abs(out - out_ref)) / np.max(np.abs(out_ref)) < 1e-13


@pytest.mark.parametrize("fused", [False, True])
def test_vmap_matches_single_calls(fused):
    df, geometry, params, _ = noisy_case(
        "kinetic.yaml", GRID, {**CASES["kinetic"][1], "fused_linear": fused, "backend": "cuda"},
        seed=7, mixed_precision=False,
    )
    pre = linear_precompute(geometry, params)
    ops = create_ops(pre, backend="cuda", mixed_precision=False)
    dfs = jax.numpy.stack([df, 0.5 * df[..., ::-1] + 0.1 * df])
    phis = dfs[:, 0, 0, 0] * 0.1

    def rhs(d, p):
        return ops.linear_rhs(d, p, geometry, params, pre)

    batched = np.asarray(jax.jit(jax.vmap(rhs))(dfs, phis))
    single = np.stack([np.asarray(jax.jit(rhs)(dfs[i], phis[i])) for i in range(2)])
    assert np.array_equal(batched, single)
