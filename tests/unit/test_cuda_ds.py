"""Double-single linear RHS (linear_arith="ds") against the FP64 CUDA kernel."""

import jax
import numpy as np
import pytest

from conftest import HAS_CUDA, noisy_case, rel_l2  # type: ignore[import-not-found]

from gyaradax.backends import create_ops
from gyaradax.operators.precompute import linear_precompute

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

GRID = dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32)

CASES = {
    "adiabatic": ("iteration_13.yaml", dict()),
    "kinetic": ("kinetic.yaml", dict()),
    "kinetic_collisions": ("kinetic.yaml", dict(collisions=True, coll_freq=1e-3)),
}


def _rhs(name, linear_arith):
    cfg_name, overrides = CASES[name]
    df, geometry, params, _ = noisy_case(
        cfg_name, GRID, {**overrides, "linear_arith": linear_arith, "backend": "cuda"},
        seed=5, mixed_precision=False,
    )
    pre = linear_precompute(geometry, params)
    ops = create_ops(pre, backend="cuda", mixed_precision=False)
    phi = df[(0,) * (df.ndim - 3)] * 0.1
    return np.asarray(jax.jit(lambda d: ops.linear_rhs(d, phi, geometry, params, pre))(df))


@pytest.mark.parametrize("name", list(CASES))
def test_ds_matches_fp64(name):
    ref, got = _rhs(name, "fp64"), _rhs(name, "ds")
    assert not np.array_equal(ref, got), "ds path did not run"
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12
    assert rel_l2(got, ref) < 1e-12


def test_ds_rejects_jax_backend():
    with pytest.raises(ValueError, match="linear_arith"):
        _, geometry, params, _ = noisy_case(
            "iteration_13.yaml", GRID, {"linear_arith": "ds", "backend": "jax"}
        )
        linear_precompute(geometry, params)
