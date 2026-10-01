"""Fast agreement check between the JAX Arnoldi and ARPACK.

Tiny ES adiabatic block: eigenvalues and the eigenvector overlap
|<u, v>| / (||u|| ||v||), which is the basis-independent statement that the
two solvers found the same mode.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gyaradax  # noqa: F401

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

if _AVAILABLE:
    from gyaradax.eigenvalue import eigensolve_linear, eigensolve_linear_jax
    from gyaradax.precompute import linear_precompute
    from test_eigenvalue import _es_adiabatic_case

K, IKY, NSTEPS = 3, 1, 50


@pytest.mark.skipif(not _AVAILABLE, reason="gyaradax is not installed")
def test_jax_arnoldi_overlaps_arpack():
    geom, params, nsp = _es_adiabatic_case()
    pre = linear_precompute(geom, params)
    kw = dict(pre=pre, n_species=nsp, k=K, mode="exp",
              n_steps_per_matvec=NSTEPS, ky_select=IKY)

    ref_vals, ref_vecs, ref_res = eigensolve_linear(
        **kw, geometry=geom, params=params, tol=1e-10, return_residuals=True)
    val, vec, res = eigensolve_linear_jax(
        **kw, geometry=geom, params=params, ncv=40, tol=1e-8,
        return_residuals=True)

    assert np.max(ref_res) < 1e-8, f"ARPACK reference not converged: {ref_res}"
    assert np.max(res) < 1e-5, f"jax arnoldi not converged: {res}"

    for i in range(K):
        rel = abs(val[i] - ref_vals[i]) / abs(ref_vals[i])
        u, v = ref_vecs[i].reshape(-1), vec[i].reshape(-1)
        overlap = abs(np.vdot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))
        assert rel < 1e-6, f"mode {i}: jax={val[i]}, arpack={ref_vals[i]}, rel={rel:.2e}"
        assert overlap > 1.0 - 1e-6, f"mode {i}: eigenvector overlap {overlap:.8f}"


if __name__ == "__main__":
    import absl.testing.absltest  # noqa: F401
    pytest.main([__file__, "-q"])
