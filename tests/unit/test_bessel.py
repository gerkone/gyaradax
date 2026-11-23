import jax.numpy as jnp
import numpy as np
import scipy.special as ss
from jax.scipy.special import bessel_jn

from gyaradax.operators.poisson import _BESSEL_SERIES_X, j0, j1_hat

X = np.concatenate([[0.0], np.logspace(-10, 1, 4001)])


def test_j0_matches_scipy():
    out = np.asarray(j0(jnp.asarray(X)))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, ss.j0(X), rtol=0, atol=1e-14)


def test_j1_hat_matches_scipy():
    out = np.asarray(j1_hat(jnp.asarray(X)))
    ref = np.where(X > 0, 2 * ss.j1(X) / np.where(X > 0, X, 1.0), 1.0)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-14)


def test_unchanged_above_series_range():
    x = jnp.asarray(X[X >= _BESSEL_SERIES_X])
    np.testing.assert_array_equal(np.asarray(j0(x)), np.asarray(bessel_jn(x, v=0)[0]))
    np.testing.assert_array_equal(np.asarray(j1_hat(x)), np.asarray(2.0 * bessel_jn(x, v=1)[1] / x))
