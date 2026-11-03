import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import JAX_BACKENDS  # type: ignore[import-not-found]

from gyaradax.core.params import GKParams
from gyaradax.operators.precompute import linear_precompute
from gyaradax.solver.rk4 import gkstep_single, default_state


@pytest.mark.parametrize("backend, use_z2z, mixed_precision", JAX_BACKENDS)
def test_gkstep_gradient_validity(lin_geom, lin_shape, backend, use_z2z, mixed_precision):
    """test that gkstep_single is fully differentiable via jax reverse-mode AD."""
    key = jax.random.PRNGKey(42)
    df0 = jax.random.normal(key, lin_shape, dtype=jnp.float64) + 0j

    params = GKParams(
        dt=0.01,
        naverage=40,
        non_linear=False,
        backend=backend,
        use_z2z=use_z2z,
        mixed_precision=mixed_precision,
    )
    state = default_state(nky=len(lin_geom["krho"]))
    pre = linear_precompute(lin_geom, params)

    def loss_fn(alpha):
        scaled_df = df0 * alpha
        next_df, _, _ = gkstep_single(scaled_df, lin_geom, params, state, pre)
        return jnp.sum(jnp.abs(next_df) ** 2)

    grad_fn = jax.grad(loss_fn)
    alpha_val = 1.0
    analytical_grad = grad_fn(alpha_val)

    eps = 1e-5
    fd_grad = (loss_fn(alpha_val + eps) - loss_fn(alpha_val - eps)) / (2 * eps)

    rel_error = jnp.abs(analytical_grad - fd_grad) / (jnp.abs(analytical_grad) + 1e-30)
    assert jnp.isfinite(analytical_grad)
    assert rel_error < 1e-4


@pytest.mark.parametrize("backend, use_z2z, mixed_precision", JAX_BACKENDS)
def test_nonlinear_gradient_validity(nonlin_geom, nonlin_shape, backend, use_z2z, mixed_precision):
    """test differentiability of the nonlinear pseudospectral solver path."""
    key = jax.random.PRNGKey(42)
    df0 = jax.random.normal(key, nonlin_shape, dtype=jnp.float64) + 0j
    # CUDA NL bracket is not AD-differentiable (FFI custom call); skip gradient check
    mp = False if backend == "jax" else mixed_precision
    params = GKParams(
        dt=0.01, naverage=40, non_linear=True, mixed_precision=mp, backend=backend, use_z2z=use_z2z
    )
    state = default_state(nky=len(nonlin_geom["krho"]))
    pre = linear_precompute(nonlin_geom, params)

    def loss_fn(alpha):
        scaled_df = df0 * alpha
        next_df, _, _ = gkstep_single(scaled_df, nonlin_geom, params, state, pre)
        return jnp.sum(jnp.abs(next_df) ** 2)

    grad_fn = jax.grad(loss_fn)
    alpha_val = 1.0
    analytical_grad = grad_fn(alpha_val)

    eps = 1e-5
    fd_grad = (loss_fn(alpha_val + eps) - loss_fn(alpha_val - eps)) / (2 * eps)

    rel_error = jnp.abs(analytical_grad - fd_grad) / (jnp.abs(analytical_grad) + 1e-30)
    assert jnp.isfinite(analytical_grad)
    assert rel_error < 1e-4


@pytest.mark.parametrize("finit", ["cosine2", "zonal"])
def test_gk_init_under_jit(finit):
    """gk_init can be traced, so a loss may build its initial state inside jax.jit."""
    import dataclasses

    from gyaradax import compute_geometry, gk_init

    geometry = compute_geometry(
        q=1.4, shat=0.78, eps=0.19, nvpar=8, nmu=4, ns=8, nkx=5, nky=4, krhomax=1.4, kxmax=2 * 5 * 0.78
    )
    params = GKParams(dt=0.01, naverage=40, non_linear=False, finit=finit, rlt=6.9)

    def init(rlt):
        return gk_init(geometry, dataclasses.replace(params, rlt=rlt))[0]

    traced, eager = np.asarray(jax.jit(init)(6.9)), np.asarray(init(6.9))
    np.testing.assert_array_equal(traced != 0, eager != 0)
    np.testing.assert_allclose(traced, eager, rtol=1e-14, atol=0)
