import jax
import jax.numpy as jnp

from gyaradax.params import GKParams
from gyaradax.solver import gkstep_single, default_state, linear_precompute


def test_gkstep_gradient_validity(lin_geom, lin_shape):
    """test that gkstep_single is fully differentiable via jax reverse-mode AD."""
    key = jax.random.PRNGKey(42)
    df0 = jax.random.normal(key, lin_shape, dtype=jnp.float64) + 0j

    params = GKParams(dt=0.01, naverage=40, non_linear=False)
    state = default_state(nky=len(lin_geom["krho"]))
    pre = linear_precompute(lin_geom, params)

    def loss_fn(alpha):
        scaled_df = df0 * alpha
        next_df, _, _ = gkstep_single(scaled_df, lin_geom, params, state, pre)
        return jnp.sum(jnp.abs(next_df) ** 2)

    grad_fn = jax.grad(loss_fn)
    alpha_val = 1.0
    analytical_grad = grad_fn(alpha_val)

    epsilon = 1e-5
    fd_grad = (loss_fn(alpha_val + epsilon) - loss_fn(alpha_val - epsilon)) / (
        2 * epsilon
    )

    rel_error = jnp.abs(analytical_grad - fd_grad) / (jnp.abs(analytical_grad) + 1e-30)
    assert jnp.isfinite(analytical_grad)
    assert rel_error < 1e-4


def test_nonlinear_gradient_validity(nonlin_geom, nonlin_shape):
    """test differentiability of the nonlinear pseudospectral solver path."""
    key = jax.random.PRNGKey(42)
    df0 = jax.random.normal(key, nonlin_shape, dtype=jnp.float64) + 0j
    params = GKParams(dt=0.01, naverage=40, non_linear=True, mixed_precision=False)
    state = default_state(nky=len(nonlin_geom["krho"]))
    pre = linear_precompute(nonlin_geom, params)

    def loss_fn(alpha):
        scaled_df = df0 * alpha
        next_df, _, _ = gkstep_single(scaled_df, nonlin_geom, params, state, pre)
        return jnp.sum(jnp.abs(next_df) ** 2)

    grad_fn = jax.grad(loss_fn)
    alpha_val = 1.0
    analytical_grad = grad_fn(alpha_val)

    epsilon = 1e-5
    fd_grad = (loss_fn(alpha_val + epsilon) - loss_fn(alpha_val - epsilon)) / (
        2 * epsilon
    )

    rel_error = jnp.abs(analytical_grad - fd_grad) / (jnp.abs(analytical_grad) + 1e-30)
    assert jnp.isfinite(analytical_grad)
    assert rel_error < 1e-4
