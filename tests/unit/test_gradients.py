import jax
import jax.numpy as jnp

from gyaradax.solver import GKParams, default_state, gkstep_single


def test_gkstep_gradient_validity(lin_geom, lin_shape):
    """
    Test that gkstep_single is fully differentiable and the analytical
    gradient (via JAX reverse-mode AD) matches finite differences.
    """
    key = jax.random.PRNGKey(42)
    # create a small random seed
    df0 = jax.random.normal(key, lin_shape, dtype=jnp.float64) + 0j

    params = GKParams(dt=0.01, naverage=40, non_linear=False)
    state = default_state(nky=len(lin_geom["krho"]))

    # Define a scalar loss function parameterized by a scalar alpha
    def loss_fn(alpha):
        scaled_df = df0 * alpha
        next_df, _, _ = gkstep_single(scaled_df, lin_geom, params, state)
        # Return a real scalar: sum of squared magnitudes
        return jnp.sum(jnp.abs(next_df) ** 2)

    # 1. Analytical gradient via JAX reverse-mode AD
    grad_fn = jax.grad(loss_fn)
    alpha_val = 1.0
    analytical_grad = grad_fn(alpha_val)

    # 2. Finite difference gradient
    epsilon = 1e-5
    loss_plus = loss_fn(alpha_val + epsilon)
    loss_minus = loss_fn(alpha_val - epsilon)
    fd_grad = (loss_plus - loss_minus) / (2 * epsilon)

    # Compare
    rel_error = jnp.abs(analytical_grad - fd_grad) / (jnp.abs(analytical_grad) + 1e-30)

    assert jnp.isfinite(analytical_grad)
    assert rel_error < 1e-4


def test_nonlinear_gradient_validity(nonlin_geom, nonlin_shape):
    """
    Test differentiability of the nonlinear pseudospectral solver path.
    """
    key = jax.random.PRNGKey(42)
    df0 = jax.random.normal(key, nonlin_shape, dtype=jnp.float64) + 0j
    params = GKParams(dt=0.01, naverage=40, non_linear=True, enable_term_iii=True)
    state = default_state(nky=len(nonlin_geom["krho"]))

    def loss_fn(alpha):
        scaled_df = df0 * alpha
        next_df, _, _ = gkstep_single(scaled_df, nonlin_geom, params, state)
        return jnp.sum(jnp.abs(next_df) ** 2)

    grad_fn = jax.grad(loss_fn)
    alpha_val = 1.0
    analytical_grad = grad_fn(alpha_val)

    epsilon = 1e-5
    loss_plus = loss_fn(alpha_val + epsilon)
    loss_minus = loss_fn(alpha_val - epsilon)
    fd_grad = (loss_plus - loss_minus) / (2 * epsilon)

    rel_error = jnp.abs(analytical_grad - fd_grad) / (jnp.abs(analytical_grad) + 1e-30)

    assert jnp.isfinite(analytical_grad)
    assert rel_error < 1e-4
