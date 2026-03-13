import jax.numpy as jnp


def _center_5pt(stencil5):
    """
    Center a 5-point finite difference stencil into a 9-point zero-padded array.

    Args:
        stencil5: Sequence of 5 coefficients representing the central stencil.

    Returns:
        List of 9 coefficients with zero-padding on both ends.
    """
    out = [0.0] * 9
    out[2:7] = stencil5
    return out


# differential stencils from linear_terms.f90::differential_scheme, order='fourth_order'.
# these correspond to the fortran implementation of upwinded fourth-order finite differences.
D1_IPW_POS = jnp.asarray(
    [
        _center_5pt([0.0, 0.0, -18.0, 24.0, -6.0]),
        [0.0, 0.0, 0.0, -4.0, -6.0, 12.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        _center_5pt([0.0, -6.0, 0.0, 0.0, 0.0]),
    ],
    dtype=jnp.float64,
)
D1_IPW_NEG = jnp.asarray(
    [
        _center_5pt([0.0, 0.0, 0.0, 6.0, 0.0]),
        [0.0, 0.0, 0.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -12.0, 6.0, 4.0, 0.0, 0.0, 0.0],
        _center_5pt([6.0, -24.0, 18.0, 0.0, 0.0]),
    ],
    dtype=jnp.float64,
)

D4_IPW_POS = jnp.asarray(
    [
        [0.0] * 9,
        [0.0] * 9,
        _center_5pt([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0, 0.0, -1.0, 4.0, -6.0, 4.0, 0.0, 0.0, 0.0],
        _center_5pt([0.0, 12.0, -24.0, 0.0, 0.0]),
    ],
    dtype=jnp.float64,
)
D4_IPW_NEG = jnp.asarray(
    [
        _center_5pt([0.0, 0.0, -24.0, 12.0, 0.0]),
        [0.0, 0.0, 0.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0],
        _center_5pt([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0] * 9,
        [0.0] * 9,
    ],
    dtype=jnp.float64,
)

VPAR_D1 = jnp.asarray([1.0, -8.0, 0.0, 8.0, -1.0], dtype=jnp.float64) / 12.0
VPAR_D4 = jnp.asarray([-1.0, 4.0, -6.0, 4.0, -1.0], dtype=jnp.float64) / 12.0
