"""
Numerical finite difference stencils for parallel and velocity coordinates.

These stencils correspond to the fourth-order upwinded and central schemes 
used in the GKW linear and nonlinear terms.
"""

import jax
import jax.numpy as jnp
import numpy as np
# enforce 64-bit precision
jax.config.update("jax_enable_x64", True)


def _center_5pt(stencil5):
    """Center a 5-point finite difference stencil into a 9-point zero-padded array."""
    out = [0.0] * 9
    out[2:7] = stencil5
    return out


# stencils from linear_terms.f90::differential_scheme, order='fourth_order'.
# correspond to the fortran implementation of upwinded fourth-order finite differences.

# D1_IPW_POS: First derivative, upwinded for positive characteristic velocity
D1_IPW_POS = np.asarray(
    [
        _center_5pt([0.0, 0.0, -18.0, 24.0, -6.0]),
        [0.0, 0.0, 0.0, -4.0, -6.0, 12.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        _center_5pt([0.0, -6.0, 0.0, 0.0, 0.0]),
    ],
    dtype=np.float64,
)

# D1_IPW_NEG: First derivative, upwinded for negative characteristic velocity
D1_IPW_NEG = np.asarray(
    [
        _center_5pt([0.0, 0.0, 0.0, 6.0, 0.0]),
        [0.0, 0.0, 0.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -12.0, 6.0, 4.0, 0.0, 0.0, 0.0],
        _center_5pt([6.0, -24.0, 18.0, 0.0, 0.0]),
    ],
    dtype=np.float64,
)

# D4_IPW_POS: Fourth derivative dissipation, corresponding to POS upwinding
D4_IPW_POS = np.asarray(
    [
        [0.0] * 9,
        [0.0] * 9,
        _center_5pt([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0, 0.0, -1.0, 4.0, -6.0, 4.0, 0.0, 0.0, 0.0],
        _center_5pt([0.0, 12.0, -24.0, 0.0, 0.0]),
    ],
    dtype=np.float64,
)

# D4_IPW_NEG: Fourth derivative dissipation, corresponding to NEG upwinding
D4_IPW_NEG = np.asarray(
    [
        _center_5pt([0.0, 0.0, -24.0, 12.0, 0.0]),
        [0.0, 0.0, 0.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0],
        _center_5pt([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0] * 9,
        [0.0] * 9,
    ],
    dtype=np.float64,
)

# VPAR_D1: Central first derivative in parallel velocity space
VPAR_D1 = np.asarray([1.0, -8.0, 0.0, 8.0, -1.0], dtype=np.float64) / 12.0

# VPAR_D4: Central fourth derivative in parallel velocity space
VPAR_D4 = np.asarray([-1.0, 4.0, -6.0, 4.0, -1.0], dtype=np.float64) / 12.0
