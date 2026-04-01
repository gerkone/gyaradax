"""Abstract base class for solver operations with backend dispatch.

Each backend (JAX, CUDA) provides a concrete implementation that is
constructed once from precomputed data and used throughout the solve.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import jax.numpy as jnp

from gyaradax.types import GKPre


class SolverOps(ABC):
    """Container for solver operations. backend selection happens at construction."""

    @abstractmethod
    def __init__(self, pre: GKPre, field_template: jnp.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        """Apply 5-point velocity-space stencil along vpar axis."""
        raise NotImplementedError

    @abstractmethod
    def _apply_vpar_dual(
        self, field: jnp.ndarray, coeffs_d1, coeffs_d4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply first and fourth derivative vpar stencils in one pass."""
        raise NotImplementedError

    @abstractmethod
    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Apply 9-point parallel stencil with mode connectivity."""
        ...

    @abstractmethod
    def _apply_parallel_dual(
        self,
        field1: jnp.ndarray,
        field2: jnp.ndarray,
        coeffs1: jnp.ndarray,
        coeffs2: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply parallel stencils to two fields simultaneously."""
        ...

    @abstractmethod
    def nonlinear_term_iii(self, df, phi, geometry, **kwargs) -> jnp.ndarray:
        """Compute term III (nonlinear ExB advection) via pseudospectral method."""
        raise NotImplementedError

    def linear_rhs(self, df, phi, geometry, params, pre) -> Optional[jnp.ndarray]:
        """Optional fused linear RHS. returns None if not implemented."""
        return None
