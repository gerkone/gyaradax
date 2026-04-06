"""Abstract base class for solver operations with backend dispatch.

Each backend (JAX, CUDA) provides a concrete implementation that is
constructed once from precomputed data and used throughout the solve.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import jax.numpy as jnp

from gyaradax.types import GKPre


class SolverOps(ABC):
    """Container for solver operations. Backend selection happens at construction."""

    def __init__(
        self,
        pre: GKPre,
        use_z2z: bool = False,
        mixed_precision: bool = True,
    ):
        self.pre = pre
        self.use_z2z = use_z2z
        self.mixed_precision = mixed_precision

    def tree_flatten(self):
        return (self.pre,), (self.use_z2z, self.mixed_precision)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (pre,) = children
        use_z2z, mixed_precision = aux_data
        return cls(pre, use_z2z=use_z2z, mixed_precision=mixed_precision)

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
        raise NotImplementedError

    @abstractmethod
    def _apply_parallel_dual(
        self,
        field1: jnp.ndarray,
        field2: jnp.ndarray,
        coeffs1: jnp.ndarray,
        coeffs2: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply parallel stencils to two fields simultaneously."""
        raise NotImplementedError

    @abstractmethod
    def nonlinear_term_iii(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        bessel: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Compute term III (nonlinear ExB advection) via pseudospectral method.

        Backend must handle both 5D (nv, nmu, ns, nkx, nky) and 6D (nsp, nv, nmu, ns, nkx, nky) df,
        or raise NotImplementedError/ValueError if unsupported.

        Mixed precision is controlled by self.mixed_precision (set at construction time).

        Args:
            df: Distribution function, 5D or 6D
            phi: Electrostatic potential (ns, nkx, nky)
            geometry: Geometry dict with grid and metric data
            efun_sign: Sign factor for ExB bracket
            fft_prefactor: Prefactor for FFT
            exclude_zero_mode: Zero out (kx=0, ky=0) mode
            bessel: Optional Bessel function array

        Returns:
            Nonlinear RHS contribution (same shape as df)

        Raises:
            NotImplementedError: If backend cannot handle this configuration (e.g., 6D with non-uniform params)
            ValueError: If df has unsupported shape
        """
        raise NotImplementedError

    @abstractmethod
    def linear_rhs(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params,
        pre,
    ) -> jnp.ndarray:
        """Compute linear RHS for 5D (single species) or 6D (multi-species) df.

        Implements Terms I, II, IV, V, VII, VIII + dissipation.
        Backend must handle both 5D and 6D cases, or raise NotImplementedError/ValueError.

        Args:
            df: Distribution function, 5D (nv, nmu, ns, nkx, nky) or 6D (nsp, nv, nmu, ns, nkx, nky)
            phi: Electrostatic potential (ns, nkx, nky)
            geometry: Geometry dict with grid and metric data
            params: GKParams with physical parameters
            pre: GKPre with precomputed coefficients

        Returns:
            RHS contribution (same shape as df)

        Raises:
            NotImplementedError: If backend cannot handle this configuration (e.g., non-uniform species params)
            ValueError: If df has unsupported shape
        """
        raise NotImplementedError
