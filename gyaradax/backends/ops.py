"""Abstract base class for solver operations with backend dispatch.

Each backend (JAX, CUDA) provides a concrete implementation that is
constructed once from precomputed data and used throughout the solve.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import jax.numpy as jnp

from gyaradax.fields import _compute_fields, g_to_f
from gyaradax.state import GKPre


def em_chi_correction(ndim: int, pre, apar=None, bpar=None):
    """Velocity-dependent EM part of chi = J0*phi + chi_correction, or None."""
    chi_corr = None
    if apar is not None and "apar_chi_factor" in pre:
        apar_b = apar[jnp.newaxis, jnp.newaxis, :, :, :]
        if ndim == 6:
            apar_b = apar_b[jnp.newaxis]
        chi_corr = pre["apar_chi_factor"] * apar_b
    if bpar is not None and "bpar_chi_factor" in pre:
        bpar_b = bpar[jnp.newaxis, jnp.newaxis, :, :, :]
        if ndim == 6:
            bpar_b = bpar_b[jnp.newaxis]
        bpar_chi = pre["bpar_chi_factor"] * bpar_b
        chi_corr = bpar_chi if chi_corr is None else chi_corr + bpar_chi
    return chi_corr


class SolverOps(ABC):
    """Container for solver operations. Backend selection happens at construction."""

    def __init__(
        self,
        pre: GKPre,
        use_z2z: bool = False,
        mixed_precision: bool = True,
        mesh=None,
    ):
        self.pre = pre
        self.use_z2z = use_z2z
        self.mixed_precision = mixed_precision
        # device mesh of a sharded run (None on a single device)
        self.mesh = mesh

    def tree_flatten(self):
        return (self.pre,), (self.use_z2z, self.mixed_precision, self.mesh)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (pre,) = children
        use_z2z, mixed_precision, mesh = aux_data
        return cls(pre, use_z2z=use_z2z, mixed_precision=mixed_precision, mesh=mesh)

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
        bessel: jnp.ndarray | None = None,
        chi_correction: jnp.ndarray | None = None,
        apar: jnp.ndarray | None = None,
        bpar: jnp.ndarray | None = None,
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
            chi_correction: Optional velocity-dependent EM correction added to J0*phi
            apar, bpar: Optional EM fields (ns, nkx, nky); the advecting potential becomes
                chi = J0*phi + apar_chi_factor*apar + bpar_chi_factor*bpar

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
        apar: jnp.ndarray | None = None,
        bpar: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute linear RHS for 5D (single species) or 6D (multi-species) df.

        Implements Terms I, II, IV, V, VII, VIII, X, XI + dissipation.
        When apar is provided (EM mode), includes electromagnetic coupling terms.
        Backend must handle both 5D and 6D cases, or raise NotImplementedError/ValueError.

        Args:
            df: Distribution function, 5D (nv, nmu, ns, nkx, nky) or 6D (nsp, nv, nmu, ns, nkx, nky)
            phi: Electrostatic potential (ns, nkx, nky)
            geometry: Geometry dict with grid and metric data
            params: GKParams with physical parameters
            pre: GKPre with precomputed coefficients
            apar: Parallel vector potential (ns, nkx, nky), None for electrostatic

        Returns:
            RHS contribution (same shape as df)

        Raises:
            NotImplementedError: If backend cannot handle this configuration (e.g., non-uniform species params)
            ValueError: If df has unsupported shape
        """
        raise NotImplementedError

    def compute_fields(self, dg: jnp.ndarray, geometry, params, pre):
        return _compute_fields(dg, geometry, params, pre)

    def linear_rhs_from_g(
        self,
        dg: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params,
        pre,
        apar: jnp.ndarray | None = None,
        bpar: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # linear terms act on the physical f; backends may fuse the g -> f conversion
        df = g_to_f(dg, apar, params, pre) if apar is not None else dg
        return self.linear_rhs(df, phi, geometry, params, pre, apar=apar, bpar=bpar)
