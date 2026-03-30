"""
Solver operations with backend dispatch.
 
The SolverOps object holds all accelerated operations as bound methods.
Each backend (JAX, CUDA) provides a factory that builds a SolverOps
instance from precomputed data.
"""
 
from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Tuple, Optional
from gyaradax.types import GKPre
 
 
class SolverOps(ABC):
    """Container for solver operations. Built once, used everywhere.
 
    Each method is a concrete implementation — no dynamic dispatch
    at call time. Backend selection happens at construction.
    """
    @abstractmethod
    def __init__(self, pre: GKPre, field_template: jnp.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        raise NotImplementedError
 

    @abstractmethod
    def _apply_vpar_dual(
        self, field: jnp.ndarray, coeffs_d1, coeffs_d4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError


    @abstractmethod
    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Apply the parallel stencil to a single field."""
        ...

    @abstractmethod
    def _apply_parallel_dual(
        self, field1: jnp.ndarray, field2: jnp.ndarray, coeffs1: jnp.ndarray, coeffs2: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply parallel stencils to two fields simultaneously (fused)."""
        ...

    @abstractmethod
    def nonlinear_term_iii(
        self, df, phi, geometry, **kwargs
    ) -> jnp.ndarray:
        raise NotImplementedError

    def linear_rhs(self, df, phi, geometry, params, pre) -> Optional[jnp.ndarray]:
        """ Optional unified linear RHS override. Returns None if not implemented. """
        return None