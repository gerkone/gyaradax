"""
Solver operations with backend dispatch.
 
The SolverOps object holds all accelerated operations as bound methods.
Each backend (JAX, CUDA) provides a factory that builds a SolverOps
instance from precomputed data.
"""
 
from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Tuple
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
        raise NotImplementedError

    
    @abstractmethod
    def nonlinear_term_iii(self, df, phi, geometry, pre, **kwargs) -> jnp.ndarray:
        raise NotImplementedError