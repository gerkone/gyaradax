"""
Backend selection and ops creation.
 
    from gyaradax.backends import create_ops
 
    ops = create_ops(pre, df)              # auto-detect
    ops = create_ops(pre, df, "jax")       # force JAX
    ops = create_ops(pre, df, "cuda")      # force CUDA (raises if unavailable)
"""
 
import os
import logging
import jax
from gyaradax.backends.ops import SolverOps
from gyaradax.backends._jax import JAXOps
from gyaradax.backends._cuda import CUDAOps, is_available
 
log = logging.getLogger(__name__)
 
 
def create_ops(pre, field_template, backend: str = 'auto') -> SolverOps:
    """Create a SolverOps instance for the given backend.
 
    Args:
        pre: GKPre precomputed data.
        field_template: example field for shape/dtype (e.g., df array).
        backend: "jax", "cuda" or "auto"
    Returns:
        SolverOps with all operations ready to call.
    """
    if backend == "jax":
        log.info("Backend: JAX")
        return JAXOps(pre, field_template)
 
    if backend in ("cuda", "auto"):
        has_gpu = any(d.platform == "gpu" for d in jax.devices())
 
        if backend == "cuda" and not has_gpu:
            raise RuntimeError("backend='cuda' but no GPU found")
 
        if has_gpu:
            if is_available():
                log.info("Backend: CUDA")
                return CUDAOps(pre, field_template)
            elif backend == "cuda":
                raise RuntimeError("backend='cuda' but extensions not compiled")
            else:
                log.info("Backend: JAX (GPU present, extensions not compiled)")
        
        if backend == "auto":
            log.info("Backend: JAX (GPU not found or CUDA not available)")
            return JAXOps(pre, field_template)
 
    raise ValueError(f"Unknown backend: {backend!r}. Use 'jax', 'cuda', or 'auto'.")