"""Backend selection and solver ops creation.

usage:
    from gyaradax.backends import create_ops

    ops = create_ops(pre)                  # auto-detect
    ops = create_ops(pre, "jax")           # force JAX
    ops = create_ops(pre, "cuda")          # force CUDA (raises if unavailable)
"""

import logging

import jax

from gyaradax.backends._jax import JAXOps
from gyaradax.backends.ops import SolverOps

log = logging.getLogger(__name__)


def _load_cuda_backend():
    """Import CUDA backend objects only when CUDA selection needs them."""
    from gyaradax.backends._cuda import CUDAOps, is_available

    return CUDAOps, is_available


def create_ops(
    pre, backend: str = "auto", use_z2z: bool = False, mixed_precision: bool = True
) -> SolverOps:
    """Create a SolverOps instance for the given backend.

    Args:
        pre: Precomputed geometry and coefficients
        backend: Backend selection ('auto', 'jax', or 'cuda')
        use_z2z: Use Z2Z (complex-to-complex) FFTs instead of R2C (real-to-complex).
                 Note: CUDA backend is Z2Z-only, this flag only affects JAX backend.
        mixed_precision: Use mixed precision (FP32 FFTs) for nonlinear bracket.
                        Set to False for full FP64 accuracy.
    """
    if backend == "jax":
        z2z_str = " (z2z)" if use_z2z else ""
        mp_str = " (mixed)" if mixed_precision else " (fp64)"
        log.info("Backend: JAX%s%s", z2z_str, mp_str)
        return JAXOps(pre, use_z2z=use_z2z, mixed_precision=mixed_precision)

    if backend in ("cuda", "auto"):
        has_gpu = any(d.platform == "gpu" for d in jax.devices())

        if backend == "cuda" and not has_gpu:
            raise RuntimeError("backend='cuda' but no GPU found")

        if has_gpu:
            try:
                CUDAOps, is_available = _load_cuda_backend()
            except ImportError as exc:
                if backend == "cuda":
                    raise RuntimeError(
                        "backend='cuda' but CUDA backend could not be imported"
                    ) from exc
                log.info("Backend: JAX (GPU present, CUDA backend import failed)")
            else:
                if is_available():
                    mp_str = " (mixed)" if mixed_precision else " (fp64)"
                    log.info("Backend: CUDA%s [Z2Z-only]", mp_str)
                    return CUDAOps(pre, use_z2z=use_z2z, mixed_precision=mixed_precision)
                if backend == "cuda":
                    raise RuntimeError("backend='cuda' but extensions not compiled")
                log.info("Backend: JAX (GPU present, extensions not compiled)")

        if backend == "auto":
            log.info("Backend: JAX (GPU not found or CUDA not available)")
            return JAXOps(pre, use_z2z=use_z2z, mixed_precision=mixed_precision)

    raise ValueError(f"Unknown backend: {backend!r}. Use 'jax', 'cuda', or 'auto'.")
