"""Backend selection: :func:`create_ops` returns the JAX, CUDA or HIP :class:`SolverOps`.

Example::

    from gyaradax.backends import create_ops

    ops = create_ops(pre)          # CUDA when the library and a GPU are present, else JAX
    ops = create_ops(pre, "jax")   # pure JAX
    ops = create_ops(pre, "cuda")  # CUDA, raises if unavailable
    ops = create_ops(pre, "hip")   # HIP on AMD GPUs, raises if unavailable
"""

import logging

import jax

from gyaradax.backends._jax import JAXOps
from gyaradax.backends.ops import SolverOps, warn_once

log = logging.getLogger(__name__)


def _load_cuda_backend():
    """Import CUDA backend objects only when CUDA selection needs them."""
    from gyaradax.backends._cuda import CUDAOps, is_available

    return CUDAOps, is_available


def _load_hip_backend():
    """Import HIP backend objects only when HIP selection needs them."""
    from gyaradax.backends._hip import HIPOps, is_available

    return HIPOps, is_available


def _on_rocm() -> bool:
    return any(
        "rocm" in str(getattr(getattr(d, "client", None), "platform_version", ""))
        for d in jax.devices()
    )


def create_ops(
    pre,
    backend: str = "auto",
    use_z2z: bool = False,
    mixed_precision: bool = True,
    mesh=None,
    fused_bracket: bool = False,
) -> SolverOps:
    """Create the solver operations for a backend.

    Args:
        pre: Precomputed coefficients from :func:`~gyaradax.operators.precompute.linear_precompute`.
        backend: ``"auto"``, ``"jax"``, ``"cuda"`` or ``"hip"``. ``"auto"`` picks the vendor
            kernels when a GPU and the compiled library are present and warns when it falls
            back to JAX on a GPU.
        use_z2z: Z2Z instead of R2C FFTs in the JAX bracket (the CUDA backend is Z2Z only).
        mixed_precision: FP32 FFTs in the nonlinear bracket; ``False`` keeps FP64.
        mesh: Device mesh of a sharded run (:func:`gyaradax.sharding.build_mesh`); the
            bracket and the CUDA kernels then run on the local blocks.
        fused_bracket: CUDA: pack and unpack the mixed-precision bracket inside the cuFFT passes.

    Returns:
        A :class:`SolverOps` instance.
    """
    if backend == "jax":
        z2z_str = " (z2z)" if use_z2z else ""
        mp_str = " (mixed)" if mixed_precision else " (fp64)"
        log.info("Backend: JAX%s%s", z2z_str, mp_str)
        return JAXOps(pre, use_z2z=use_z2z, mixed_precision=mixed_precision, mesh=mesh)

    if backend == "hip" or (backend == "auto" and _on_rocm()):
        HIPOps, is_available = _load_hip_backend()
        if is_available():
            mp_str = " (mixed)" if mixed_precision else " (fp64)"
            log.info("Backend: HIP%s", mp_str)
            return HIPOps(
                pre,
                use_z2z=use_z2z,
                mixed_precision=mixed_precision,
                mesh=mesh,
                fused_bracket=fused_bracket,
            )
        if backend == "hip":
            raise RuntimeError("backend='hip' but no GPU or the HIP library is not built")
        warn_once(log, "backend 'auto' runs JAX on a GPU: the HIP library is not built")
        return JAXOps(pre, use_z2z=use_z2z, mixed_precision=mixed_precision, mesh=mesh)

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
                warn_once(log, f"backend 'auto' runs JAX on a GPU: CUDA import failed ({exc})")
            else:
                if is_available():
                    mp_str = " (mixed)" if mixed_precision else " (fp64)"
                    log.info("Backend: CUDA%s [Z2Z-only]", mp_str)
                    return CUDAOps(
                        pre,
                        use_z2z=use_z2z,
                        mixed_precision=mixed_precision,
                        mesh=mesh,
                        fused_bracket=fused_bracket,
                    )
                if backend == "cuda":
                    raise RuntimeError("backend='cuda' but extensions not compiled")
                warn_once(log, "backend 'auto' runs JAX on a GPU: the CUDA library is not built")

        if backend == "auto":
            log.info("Backend: JAX (GPU not found or CUDA not available)")
            return JAXOps(pre, use_z2z=use_z2z, mixed_precision=mixed_precision, mesh=mesh)

    raise ValueError(f"Unknown backend: {backend!r}. Use 'jax', 'cuda', 'hip' or 'auto'.")
