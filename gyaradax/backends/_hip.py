"""HIP backend for AMD GPUs: the CUDA kernels built with hipcc, and a hipFFT v6 bracket."""

import logging
from pathlib import Path

import jax

from gyaradax.backends import _cuda
from gyaradax.backends.ops import warn_once
from gyaradax.core.state import GKPre

log = logging.getLogger(__name__)

LIB_PATH = Path(__file__).parent / "hip_kernels" / "libgyaradax_hip.so"


def is_available() -> bool:
    """Whether the HIP kernels are compiled and JAX sees a ROCm GPU."""
    if not LIB_PATH.exists():
        return False
    try:
        return bool(jax.devices("rocm"))
    except RuntimeError:
        return False


@jax.tree_util.register_pytree_node_class
class HIPOps(_cuda.CUDAOps):
    """Solver operations on the HIP kernels; same call layout as :class:`CUDAOps`. Not differentiable."""

    def __init__(
        self,
        pre: GKPre,
        use_z2z: bool = False,
        mixed_precision: bool = True,
        mesh=None,
        fused_bracket: bool = False,
    ):
        if mesh is not None:
            raise NotImplementedError("backend='hip' runs on a single GPU")
        if fused_bracket:
            warn_once(log, "fused_bracket is ignored: the hipFFT bracket has no callbacks")
        _cuda._register_ffi(LIB_PATH, "ROCM")
        super().__init__(pre, use_z2z, mixed_precision, mesh, fused_bracket)

    def _bracket_row_size(self, fp64, mphi, nky, b_df, nsp, b_inner):
        return mphi

    def _bracket_kx_size(self, native, nkx, ixzero, strides, fp64):
        return native
