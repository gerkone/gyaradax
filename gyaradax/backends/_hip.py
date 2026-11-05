"""HIP backend for AMD GPUs: the CUDA kernels built with hipcc, and a hipFFT v6 bracket."""

import logging
import os
from pathlib import Path

import jax

from gyaradax.backends import _cuda
from gyaradax.backends.ops import warn_once
from gyaradax.core.state import GKPre
from gyaradax.operators.precompute import prime_factors_smallereq_than

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
        """Padded kx size: the smallest alias-free 5-smooth length; ``GYARADAX_BRACKET_MRAD`` overrides."""
        kn, kp = ixzero, nkx - 1 - ixzero
        m_min = max(2 * kp + kn, 2 * kn + kp) + 1
        choice = os.environ.get("GYARADAX_BRACKET_MRAD", "auto")
        if choice == "native":
            return native
        if choice != "auto":
            m = int(choice)
            if m < m_min:
                raise ValueError(f"GYARADAX_BRACKET_MRAD={m} aliases: the kx product needs {m_min}")
            return m
        return next((m for m in range(m_min, native) if prime_factors_smallereq_than(m, 5)), native)
