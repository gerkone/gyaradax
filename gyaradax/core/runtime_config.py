"""Pre-JAX process runtime configuration helpers.

This module intentionally does not import JAX.  Entry points that need to
select CUDA devices or XLA runtime flags should call these helpers before any
JAX import or package import that may import JAX.
"""

from __future__ import annotations

import os
import warnings

_MULTI_GPU_XLA_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_pipelined_all_reduce=true",
    "--xla_gpu_enable_pipelined_all_gather=true",
    "--xla_gpu_enable_while_loop_double_buffering=true",
)


def configure_runtime_env(
    *,
    device: int | None = -1,
    device_list: str | None = None,
    preallocate: bool | str | None = False,
    n_gpus_sp: int = 1,
    n_gpus_vp: int = 1,
    n_gpus_mu: int = 1,
) -> None:
    """Configure process environment that must be set before importing JAX.

    Selecting devices sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` unless it is already
    set, and warns when it replaces a different ``CUDA_VISIBLE_DEVICES``.

    Args:
        device: Single CUDA device index, numbered as ``nvidia-smi`` does.
            ``-1`` or ``None`` leaves ``CUDA_VISIBLE_DEVICES`` unchanged unless
            ``device_list`` is set.
        device_list: Comma-separated CUDA device list.  When provided, this
            takes precedence over ``device``.
        preallocate: Default for ``XLA_PYTHON_CLIENT_PREALLOCATE``.  ``None``
            leaves the variable untouched; bools are rendered as lowercase
            strings. Existing environment values are preserved.
        n_gpus_sp: Number of devices requested on the species axis.
        n_gpus_vp: Number of devices requested on the vparallel axis.
        n_gpus_mu: Number of devices requested on the mu axis.
    """
    visible = device_list or (str(device) if device is not None and device != -1 else None)
    if visible:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        if previous is not None and previous != visible:
            warnings.warn(f"CUDA_VISIBLE_DEVICES={previous} is replaced by {visible}", stacklevel=2)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if preallocate is not None:
        if isinstance(preallocate, bool):
            preallocate_value = str(preallocate).lower()
        else:
            preallocate_value = preallocate
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", preallocate_value)

    if n_gpus_sp * n_gpus_vp * n_gpus_mu > 1:
        append_xla_flags(_MULTI_GPU_XLA_FLAGS)


def append_xla_flags(flags: tuple[str, ...]) -> None:
    """Append XLA flags to the process environment."""
    joined = " ".join(flags)
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + joined).strip()
