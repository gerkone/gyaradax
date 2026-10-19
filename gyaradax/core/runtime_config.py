"""Pre-JAX process runtime configuration helpers.

This module intentionally does not import JAX.  Entry points that need to
select CUDA devices or XLA runtime flags should call these helpers before any
JAX import or package import that may import JAX.
"""

from __future__ import annotations

import os

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

    Args:
        device: Single CUDA device index.  ``-1`` or ``None`` leaves existing
            ``CUDA_VISIBLE_DEVICES`` unchanged unless ``device_list`` is set.
        device_list: Comma-separated CUDA device list.  When provided, this
            takes precedence over ``device``.
        preallocate: Default for ``XLA_PYTHON_CLIENT_PREALLOCATE``.  ``None``
            leaves the variable untouched; bools are rendered as lowercase
            strings. Existing environment values are preserved.
        n_gpus_sp: Number of devices requested on the species axis.
        n_gpus_vp: Number of devices requested on the vparallel axis.
        n_gpus_mu: Number of devices requested on the mu axis.
    """
    if device_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_list
    elif device is not None and device != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

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
