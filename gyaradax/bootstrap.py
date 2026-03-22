"""Centralized jax configuration. call init_jax() before any jax imports."""

import os
import sys

_initialized = False


def init_jax(device: int = None):
    """Configure jax: fp64 precision and optional gpu device isolation."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    if device is None:
        for i, arg in enumerate(sys.argv):
            if arg == "--device" and i + 1 < len(sys.argv):
                try:
                    device = int(sys.argv[i + 1])
                except ValueError:
                    pass
                break

    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
