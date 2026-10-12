"""Centralized JAX configuration helpers."""

from __future__ import annotations

import jax


def enable_x64() -> None:
    """Enable JAX 64-bit mode.

    JAX config updates are idempotent when repeated with the same value.  Call
    this before arrays or JIT-sensitive objects are created.
    """
    jax.config.update("jax_enable_x64", True)
