"""Tests for centralized JAX configuration helpers."""

from __future__ import annotations

import jax

from gyaradax.jax_config import enable_x64


def test_enable_x64_sets_jax_config() -> None:
    enable_x64()

    assert jax.config.read("jax_enable_x64") is True


def test_enable_x64_is_idempotent() -> None:
    enable_x64()
    enable_x64()

    assert jax.config.read("jax_enable_x64") is True
