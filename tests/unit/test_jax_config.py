"""Tests for centralized JAX configuration helpers."""

from __future__ import annotations

import subprocess
import sys

import jax
import pytest

from gyaradax.jax_config import enable_x64


def test_enable_x64_sets_jax_config() -> None:
    enable_x64()

    assert jax.config.read("jax_enable_x64") is True


def test_enable_x64_is_idempotent() -> None:
    enable_x64()
    enable_x64()

    assert jax.config.read("jax_enable_x64") is True


def _run_fresh_python(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_gyaradax_enables_x64_in_fresh_process() -> None:
    _run_fresh_python(
        """
import jax
import gyaradax
assert jax.config.read('jax_enable_x64') is True
"""
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "gyaradax.params",
        "gyaradax.simulate",
        "gyaradax.solver",
        "gyaradax.stencils",
    ],
)
def test_direct_config_relevant_submodule_import_enables_x64_in_fresh_process(
    module_name: str,
) -> None:
    _run_fresh_python(
        f"""
import importlib
import jax
importlib.import_module({module_name!r})
assert jax.config.read('jax_enable_x64') is True
"""
    )
