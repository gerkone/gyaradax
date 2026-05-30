"""Tests for lightweight package import and lazy top-level exports."""

from __future__ import annotations

import subprocess
import sys


def _run_fresh_python(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_gyaradax_keeps_heavy_modules_lazy_in_fresh_process() -> None:
    _run_fresh_python(
        """
import sys
import jax
import gyaradax
assert jax.config.read('jax_enable_x64') is True
for name in (
    'gyaradax.solver',
    'gyaradax.simulate',
    'gyaradax.integrals',
    'gyaradax.backends',
    'gyaradax.geometry',
):
    assert name not in sys.modules, name
"""
    )


def test_top_level_solver_export_is_lazy_and_cached_in_fresh_process() -> None:
    _run_fresh_python(
        """
import sys
import gyaradax
assert 'gyaradax.solver' not in sys.modules
first = gyaradax.gksolve
assert 'gyaradax.solver' in sys.modules
second = gyaradax.gksolve
assert first is second
from gyaradax import gksolve
assert gksolve is first
"""
    )


def test_common_top_level_imports_still_work_in_fresh_process() -> None:
    _run_fresh_python(
        """
from gyaradax import GKParams, GKPre, gk_init, gksolve, load_config
assert callable(load_config)
assert callable(gksolve)
assert callable(gk_init)
assert GKParams.__name__ == 'GKParams'
assert GKPre.__name__ == 'GKPre'
"""
    )


def test_current_all_exports_resolve_lazily_in_fresh_process() -> None:
    _run_fresh_python(
        """
import gyaradax
for name in gyaradax.__all__:
    assert getattr(gyaradax, name) is not None, name
"""
    )


def test_legacy_eps_alias_remains_available_in_fresh_process() -> None:
    _run_fresh_python(
        """
import gyaradax
from gyaradax.constants import EPS
assert gyaradax._EPS == EPS == 1e-30
"""
    )
