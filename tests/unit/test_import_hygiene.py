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
    'gyaradax.solver.rk4',
    'gyaradax.operators',
    'gyaradax.diagnostics',
    'gyaradax.io',
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
assert 'gyaradax.solver.rk4' not in sys.modules
first = gyaradax.gksolve
assert 'gyaradax.solver.rk4' in sys.modules
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


def test_old_module_paths_alias_the_new_modules_in_fresh_process() -> None:
    _run_fresh_python(
        """
import gyaradax.core.params
import gyaradax.params
import gyaradax.simulate
import gyaradax.solver.driver
assert gyaradax.params is gyaradax.core.params
assert gyaradax.simulate is gyaradax.solver.driver
from gyaradax.params import GKParams
assert GKParams is gyaradax.core.params.GKParams
from gyaradax.integrals import calculate_apar, calculate_fluxes, calculate_phi, geom_tensors
from gyaradax.utils import load_gkw_k_dump, pack_half_spectrum, save_dumps
from gyaradax.simulate import _ensure_species_arrays
from gyaradax.solver import gksolve, linear_precompute, _compute_fields, estimate_timestep
from gyaradax.operators.ampere import calculate_apar as apar
assert apar is calculate_apar
"""
    )
