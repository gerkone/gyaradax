"""Top-level convenience imports for gyaradax.

The package import is intentionally lightweight: it enables the project-wide
JAX x64 policy and exposes compatibility constants, while heavier solver,
simulation, geometry, and diagnostics objects are imported lazily on first use.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gyaradax.constants import EPS
from gyaradax.jax_config import enable_x64

enable_x64()

# Compatibility alias used by older internal and external imports.
_EPS = EPS

__all__ = [
    "GKParams",
    "GKPre",
    "default_state",
    "gksolve",
    "gkstep_single",
    "gksimulate",
    "gk_init",
    "gk_run",
    "gk_from_config",
    "gk_from_gkw_dir",
    "init_f",
    "load_config",
    "gkparams_from_config",
    "gkparams_from_input_and_geometry",
    "compute_geometry_from_input",
    "geometry_from_geom_dat_and_input",
    "load_geometry",
    "compute_geometry",
    "get_integrals",
    "load_gkw_k_dump",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "GKParams": ("gyaradax.params", "GKParams"),
    "load_config": ("gyaradax.params", "load_config"),
    "gkparams_from_config": ("gyaradax.params", "gkparams_from_config"),
    "gkparams_from_input_and_geometry": (
        "gyaradax.params",
        "gkparams_from_input_and_geometry",
    ),
    "GKPre": ("gyaradax.state", "GKPre"),
    "default_state": ("gyaradax.solver", "default_state"),
    "gksolve": ("gyaradax.solver", "gksolve"),
    "gkstep_single": ("gyaradax.solver", "gkstep_single"),
    "init_f": ("gyaradax.solver", "init_f"),
    "gksimulate": ("gyaradax.simulate", "gksimulate"),
    "gk_init": ("gyaradax.simulate", "gk_init"),
    "gk_run": ("gyaradax.simulate", "gk_run"),
    "gk_from_config": ("gyaradax.simulate", "gk_from_config"),
    "gk_from_gkw_dir": ("gyaradax.simulate", "gk_from_gkw_dir"),
    "compute_geometry": ("gyaradax.geometry", "compute_geometry"),
    "compute_geometry_from_input": ("gyaradax.geometry", "compute_geometry_from_input"),
    "geometry_from_geom_dat_and_input": (
        "gyaradax.geometry",
        "geometry_from_geom_dat_and_input",
    ),
    "get_integrals": ("gyaradax.integrals", "get_integrals"),
    "load_geometry": ("gyaradax.utils", "load_geometry"),
    "load_gkw_k_dump": ("gyaradax.utils", "load_gkw_k_dump"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve top-level convenience exports."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes including lazy convenience exports."""
    return sorted(set(globals()) | set(__all__))
