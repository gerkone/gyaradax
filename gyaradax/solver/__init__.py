"""Time integration (RK4, adaptive CFL), simulation drivers and the eigenvalue solver.

Attributes of the package resolve to the modules the former ``gyaradax/solver.py``
exported from, so ``from gyaradax.solver import gksolve, linear_precompute`` keeps working.
"""

from importlib import import_module

_FALLBACK = (
    "gyaradax.solver.rk4",
    "gyaradax.solver.cfl",
    "gyaradax.operators.fields",
    "gyaradax.operators.precompute",
)


def __getattr__(name):
    if not name.startswith("__"):
        for module in _FALLBACK:
            mod = import_module(module)
            if hasattr(mod, name):
                return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
