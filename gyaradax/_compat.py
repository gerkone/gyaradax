"""Import aliases that keep the pre-subpackage module paths working.

``import gyaradax.params`` returns ``gyaradax.core.params`` itself; a module that was
split (``gyaradax.integrals``, ``gyaradax.utils``) becomes a module holding the public
and private names of all its successors.
"""

import importlib
import importlib.abc
import importlib.util
import sys

MOVED = {
    "gyaradax.params": ("gyaradax.core.params",),
    "gyaradax.state": ("gyaradax.core.state",),
    "gyaradax.constants": ("gyaradax.core.constants",),
    "gyaradax.jax_config": ("gyaradax.core.jax_config",),
    "gyaradax.runtime_config": ("gyaradax.core.runtime_config",),
    "gyaradax.precompute": ("gyaradax.operators.precompute",),
    "gyaradax.stencils": ("gyaradax.operators.stencils",),
    "gyaradax.collisions": ("gyaradax.operators.collisions",),
    "gyaradax.fields": ("gyaradax.operators.fields",),
    "gyaradax.simulate": ("gyaradax.solver.driver",),
    "gyaradax.cfl": ("gyaradax.solver.cfl",),
    "gyaradax.eigenvalue": ("gyaradax.solver.eigenvalue",),
    "gyaradax.diag": ("gyaradax.diagnostics.diag",),
    "gyaradax.telemetry": ("gyaradax.diagnostics.telemetry",),
    "gyaradax.plot_utils": ("gyaradax.diagnostics.plot_utils",),
    "gyaradax.gkw_input": ("gyaradax.io.gkw",),
    "gyaradax.integrals": (
        "gyaradax.operators.poisson",
        "gyaradax.operators.ampere",
        "gyaradax.diagnostics.fluxes",
    ),
    "gyaradax.utils": (
        "gyaradax.io.gkw",
        "gyaradax.io.rundir",
        "gyaradax.diagnostics.fluxes",
        "gyaradax.diagnostics.diag",
        "gyaradax.operators.precompute",
    ),
}


class _AliasLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        targets = [importlib.import_module(name) for name in MOVED[module.__name__]]
        if len(targets) == 1:
            # the import system returns whatever sys.modules holds after exec_module
            sys.modules[module.__name__] = targets[0]
            return
        for target in reversed(targets):
            module.__dict__.update({k: v for k, v in vars(target).items() if not k.startswith("__")})


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in MOVED:
            return importlib.util.spec_from_loader(fullname, _AliasLoader())
        return None


def install():
    # first, so a stale copy of an old module (e.g. another editable checkout) never wins
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
