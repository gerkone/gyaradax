# lazy-loading package. jax config is handled by gyaradax.bootstrap.init_jax().

_submodule_mapping = {
    "GKParams": "gyaradax.params",
    "load_config": "gyaradax.params",
    "gkparams_from_config": "gyaradax.params",
    "gksolve": "gyaradax.solver",
    "GKPre": "gyaradax.solver",
    "default_state": "gyaradax.solver",
    "gkstep_single": "gyaradax.solver",
    "init_f": "gyaradax.solver",
    "gksimulate": "gyaradax.simulate",
    "load_geometry": "gyaradax.geometry",
    "get_integrals": "gyaradax.integrals",
    "load_gkw_k_dump": "gyaradax.utils",
}


def __getattr__(name):
    if name in _submodule_mapping:
        import importlib

        module = importlib.import_module(_submodule_mapping[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_submodule_mapping.keys())
