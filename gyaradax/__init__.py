# lazy-loading package. jax config is handled by gyaradax.bootstrap.init_jax().

import jax

jax.config.update("jax_enable_x64", True)

from gyaradax.params import GKParams, load_config, gkparams_from_config, gkparams_from_input_and_geometry
from gyaradax.solver import (
    gksolve,
    GKPre,
    default_state,
    gkstep_single,
    init_f,
)
from gyaradax.simulate import gksimulate, gk_init, gk_run, gk_from_config
from gyaradax.utils import load_geometry
from gyaradax.geometry import compute_geometry, compute_geometry_from_input, geometry_from_geom_dat_and_input
from gyaradax.integrals import get_integrals
from gyaradax.utils import load_gkw_k_dump

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
    "init_f",
    "load_config",
    "gkparams_from_config",
    "gkparams_from_input_and_geometry",
    "compute_geometry_from_input",
    "load_geometry",
    "compute_geometry",
    "get_integrals",
    "load_gkw_k_dump",
]
