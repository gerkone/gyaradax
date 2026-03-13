import jax

# Enforce 64-bit precision for all JAX calculations.
jax.config.update("jax_enable_x64", True)

from gyaradax.solver import (  # noqa: E402
    GKParams,
    GKState,
    default_state,
    gksolve,
    gkstep_single,
    simulate,
    init_f,
    load_config,
    gkparams_from_config,
    linear_precompute,
    linear_rhs,
    nonlinear_term_iii,
    normalize_per_ky,
    kx_ky_grids,
    mode_amplitude,
)
from gyaradax.geometry import load_geometry, parse_input_dat  # noqa: E402
from gyaradax.integrals import get_integrals  # noqa: E402
from gyaradax.utils import (  # noqa: E402
    load_gkw_dump,
    load_gkw_k_dump,
    read_gkw_dump_time,
    save_checkpoint,
    load_checkpoint,
)
from gyaradax.diag import (  # noqa: E402
    kx0_mode_columns,
    project_all_modes_to_kx0,
    term_iii_rhs,
    term_iii_fft_pack_roundtrip,
)

__all__ = [
    "GKParams",
    "GKState",
    "default_state",
    "gksolve",
    "gkstep_single",
    "simulate",
    "init_f",
    "load_config",
    "gkparams_from_config",
    "load_geometry",
    "parse_input_dat",
    "get_integrals",
    "load_gkw_dump",
    "load_gkw_k_dump",
    "read_gkw_dump_time",
    "save_checkpoint",
    "load_checkpoint",
    "kx0_mode_columns",
    "project_all_modes_to_kx0",
    "term_iii_rhs",
    "term_iii_fft_pack_roundtrip",
    "linear_precompute",
    "linear_rhs",
    "nonlinear_term_iii",
    "normalize_per_ky",
    "kx_ky_grids",
    "mode_amplitude",
]
