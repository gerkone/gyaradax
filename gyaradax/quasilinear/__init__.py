"""gyaradax.quasilinear — quasilinear transport model.

JAX-native QL model built on gyaradax linear runs, calibrated against
nonlinear gyaradax/GKW simulations.

Public API:
  ql_flux                          canonical saturation rule (γ/⟨k⊥²⟩ · W)
  ql_flux_diagnostics              same + intermediate quantities
  linear_from_fds                  load + compute QL inputs from a GKW _Lin dir
  linear_run                       run gyaradax linearly (disable_per_ky_norm)
  harvest                          parallel-IO harvest of (X, Y, features) from (Lin, NL) triples
  fit_cn                           scalar amplitude calibration
  fit_cn_parametric                affine C_n(ŝ, q, R/L_T, R/L_n) calibration
  ParametricCn                     fitted parametric C_n with .predict(X, F)
"""

from .saturation import (
    ql_flux,
    ql_flux_diagnostics,
    k_perp_squared,
    k_perp_eff_squared,
)
from .data import (
    load_linear_outputs,
    load_nonlinear_target,
    pair_sims,
    parse_input_dat,
    gradient_labels,
    physics_features,
    is_unstable,
    growth_rate_max,
    FEATURE_NAMES,
)
from .calibration import (
    fit_cn,
    fit_cn_log,
    fit_cn_parametric,
    ParametricCn,
    r2_score,
    DEFAULT_PARAM_FEATURES,
)
from .linear_pipeline import linear_from_fds, linear_run, harvest, root_mse, root_mse_log

__all__ = [
    "ql_flux",
    "ql_flux_diagnostics",
    "k_perp_squared",
    "k_perp_eff_squared",
    "load_linear_outputs",
    "load_nonlinear_target",
    "pair_sims",
    "parse_input_dat",
    "gradient_labels",
    "physics_features",
    "is_unstable",
    "growth_rate_max",
    "FEATURE_NAMES",
    "fit_cn",
    "fit_cn_log",
    "fit_cn_parametric",
    "ParametricCn",
    "r2_score",
    "DEFAULT_PARAM_FEATURES",
    "linear_from_fds",
    "linear_run",
    "harvest",
    "root_mse",
    "root_mse_log",
]
