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
  fit_cn_polynomial                polynomial C_n(features) calibration (degree tunable)
  PolynomialCn                     fitted polynomial C_n with .predict(X, F)
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
    fit_cn_heads,
    fit_cn_log,
    fit_cn_parametric,
    fit_cn_polynomial,
    fit_cn_polynomial_log,
    ParametricCn,
    PolynomialCn,
    r2_score,
    DEFAULT_PARAM_FEATURES,
)
# advanced calibration heads need scikit-learn, an optional dependency. keep the
# package importable (ql_flux, parametric/polynomial heads) when it is absent.
try:
    from .calibration_advanced import (
        fit_cn_ridge_polynomial,
        fit_cn_gbm,
        fit_cn_gp_log,
        RidgePolynomialCn,
        GBMCn,
        GPLogCn,
    )
except ImportError:
    fit_cn_ridge_polynomial = fit_cn_gbm = fit_cn_gp_log = None
    RidgePolynomialCn = GBMCn = GPLogCn = None
from .linear_pipeline import linear_from_fds, linear_run, harvest, root_mse, root_mse_log
from .models import load_default_weights as load_default_cn_weights
from .models import load_weights_from_name as load_cn_weights_from_name

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
    "fit_cn_heads",
    "fit_cn_log",
    "fit_cn_parametric",
    "fit_cn_polynomial",
    "fit_cn_polynomial_log",
    "fit_cn_ridge_polynomial",
    "fit_cn_gbm",
    "fit_cn_gp_log",
    "ParametricCn",
    "PolynomialCn",
    "RidgePolynomialCn",
    "GBMCn",
    "GPLogCn",
    "r2_score",
    "DEFAULT_PARAM_FEATURES",
    "linear_from_fds",
    "linear_run",
    "harvest",
    "root_mse",
    "root_mse_log",
    "load_default_cn_weights",
    "load_cn_weights_from_name",
]
