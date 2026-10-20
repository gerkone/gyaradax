"""Quasilinear transport: saturation rules, the single-point linear solve and the sat2 fit.

  rules            the saturation rules (canonical, sat0, sat2)
  point_eval       linear solve + rule at one operating point, the entry point the TORAX
                   plugin calls; nl_at_point for the nonlinear ground truth
  linear_pipeline  QL inputs from a GKW run directory
  calibration      bundled calibrations (weights/*.json): load, save, and the sat2 fit
"""

from . import calibration
from .linear_pipeline import linear_from_fds
from .point_eval import nl_at_point, ql_at_point
from .rules import (
    RULES,
    RULES_WITH_DLNPDR,
    RULES_WITH_KTHNORM,
    dlnpdr_tglf,
    geom_sat2_factors,
    k_perp_eff_squared,
    k_perp_squared,
    ql_flux,
    ql_flux_per_species,
    ql_flux_sat0,
    ql_flux_sat2,
)

__all__ = [
    "calibration",
    "RULES",
    "RULES_WITH_DLNPDR",
    "RULES_WITH_KTHNORM",
    "ql_flux",
    "ql_flux_sat0",
    "ql_flux_sat2",
    "ql_flux_per_species",
    "dlnpdr_tglf",
    "geom_sat2_factors",
    "k_perp_squared",
    "k_perp_eff_squared",
    "ql_at_point",
    "nl_at_point",
    "linear_from_fds",
]
