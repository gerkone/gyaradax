"""Quasilinear transport: saturation rules, the single-point linear solve and the sat2 fit.

  rules            the saturation rules (canonical, sat0, sat2) and CALIBRATION
  point_eval       linear solve + rule at one operating point, the entry point the TORAX
                   plugin calls; nl_at_point for the nonlinear ground truth
  linear_pipeline  QL inputs from a GKW run directory
  calibration      fit_sat2 on the summary tables, and the Cn head classes
  models           bundled Cn calibration heads
"""

from .linear_pipeline import linear_from_fds
from .models import load_cn_payload, select_cn_head
from .point_eval import nl_at_point, ql_at_point
from .rules import (
    CALIBRATION,
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
    "CALIBRATION",
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
    "load_cn_payload",
    "select_cn_head",
]
