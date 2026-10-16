"""Quasilinear transport: saturation rules and the single-point linear solve.

  rules / saturation            the saturation rules and their shared pieces
  point_eval                    linear solve + rule at one operating point,
                                the entry point the TORAX plugin calls
  linear_pipeline               linear run from a GKW directory, and the
                                harvest of one run's spectra
  data                          GKW output readers
  models                        bundled Cn calibration heads

Calibration data no longer comes from walking paired run directories: the
summary parquets under /restricteddata/ukaea/gyrokinetics/preprocessed
carry gamma(ky), the quasilinear weights and the spectra, so the campaign
harvest, its storage layer and the Cn fitting heads that fed on it are gone.
"""

from .data import load_linear_outputs, parse_input_dat
from .linear_pipeline import linear_from_fds, linear_run
from .models import load_cn_payload, select_cn_head
from .point_eval import ql_at_point, ql_rules_at_point
from .rules import (
    RULES,
    RULES_WITH_DLNPDR,
    RULES_WITH_KTHNORM,
    dlnpdr_tglf,
    geom_sat2_factors,
    ql_flux_per_species,
    ql_flux_sat0,
    ql_flux_sat2,
)
from .saturation import k_perp_eff_squared, k_perp_squared, ql_flux

__all__ = [
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
    "ql_rules_at_point",
    "linear_from_fds",
    "linear_run",
    "load_linear_outputs",
    "parse_input_dat",
    "load_cn_payload",
    "select_cn_head",
]
