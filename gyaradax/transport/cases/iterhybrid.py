"""TORAX's iterhybrid_predictor_corrector example with ql-gyx in place of QLKNN.

Use it as a TORAX config file (`run_torax --config=<this file>`), or
`torax.ToraxConfig.from_dict(gyaradax.transport.cases.iterhybrid.CONFIG)`.
"""

import copy

from torax.examples import iterhybrid_predictor_corrector

import gyaradax.transport  # noqa: F401

QL = {
    "model_name": "ql-gyx",
    "rho_min": 0.15,
    "rho_max": 0.95,
    "rho_match": (0.3, 0.45, 0.6, 0.75, 0.85, 0.9),
    "kinetic": True,
    "rule": "sat2",
    "backend": "cuda",
    "fused_linear": True,
    "DV_effective": True,
    "An_min": 0.05,
}

CONFIG = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
CONFIG["transport"]["core_transport_models"] = {
    "turbulent" if name == "qlknn" else name: QL if name == "qlknn" else model
    for name, model in CONFIG["transport"]["core_transport_models"].items()
}
CONFIG["time_step_calculator"]["max_steps"] = 2000
