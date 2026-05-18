"""TORAX integration for gyaradax.

Two transport models exposed via TORAX's `model_name` dispatch:

    gyaradax-ql  - QL gyrokinetic, pure JAX, AD-compatible (with backend='jax'),
                   fast enough to live inside TORAX's Newton iteration.
    gyaradax-nl  - NL gyrokinetic. Two sub-modes (set via config.mode):
                   'direct'    - run NL gyaradax inside the call. Slow; only
                                 sensible for one-shot forward runs or off-line
                                 surrogate-data generation.
                   'callback'  - read a precomputed (qi, qe, pfe) table from
                                 disk; the right mode for PORTALS-style outer
                                 loops.

Both subclass `QuasilinearTransportModel` via the shared
`GyaradaxBasedTransportModel` base, mirroring TORAX's
`qualikiz_based_transport_model` / `tglf_based_transport_model` pattern.

Activation in a TORAX config:

    'transport': {
        'model_name': 'gyaradax-ql',
        'rho_match': (0.35, 0.55, 0.75, 0.875),
        'backend': 'cuda',
    }
"""

from gyaradax.torax_plugin.gyaradax_ql_transport_model import GyaradaxQLTransportModel
from gyaradax.torax_plugin.gyaradax_nl_transport_model import GyaradaxNLTransportModel
from gyaradax.torax_plugin.pydantic_model import (
    GyaradaxQLConfig,
    GyaradaxNLConfig,
)
from gyaradax.torax_plugin.register import register_all

__all__ = [
    "GyaradaxQLTransportModel",
    "GyaradaxNLTransportModel",
    "GyaradaxQLConfig",
    "GyaradaxNLConfig",
    "register_all",
]
