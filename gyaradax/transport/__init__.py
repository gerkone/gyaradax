"""TORAX transport models backed by gyaradax: ql-gyx and nl-gyx.

Importing this package registers both with TORAX; select them by `model_name` in
`transport.core_transport_models`.
"""

from torax import transport as _transport

from .base import GyaradaxConfig, GyaradaxTransportModel, ikxspace_for_shat
from .nl import GyaradaxNLConfig, GyaradaxNLTransportModel
from .ql import GyaradaxQLConfig, GyaradaxQLTransportModel

_transport.register_transport_model(GyaradaxQLConfig)
_transport.register_transport_model(GyaradaxNLConfig)

__all__ = [
    "GyaradaxConfig",
    "GyaradaxNLConfig",
    "GyaradaxNLTransportModel",
    "GyaradaxQLConfig",
    "GyaradaxQLTransportModel",
    "GyaradaxTransportModel",
    "ikxspace_for_shat",
    "register",
]
