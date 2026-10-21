"""TORAX transport models backed by gyaradax: gyaradax-ql and gyaradax-nl.

Importing this package registers both with TORAX; select them by `model_name` in
`transport.core_transport_models`.
"""

from torax import transport as _transport

from .base import GyaradaxConfig, GyaradaxTransportModel, ikxspace_for_shat
from .nl import GyaradaxNLConfig, GyaradaxNLTransportModel
from .ql import GyaradaxQLConfig, GyaradaxQLTransportModel

_REGISTERED = set()


def register(*configs) -> None:
    for config in configs or (GyaradaxQLConfig, GyaradaxNLConfig):
        if config not in _REGISTERED:
            _transport.register_transport_model(config)
            _REGISTERED.add(config)


register()

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
