"""TORAX transport models backed by gyaradax: ql-gyx and nl-gyx.

Importing this package registers both with TORAX; select them by `model_name` in
`transport.core_transport_models`. It also makes the CPU JAX's default device, so TORAX itself runs on
the CPU while the flux tubes run one per GPU, called from the host.
"""

import jax as _jax
from torax import transport as _transport

# a flux-tube solve started from inside a TORAX computation on the same GPU would wait on it
_jax.config.update("jax_default_device", _jax.devices("cpu")[0])

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
]
