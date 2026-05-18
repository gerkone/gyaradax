"""Register gyaradax transport models with TORAX.

After `from gyaradax.torax_plugin import register_all; register_all()`,
the names 'gyaradax-ql' and 'gyaradax-nl' are selectable via
`transport: {model_name: ...}` in any TORAX config dict.
"""

from torax._src.transport_model.register_model import register_transport_model

from gyaradax.torax_plugin.pydantic_model import (
    GyaradaxQLConfig,
    GyaradaxNLConfig,
)

_REGISTERED = False


def register_all() -> None:
    """Register both gyaradax transport models. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_transport_model(GyaradaxQLConfig)
    register_transport_model(GyaradaxNLConfig)
    _REGISTERED = True
