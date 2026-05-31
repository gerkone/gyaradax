"""Pydantic config schemas for gyaradax TORAX transport models.

Same pattern as `torax._src.transport_model.pydantic_model`: every model
exposes a frozen `*Config(TransportBase)` with a hardcoded `model_name`
Literal, a `build_transport_model()` returning the frozen-dataclass
TransportModel, and a `build_runtime_params(t)` returning per-step config.
"""

from typing import Annotated, Literal, Optional, Tuple
import dataclasses

import chex
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base


class _GyaradaxBase(pydantic_model_base.TransportBase):
    """Shared knobs for every gyaradax transport model."""

    rho_match: Annotated[
        Tuple[float, ...], torax_pydantic.JAX_STATIC
    ] = (0.35, 0.55, 0.75, 0.875)
    backend: Annotated[str, torax_pydantic.JAX_STATIC] = "jax"

    nvpar: Annotated[int, torax_pydantic.JAX_STATIC] = 32
    nmu: Annotated[int, torax_pydantic.JAX_STATIC] = 8
    ns: Annotated[int, torax_pydantic.JAX_STATIC] = 16
    nkx: Annotated[int, torax_pydantic.JAX_STATIC] = 43
    nky: Annotated[int, torax_pydantic.JAX_STATIC] = 16
    ikxspace: Annotated[int, torax_pydantic.JAX_STATIC] = 5


class GyaradaxQLConfig(_GyaradaxBase):
    """QL gyaradax: differentiable, in-loop."""

    model_name: Annotated[
        Literal["gyaradax-ql"], torax_pydantic.JAX_STATIC
    ] = "gyaradax-ql"
    n_steps_linear: Annotated[int, torax_pydantic.JAX_STATIC] = 200
    ncv_eigensolve: Annotated[int, torax_pydantic.JAX_STATIC] = 0
    cn_calibration_path: Annotated[
        Optional[str], torax_pydantic.JAX_STATIC
    ] = "auto"
    early_stop: Annotated[bool, torax_pydantic.JAX_STATIC] = True
    early_stop_block: Annotated[int, torax_pydantic.JAX_STATIC] = 25
    early_stop_atol: Annotated[float, torax_pydantic.JAX_STATIC] = 1e-4
    early_stop_rtol: Annotated[float, torax_pydantic.JAX_STATIC] = 1e-3
    early_stop_min_steps: Annotated[int, torax_pydantic.JAX_STATIC] = 50

    def build_transport_model(self):
        from gyaradax.torax_plugin.gyaradax_ql_transport_model import (
            GyaradaxQLTransportModel,
        )
        return GyaradaxQLTransportModel.from_config(self)

    def build_runtime_params(self, t: chex.Numeric):
        from gyaradax.torax_plugin.gyaradax_ql_transport_model import RuntimeParams
        base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
        return RuntimeParams(DV_effective=True, An_min=0.05, **base_kwargs)


class GyaradaxNLConfig(_GyaradaxBase):
    """NL gyaradax: outer-loop callback by default; direct mode for one-shot runs."""

    model_name: Annotated[
        Literal["gyaradax-nl"], torax_pydantic.JAX_STATIC
    ] = "gyaradax-nl"
    mode: Annotated[
        Literal["callback", "direct"], torax_pydantic.JAX_STATIC
    ] = "callback"
    n_steps_nl: Annotated[int, torax_pydantic.JAX_STATIC] = 4000
    n_average: Annotated[int, torax_pydantic.JAX_STATIC] = 1024
    flux_table_path: Annotated[
        Optional[str], torax_pydantic.JAX_STATIC
    ] = None

    def build_transport_model(self):
        from gyaradax.torax_plugin.gyaradax_nl_transport_model import (
            GyaradaxNLTransportModel,
        )
        return GyaradaxNLTransportModel.from_config(self)

    def build_runtime_params(self, t: chex.Numeric):
        from gyaradax.torax_plugin.gyaradax_nl_transport_model import RuntimeParams
        base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
        return RuntimeParams(DV_effective=True, An_min=0.05, **base_kwargs)
