"""nl-gyx: a nonlinear gyaradax run per radius, averaged over trailing blocks."""

import dataclasses
from typing import Annotated, Any, Dict, Literal

import jax.numpy as jnp

from gyaradax.core.params import GKParams
from gyaradax.quasilinear import point_eval

from . import base
from .base import STATIC


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxNLTransportModel(base.GyaradaxTransportModel):
    """Nonlinear gyaradax transport model (ground truth, not for production loops)."""

    n_steps_nl: int
    nl_tail_blocks: int
    nl_block: int

    @classmethod
    def from_config(cls, cfg) -> "GyaradaxNLTransportModel":
        if cfg.kinetic:
            raise NotImplementedError("nl-gyx runs adiabatic electrons only.")
        return super().from_config(cfg)

    def _per_radius(self, params: GKParams, geom: Dict[str, Any]):
        q_i, q_e, pfe = point_eval.nl_at_point(
            params,
            geom,
            (self.nvpar, self.nmu, self.ns, self.nkx, self.nky),
            n_steps=self.n_steps_nl,
            tail_blocks=self.nl_tail_blocks,
            block=self.nl_block,
        )
        return q_i, q_e, pfe, {"n_steps": jnp.asarray(self.n_steps_nl)}


class GyaradaxNLConfig(base.SolverConfig):
    """Config of the nl-gyx transport model (fields in docs/applications/torax.md)."""

    model_name: Annotated[Literal["nl-gyx"], STATIC] = "nl-gyx"
    n_steps_nl: Annotated[int, STATIC] = 30000
    nl_tail_blocks: Annotated[int, STATIC] = 12
    nl_block: Annotated[int, STATIC] = 500

    def build_transport_model(self) -> GyaradaxNLTransportModel:
        return GyaradaxNLTransportModel.from_config(self)
