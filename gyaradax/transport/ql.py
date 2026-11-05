"""ql-gyx: a linear gyaradax solve per radius, saturated by a quasilinear rule."""

import dataclasses
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

import jax.numpy as jnp

from gyaradax.core.params import GKParams
from gyaradax.quasilinear import point_eval
from gyaradax.quasilinear import calibration

from . import base
from .base import STATIC

RuntimeParams = base.RuntimeParams


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxQLTransportModel(base.GyaradaxTransportModel):
    """Quasilinear gyaradax transport model."""

    n_steps_linear: int
    rule: str
    rule_options: Tuple[Tuple[str, Any], ...]
    cn: float
    calibration: str
    early_stop: bool
    early_stop_block: int
    early_stop_atol: float
    early_stop_rtol: float
    early_stop_min_steps: int
    early_stop_patience: int

    @classmethod
    def from_config(cls, cfg) -> "GyaradaxQLTransportModel":
        cal = calibration.load(cfg.calibration or "", cfg.rule)
        if cfg.cn is None and cal is None:
            raise ValueError(f"no calibration for rule {cfg.rule!r}: set `calibration` or `cn`.")
        if cfg.kinetic and cfg.cn is None and "kinetic" not in cal.get("electrons", ()):
            raise ValueError(f"calibration {cfg.calibration!r} has no kinetic fit: set `cn`.")
        cn = float(cfg.cn) if cfg.cn is not None else cal["parameters"]["cn"]
        # records with fitted arrays stay in the calibration cache, not in the static options
        fitted = {} if cfg.rule in point_eval.RULES_WITH_OMEGA else (calibration.rule_kwargs(cal) if cal else {})
        options = {**fitted, **(cfg.rule_options or {})}
        return super().from_config(
            cfg, rule_options=tuple(sorted(options.items())), cn=cn, calibration=cfg.calibration or ""
        )

    def _rule_options(self):
        options = dict(self.rule_options)
        if self.rule in point_eval.RULES_WITH_OMEGA:
            cal = calibration.load(self.calibration, self.rule)
            options = {**calibration.rule_kwargs(cal), **options}
        return options

    def _per_radius(self, params: GKParams, geom: Dict[str, Any]):
        return point_eval.ql_at_point(
            params,
            geom,
            (self.nvpar, self.nmu, self.ns, self.nkx, self.nky),
            cn=jnp.asarray(self.cn),
            rule=self.rule,
            rule_options=self._rule_options(),
            n_steps_linear=self.n_steps_linear,
            early_stop=self.early_stop,
            early_stop_opts=dict(
                block=self.early_stop_block,
                min_steps=self.early_stop_min_steps,
                atol=self.early_stop_atol,
                rtol=self.early_stop_rtol,
                patience=self.early_stop_patience,
            ),
            return_diagnostics=True,
        )


class GyaradaxQLConfig(base.SolverConfig):
    """Config of the ql-gyx transport model (fields in docs/applications/torax.md)."""

    model_name: Annotated[Literal["ql-gyx"], STATIC] = "ql-gyx"
    n_steps_linear: Annotated[int, STATIC] = 2000
    rule: Annotated[str, STATIC] = "sat2"
    rule_options: Annotated[Optional[Dict[str, Any]], STATIC] = None
    calibration: Annotated[Optional[str], STATIC] = "auto"
    cn: Annotated[Optional[float], STATIC] = None
    early_stop: Annotated[bool, STATIC] = True
    early_stop_block: Annotated[int, STATIC] = 100
    early_stop_atol: Annotated[float, STATIC] = 1e-4
    early_stop_rtol: Annotated[float, STATIC] = 1e-3
    early_stop_min_steps: Annotated[int, STATIC] = 200
    early_stop_patience: Annotated[int, STATIC] = 2

    def build_transport_model(self) -> GyaradaxQLTransportModel:
        return GyaradaxQLTransportModel.from_config(self)
