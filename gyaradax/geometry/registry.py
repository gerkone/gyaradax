"""Geometry model registry groundwork.

The registry mirrors the backend factory shape at a small scale: a geometry
model is an object with a name and a ``compute(spec)`` method.  Existing public
APIs continue to live in ``geom.py``; this module provides the extension point
for future model modules.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from gyaradax.geometry.spec import GeometrySpec


class GeometryModel(Protocol):
    """Protocol implemented by geometry model builders."""

    name: str

    def compute(self, spec: GeometrySpec) -> dict[str, Any]: ...


class ContinuousGeometryModel(GeometryModel, Protocol):
    """Geometry model that supplies continuous geometry before shared assembly.

    Implementations own only model-specific continuous geometry formulas.
    Shared grids, tensor assembly, topology, and final solver-dict assembly stay
    outside the model implementation.
    """

    def continuous_geometry(
        self,
        *,
        sgrid: Any,
        q: float,
        shat: float,
        eps: float,
        nperiod: int,
        signB: float,
        signJ: float,
        model_params: Mapping[str, Any],
    ) -> dict[str, Any]: ...


_MODELS: dict[str, GeometryModel] = {}


def register_geometry_model(model: GeometryModel) -> GeometryModel:
    """Register a geometry model by its canonical name."""
    _MODELS[model.name] = model
    return model


def get_geometry_model(name: str) -> GeometryModel:
    """Return a registered geometry model."""
    return _MODELS[name]


def list_geometry_models() -> tuple[str, ...]:
    """List registered model names in deterministic order."""
    return tuple(sorted(_MODELS))
