"""Circular and s-alpha geometry model adapter registration.

This module contains dedicated registry adapters for the current circular
analytic geometry names.  The numerical implementation still lives in
``geom.py`` for now; these adapters only provide distinct model classes so the
registry shape can evolve toward one implementation module per geometry without
changing formulas or public entry points. Miller has its own adapter in
``miller.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from gyaradax.geometry.lapillonne import _circular_geometry, _poloidal_angle
from gyaradax.geometry.registry import register_geometry_model
from gyaradax.geometry.spec import GeometrySpec


class _DelegatingCircularGeometryModel:
    """Base adapter that delegates shared assembly to the current builder."""

    name: str

    def __init__(self, compute_impl: Callable[[GeometrySpec], dict[str, Any]]) -> None:
        self._compute_impl = compute_impl

    def compute(self, spec: GeometrySpec) -> dict[str, Any]:
        return self._compute_impl(spec)

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
    ) -> dict[str, Any]:
        """Build the model-specific continuous geometry dict.

        Shared grid, tensor, velocity, wavevector, and topology assembly stays
        in ``geom.py``; this method owns only the existing circular/s-alpha
        formula selection.
        """
        theta = _poloidal_angle(sgrid, eps, geom_type=self.name)
        return _circular_geometry(
            theta,
            q,
            shat,
            eps,
            signB=signB,
            signJ=signJ,
            geom_type=self.name,
        )


class CircularGeometryModel(_DelegatingCircularGeometryModel):
    """Registry adapter for Lapillonne circular geometry (``geom_type='circ'``)."""

    name = "circ"


class SAlphaGeometryModel(_DelegatingCircularGeometryModel):
    """Registry adapter for s-alpha geometry (``geom_type='s-alpha'``)."""

    name = "s-alpha"


def register_circular_geometry_models(
    compute_impl: Callable[[GeometrySpec], dict[str, Any]],
) -> None:
    """Register circular and s-alpha geometry models with the registry."""
    register_geometry_model(CircularGeometryModel(compute_impl))
    register_geometry_model(SAlphaGeometryModel(compute_impl))
