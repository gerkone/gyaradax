"""Analytic geometry model adapter registration.

This module contains dedicated registry adapters for the current circular
analytic geometry names.  The numerical implementation still lives in
``geom.py`` for now; these adapters only provide distinct model classes so the
registry shape can evolve toward one implementation module per geometry without
changing formulas or public entry points. Miller has its own adapter in
``miller.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gyaradax.geometry.registry import register_geometry_model
from gyaradax.geometry.spec import GeometrySpec


class _DelegatingAnalyticGeometryModel:
    """Base adapter that delegates to the current monolithic analytic builder."""

    name: str

    def __init__(self, compute_impl: Callable[[GeometrySpec], dict[str, Any]]) -> None:
        self._compute_impl = compute_impl

    def compute(self, spec: GeometrySpec) -> dict[str, Any]:
        return self._compute_impl(spec)


class CircularGeometryModel(_DelegatingAnalyticGeometryModel):
    """Registry adapter for Lapillonne circular geometry (``geom_type='circ'``)."""

    name = "circ"


class SAlphaGeometryModel(_DelegatingAnalyticGeometryModel):
    """Registry adapter for s-alpha geometry (``geom_type='s-alpha'``)."""

    name = "s-alpha"


def register_analytic_geometry_models(
    compute_impl: Callable[[GeometrySpec], dict[str, Any]],
) -> None:
    """Register circular analytic geometry models with the registry."""
    register_geometry_model(CircularGeometryModel(compute_impl))
    register_geometry_model(SAlphaGeometryModel(compute_impl))
