"""Analytic geometry model adapter registration.

This module is the first split point for analytic geometry models.  The
numerical implementation still lives in ``geom.py`` for now; this adapter gives
``circ``, ``s-alpha``, and ``miller`` dedicated registry entries without
changing formulas or public entry points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gyaradax.geometry.registry import register_geometry_model
from gyaradax.geometry.spec import GeometrySpec


class AnalyticGeometryModel:
    """Registry adapter for one analytic geometry model name."""

    def __init__(self, name: str, compute_impl: Callable[[GeometrySpec], dict[str, Any]]) -> None:
        self.name = name
        self._compute_impl = compute_impl

    def compute(self, spec: GeometrySpec) -> dict[str, Any]:
        return self._compute_impl(spec)


def register_analytic_geometry_models(
    compute_impl: Callable[[GeometrySpec], dict[str, Any]],
) -> None:
    """Register the current analytic geometry models with the shared registry."""
    for name in ("circ", "s-alpha", "miller"):
        register_geometry_model(AnalyticGeometryModel(name, compute_impl))
