"""Typed geometry construction specifications.

``GeometrySpec`` is neutral configuration data: it records which analytic
geometry model to build plus the grid/equilibrium parameters needed by the
current ``compute_geometry`` public API.  It intentionally does not implement
geometry math; model implementations consume the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GeometrySpec:
    """Normalized specification for analytic geometry construction.

    The fields mirror the existing ``compute_geometry`` arguments so adopting
    the spec can be behavior-preserving.  Model-specific parameters (currently
    Miller shape parameters) are stored separately in ``model_params``.
    """

    model: str
    q: float
    shat: float
    eps: float
    ns: int
    nkx: int
    nky: int
    nvpar: int
    nmu: int
    vpar_max: float = 3.0
    nperiod: int = 1
    kxmax: float = 0.0
    krhomax: float = 1.4
    ikxspace: int = 5
    signB: float = 1.0
    Rref: float = 100.0
    model_params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def geom_type(self) -> str:
        """Compatibility alias for the historical ``compute_geometry`` name."""
        return self.model

    def compute_kwargs(self) -> dict[str, Any]:
        """Return kwargs equivalent to the existing ``compute_geometry`` API."""
        return {
            "q": self.q,
            "shat": self.shat,
            "eps": self.eps,
            "ns": self.ns,
            "nkx": self.nkx,
            "nky": self.nky,
            "nvpar": self.nvpar,
            "nmu": self.nmu,
            "vpar_max": self.vpar_max,
            "nperiod": self.nperiod,
            "kxmax": self.kxmax,
            "krhomax": self.krhomax,
            "ikxspace": self.ikxspace,
            "signB": self.signB,
            "Rref": self.Rref,
            "geom_type": self.model,
            **dict(self.model_params),
        }


def geometry_spec_from_compute_kwargs(
    *,
    q: float,
    shat: float,
    eps: float,
    ns: int,
    nkx: int,
    nky: int,
    nvpar: int,
    nmu: int,
    vpar_max: float = 3.0,
    nperiod: int = 1,
    kxmax: float = 0.0,
    krhomax: float = 1.4,
    ikxspace: int = 5,
    signB: float = 1.0,
    Rref: float = 100.0,
    geom_type: str = "circ",
    **model_params: Any,
) -> GeometrySpec:
    """Build a ``GeometrySpec`` from the current direct Python API kwargs."""
    return GeometrySpec(
        model=geom_type,
        q=q,
        shat=shat,
        eps=eps,
        ns=ns,
        nkx=nkx,
        nky=nky,
        nvpar=nvpar,
        nmu=nmu,
        vpar_max=vpar_max,
        nperiod=nperiod,
        kxmax=kxmax,
        krhomax=krhomax,
        ikxspace=ikxspace,
        signB=signB,
        Rref=Rref,
        model_params=model_params,
    )
