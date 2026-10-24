"""Shared TORAX plumbing of the gyaradax transport models.

`GyaradaxTransportModel` owns what does not depend on how the fluxes are produced: the
flux-tube grid, the local GKParams and geometry at each `rho_match` radius, the batching over
radii, the GKW gyroBohm -> TORAX conversion and the diagnostics sink. Subclasses supply
the flux model through `_per_radius` (or `_per_radius_from_profiles`).
"""

import dataclasses
import math
import warnings
from functools import lru_cache
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import pydantic
import torax
from torax import transport
from torax._src import constants
from torax._src.physics import psi_calculations
from torax._src.transport_model import quasilinear_utils

from gyaradax.core.params import GKParams
from gyaradax.geometry import build_topology, compute_continuous_geometry

from . import diagnostics as diag_lib
from . import units

STATIC = torax.JAX_STATIC

_RLT_MIN, _RLT_MAX = 0.0, 30.0
_RLN_MIN, _RLN_MAX = -15.0, 15.0
_Q_MIN, _Q_MAX = 0.5, 10.0
_SHAT_MIN, _SHAT_MAX = -3.0, 6.0
_EPS_MIN, _EPS_MAX = 0.02, 0.5
_BETA_MAX = 0.05
# GKW deuterium-reference electron mass ratio
_ELECTRON_MASS_RATIO = 2.72e-4

_GKPARAMS_DT = 0.005
_VPAR_MAX = 3.0
_RREF = 100.0
_NPERIOD = 1
_GEOM_TYPE = "circ"

_GRID_KEYS = ("nvpar", "nmu", "ns", "nkx", "nky", "ikxspace", "krhomax")
_GRID_FLOOR = {"nkx": 85, "nky": 32}
_GRID_PRESETS = {
    "adiabatic": dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32, ikxspace=5, krhomax=1.4),
    "kinetic": dict(nvpar=32, nmu=12, ns=18, nkx=191, nky=32, ikxspace=16, krhomax=1.5),
}


def ikxspace_for_shat(shat, lo: int = 1, hi: int = 32) -> int:
    """ikxspace = floor(6.53 |shat|), clamped to [lo, hi], as in the stored runs."""
    return int(min(max(int(abs(float(shat)) * 6.53), lo), hi))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport.ComponentRuntimeParams):
    DV_effective: bool = dataclasses.field(metadata={"static": True})
    An_min: float
    DV_effective_smooth_width: float


@lru_cache(maxsize=8)
def topology(nkx: int, nky: int, ikxspace: int, ns: int):
    return build_topology(nkx=nkx, nky=nky, ikxspace=ikxspace, ns=ns)


def resolve_grid(cfg) -> dict:
    """Grid from the explicit config, then `grid_preset` (default: kinetic or adiabatic)."""
    preset = cfg.grid_preset or ("kinetic" if cfg.kinetic else "adiabatic")
    grid = {
        key: getattr(cfg, key) if getattr(cfg, key) is not None else _GRID_PRESETS[preset][key]
        for key in _GRID_KEYS
    }
    if int(grid["nkx"]) % 2 == 0:
        raise ValueError(
            f"gyaradax nkx must be odd (got {grid['nkx']}): the kx grid is symmetric about "
            f"kx=0. Use {int(grid['nkx']) - 1}."
        )
    for key, floor in _GRID_FLOOR.items():
        if int(grid[key]) < floor:
            warnings.warn(
                f"gyaradax: {key}={grid[key]} is below the {floor} resolution floor; raising to "
                f"{floor}.",
                RuntimeWarning,
                stacklevel=2,
            )
            grid[key] = floor
    return grid


def radius_shares(n: int, batching: str) -> Tuple[int, int]:
    per = -(-n // min(jax.local_device_count(), n)) if batching == "devices" else n
    return -(-n // per), per


def over_radii(fn, idx, batching: str = "devices"):
    """Map fn over the radii: in a loop, vmapped, or vmapped in shards over the local devices."""
    if batching == "loop":
        return jax.lax.map(fn, idx)
    n = idx.shape[0]
    n_dev, per = radius_shares(n, batching)
    if n_dev == 1:
        return jax.vmap(fn)(idx)
    # the last radius pads the final shard and its result is dropped
    padded = jnp.concatenate([idx, jnp.repeat(idx[-1:], n_dev * per - n)])
    mesh = jax.make_mesh(
        (n_dev,),
        ("radii",),
        devices=jax.local_devices()[:n_dev],
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    spec = jax.sharding.PartitionSpec("radii")
    sharded = jax.shard_map(jax.vmap(fn), mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
    out = sharded(padded)
    return tuple(o[:n] for o in out)


def face_indices_for_radii(geo, rho_match: Tuple[float, ...]) -> jax.Array:
    rho_face = geo.rho_face_norm
    return jnp.argmin(jnp.abs(rho_face[:, None] - jnp.asarray(rho_match)[None, :]), axis=0)


def gkparams_for_radius(rho_idx, ql_inputs, core_profiles, geo, config) -> GKParams:
    """Local GKParams of one flux tube, clipped into the model's operating box."""
    rlt = jnp.clip(ql_inputs.lref_over_lti[rho_idx], *config.clip_rlt)
    rln = jnp.clip(ql_inputs.lref_over_lne[rho_idx], *config.clip_rln)
    q = jnp.clip(core_profiles.q_face[rho_idx], *config.clip_q)
    shat = jnp.clip(
        psi_calculations.calc_s_rmid(geo, core_profiles.psi)[rho_idx], *config.clip_shat
    )
    eps = jnp.clip(jnp.asarray(geo.epsilon_face)[rho_idx], *config.clip_eps)
    if config.em:
        n_e = core_profiles.n_e.face_value()[rho_idx]
        t_i = core_profiles.T_i.face_value()[rho_idx] * constants.CONSTANTS.keV_to_J
        beta = jnp.clip(
            2.0 * constants.CONSTANTS.mu_0 * n_e * t_i / geo.B_0**2, 0.0, config.beta_max
        )
    else:
        beta = jnp.asarray(0.0)
    if config.kinetic:
        # [ion, electron] with T_ref = T_i; one rln for both (gradient quasineutrality)
        rlt_e = jnp.clip(ql_inputs.lref_over_lte[rho_idx], *config.clip_rlt)
        te_ti = core_profiles.T_e.face_value()[rho_idx] / core_profiles.T_i.face_value()[rho_idx]
        tmp = jnp.stack([jnp.ones_like(te_ti), te_ti])
        mas = jnp.array([1.0, config.electron_mass_ratio])
        species = dict(
            rlt=jnp.stack([rlt, rlt_e]),
            rln=jnp.stack([rln, rln]),
            tmp=tmp,
            mas=mas,
            de=jnp.ones(2),
            signz=jnp.array([1.0, -1.0]),
            vthrat=jnp.sqrt(tmp / mas),
            adiabatic_electrons=False,
        )
    else:
        species = dict(rlt=rlt, rln=rln, adiabatic_electrons=True)
    return GKParams(
        q=q,
        shat=shat,
        eps=eps,
        beta=beta,
        nlapar=config.em,
        non_linear=False,
        dt=config.gkparams_dt,
        backend=config.backend,
        mixed_precision=config.mixed_precision,
        fused_linear=config.fused_linear,
        linear_arith=config.linear_arith,
        **species,
    )


def geometry_at(q, shat, eps, config, topo: Dict[str, Any]) -> Dict[str, Any]:
    return compute_continuous_geometry(
        q=q,
        shat=shat,
        eps=eps,
        ns=config.ns,
        nkx=config.nkx,
        nky=config.nky,
        nvpar=config.nvpar,
        nmu=config.nmu,
        vpar_max=config.vpar_max,
        nperiod=config.nperiod,
        kxmax=0.0,
        krhomax=config.krhomax,
        ikxspace=config.ikxspace,
        signB=1.0,
        Rref=config.rref,
        geom_type=config.geom_type,
        topology=topo,
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxTransportModel(transport.ComponentTransportModel):
    """Grid, geometry and TORAX wiring shared by the gyaradax transport models."""

    rho_match: Tuple[float, ...] = (0.35, 0.55, 0.75, 0.875)
    backend: str = "jax"
    mixed_precision: bool = True
    fused_linear: bool = True
    linear_arith: str = "fp64"
    batching: str = "devices"
    nvpar: int = 32
    nmu: int = 8
    ns: int = 16
    nkx: int = 85
    nky: int = 32
    ikxspace: int = 5
    krhomax: float = 1.4
    em: bool = False
    grid_preset: str = ""
    ikxspace_per_radius: Tuple[int, ...] = ()
    vpar_max: float = _VPAR_MAX
    nperiod: int = _NPERIOD
    geom_type: str = _GEOM_TYPE
    gkparams_dt: float = _GKPARAMS_DT
    rref: float = _RREF
    electron_mass_ratio: float = _ELECTRON_MASS_RATIO
    beta_max: float = _BETA_MAX
    clip_rlt: Tuple[float, float] = (_RLT_MIN, _RLT_MAX)
    clip_rln: Tuple[float, float] = (_RLN_MIN, _RLN_MAX)
    clip_q: Tuple[float, float] = (_Q_MIN, _Q_MAX)
    clip_shat: Tuple[float, float] = (_SHAT_MIN, _SHAT_MAX)
    clip_eps: Tuple[float, float] = (_EPS_MIN, _EPS_MAX)
    kinetic: bool = False
    diagnostics_path: str = ""

    def _make_core_transport(
        self, qi, qe, pfe, ql_inputs, transport_params, geo, core_profiles, two_point_mask
    ) -> transport.TransportCoeffs:
        """TORAX transport coefficients from face-grid fluxes in the TORAX_QL convention."""
        lref_grad = units.gradient_reference_length(geo)
        lref_flux = units.flux_reference_length(units.TORAX_QL, geo)
        pfe_si = pfe * core_profiles.n_e.face_value() * ql_inputs.chiGB / lref_flux
        # chi[GB] = Q[GB] / (Lref/LT), rescaled when the two reference lengths differ
        chi_face_ion = ((lref_grad / lref_flux) * qi) / ql_inputs.lref_over_lti * ql_inputs.chiGB
        chi_face_el = ((lref_grad / lref_flux) * qe) / ql_inputs.lref_over_lte * ql_inputs.chiGB
        if transport_params.DV_effective:
            d_face_el, v_face_el = quasilinear_utils.calculate_dv_effective(
                particle_flux_SI=pfe_si,
                normalized_particle_flux=pfe,
                n_e=core_profiles.n_e,
                geo=geo,
                gradient_reference_length=lref_grad,
                An_min=transport_params.An_min,
                DV_effective_smooth_width=transport_params.DV_effective_smooth_width,
                two_point_mask=two_point_mask,
            )
        else:
            # D tied to chi_e, V set so the total particle flux matches
            chex.assert_rank(pfe, 1)
            d_face_el = chi_face_el
            v_face_el = (
                pfe_si / core_profiles.n_e.face_value()
                - ql_inputs.lref_over_lne
                * d_face_el
                / lref_grad
                * geo.g1_over_vpr2_face
                * geo.rho_b**2
            ) / (geo.g0_over_vpr_face * geo.rho_b)
        return transport.TransportCoeffs(
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    @classmethod
    def _base_kwargs(cls, cfg) -> dict:
        """Constructor arguments of the fields defined on this base, from a config."""
        fis = getattr(cfg, "fast_ion_stabilization", None)
        if fis is not None and float(fis.get_value(0.0)) != 0.0:
            raise NotImplementedError(f"{cls.__name__} does not implement fast_ion_stabilization.")
        grid = resolve_grid(cfg)
        # warm the cache outside jit, so the topology arrays never become tracers
        topology(int(grid["nkx"]), int(grid["nky"]), int(grid["ikxspace"]), int(grid["ns"]))
        optional = (
            "vpar_max",
            "nperiod",
            "geom_type",
            "gkparams_dt",
            "rref",
            "electron_mass_ratio",
            "beta_max",
            "clip_rlt",
            "clip_rln",
            "clip_q",
            "clip_shat",
            "clip_eps",
        )
        return dict(
            rho_match=tuple(cfg.rho_match),
            backend=cfg.backend,
            mixed_precision=cfg.mixed_precision,
            fused_linear=cfg.fused_linear,
            linear_arith=cfg.linear_arith,
            batching=cfg.batching,
            nvpar=int(grid["nvpar"]),
            nmu=int(grid["nmu"]),
            ns=int(grid["ns"]),
            nkx=int(grid["nkx"]),
            nky=int(grid["nky"]),
            ikxspace=int(grid["ikxspace"]),
            krhomax=float(grid["krhomax"]),
            em=cfg.em,
            grid_preset=cfg.grid_preset or "",
            **{k: getattr(cfg, k) for k in optional if getattr(cfg, k, None) is not None},
            ikxspace_per_radius=tuple(cfg.ikxspace_per_radius or ()),
            kinetic=cfg.kinetic,
            diagnostics_path=cfg.diagnostics_path or "",
        )

    @classmethod
    def from_config(cls, cfg):
        return cls(**cls._base_kwargs(cfg))

    @property
    def topology(self):
        return topology(self.nkx, self.nky, self.ikxspace, self.ns)

    def topology_at(self, k: int):
        """Topology of the k-th radius: ikxspace sizes the connectivity, so it is static."""
        if not self.ikxspace_per_radius:
            return self.topology
        return topology(self.nkx, self.nky, int(self.ikxspace_per_radius[k]), self.ns)

    @property
    def sink(self) -> Any:
        return diag_lib.get_sink(
            self.diagnostics_path, math.prod(radius_shares(len(self.rho_match), self.batching))
        )

    def _per_radius_from_profiles(self, rho_idx, ql_inputs, core_profiles, geo, topo=None):
        params = gkparams_for_radius(rho_idx, ql_inputs, core_profiles, geo, self)
        geom = geometry_at(params.q, params.shat, params.eps, self, topo or self.topology)
        q_i, q_e, pfe, extra = self._per_radius(params, geom)
        diag_lib.record(
            self.sink,
            {"model": type(self).__name__},
            {
                "rho_idx": rho_idx,
                "rho_face": jnp.asarray(geo.rho_face_norm)[rho_idx],
                "rlt": params.rlt,
                "rln": params.rln,
                "q": params.q,
                "shat": params.shat,
                "eps": params.eps,
                "beta": params.beta,
                "qi_gb": q_i,
                "qe_gb": q_e,
                "pfe_gb": pfe,
                **extra,
            },
        )
        return q_i, q_e, pfe

    def _per_radius(self, params: GKParams, geom: Dict[str, Any]):
        """(qi, qe, pfe) in GKW gyroBohm units and a dict of per-call diagnostics."""
        raise NotImplementedError

    def call_implementation(
        self, transport_runtime_params, runtime_params, geo, core_profiles, two_point_mask
    ) -> transport.TransportCoeffs:
        del runtime_params
        ql_inputs = units.build_quasilinear_inputs(core_profiles, geo)
        match_idx = face_indices_for_radii(geo, self.rho_match)
        sink = self.sink

        def per_radius(idx, topo=None):
            idx = diag_lib.mark_begin(sink, idx)
            return self._per_radius_from_profiles(idx, ql_inputs, core_profiles, geo, topo)

        if self.ikxspace_per_radius:
            # one topology per radius, so the radii cannot share a vmap
            out = [
                per_radius(match_idx[k], self.topology_at(k)) for k in range(len(self.rho_match))
            ]
            qi_m, qe_m, pfe_m = (jnp.stack([o[i] for o in out]) for i in range(3))
        else:
            qi_m, qe_m, pfe_m = over_radii(per_radius, match_idx, self.batching)

        rho_face = geo.rho_face_norm
        rho_match = jnp.asarray(self.rho_match)
        gb = units.gb_flux_conversion_factor(units.GYARADAX, units.TORAX_QL, geo)
        return self._make_core_transport(
            qi=jnp.interp(rho_face, rho_match, qi_m) * gb,
            qe=jnp.interp(rho_face, rho_match, qe_m) * gb,
            pfe=jnp.interp(rho_face, rho_match, pfe_m) * gb,
            ql_inputs=ql_inputs,
            transport_params=transport_runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            two_point_mask=two_point_mask,
        )


class GyaradaxConfig(transport.ComponentTransportBase):
    """Config fields shared by every gyaradax transport model (see docs/TORAX.md)."""

    rho_match: Annotated[Tuple[float, ...], STATIC] = (0.35, 0.55, 0.75, 0.875)
    DV_effective: Annotated[bool, STATIC] = True
    An_min: Annotated[float, STATIC] = 0.05
    DV_effective_smooth_width: pydantic.NonNegativeFloat = 0.01
    diagnostics_path: Annotated[Optional[str], STATIC] = None

    def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
        base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
        return RuntimeParams(
            DV_effective=self.DV_effective,
            An_min=self.An_min,
            DV_effective_smooth_width=self.DV_effective_smooth_width,
            **base_kwargs,
        )


class SolverConfig(GyaradaxConfig):
    """Adds the flux-tube grid and linear-solve settings, for models that run gyaradax."""

    backend: Annotated[str, STATIC] = "jax"
    mixed_precision: Annotated[bool, STATIC] = True
    fused_linear: Annotated[bool, STATIC] = True
    linear_arith: Annotated[Literal["fp64", "ds"], STATIC] = "fp64"
    batching: Annotated[Literal["devices", "vmap", "loop"], STATIC] = "devices"
    nvpar: Annotated[Optional[int], STATIC] = None
    nmu: Annotated[Optional[int], STATIC] = None
    ns: Annotated[Optional[int], STATIC] = None
    nkx: Annotated[Optional[int], STATIC] = None
    nky: Annotated[Optional[int], STATIC] = None
    ikxspace: Annotated[Optional[int], STATIC] = None
    krhomax: Annotated[Optional[float], STATIC] = None
    em: Annotated[bool, STATIC] = False
    grid_preset: Annotated[Optional[str], STATIC] = None
    ikxspace_per_radius: Annotated[Optional[Tuple[int, ...]], STATIC] = None
    vpar_max: Annotated[Optional[float], STATIC] = None
    nperiod: Annotated[Optional[int], STATIC] = None
    geom_type: Annotated[Optional[str], STATIC] = None
    gkparams_dt: Annotated[Optional[float], STATIC] = None
    rref: Annotated[Optional[float], STATIC] = None
    electron_mass_ratio: Annotated[Optional[float], STATIC] = None
    beta_max: Annotated[Optional[float], STATIC] = None
    clip_rlt: Annotated[Optional[Tuple[float, float]], STATIC] = None
    clip_rln: Annotated[Optional[Tuple[float, float]], STATIC] = None
    clip_q: Annotated[Optional[Tuple[float, float]], STATIC] = None
    clip_shat: Annotated[Optional[Tuple[float, float]], STATIC] = None
    clip_eps: Annotated[Optional[Tuple[float, float]], STATIC] = None
    kinetic: Annotated[bool, STATIC] = False
