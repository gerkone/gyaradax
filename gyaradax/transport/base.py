"""Shared TORAX plumbing of the gyaradax transport models.

`GyaradaxTransportModel` owns what does not depend on how the fluxes are produced: the
flux-tube grid, the local GKParams and geometry at each `rho_match` radius, the batching over
radii and the diagnostics sink. Subclasses supply the flux model through `_per_radius`. TORAX's
QuaLiKiz base class turns the gyroBohm fluxes into transport coefficients.
"""

import dataclasses
import logging
import math
import warnings
from functools import lru_cache
from typing import Annotated, Any, Dict, Literal, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax import constants, torax_pydantic, transport
from torax._src.physics import psi_calculations
from torax._src.transport_model import qualikiz_based_transport_model as qualikiz
from torax._src.transport_model import quasilinear_utils
from torax._src.transport_model import transport_state

from gyaradax.core.params import GKParams
from gyaradax.geometry import build_topology, compute_continuous_geometry

from . import diagnostics as diag_lib

STATIC = torax_pydantic.JAX_STATIC
_LOG = logging.getLogger("gyaradax.transport")

_RLT_MIN, _RLT_MAX = 0.0, 30.0
_RLN_MIN, _RLN_MAX = -15.0, 15.0
_Q_MIN, _Q_MAX = 0.5, 10.0
_SHAT_MIN, _SHAT_MAX = -3.0, 6.0
_EPS_MIN, _EPS_MAX = 0.02, 0.5
_BETA_MAX = 0.05
# GKW deuterium-reference electron mass ratio
_ELECTRON_MASS_RATIO = 2.72e-4
# unit of gyaradax's coll_nref [m^-3]
_COLL_NREF_UNIT = 1e19

_GKPARAMS_DT = {"adiabatic": 0.005, "kinetic": 0.004}
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
# ikxspace = floor(slope |shat|) per radius
_IKXSPACE_PER_SHAT = 6.53
# (sqrt 2)^3 from gyaradax's v_th = sqrt(2T/m) against sqrt(T/m)
_VTH_SQRT2_FLUX_FACTOR = 2.0 * math.sqrt(2.0)
# ikxspace values the shat rule snaps to; each costs one prebuilt topology
_IKXSPACE_CHOICES = {"adiabatic": (), "kinetic": (1, 2, 3, 4, 6, 8, 12, 16)}


def ikxspace_for_shat(shat, choices, slope: float = _IKXSPACE_PER_SHAT):
    """Index into ``choices`` (ascending) of the largest one <= floor(slope |shat|); traceable."""
    target = jnp.floor(slope * jnp.abs(shat))
    n_le = jnp.sum(jnp.asarray(choices) <= target)
    return jnp.clip(n_le - 1, 0, len(choices) - 1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport.ComponentRuntimeParams):
    DV_effective: bool = jax.tree.static()
    An_min: float
    DV_effective_smooth_width: float


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WarmStartState(transport_state.ComponentTransportState):
    """Final distribution per radius of the previous transport step, with the ikxspace index it was run at (-1: none)."""

    df: jax.Array
    ikxspace_index: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QuasilinearInputs:
    """Face-grid chiGB (L_ref = a_minor) and R/L gradients, as TORAX's QuaLiKiz models use them."""

    chiGB: jax.Array
    lref_over_lti: jax.Array
    lref_over_lte: jax.Array
    lref_over_lne: jax.Array


def quasilinear_inputs(core_profiles, geo, two_point_mask=None) -> QuasilinearInputs:
    grads = quasilinear_utils.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=geo.r_mid,
        radial_face_coordinate=geo.r_mid_face,
        reference_length=geo.R_major,
        two_point_mask=two_point_mask,
    )
    chi_gb = quasilinear_utils.calculate_chiGB(
        reference_temperature=core_profiles.kinetic.T_i.face_value(),
        reference_magnetic_field=geo.B_0,
        reference_mass=core_profiles.ions.A_i,
        reference_length=geo.a_minor,
    )
    return QuasilinearInputs(chi_gb, grads.lref_over_lti, grads.lref_over_lte, grads.lref_over_lne)


def gkw_to_qualikiz_flux(geo):
    """Factor from GKW gyroBohm fluxes (R, v_th with sqrt 2) to TORAX's QuaLiKiz ones (a_minor, T_i)."""
    return _VTH_SQRT2_FLUX_FACTOR * (geo.a_minor / geo.R_major) ** 2


@lru_cache(maxsize=8)
def topology(nkx: int, nky: int, ikxspace: int, ns: int):
    return build_topology(nkx=nkx, nky=nky, ikxspace=ikxspace, ns=ns)


@lru_cache(maxsize=4)
def topology_bank(nkx: int, nky: int, ns: int, choices: Tuple[int, ...]):
    """Topologies of every ikxspace in ``choices``, stacked along a leading axis."""
    topos = [topology(nkx, nky, int(k), ns) for k in choices]

    def merge(values):
        # entries equal for every choice stay concrete; the rest are stacked
        if all(np.array_equal(np.asarray(v), np.asarray(values[0])) for v in values):
            return values[0]
        if not hasattr(values[0], "shape"):
            raise ValueError("a scalar topology entry differs between ikxspace values")
        return _Stacked(jnp.stack(values))

    bank = {}
    for key, v in topos[0].items():
        if key == "flipped":
            bank[key] = {k: merge([t[key][k] for t in topos]) for k in v}
        else:
            bank[key] = merge([t[key] for t in topos])
    return bank


@dataclasses.dataclass(frozen=True)
class _Stacked:
    values: Any


def select_topology(bank, i):
    """The topology at (traced) index ``i`` of a :func:`topology_bank`."""
    def pick(v):
        return v.values[i] if isinstance(v, _Stacked) else v

    return {k: ({kk: pick(vv) for kk, vv in v.items()} if k == "flipped" else pick(v))
            for k, v in bank.items()}


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
    grid = {k: int(v) if k != "krhomax" else float(v) for k, v in grid.items()}
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


def over_radii(fn, args, batching: str = "devices"):
    """Map fn over the leading (radius) axis of the pytree args: in a loop, vmapped, or vmapped in shards over the local devices."""
    if batching == "loop":
        return jax.lax.map(fn, args)
    n = jax.tree.leaves(args)[0].shape[0]
    n_dev, per = radius_shares(n, batching)
    if n_dev == 1:
        return jax.vmap(fn)(args)
    padded = args
    if n_dev * per > n:
        # the last radius pads the final shard and its result is dropped
        padded = jax.tree.map(lambda a: jnp.concatenate([a, jnp.repeat(a[-1:], n_dev * per - n, axis=0)]), args)
    mesh = jax.make_mesh(
        (n_dev,),
        ("radii",),
        devices=jax.local_devices()[:n_dev],
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    spec = jax.sharding.PartitionSpec("radii")
    sharded = jax.shard_map(jax.vmap(fn), mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
    return jax.tree.map(lambda o: o[:n], sharded(padded))


def _report_divergence(finite, model, rho, q, shat, eps, rlt, rln):
    if not bool(finite):
        _LOG.error(
            "%s: the solve at rho = %.3f is not finite (q = %.3f, shat = %.3f, eps = %.3f, R/LT = %s, R/Ln = %s); "
            "the NaN is passed to TORAX, which stops the run",
            model, float(rho), float(q), float(shat), float(eps), np.round(rlt, 3).tolist(), np.round(rln, 3).tolist(),
        )


def face_indices_for_radii(geo, rho_match: Tuple[float, ...]) -> jax.Array:
    rho_face = geo.rho_face_norm
    return jnp.argmin(jnp.abs(rho_face[:, None] - jnp.asarray(rho_match)[None, :]), axis=0)


def face_profiles(core_profiles, geo) -> Dict[str, jax.Array]:
    """Face-grid profiles the flux tubes need, computed once outside the loop over radii.

    TORAX caches derived grids on its profile objects (cached properties); computing them
    inside a scan or shard_map over radii would leak tracers into that cache.
    """
    kinetic = core_profiles.kinetic
    return dict(
        q=core_profiles.poloidal_flux.q_face,
        shat=psi_calculations.calc_s_rmid(geo, core_profiles.poloidal_flux.psi),
        eps=jnp.asarray(geo.epsilon_face),
        n_e=kinetic.n_e.face_value(),
        T_i=kinetic.T_i.face_value(),
        T_e=kinetic.T_e.face_value(),
    )


def gkparams_for_radius(rho_idx, ql_inputs, faces, geo, config) -> GKParams:
    """Local GKParams of one flux tube, clipped into the model's operating box."""
    rlt = jnp.clip(ql_inputs.lref_over_lti[rho_idx], *config.clip_rlt)
    rln = jnp.clip(ql_inputs.lref_over_lne[rho_idx], *config.clip_rln)
    q = jnp.clip(faces["q"][rho_idx], *config.clip_q)
    shat = jnp.clip(faces["shat"][rho_idx], *config.clip_shat)
    eps = jnp.clip(faces["eps"][rho_idx], *config.clip_eps)
    if config.em:
        n_e = faces["n_e"][rho_idx]
        t_i = faces["T_i"][rho_idx] * constants.CONSTANTS.keV_to_J
        beta = jnp.clip(
            2.0 * constants.CONSTANTS.mu_0 * n_e * t_i / geo.B_0**2, 0.0, config.beta_max
        )
    else:
        beta = jnp.asarray(0.0)
    if config.kinetic:
        # [ion, electron] with T_ref = T_i; one rln for both (gradient quasineutrality)
        rlt_e = jnp.clip(ql_inputs.lref_over_lte[rho_idx], *config.clip_rlt)
        te_ti = faces["T_e"][rho_idx] / faces["T_i"][rho_idx]
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
    coll = {}
    if config.collisions:
        # coulomb-log frequencies from the local n_e, T_i (T_ref) and the major radius (R_ref)
        coll = dict(
            collisions=True,
            coll_rref=jnp.asarray(geo.R_major, dtype=jnp.float64),
            coll_nref=faces["n_e"][rho_idx] / _COLL_NREF_UNIT,
            coll_tref=faces["T_i"][rho_idx],
        )
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
        **coll,
    )


def geometry_at(q, shat, eps, config, topo: Dict[str, Any], ikxspace=None) -> Dict[str, Any]:
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
        ikxspace=config.ikxspace if ikxspace is None else ikxspace,
        signB=1.0,
        Rref=config.rref,
        geom_type=config.geom_type,
        topology=topo,
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxTransportModel(qualikiz.QualikizBasedTransportModel):
    """Grid, geometry and TORAX wiring shared by the gyaradax transport models."""

    rho_match: Tuple[float, ...]
    backend: str
    mixed_precision: bool
    fused_linear: bool
    linear_arith: str
    batching: str
    nvpar: int
    nmu: int
    ns: int
    nkx: int
    nky: int
    ikxspace: int
    krhomax: float
    em: bool
    ikxspace_choices: Tuple[int, ...]
    ikxspace_per_shat: float
    collisions: bool
    vpar_max: float
    nperiod: int
    geom_type: str
    gkparams_dt: float
    rref: float
    electron_mass_ratio: float
    beta_max: float
    clip_rlt: Tuple[float, float]
    clip_rln: Tuple[float, float]
    clip_q: Tuple[float, float]
    clip_shat: Tuple[float, float]
    clip_eps: Tuple[float, float]
    kinetic: bool
    diagnostics_path: str

    @classmethod
    def from_config(cls, cfg, **overrides):
        """The model from a config: shared fields by name, then the resolved grid and overrides."""
        if float(cfg.fast_ion_stabilization.get_value(0.0)) != 0.0:
            raise NotImplementedError(f"{cls.__name__} does not implement fast_ion_stabilization.")
        grid = resolve_grid(cfg)
        preset = cfg.grid_preset or ("kinetic" if cfg.kinetic else "adiabatic")
        choices = cfg.ikxspace_choices if cfg.ikxspace_choices is not None else _IKXSPACE_CHOICES[preset]
        choices = tuple(sorted(int(k) for k in choices))
        # warm the caches outside jit, so the topology arrays never become tracers
        topology(int(grid["nkx"]), int(grid["nky"]), int(grid["ikxspace"]), int(grid["ns"]))
        if choices:
            topology_bank(int(grid["nkx"]), int(grid["nky"]), int(grid["ns"]), choices)
        fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cls) if hasattr(cfg, f.name)}
        fields.update(grid, ikxspace_choices=choices, diagnostics_path=cfg.diagnostics_path or "")
        fields["gkparams_dt"] = cfg.gkparams_dt or _GKPARAMS_DT["kinetic" if cfg.kinetic else "adiabatic"]
        return cls(**{**fields, **overrides})

    @property
    def topology(self):
        return topology(self.nkx, self.nky, self.ikxspace, self.ns)

    @property
    def sink(self) -> Any:
        return diag_lib.get_sink(
            self.diagnostics_path, math.prod(radius_shares(len(self.rho_match), self.batching))
        )

    def _warm_df_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the distribution carried between transport steps, or None for a stateless model."""
        return None

    def initial_state(self, runtime_params):
        shape = self._warm_df_shape()
        if shape is None:
            return super().initial_state(runtime_params)
        n = len(self.rho_match)
        return WarmStartState(
            df=jnp.zeros((n, *shape), jnp.complex64), ikxspace_index=jnp.full((n,), -1, jnp.int32)
        )

    def _per_radius_from_profiles(self, rho_idx, ql_inputs, faces, geo, warm=None):
        """Fluxes at one radius; with a (df, ikxspace_index) warm start also the new one."""
        params = gkparams_for_radius(rho_idx, ql_inputs, faces, geo, self)
        topo, ikxspace, i = self.topology, None, jnp.asarray(0, jnp.int32)
        if self.ikxspace_choices:
            i = ikxspace_for_shat(params.shat, self.ikxspace_choices, self.ikxspace_per_shat).astype(jnp.int32)
            topo = select_topology(
                topology_bank(self.nkx, self.nky, self.ns, self.ikxspace_choices), i
            )
            ikxspace = jnp.asarray(self.ikxspace_choices)[i]
        geom = geometry_at(params.q, params.shat, params.eps, self, topo, ikxspace)
        if warm is None:
            q_i, q_e, pfe, extra = self._per_radius(params, geom)
        else:
            # a distribution on another kx connectivity is no start
            df_prev, i_prev = warm
            q_i, q_e, pfe, extra, df_new = self._per_radius(params, geom, df_prev, i_prev == i)
            extra = {**extra, "warm": (i_prev == i).astype(jnp.int32)}
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
                "ikxspace": self.ikxspace if ikxspace is None else ikxspace,
                "coll_nref": params.coll_nref,
                "coll_tref": params.coll_tref,
                "qi_gb": q_i,
                "qe_gb": q_e,
                "pfe_gb": pfe,
                **extra,
            },
        )
        finite = jnp.isfinite(q_i) & jnp.isfinite(q_e) & jnp.isfinite(pfe)
        if warm is not None:
            finite = finite & jnp.all(jnp.isfinite(df_new))
        jax.debug.callback(
            _report_divergence, finite, type(self).__name__, jnp.asarray(geo.rho_face_norm)[rho_idx],
            params.q, params.shat, params.eps, jnp.atleast_1d(params.rlt), jnp.atleast_1d(params.rln),
        )
        if warm is None:
            return q_i, q_e, pfe
        return q_i, q_e, pfe, (df_new.astype(jnp.complex64), i)

    def _per_radius(self, params: GKParams, geom: Dict[str, Any], df_prev=None, warm=None):
        """(qi, qe, pfe) in GKW gyroBohm units and a dict of per-call diagnostics.

        Models with a warm start (`_warm_df_shape`) get the previous distribution and whether to use it,
        and also return the final distribution.
        """
        raise NotImplementedError

    def call_implementation(
        self, transport_runtime_params, runtime_params, geo, core_profiles, two_point_mask, component_state
    ):
        del runtime_params
        ql_inputs = quasilinear_inputs(core_profiles, geo, two_point_mask)
        match_idx = face_indices_for_radii(geo, self.rho_match)
        faces = face_profiles(core_profiles, geo)
        sink = self.sink

        stateful = isinstance(component_state, WarmStartState)

        def per_radius(args):
            idx = diag_lib.mark_begin(sink, args[0])
            warm = args[1:] if stateful else None
            return self._per_radius_from_profiles(idx, ql_inputs, faces, geo, warm)

        args = (match_idx, component_state.df, component_state.ikxspace_index) if stateful else (match_idx,)
        out = over_radii(per_radius, args, self.batching)
        fluxes = out[:3]
        if stateful:
            component_state = WarmStartState(df=out[3][0], ikxspace_index=out[3][1])
        rho_tubes = geo.rho_face_norm[match_idx]
        qi, qe, pfe = (jnp.interp(geo.rho_face_norm, rho_tubes, f) * gkw_to_qualikiz_flux(geo) for f in fluxes)
        coeffs = self._make_transport_coeffs(
            qi, qe, pfe, ql_inputs, transport_runtime_params, geo, core_profiles, two_point_mask
        )
        return coeffs, component_state


class GyaradaxConfig(transport.ComponentTransportConfigBase):
    """Config fields shared by every gyaradax transport model (see docs/applications/torax.md)."""

    rho_match: Annotated[Tuple[float, ...], STATIC] = (0.35, 0.55, 0.75, 0.875)
    DV_effective: Annotated[bool, STATIC] = True
    An_min: pydantic.PositiveFloat = 0.05
    DV_effective_smooth_width: pydantic.NonNegativeFloat = 0.01
    diagnostics_path: Annotated[Optional[str], STATIC] = None

    def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
        return RuntimeParams(
            DV_effective=self.DV_effective,
            An_min=self.An_min,
            DV_effective_smooth_width=self.DV_effective_smooth_width,
            **vars(super().build_runtime_params(t)),
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
    ikxspace_choices: Annotated[Optional[Tuple[int, ...]], STATIC] = None
    ikxspace_per_shat: Annotated[float, STATIC] = _IKXSPACE_PER_SHAT
    collisions: Annotated[bool, STATIC] = True
    vpar_max: Annotated[float, STATIC] = _VPAR_MAX
    nperiod: Annotated[int, STATIC] = _NPERIOD
    geom_type: Annotated[str, STATIC] = _GEOM_TYPE
    gkparams_dt: Annotated[Optional[float], STATIC] = None
    rref: Annotated[float, STATIC] = _RREF
    electron_mass_ratio: Annotated[float, STATIC] = _ELECTRON_MASS_RATIO
    beta_max: Annotated[float, STATIC] = _BETA_MAX
    clip_rlt: Annotated[Tuple[float, float], STATIC] = (_RLT_MIN, _RLT_MAX)
    clip_rln: Annotated[Tuple[float, float], STATIC] = (_RLN_MIN, _RLN_MAX)
    clip_q: Annotated[Tuple[float, float], STATIC] = (_Q_MIN, _Q_MAX)
    clip_shat: Annotated[Tuple[float, float], STATIC] = (_SHAT_MIN, _SHAT_MAX)
    clip_eps: Annotated[Tuple[float, float], STATIC] = (_EPS_MIN, _EPS_MAX)
    kinetic: Annotated[bool, STATIC] = False
