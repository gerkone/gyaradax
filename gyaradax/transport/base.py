"""Shared TORAX plumbing of the gyaradax transport models.

`GyaradaxTransportModel` owns what does not depend on how the fluxes are produced: the
flux-tube grid, the local GKParams and geometry at each `rho_match` radius, one solve per radius
on its own device, and the jsonl diagnostics. Subclasses supply the flux model through `_per_radius`. TORAX's
QuaLiKiz base class turns the gyroBohm fluxes into transport coefficients.
"""

import dataclasses
import functools
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import chex
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pydantic
import torax
from torax import constants, torax_pydantic, transport
from torax._src.physics import psi_calculations
from torax._src.transport_model import qualikiz_based_transport_model as qualikiz
from torax._src.transport_model import quasilinear_utils

from gyaradax.core.params import GKParams
from gyaradax.geometry import build_topology, compute_continuous_geometry

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

_GRID_KEYS = ("nvpar", "nmu", "ns", "nkx", "nky", "ikxspace", "krhomax")
_GRID_PRESETS = {
    "adiabatic": dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32, ikxspace=5, krhomax=1.4),
    "kinetic": dict(nvpar=32, nmu=12, ns=18, nkx=191, nky=32, ikxspace=16, krhomax=1.5),
}
# ikxspace = floor(slope |shat|) per radius
_IKXSPACE_PER_SHAT = 6.53
# (sqrt 2)^3 from gyaradax's v_th = sqrt(2T/m) against sqrt(T/m)
_VTH_SQRT2_FLUX_FACTOR = 2.0 * math.sqrt(2.0)
# ikxspace values the shat rule snaps to
_IKXSPACE_CHOICES = {"adiabatic": (), "kinetic": (1, 2, 3, 4, 6, 8, 12, 16)}


def ikxspace_for_shat(shat: float, choices: Tuple[int, ...], slope: float = _IKXSPACE_PER_SHAT) -> int:
    """The largest of ``choices`` (ascending) <= floor(slope |shat|), else the smallest."""
    target = math.floor(slope * abs(float(shat)))
    return max((k for k in choices if k <= target), default=choices[0])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport.ComponentRuntimeParams):
    DV_effective: bool = jax.tree.static()
    An_min: float
    DV_effective_smooth_width: float


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QuasilinearInputs:
    """Face-grid chiGB (L_ref = a_minor) and R/L gradients, as TORAX's QuaLiKiz models use them."""

    chiGB: jax.Array
    lref_over_lti: jax.Array
    lref_over_lte: jax.Array
    lref_over_lne: jax.Array


def quasilinear_inputs(
    core_profiles: torax.CoreProfiles, geo: torax.Geometry, two_point_mask: Optional[jax.Array] = None
) -> QuasilinearInputs:
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


def gkw_to_qualikiz_flux(geo: torax.Geometry) -> jax.Array:
    """Factor from GKW gyroBohm fluxes (R, v_th with sqrt 2) to TORAX's QuaLiKiz ones (a_minor, T_i)."""
    return _VTH_SQRT2_FLUX_FACTOR * (geo.a_minor / geo.R_major) ** 2


def resolve_grid(cfg: "GyaradaxConfig") -> Dict[str, Any]:
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
    return {k: int(v) if k != "krhomax" else float(v) for k, v in grid.items()}


def local_inputs(
    core_profiles: torax.CoreProfiles, geo: torax.Geometry, qli: QuasilinearInputs, rho_match: Tuple[float, ...]
) -> Dict[str, jax.Array]:
    """Per-radius local inputs at the faces nearest ``rho_match``, clipped into the model's operating box."""
    idx = jnp.argmin(jnp.abs(geo.rho_face_norm[:, None] - jnp.asarray(rho_match)[None, :]), axis=0)
    kinetic = core_profiles.kinetic
    n_e, t_i = kinetic.n_e.face_value()[idx], kinetic.T_i.face_value()[idx]
    shat = psi_calculations.calc_s_rmid(geo, core_profiles.poloidal_flux.psi)
    beta = 2.0 * constants.CONSTANTS.mu_0 * n_e * t_i * constants.CONSTANTS.keV_to_J / geo.B_0**2
    return dict(
        rho_face=geo.rho_face_norm[idx],
        q=jnp.clip(core_profiles.poloidal_flux.q_face[idx], _Q_MIN, _Q_MAX),
        shat=jnp.clip(shat[idx], _SHAT_MIN, _SHAT_MAX),
        eps=jnp.clip(jnp.asarray(geo.epsilon_face)[idx], _EPS_MIN, _EPS_MAX),
        rlt_i=jnp.clip(qli.lref_over_lti[idx], _RLT_MIN, _RLT_MAX),
        rlt_e=jnp.clip(qli.lref_over_lte[idx], _RLT_MIN, _RLT_MAX),
        rln=jnp.clip(qli.lref_over_lne[idx], _RLN_MIN, _RLN_MAX),
        te_ti=kinetic.T_e.face_value()[idx] / t_i,
        n_e=n_e,
        t_i=t_i,
        beta=jnp.clip(beta, 0.0, _BETA_MAX),
        r_major=jnp.broadcast_to(jnp.asarray(geo.R_major, jnp.float64), idx.shape),
    )


def _write_rows(path: str, model: str, t: float, rows: List[Dict[str, Any]]) -> None:
    call = _CALLS.get(path, -1) + 1
    _CALLS[path] = call
    if call == 0:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w" if call == 0 else "a") as f:
        for j, row in enumerate(rows):
            native = {k: np.asarray(v).tolist() for k, v in row.items()}
            f.write(json.dumps({"call": call, "radius": j, "model": model, "t": t, **native}) + "\n")


def _report_divergence(model: str, rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        if r["finite"]:
            continue
        _LOG.error(
            "%s: the solve at rho = %.3f is not finite (q = %.3f, shat = %.3f, eps = %.3f, R/LT = %s, R/Ln = %s); "
            "the NaN is passed to TORAX, which stops the run",
            model, float(r["rho_face"]), float(r["q"]), float(r["shat"]), float(r["eps"]),
            np.round(np.atleast_1d(r["rlt"]), 3).tolist(), np.round(np.atleast_1d(r["rln"]), 3).tolist(),
        )


# transport calls written so far per diagnostics path
_CALLS: Dict[str, int] = {}
# final distribution and ikxspace per (model, rho_match slot), kept on the radius's device between calls
_WARM: Dict[Tuple[int, int], Tuple[jax.Array, int]] = {}


@functools.lru_cache(maxsize=32)
def _compiled(model: "GyaradaxTransportModel", ikxspace: int):
    topo = build_topology(nkx=model.nkx, nky=model.nky, ikxspace=ikxspace, ns=model.ns)
    return jax.jit(functools.partial(model._solve, ikxspace=ikxspace, topo=topo))


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxTransportModel(qualikiz.QualikizBasedTransportModel):
    """Grid, geometry and TORAX wiring shared by the gyaradax transport models."""

    rho_match: Tuple[float, ...]
    backend: str
    mixed_precision: bool
    fused_linear: bool
    linear_arith: str
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
    gkparams_dt: float
    kinetic: bool
    diagnostics_path: str

    @classmethod
    def from_config(cls, cfg: "GyaradaxConfig", **overrides: Any) -> "GyaradaxTransportModel":
        """The model from a config: shared fields by name, then the resolved grid and overrides."""
        if float(cfg.fast_ion_stabilization.get_value(0.0)) != 0.0:
            raise NotImplementedError(f"{cls.__name__} does not implement fast_ion_stabilization.")
        grid = resolve_grid(cfg)
        preset = cfg.grid_preset or ("kinetic" if cfg.kinetic else "adiabatic")
        choices = cfg.ikxspace_choices if cfg.ikxspace_choices is not None else _IKXSPACE_CHOICES[preset]
        choices = tuple(sorted(int(k) for k in choices))
        fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cls) if hasattr(cfg, f.name)}
        fields.update(grid, ikxspace_choices=choices, diagnostics_path=cfg.diagnostics_path or "")
        fields["gkparams_dt"] = cfg.gkparams_dt or _GKPARAMS_DT["kinetic" if cfg.kinetic else "adiabatic"]
        return cls(**{**fields, **overrides})

    def _warm_df_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the distribution carried between transport steps, or None for a stateless model."""
        return None

    def gkparams(self, loc: Dict[str, Any]) -> GKParams:
        """GKParams of one flux tube from its entries of :func:`local_inputs`."""
        if self.kinetic:
            # [ion, electron] with T_ref = T_i; one rln for both (gradient quasineutrality)
            tmp = jnp.stack([jnp.ones_like(loc["te_ti"]), loc["te_ti"]])
            mas = jnp.array([1.0, _ELECTRON_MASS_RATIO])
            species = dict(
                rlt=jnp.stack([loc["rlt_i"], loc["rlt_e"]]),
                rln=jnp.stack([loc["rln"], loc["rln"]]),
                tmp=tmp,
                mas=mas,
                de=jnp.ones(2),
                signz=jnp.array([1.0, -1.0]),
                vthrat=jnp.sqrt(tmp / mas),
                adiabatic_electrons=False,
            )
        else:
            species = dict(rlt=loc["rlt_i"], rln=loc["rln"], adiabatic_electrons=True)
        coll = {}
        if self.collisions:
            coll = dict(
                collisions=True,
                coll_rref=loc["r_major"],
                coll_nref=loc["n_e"] / _COLL_NREF_UNIT,
                coll_tref=loc["t_i"],
            )
        return GKParams(
            q=loc["q"],
            shat=loc["shat"],
            eps=loc["eps"],
            beta=loc["beta"] if self.em else jnp.asarray(0.0),
            nlapar=self.em,
            non_linear=False,
            dt=self.gkparams_dt,
            backend=self.backend,
            mixed_precision=self.mixed_precision,
            fused_linear=self.fused_linear,
            linear_arith=self.linear_arith,
            **species,
            **coll,
        )

    def ikxspace_at(self, shat: float) -> int:
        return ikxspace_for_shat(shat, self.ikxspace_choices, self.ikxspace_per_shat) if self.ikxspace_choices else self.ikxspace

    def geometry(self, params: GKParams, ikxspace: int, topo: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if topo is None:
            topo = build_topology(nkx=self.nkx, nky=self.nky, ikxspace=ikxspace, ns=self.ns)
        return compute_continuous_geometry(
            q=params.q, shat=params.shat, eps=params.eps, ns=self.ns, nkx=self.nkx, nky=self.nky,
            nvpar=self.nvpar, nmu=self.nmu, vpar_max=self.vpar_max, krhomax=self.krhomax,
            ikxspace=ikxspace, topology=topo,
        )

    def _per_radius(
        self, params: GKParams, geom: Dict[str, Any], df_prev: Optional[jax.Array] = None, warm: Optional[jax.Array] = None
    ) -> Tuple[Any, ...]:
        """(qi, qe, pfe) in GKW gyroBohm units and a dict of per-call diagnostics.

        Models with a warm start (`_warm_df_shape`) get the previous distribution and whether to use it,
        and also return the final distribution.
        """
        raise NotImplementedError

    def _solve(
        self,
        loc: Dict[str, Any],
        df_prev: Optional[jax.Array] = None,
        warm: Optional[jax.Array] = None,
        *,
        ikxspace: int,
        topo: Dict[str, Any],
    ) -> Tuple[Tuple[jax.Array, ...], Dict[str, jax.Array], Optional[jax.Array]]:
        """One flux tube: fluxes, its diagnostics row and, with a warm start, the final distribution."""
        params = self.gkparams(loc)
        out = self._per_radius(params, self.geometry(params, ikxspace, topo), *(() if df_prev is None else (df_prev, warm)))
        q_i, q_e, pfe, extra = out[:4]
        finite = jnp.isfinite(q_i) & jnp.isfinite(q_e) & jnp.isfinite(pfe)
        if df_prev is not None:
            finite = finite & jnp.all(jnp.isfinite(out[4]))
        row = dict(
            rho_face=loc["rho_face"], rlt=params.rlt, rln=params.rln, q=params.q, shat=params.shat,
            eps=params.eps, beta=params.beta, coll_nref=params.coll_nref, coll_tref=params.coll_tref,
            qi_gb=q_i, qe_gb=q_e, pfe_gb=pfe, finite=finite.astype(jnp.int32), **extra,
        )
        return (q_i, q_e, pfe), row, (None if df_prev is None else out[4].astype(jnp.complex64))

    def _dispatch(self, loc: Dict[str, np.ndarray], t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every radius as its own jitted solve, radius j on device j mod the device count, the devices in parallel
        from host threads; a warm start keeps each final distribution on its device. Writes the diagnostics rows
        and returns the fluxes, NaN where a solve diverged.
        """
        try:
            devices = jax.devices("cuda")
        except RuntimeError:
            devices = jax.local_devices()
        n, shape = len(loc["q"]), self._warm_df_shape()
        locs = [{k: v[j] for k, v in loc.items()} for j in range(n)]
        ikx = [self.ikxspace_at(lj["shat"]) for lj in locs]
        solvers = {k: _compiled(self, k) for k in set(ikx)}

        def run_device(k):
            out = {}
            for j in range(k, n, len(devices)):
                start = time.perf_counter()
                args = [jax.device_put(locs[j], devices[k])]
                if shape is not None:
                    df_prev, ikx_prev = _WARM.get((id(self), j), (None, None))
                    warm = ikx_prev == ikx[j]
                    if df_prev is None:
                        df_prev = jax.device_put(jnp.zeros(shape, jnp.complex64), devices[k])
                    args += [df_prev, jax.device_put(np.bool_(warm), devices[k])]
                fluxes, row, df_new = solvers[ikx[j]](*args)
                if shape is not None:
                    _WARM[(id(self), j)] = (df_new, ikx[j])
                    row = {**row, "warm": int(warm)}
                fluxes, row = jax.device_get((fluxes, row))
                out[j] = (fluxes, {**row, "ikxspace": ikx[j], "wall_s": time.perf_counter() - start})
            return out

        parts = {}
        with ThreadPoolExecutor(len(devices)) as pool:
            for d in pool.map(run_device, range(min(len(devices), n))):
                parts.update(d)
        rows = [parts[j][1] for j in range(n)]
        if self.diagnostics_path:
            _write_rows(self.diagnostics_path, type(self).__name__, float(t), rows)
        _report_divergence(type(self).__name__, rows)
        dtype = loc["q"].dtype
        return tuple(
            np.asarray([parts[j][0][c] if rows[j]["finite"] else np.nan for j in range(n)], dtype) for c in range(3)
        )

    def call_implementation(
        self,
        transport_runtime_params: RuntimeParams,
        runtime_params: torax.RuntimeParams,
        geo: torax.Geometry,
        core_profiles: torax.CoreProfiles,
        two_point_mask: jax.Array,
        component_state: Any,
    ) -> Tuple[transport.TransportCoeffs, Any]:
        ql_inputs = quasilinear_inputs(core_profiles, geo, two_point_mask)
        loc = local_inputs(core_profiles, geo, ql_inputs, self.rho_match)
        # the solves leave this computation: each radius runs on its own device, driven from the host
        spec = (jax.ShapeDtypeStruct(loc["q"].shape, loc["q"].dtype),) * 3
        fluxes = jax.experimental.io_callback(self._dispatch, spec, loc, runtime_params.t, ordered=False)
        qi, qe, pfe = (jnp.interp(geo.rho_face_norm, loc["rho_face"], f) * gkw_to_qualikiz_flux(geo) for f in fluxes)
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

    backend: Annotated[str, STATIC] = "jax"
    mixed_precision: Annotated[bool, STATIC] = True
    fused_linear: Annotated[bool, STATIC] = True
    linear_arith: Annotated[Literal["fp64", "ds"], STATIC] = "fp64"
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
    vpar_max: Annotated[float, STATIC] = 3.0
    gkparams_dt: Annotated[Optional[float], STATIC] = None
    kinetic: Annotated[bool, STATIC] = False

    def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
        return RuntimeParams(
            DV_effective=self.DV_effective,
            An_min=self.An_min,
            DV_effective_smooth_width=self.DV_effective_smooth_width,
            **vars(super().build_runtime_params(t)),
        )
