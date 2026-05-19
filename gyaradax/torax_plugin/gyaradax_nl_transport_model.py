"""NL gyaradax as a TORAX transport model.

Two modes:
  'callback': read precomputed (qi, qe, pfe) at rho_match from a flux table
              on disk. Right mode for PORTALS-style outer-loop flux matching.
  'direct':   run the full NL gyaradax sim inside call_implementation. Slow;
              for one-shot benchmarks or training-data generation.
"""

from typing import Any, Dict, Tuple
from functools import lru_cache
import dataclasses
import pickle
import warnings

import jax
import jax.numpy as jnp

from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib

from gyaradax.solver import gksolve, linear_precompute, default_state
from gyaradax.params import GKParams
from gyaradax.torax_plugin.gyaradax_based_transport_model import (
    GyaradaxBasedTransportModel,
    RuntimeParams as _BaseRuntimeParams,
    _get_topology_cached,
    build_quasilinear_inputs,
    face_indices_for_radii,
    gkparams_for_radius,
    gyaradax_geometry_at,
)


_NL_BLOCK = 50              # gksolve steps per averaging block
_QI_CLIP_ABS = 1e3          # nan-guard / clip on per-radius q_i


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(_BaseRuntimeParams):
    """Runtime parameters for the gyaradax-NL transport model."""


@lru_cache(maxsize=8)
def _get_flux_table(path: str):
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        warnings.warn(
            f"gyaradax-NL (callback): flux_table_path='{path}' does not exist; "
            "TORAX will see zero transport. Pickle a {qi, qe, pfe} dict and re-run.",
            RuntimeWarning, stacklevel=2,
        )
        return None


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxNLTransportModel(GyaradaxBasedTransportModel):
    """NL gyaradax transport model."""

    mode: str = "callback"
    n_steps_nl: int = 4000
    n_average: int = 1024
    flux_table_path: str = ""

    @classmethod
    def from_config(cls, cfg) -> "GyaradaxNLTransportModel":
        # warm caches outside any jit; build_topology allocates int8 arrays
        # that would otherwise become tracers inside torax's jit
        _get_topology_cached(cfg.nkx, cfg.nky, cfg.ikxspace, cfg.ns)
        _get_flux_table(cfg.flux_table_path or "")
        return cls(
            rho_match=tuple(cfg.rho_match),
            backend=cfg.backend,
            mode=cfg.mode,
            n_steps_nl=cfg.n_steps_nl,
            n_average=cfg.n_average,
            nvpar=cfg.nvpar, nmu=cfg.nmu, ns=cfg.ns,
            nkx=cfg.nkx, nky=cfg.nky, ikxspace=cfg.ikxspace,
            flux_table_path=cfg.flux_table_path or "",
        )

    @property
    def flux_table(self):
        return _get_flux_table(self.flux_table_path)

    def _per_radius(
        self, params: GKParams, geom: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        params = dataclasses.replace(params, non_linear=True,
                                     disable_per_ky_norm=False)
        return self._gyaradax_nl_at_radius(params, geom)

    def call_implementation(
        self,
        transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
        runtime_params: runtime_params_lib.RuntimeParams,
        geo: geometry_lib.Geometry,
        core_profiles: state.CoreProfiles,
        pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
    ) -> transport_model_lib.TurbulentTransport:
        if self.mode == "direct":
            return super().call_implementation(
                transport_runtime_params, runtime_params, geo, core_profiles,
                pedestal_model_output,
            )
        if self.mode != "callback":
            raise ValueError(f"unknown mode: {self.mode}")

        del pedestal_model_output, runtime_params
        ql_inputs = build_quasilinear_inputs(core_profiles, geo)
        rho_face = geo.rho_face_norm
        rho_match_arr = jnp.asarray(self.rho_match)
        qi_m, qe_m, pfe_m = self._fluxes_from_table()
        qi_face = jnp.interp(rho_face, rho_match_arr, qi_m)
        qe_face = jnp.interp(rho_face, rho_match_arr, qe_m)
        pfe_face = jnp.interp(rho_face, rho_match_arr, pfe_m)
        return self._make_core_transport(
            qi=qi_face, qe=qe_face, pfe=pfe_face,
            quasilinear_inputs=ql_inputs,
            transport=transport_runtime_params,
            geo=geo, core_profiles=core_profiles,
            gradient_reference_length=geo.R_major,
            gyrobohm_flux_reference_length=geo.a_minor,
        )

    def _fluxes_from_table(self):
        """Read (qi, qe, pfe) from the static flux table. Zeros if path missing."""
        if self.flux_table is None:
            zeros = jnp.zeros(len(self.rho_match))
            return zeros, zeros, zeros
        return (
            jnp.asarray(self.flux_table["qi"]),
            jnp.asarray(self.flux_table["qe"]),
            jnp.asarray(self.flux_table["pfe"]),
        )

    def _gyaradax_nl_at_radius(self, params, geom):
        """Full NL gyaradax sim at one radius: warmup + block-averaged tail."""
        n_avg_blocks = max(self.n_average // _NL_BLOCK, 1)
        n_warmup = max(self.n_steps_nl - n_avg_blocks * _NL_BLOCK, 0)

        df = self._initial_df()
        sim_state = default_state(nky=self.nky)
        pre = linear_precompute(geom, params)

        if n_warmup > 0:
            df, _, sim_state = gksolve(df, geom, params, sim_state,
                                        n_steps=n_warmup, pre=pre)

        def body(carry, _):
            df, st = carry
            df, (_phi, fluxes), st = gksolve(df, geom, params, st,
                                              n_steps=_NL_BLOCK, pre=pre)
            return (df, st), fluxes[1]  # eflux
        (_df_final, _state_final), efluxes = jax.lax.scan(
            body, (df, sim_state), None, length=n_avg_blocks,
        )
        qi = jnp.mean(efluxes)
        qi = jnp.where(jnp.isfinite(qi),
                       jnp.clip(qi, -_QI_CLIP_ABS, _QI_CLIP_ABS), 0.0)
        # qe = qi, pfe = 0 placeholder (ITG-adiabatic)
        return qi, qi, jnp.asarray(0.0)
