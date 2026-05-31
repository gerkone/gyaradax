"""QL gyaradax as a TORAX transport model.

Pure-JAX linear gyrokinetic solve + saturation rule + calibration head.
Inherits the vmap/interp/_make_core_transport machinery from
`GyaradaxBasedTransportModel` and only specifies what happens at one radius.
"""

from typing import Any, Dict, Tuple
from functools import lru_cache
import dataclasses
import glob
import pathlib
import pickle
import warnings

import jax
import jax.numpy as jnp

from gyaradax.solver import gksolve, linear_precompute, default_state
from gyaradax.integrals import calculate_fluxes, geom_tensors
from gyaradax.params import GKParams
from gyaradax.quasilinear.saturation import ql_flux
from gyaradax.torax_plugin.gyaradax_based_transport_model import (
    GyaradaxBasedTransportModel,
    RuntimeParams as _BaseRuntimeParams,
    _get_topology_cached,
)


_QI_CLIP_ABS = 1e3  # nan-guard / clip on per-radius q_i (poisons TORAX Newton otherwise)
# key the cn-head 'auto' search off the gyaradax install, not this file's location,
# so it resolves correctly when dropped into torax/_src/transport_model
import gyaradax as _gyaradax_pkg

_GYARADAX_DATA = pathlib.Path(_gyaradax_pkg.__file__).resolve().parent.parent / "data"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(_BaseRuntimeParams):
    """Runtime parameters for the gyaradax-QL transport model."""


@lru_cache(maxsize=8)
def _get_cn_head(path: str):
    """Cache calibration heads by path. 'auto' = newest data/cn_iter_hybrid_*.pkl."""
    if not path:
        return None
    if path == "auto":
        candidates = sorted(
            glob.glob(str(_GYARADAX_DATA / "cn_iter_hybrid_*.pkl"))
        )
        if not candidates:
            warnings.warn(
                "gyaradax-QL: cn_calibration_path='auto' but no "
                "data/cn_iter_hybrid_*.pkl fit found; falling back to cn_scalar=1.",
                RuntimeWarning, stacklevel=2,
            )
            return None
        path = candidates[-1]
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except FileNotFoundError:
        warnings.warn(
            f"gyaradax-QL: cn_calibration_path='{path}' does not exist; "
            "falling back to cn_scalar=1.",
            RuntimeWarning, stacklevel=2,
        )
        return None
    if isinstance(obj, dict) and "polynomial" in obj:
        return obj["polynomial"]
    return obj


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GyaradaxQLTransportModel(GyaradaxBasedTransportModel):
    """QL gyaradax transport model."""

    n_steps_linear: int = 200
    ncv_eigensolve: int = 0
    cn_calibration_path: str = "auto"
    cn_scalar: float = 1.0
    # early-stop knobs: skip remaining gksolve steps once per-ky growth rates
    # stop moving. solver itself is untouched -- we chunk the call and check
    # convergence between chunks via lax.while_loop.
    early_stop: bool = True
    early_stop_block: int = 25
    early_stop_atol: float = 1e-4
    early_stop_rtol: float = 1e-3
    early_stop_min_steps: int = 50

    @classmethod
    def from_config(cls, cfg) -> "GyaradaxQLTransportModel":
        # warm caches outside any jit; build_topology allocates int8 arrays
        # that would otherwise become tracers inside torax's jit
        _get_topology_cached(cfg.nkx, cfg.nky, cfg.ikxspace, cfg.ns)
        _get_cn_head(cfg.cn_calibration_path or "")
        return cls(
            rho_match=tuple(cfg.rho_match),
            backend=cfg.backend,
            n_steps_linear=cfg.n_steps_linear,
            ncv_eigensolve=cfg.ncv_eigensolve,
            nvpar=cfg.nvpar, nmu=cfg.nmu, ns=cfg.ns,
            nkx=cfg.nkx, nky=cfg.nky, ikxspace=cfg.ikxspace,
            cn_calibration_path=cfg.cn_calibration_path or "",
            early_stop=cfg.early_stop,
            early_stop_block=cfg.early_stop_block,
            early_stop_atol=cfg.early_stop_atol,
            early_stop_rtol=cfg.early_stop_rtol,
            early_stop_min_steps=cfg.early_stop_min_steps,
        )

    @property
    def cn_head(self):
        return _get_cn_head(self.cn_calibration_path)

    def _per_radius(
        self, params: GKParams, geom: Dict[str, Any]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._gyaradax_ql_at_radius(params, geom)

    def _linear_with_early_stop(self, df, geom, params, sim_state, pre):
        """Chunked linear gksolve with per-ky growth-rate convergence check.

        Runs gksolve in blocks of `early_stop_block` and exits early once the
        per-ky growth-rate vector stops moving within (atol, rtol). Pure
        wrapper -- never touches the solver internals. Hard-caps the total
        iteration count at `n_steps_linear`. Compatible with jit + vmap
        (lax.while_loop runs until *all* batch elements converge).
        """
        block = int(self.early_stop_block)
        max_blocks = max(int(self.n_steps_linear) // block, 1)
        min_blocks = max(int(self.early_stop_min_steps) // block, 1)
        atol = float(self.early_stop_atol)
        rtol = float(self.early_stop_rtol)

        # bootstrap one block so `phi_last` has a concrete dtype/shape to carry
        df1, (phi1, _flx), sim1 = gksolve(df, geom, params, sim_state,
                                          n_steps=block, pre=pre)
        g0 = sim1.last_growth_rate

        def cond(state):
            i, _df, _phi, _sim, _prev, conv = state
            return jnp.logical_and(i < max_blocks, jnp.logical_not(conv))

        def body(state):
            i, df_c, _phi_c, sim_c, prev_g, _ = state
            df_n, (phi_n, _f), sim_n = gksolve(df_c, geom, params, sim_c,
                                                n_steps=block, pre=pre)
            new_g = sim_n.last_growth_rate
            delta = jnp.max(jnp.abs(new_g - prev_g))
            scale = atol + rtol * jnp.max(jnp.abs(new_g))
            conv = jnp.logical_and(i + 1 >= min_blocks, delta <= scale)
            return (i + 1, df_n, phi_n, sim_n, new_g, conv)

        init = (jnp.asarray(1), df1, phi1, sim1, g0, jnp.array(False))
        _i, df_f, phi_f, sim_f, _g, _c = jax.lax.while_loop(cond, body, init)
        return df_f, phi_f, sim_f

    def _gyaradax_ql_at_radius(self, params, geom):
        """Linear gksolve + QL saturation rule + Cn head at one radius."""
        df = self._initial_df()
        sim_state = default_state(nky=self.nky)
        pre = linear_precompute(geom, params)
        if self.early_stop:
            df_final, phi, sim_state_final = self._linear_with_early_stop(
                df, geom, params, sim_state, pre,
            )
        else:
            df_final, (phi, _fluxes), sim_state_final = gksolve(
                df, geom, params, sim_state, n_steps=self.n_steps_linear, pre=pre,
            )

        gt = geom_tensors(geom)
        _pflux, eflux_kxy, _vflux = calculate_fluxes(gt, df_final, phi, reduce=False)
        ints = jnp.asarray(geom["ints"])
        ds = jnp.mean(ints)
        phi2 = jnp.abs(phi) ** 2
        phi2_kxy = jnp.sum(phi2 * ints[:, None, None], axis=0)
        lg = jnp.asarray(geom["little_g"])
        little_g = lg.T if lg.shape[0] != 3 else lg
        krho = jnp.asarray(geom["krho"], dtype=jnp.float64)
        kxrh = jnp.asarray(geom["kxrh"], dtype=jnp.float64)
        gamma = sim_state_final.last_growth_rate

        # FEATURE_NAMES = (rlt_i, rln_i, rlt_e, rln_e, shat, q, eps, beta)
        head = self.cn_head
        if head is not None and hasattr(head, "cn_jax"):
            features = jnp.array([[
                params.rlt, params.rln, params.rlt, params.rln,
                params.shat, params.q, params.eps, params.beta,
            ]])
            cn = head.cn_jax(features)[0]
        else:
            cn = jnp.asarray(self.cn_scalar)

        q_i = ql_flux(
            growth_rate=gamma, phi2=phi2, phi2_kxy=phi2_kxy,
            flux_kxy=eflux_kxy, krho=krho, kxrh=kxrh,
            little_g=little_g, ds=ds, cn=cn,
        )
        q_i = jnp.where(jnp.isfinite(q_i),
                        jnp.clip(q_i, -_QI_CLIP_ABS, _QI_CLIP_ABS), 0.0)
        # qe = qi, pfe = 0 placeholder (ITG-adiabatic)
        return q_i, q_i, jnp.asarray(0.0)
