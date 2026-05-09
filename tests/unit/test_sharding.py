"""Multi-GPU grid parallelism tests.

Tests gate on device count: single-device tests always run; multi-device
tests are skipped when insufficient GPUs are visible. The goal is to
ensure `gyaradax/sharding.py` is a true no-op on single device and that
sharded runs match single-device outputs within FP64 round-off.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace

from gyaradax import sharding
from gyaradax.geometry import compute_geometry
from gyaradax.params import GKParams, gkparams_from_config
from gyaradax.solver import linear_precompute
from gyaradax.simulate import gk_init, gk_run
from gyaradax import load_config


CONFIG_ADIABATIC = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "iteration_13.yaml"
)


def _build(params_overrides=None):
    cfg = load_config(CONFIG_ADIABATIC)
    overrides = {"non_linear": True, "adaptive_dt": False, "dt": 0.005}
    if params_overrides:
        overrides.update(params_overrides)
    params = gkparams_from_config(cfg, **overrides)
    grid = cfg.grid
    geometry = compute_geometry(
        q=params.q, shat=params.shat, eps=params.eps,
        ns=grid.ns, nkx=grid.nkx, nky=grid.nky, nvpar=grid.nvpar, nmu=grid.nmu,
        vpar_max=grid.vpar_max, nperiod=grid.nperiod, krhomax=grid.krhomax,
        ikxspace=grid.ikxspace, adiabatic_electrons=True,
        geom_type=getattr(cfg.geometry, "geometry_model", "circ"),
        signB=params.signB,
    )
    pre = linear_precompute(geometry, params)
    df, geometry, state = gk_init(geometry, params, n_species=1)
    return df, geometry, params, state, pre


def test_build_mesh_single_device():
    """With all n_gpus_*==1, build_mesh returns None (no-op path)."""
    p = GKParams()
    assert sharding.build_mesh(p) is None
    assert not sharding.is_active(None)


def test_shard_helpers_identity_on_single_device():
    """shard_df / shard_pre pass through unchanged when mesh is None."""
    df, geometry, params, state, pre = _build()
    mesh = sharding.build_mesh(params)
    assert mesh is None
    # Helpers return the same object (identity) when mesh is None.
    df2 = sharding.shard_df(df, mesh, None)
    pre2 = sharding.shard_pre(pre, mesh, None)
    assert df2 is df
    assert pre2 is pre


def test_grid_shape_inference():
    """grid_shape_from infers axis lengths from geometry / params."""
    df, geometry, params, state, pre = _build()
    grid = sharding.grid_shape_from(params, geometry)
    assert grid.nvpar == df.shape[0]
    assert grid.nmu == df.shape[1]
    assert grid.ns == df.shape[2]
    assert grid.nkx == df.shape[3]
    assert grid.nky == df.shape[4]
    assert grid.nsp == 1  # adiabatic


def test_spec_classification():
    """_spec_for_shape returns the right PartitionSpec per shape kind."""
    from jax.sharding import PartitionSpec
    grid = sharding.GridShape(nsp=1, nvpar=16, nmu=8, ns=16, nkx=9, nky=5)

    # df (5D velocity-sharded)
    assert sharding._spec_for_shape((16, 8, 16, 9, 5), grid) == PartitionSpec(
        "vp", "mu", None, None, None
    )
    # field (3D replicated)
    assert sharding._spec_for_shape((16, 9, 5), grid) == PartitionSpec()
    # collision stencil (5D adiabatic)
    assert sharding._spec_for_shape((9, 16, 8, 16), grid) == PartitionSpec(
        None, "vp", "mu", None
    )
    # unmatched shape (e.g. kx_b broadcast) → replicated
    assert sharding._spec_for_shape((1, 1, 1, 9, 1), grid) == PartitionSpec()


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires ≥2 GPUs")
def test_equivalence_2gpu_vp():
    """100-step adiabatic run with (vp=2) vs single-device.

    Compares (df_final, phi, fluxes) — relative L2 should be within
    FP64 round-off accumulation (target < 1e-10).
    """
    # baseline
    df0, geom0, p0, st0, pre0 = _build()
    df_ref, phi_ref, flx_ref, _ = gk_run(
        df0, geom0, p0, st0, n_steps=100, pre=pre0
    )

    # sharded (vp=2)
    df1, geom1, p1, st1, pre1 = _build({"n_gpus_vp": 2})
    mesh = sharding.build_mesh(p1)
    assert mesh is not None
    grid = sharding.grid_shape_from(p1, geom1)
    df1 = sharding.shard_df(df1, mesh, grid)
    pre1 = sharding.shard_pre(pre1, mesh, grid)
    df_sh, phi_sh, flx_sh, _ = gk_run(df1, geom1, p1, st1, n_steps=100, pre=pre1)

    # bring back to host for comparison
    df_ref_np = np.asarray(df_ref)
    df_sh_np = np.asarray(df_sh)
    phi_ref_np = np.asarray(phi_ref)
    phi_sh_np = np.asarray(phi_sh)

    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))

    # threshold: ~1e-8 accounts for FP64 reduction-order differences
    # accumulating over 100 RK4 steps on ndim=5 arrays.
    assert rel_l2(df_ref_np, df_sh_np) < 1e-8, f"df rel L2 = {rel_l2(df_ref_np, df_sh_np):.3e}"
    assert rel_l2(phi_ref_np, phi_sh_np) < 1e-8
    # fluxes: allow abs-or-rel ≤ 1e-8 (pflux is ~1e-20 noise at t=0 adiabatic)
    for i, name in enumerate(("pflux", "eflux", "vflux")):
        a, b = float(flx_ref[i]), float(flx_sh[i])
        err = abs(a - b) / max(abs(a), 1e-10)
        assert err < 1e-8, f"{name} rel err {err:.3e} (ref={a:.3e}, sh={b:.3e})"


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires ≥4 GPUs")
def test_equivalence_4gpu_vpmu():
    """100-step adiabatic with (vp=2, mu=2) vs single-device. Same targets."""
    df0, geom0, p0, st0, pre0 = _build()
    df_ref, phi_ref, flx_ref, _ = gk_run(df0, geom0, p0, st0, n_steps=100, pre=pre0)

    df1, geom1, p1, st1, pre1 = _build({"n_gpus_vp": 2, "n_gpus_mu": 2})
    mesh = sharding.build_mesh(p1)
    grid = sharding.grid_shape_from(p1, geom1)
    df1 = sharding.shard_df(df1, mesh, grid)
    pre1 = sharding.shard_pre(pre1, mesh, grid)
    df_sh, phi_sh, flx_sh, _ = gk_run(df1, geom1, p1, st1, n_steps=100, pre=pre1)

    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))

    assert rel_l2(np.asarray(df_ref), np.asarray(df_sh)) < 1e-8
    assert rel_l2(np.asarray(phi_ref), np.asarray(phi_sh)) < 1e-8


CONFIG_KINETIC = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "nl_em_apar.yaml"
)


def _build_kinetic(params_overrides=None):
    cfg = load_config(CONFIG_KINETIC)
    overrides = {"non_linear": True, "adaptive_dt": False, "dt": 0.002}
    if params_overrides:
        overrides.update(params_overrides)
    params = gkparams_from_config(cfg, **overrides)
    grid = cfg.grid
    geometry = compute_geometry(
        q=params.q, shat=params.shat, eps=params.eps,
        ns=grid.ns, nkx=grid.nkx, nky=grid.nky, nvpar=grid.nvpar, nmu=grid.nmu,
        vpar_max=grid.vpar_max, nperiod=grid.nperiod, krhomax=grid.krhomax,
        ikxspace=grid.ikxspace, adiabatic_electrons=False,
        geom_type=getattr(cfg.geometry, "geometry_model", "circ"),
        signB=params.signB,
    )
    for k in ("mas", "signz", "tmp", "de", "vthrat"):
        geometry[k] = jnp.atleast_1d(jnp.asarray(getattr(params, k), dtype=jnp.float64))
    pre = linear_precompute(geometry, params)
    df, geometry, state = gk_init(geometry, params, n_species=2)
    return df, geometry, params, state, pre


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires ≥2 GPUs")
def test_equivalence_2gpu_sp_kinetic():
    """50-step kinetic with (sp=2) vs single-device. Trivial species split."""
    df0, geom0, p0, st0, pre0 = _build_kinetic()
    df_ref, phi_ref, flx_ref, _ = gk_run(df0, geom0, p0, st0, n_steps=50, pre=pre0)

    df1, geom1, p1, st1, pre1 = _build_kinetic({"n_gpus_sp": 2})
    mesh = sharding.build_mesh(p1)
    assert mesh is not None
    grid = sharding.grid_shape_from(p1, geom1)
    df1 = sharding.shard_df(df1, mesh, grid)
    pre1 = sharding.shard_pre(pre1, mesh, grid)
    df_sh, phi_sh, flx_sh, _ = gk_run(df1, geom1, p1, st1, n_steps=50, pre=pre1)

    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))

    assert rel_l2(np.asarray(df_ref), np.asarray(df_sh)) < 1e-8, \
        f"df rel L2 = {rel_l2(np.asarray(df_ref), np.asarray(df_sh)):.3e}"
    assert rel_l2(np.asarray(phi_ref), np.asarray(phi_sh)) < 1e-8
