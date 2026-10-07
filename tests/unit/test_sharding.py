"""Multi-GPU grid parallelism tests.

Tests gate on device count: single-device tests always run; multi-device
tests are skipped when insufficient GPUs are visible. The goal is to
ensure `gyaradax/sharding.py` is a true no-op on single device and that
sharded runs match single-device outputs within FP64 round-off.
"""

import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import HAS_CUDA, noisy_case, rel_l2  # type: ignore[import-not-found]

from gyaradax import sharding
from gyaradax.geometry import compute_geometry
from gyaradax.params import GKParams, gkparams_from_config
from gyaradax.precompute import linear_precompute
from gyaradax.simulate import gk_init, gk_run
from gyaradax import load_config


CONFIG_ADIABATIC = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "iteration_13.yaml"
)


def _build(params_overrides=None):
    cfg = load_config(CONFIG_ADIABATIC)
    overrides = {"non_linear": True, "adaptive_dt": False, "dt": 0.005, "mixed_precision": False}
    if params_overrides:
        overrides.update(params_overrides)
    params = gkparams_from_config(cfg, **overrides)
    grid = cfg.grid
    geometry = compute_geometry(
        q=params.q,
        shat=params.shat,
        eps=params.eps,
        ns=grid.ns,
        nkx=grid.nkx,
        nky=grid.nky,
        nvpar=grid.nvpar,
        nmu=grid.nmu,
        vpar_max=grid.vpar_max,
        nperiod=grid.nperiod,
        krhomax=grid.krhomax,
        ikxspace=grid.ikxspace,
        adiabatic_electrons=True,
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
    assert sharding._spec_for_shape((9, 16, 8, 16), grid) == PartitionSpec(None, "vp", "mu", None)
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
    df_ref, phi_ref, flx_ref, _ = gk_run(df0, geom0, p0, st0, n_steps=100, pre=pre0)

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


    # threshold: ~1e-8 accounts for FP64 reduction-order differences
    # accumulating over 100 RK4 steps on ndim=5 arrays.
    assert rel_l2(df_sh_np, df_ref_np) < 1e-8, f"df rel L2 = {rel_l2(df_sh_np, df_ref_np):.3e}"
    assert rel_l2(phi_sh_np, phi_ref_np) < 1e-8
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


    assert rel_l2(df_sh, df_ref) < 1e-8
    assert rel_l2(phi_sh, phi_ref) < 1e-8


CONFIG_KINETIC = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "nl_em_apar.yaml")


def _build_kinetic(params_overrides=None):
    from gyaradax.geometry import compute_geometry_from_config

    cfg = load_config(CONFIG_KINETIC)
    overrides = {"non_linear": True, "adaptive_dt": False, "dt": 0.002, "mixed_precision": False}
    if params_overrides:
        overrides.update(params_overrides)
    params = gkparams_from_config(cfg, **overrides)
    geometry = compute_geometry_from_config(cfg)
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


    assert rel_l2(df_sh, df_ref) < 1e-8, (
        f"df rel L2 = {rel_l2(df_sh, df_ref):.3e}"
    )
    assert rel_l2(phi_sh, phi_ref) < 1e-8


def _em_case(backend, mesh_axes=None):
    grid = dict(nvpar=16, nmu=4, ns=16, nkx=21, nky=5)
    overrides = dict(nlbpar=True, adaptive_dt=False, **(mesh_axes or {}))
    return noisy_case(
        "nl_em_waltz_b01.yaml", grid, overrides, seed=7, backend=backend, mixed_precision=False
    )


def _all_gather_bytes(compiled_text):
    import re

    nbytes = {"c128": 16, "c64": 8, "f64": 8, "f32": 4}
    total = 0
    for line in compiled_text.splitlines():
        m = re.search(r"= (.*?) all-gather(-start)?\(", line)
        if m:
            dt, shp = re.findall(r"([a-z]+[0-9]*)\[([0-9,]*)\]", m.group(1))[-1]
            total += int(np.prod([int(x) for x in shp.split(",") if x])) * nbytes.get(dt, 8)
    return total


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires ≥2 GPUs")
def test_vpar_halo_matches_unsharded_stencil():
    from jax.sharding import NamedSharding, PartitionSpec

    mesh = sharding.build_mesh(GKParams(n_gpus_vp=2))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 3, 5))
    coef = jnp.array([1.0, -8.0, 0.5, 8.0, -1.0])

    def stencil(f, v_axis):
        pad = [(0, 0)] * f.ndim
        pad[v_axis] = (2, 2)
        fp = jnp.pad(f, pad)
        n = f.shape[v_axis]
        return sum(c * jax.lax.slice_in_dim(fp, k, k + n, axis=v_axis) for k, c in enumerate(coef))

    ref = stencil(x, 1)

    def local(f):
        ext = sharding.halo(f, 1, "vp", 2)
        out = sum(
            c * jax.lax.slice_in_dim(ext, k, k + f.shape[1], axis=1) for k, c in enumerate(coef)
        )
        return out

    xs = jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, "vp")))
    out = jax.jit(
        lambda f: sharding.velocity_map(
            local, mesh, (f,), ((None, "vp"),), PartitionSpec(None, "vp")
        )
    )(xs)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def _assert_sharded_matches_single(df, geometry, params, state, sharded_params):
    """3 steps on the 2-device mesh of ``sharded_params`` match one device and never gather df."""
    from gyaradax.solver import gksolve

    if params.backend == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    ref = gksolve(df, geometry, params, state, n_steps=3, pre=linear_precompute(geometry, params))
    mesh = sharding.build_mesh(sharded_params)
    grid = sharding.grid_shape_from(sharded_params, geometry)
    pre = sharding.precompute_sharded(geometry, sharded_params, mesh, grid)
    df_sharded = sharding.shard_df(df, mesh, grid)
    run = jax.jit(lambda d, s, p: gksolve(d, geometry, sharded_params, s, n_steps=3, pre=p))
    out = run(df_sharded, state, pre)
    compiled = run.lower(df_sharded, state, pre).compile().as_text()
    assert _all_gather_bytes(compiled) < df.nbytes / 100
    assert rel_l2(out[0], ref[0]) < 1e-12
    assert rel_l2(out[1][0], ref[1][0]) < 1e-12


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires ≥2 GPUs")
@pytest.mark.parametrize("backend", ["cuda", "jax"])
@pytest.mark.parametrize("axis", ["n_gpus_sp", "n_gpus_vp", "n_gpus_mu"])
def test_em_sharded_matches_single_device(backend, axis):
    df, geometry, params, state = _em_case(backend)
    _assert_sharded_matches_single(df, geometry, params, state, replace(params, **{axis: 2}))


@pytest.mark.skipif(len(jax.devices()) < 2, reason="requires ≥2 GPUs")
@pytest.mark.parametrize("backend", ["cuda", "jax"])
@pytest.mark.parametrize("axis", ["n_gpus_vp", "n_gpus_mu"])
@pytest.mark.parametrize("conserve", [False, True])
def test_collisions_sharded_matches_single_device(backend, axis, conserve):
    coll = {
        "backend": backend,
        "collisions": True,
        "coll_freq": 0.05,
        "coll_mom_conservation": conserve,
        "coll_ene_conservation": conserve,
    }
    df, geometry, params, state, _ = _build(coll)
    _assert_sharded_matches_single(df, geometry, params, state, replace(params, **{axis: 2}))
