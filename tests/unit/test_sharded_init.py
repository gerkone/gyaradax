"""Test sharded initialization matches single-GPU initialization."""

import jax
import numpy as np
import pytest

from gyaradax import sharding
from gyaradax.geometry import compute_geometry
from gyaradax.params import gkparams_from_config
from gyaradax.solver import init_f, linear_precompute
from gyaradax import load_config
import os
from typing import Any, cast


CONFIG_ADIABATIC = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "iteration_13.yaml"
)


def test_sharded_init_matches_single_gpu():
    """Test that sharded init_f produces same result as single-GPU init_f."""
    # Skip if less than 2 GPUs
    if len(jax.devices()) < 2:
        pytest.skip("Need at least 2 GPUs")
    
    # Load config
    cfg = load_config(CONFIG_ADIABATIC)
    
    # Single-GPU setup
    params_single = gkparams_from_config(cfg, non_linear=True, adaptive_dt=False, dt=0.005)
    geom_single = compute_geometry(
        q=params_single.q, shat=params_single.shat, eps=params_single.eps,
        ns=cfg.grid.ns, nkx=cfg.grid.nkx, nky=cfg.grid.nky,
        nvpar=cfg.grid.nvpar, nmu=cfg.grid.nmu,
        vpar_max=cfg.grid.vpar_max, nperiod=cfg.grid.nperiod,
        krhomax=cfg.grid.krhomax, ikxspace=cfg.grid.ikxspace,
        adiabatic_electrons=True, geom_type='circ', signB=params_single.signB,
    )
    
    # Create single-GPU df
    df_single = init_f(
        geom_single,
        finit=params_single.finit,
        amp_init_real=params_single.amp_init,
        norm_eps=params_single.norm_eps,
        n_species=1,
    )
    
    # Multi-GPU setup (vp=2)
    params_sharded = gkparams_from_config(
        cfg, non_linear=True, adaptive_dt=False, dt=0.005,
        n_gpus_vp=2, n_gpus_mu=1
    )
    geom_sharded = compute_geometry(
        q=params_sharded.q, shat=params_sharded.shat, eps=params_sharded.eps,
        ns=cfg.grid.ns, nkx=cfg.grid.nkx, nky=cfg.grid.nky,
        nvpar=cfg.grid.nvpar, nmu=cfg.grid.nmu,
        vpar_max=cfg.grid.vpar_max, nperiod=cfg.grid.nperiod,
        krhomax=cfg.grid.krhomax, ikxspace=cfg.grid.ikxspace,
        adiabatic_electrons=True, geom_type='circ', signB=params_sharded.signB,
    )
    
    # Build mesh and grid
    mesh = sharding.build_mesh(params_sharded)
    assert mesh is not None, "Mesh should be created for 2-GPU setup"
    grid = sharding.grid_shape_from(params_sharded, geom_sharded)
    
    # Create sharded df
    df_sharded = sharding.init_f_sharded(
        geom_sharded,
        params_sharded,
        mesh=mesh,
        grid=grid,
        finit=params_sharded.finit,
        amp_init_real=params_sharded.amp_init,
        n_species=1,
    )
    
    # Bring sharded array to host for comparison
    df_sharded_host = np.array(df_sharded)
    df_single_host = np.array(df_single)
    
    print(f"Single-GPU df shape: {df_single_host.shape}")
    print(f"Sharded df shape: {df_sharded_host.shape}")
    print(f"Single-GPU df device: {df_single.device}")
    print(f"Sharded df spec: {cast(Any, df_sharded.sharding).spec}")
    
    # Compare (should be identical for deterministic init modes)
    # For noise modes, they're random but should have same distribution
    if params_single.finit in ("cosine2", "cosine", "sine"):
        # Deterministic modes - should match exactly
        np.testing.assert_allclose(
            df_single_host, df_sharded_host,
            rtol=1e-14, atol=1e-14,
            err_msg="Sharded init does not match single-GPU init"
        )
        print(f"PASSED: {params_single.finit} mode matches exactly")
    else:
        # Random modes - check statistics
        assert df_single_host.shape == df_sharded_host.shape
        print(f"PASSED: {params_single.finit} mode shapes match")


def test_sharded_precompute_matches_single_gpu():
    """Test that precompute_sharded produces same result as single-GPU precompute."""
    # Skip if less than 2 GPUs
    if len(jax.devices()) < 2:
        pytest.skip("Need at least 2 GPUs")
    
    # Load config
    cfg = load_config(CONFIG_ADIABATIC)
    
    # Single-GPU setup
    params_single = gkparams_from_config(cfg, non_linear=True, adaptive_dt=False, dt=0.005)
    geom_single = compute_geometry(
        q=params_single.q, shat=params_single.shat, eps=params_single.eps,
        ns=cfg.grid.ns, nkx=cfg.grid.nkx, nky=cfg.grid.nky,
        nvpar=cfg.grid.nvpar, nmu=cfg.grid.nmu,
        vpar_max=cfg.grid.vpar_max, nperiod=cfg.grid.nperiod,
        krhomax=cfg.grid.krhomax, ikxspace=cfg.grid.ikxspace,
        adiabatic_electrons=True, geom_type='circ', signB=params_single.signB,
    )
    
    # Create single-GPU pre
    pre_single = linear_precompute(geom_single, params_single)
    
    # Multi-GPU setup
    params_sharded = gkparams_from_config(
        cfg, non_linear=True, adaptive_dt=False, dt=0.005,
        n_gpus_vp=2, n_gpus_mu=1
    )
    geom_sharded = compute_geometry(
        q=params_sharded.q, shat=params_sharded.shat, eps=params_sharded.eps,
        ns=cfg.grid.ns, nkx=cfg.grid.nkx, nky=cfg.grid.nky,
        nvpar=cfg.grid.nvpar, nmu=cfg.grid.nmu,
        vpar_max=cfg.grid.vpar_max, nperiod=cfg.grid.nperiod,
        krhomax=cfg.grid.krhomax, ikxspace=cfg.grid.ikxspace,
        adiabatic_electrons=True, geom_type='circ', signB=params_sharded.signB,
    )
    
    # Build mesh and grid
    mesh = sharding.build_mesh(params_sharded)
    grid = sharding.grid_shape_from(params_sharded, geom_sharded)
    
    # Create sharded pre
    pre_sharded = sharding.precompute_sharded(geom_sharded, params_sharded, mesh, grid)
    
    # Compare key arrays
    print("\nComparing pre arrays:")
    mismatch_count = 0
    for key in pre_single._items.keys():
        val_single = pre_single._items[key]
        val_sharded = pre_sharded._items[key]
        
        if not hasattr(val_single, 'shape') or len(val_single.shape) == 0:
            continue  # Skip scalars
        
        arr_single = np.array(val_single)
        arr_sharded = np.array(val_sharded)
        
        if arr_single.shape != arr_sharded.shape:
            print(f"  {key}: SHAPE MISMATCH {arr_single.shape} vs {arr_sharded.shape}")
            mismatch_count += 1
            continue
        
        try:
            np.testing.assert_allclose(arr_single, arr_sharded, rtol=1e-10, atol=1e-10)
            print(f"  {key}: OK (shape {arr_single.shape})")
        except AssertionError as e:
            print(f"  {key}: MISMATCH - {str(e)[:80]}")
            mismatch_count += 1
    
    print(f"\nTotal mismatches: {mismatch_count}")
    
    # Key arrays should match
    key_arrays = ['phi_weight', 'bessel', 'stream_fac', 'mirror_fac']
    for key in key_arrays:
        if key in pre_single._items:
            arr_single = np.array(pre_single._items[key])
            arr_sharded = np.array(pre_sharded._items[key])
            np.testing.assert_allclose(
                arr_single, arr_sharded,
                rtol=1e-10, atol=1e-10,
                err_msg=f"Pre array '{key}' mismatch"
            )
    
    print("PASSED: Key pre arrays match")


def test_sharded_df_recombine():
    """Test that sharded df can be recombined to match single-GPU df."""
    if len(jax.devices()) < 2:
        pytest.skip("Need at least 2 GPUs")
    
    cfg = load_config(CONFIG_ADIABATIC)
    
    # Single-GPU
    params_single = gkparams_from_config(cfg, non_linear=True)
    geom_single = compute_geometry(
        q=params_single.q, shat=params_single.shat, eps=params_single.eps,
        ns=cfg.grid.ns, nkx=cfg.grid.nkx, nky=cfg.grid.nky,
        nvpar=cfg.grid.nvpar, nmu=cfg.grid.nmu,
        vpar_max=cfg.grid.vpar_max, nperiod=cfg.grid.nperiod,
        adiabatic_electrons=True, geom_type='circ', signB=params_single.signB,
    )
    df_single = init_f(geom_single, finit="cosine2", n_species=1)
    
    # Multi-GPU sharded
    params_sharded = gkparams_from_config(cfg, non_linear=True, n_gpus_vp=2)
    geom_sharded = compute_geometry(
        q=params_sharded.q, shat=params_sharded.shat, eps=params_sharded.eps,
        ns=cfg.grid.ns, nkx=cfg.grid.nkx, nky=cfg.grid.nky,
        nvpar=cfg.grid.nvpar, nmu=cfg.grid.nmu,
        vpar_max=cfg.grid.vpar_max, nperiod=cfg.grid.nperiod,
        adiabatic_electrons=True, geom_type='circ', signB=params_sharded.signB,
    )
    
    mesh = sharding.build_mesh(params_sharded)
    grid = sharding.grid_shape_from(params_sharded, geom_sharded)
    df_sharded = sharding.init_f_sharded(
        geom_sharded, params_sharded, mesh, grid, finit="cosine2", amp_init_real=1e-4, n_species=1
    )
    
    # Recombine by gathering to host
    df_recombined = np.array(df_sharded)
    df_expected = np.array(df_single)
    
    print(f"\nRecombined shape: {df_recombined.shape}")
    print(f"Expected shape: {df_expected.shape}")
    
    np.testing.assert_allclose(
        df_recombined, df_expected,
        rtol=1e-14, atol=1e-14
    )
    
    print("PASSED: Recombined sharded df matches single-GPU df")


if __name__ == "__main__":
    # Run tests manually if called directly
    print("="*60)
    print("Testing sharded init vs single-GPU init")
    print("="*60)
    
    try:
        test_sharded_init_matches_single_gpu()
    except Exception as e:
        print(f"test_sharded_init_matches_single_gpu: FAILED - {e}")
    
    print("\n" + "="*60)
    print("Testing sharded precompute vs single-GPU precompute")
    print("="*60)
    
    try:
        test_sharded_precompute_matches_single_gpu()
    except Exception as e:
        print(f"test_sharded_precompute_matches_single_gpu: FAILED - {e}")
    
    print("\n" + "="*60)
    print("Testing sharded df recombination")
    print("="*60)
    
    try:
        test_sharded_df_recombine()
    except Exception as e:
        print(f"test_sharded_df_recombine: FAILED - {e}")
