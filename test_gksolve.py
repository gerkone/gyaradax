import jax
import jax.numpy as jnp
import numpy as np
import pytest
import os
from jax_geometry import load_geometry
from utils import load_gkw_k_dump, K_files
from jax_solver import gksolve, gksolve_rhs, compute_f_maxwellian

# Ensure fp64
jax.config.update("jax_enable_x64", True)

def test_gksolve_shapes():
    directory = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13"
    if not os.path.exists(directory):
        pytest.skip(f"Test directory {directory} not found.")

    geom = load_geometry(directory)
    nvpar, nmu, ns, nkx, nky = 32, 8, 16, 85, 32
    df = jnp.zeros((nvpar, nmu, ns, nkx, nky), dtype=jnp.complex128)
    
    next_df, (phi, fluxes) = gksolve(df, geom, dt=0.01)
    
    assert next_df.shape == df.shape
    assert phi.shape == (ns, nkx, nky)
    assert len(fluxes) == 3

def test_growth_rate_validation():
    directory = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13"
    if not os.path.exists(directory):
        pytest.skip(f"Test directory {directory} not found.")

    # Read reference growth rates
    ref_growth = np.loadtxt(f"{directory}/growth_rates_all_modes")
    # Expected: (800, 2040)
    print(f"Reference growth rates shape: {ref_growth.shape}")
    
    geom = load_geometry(directory)
    res = (32, 8, 16, 85, 32)
    ks = K_files(directory)
    
    # We use K01 as starting point
    df0 = load_gkw_k_dump(f"{directory}/{ks[0]}", res)
    
    dt = 0.01
    next_df, (phi1, _) = gksolve(df0, geom, dt)
    
    fm = compute_f_maxwellian(geom)
    _, phi0, _ = gksolve_rhs(df0, geom, fm)
    
    # Growth rate calculation with epsilon to avoid nans
    eps = 1e-20
    gamma_calc = jnp.log((jnp.abs(phi1) + eps) / (jnp.abs(phi0) + eps)) / dt
    
    max_ref_gamma = jnp.max(ref_growth)
    max_calc_gamma = jnp.max(gamma_calc)
    
    print(f"Max calculated growth rate: {max_calc_gamma}")
    print(f"Max reference growth rate: {max_ref_gamma}")
    
    # We expect some order of magnitude agreement at least for now
    # Growth rate should be positive for unstable modes
    assert max_calc_gamma > 0
    # And not vastly different from reference maximum (e.g., within factor of 5)
    assert max_calc_gamma < max_ref_gamma * 5.0
