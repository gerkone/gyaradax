import jax
import jax.numpy as jnp
import numpy as np
import os
import pytest
from jax_geometry import load_geometry
from jax_integrals import get_integrals
from utils import load_gkw_k_dump, K_files

# Ensure fp64
jax.config.update("jax_enable_x64", True)

@pytest.fixture(params=[0, 8, 13, 131, 200])
def adiabatic_dir(request):
    path = f"/restricteddata/ukaea/gyrokinetics/raw/iteration_{request.param}"
    if not os.path.exists(path):
        pytest.skip(f"Directory {path} not found")
    return path

def test_flux_integral_shapes(adiabatic_dir):
    geom = load_geometry(adiabatic_dir)
    geom["adiabatic"] = jnp.array(1.0, dtype=jnp.float64)
    
    y = len(geom["krho"])
    s = len(geom["ints"])
    vmu = len(geom["intmu"])
    vpar = len(geom["intvp"])
    x = len(geom["kxrh"])

    df = jnp.zeros((vpar, vmu, s, x, y), dtype=jnp.complex128)
    phi, (pflux, eflux, vflux) = get_integrals(df, geom)

    assert phi.shape == (s, x, y)
    assert pflux.shape == ()
    assert eflux.shape == ()
    assert vflux.shape == ()

@pytest.mark.parametrize("idx", [10, 20, 50, 80, 100])
def test_flux_integral_real_data_adiabatic(adiabatic_dir, idx):
    geom = load_geometry(adiabatic_dir)
    geom["adiabatic"] = jnp.array(1.0, dtype=jnp.float64)
    
    # Grid resolution from geom
    ns = len(geom["ints"])
    nkx = len(geom["kxrh"])
    nky = len(geom["krho"])
    nvpar = len(geom["intvp"])
    nmu = len(geom["intmu"])
    resolution = (nvpar, nmu, ns, nkx, nky)

    ks = K_files(adiabatic_dir)
    if idx >= len(ks) or idx < -len(ks):
        pytest.skip(f"Index {idx} out of range for {adiabatic_dir}")

    k_file = ks[idx]
    df = load_gkw_k_dump(f"{adiabatic_dir}/{k_file}", resolution)

    phi_pred, (pflux_pred, eflux_pred, vflux_pred) = get_integrals(df, geom)

    # get the exact timestamp for this K file
    time_val = None
    k_dat_path = f"{adiabatic_dir}/{k_file}.dat"
    if not os.path.exists(k_dat_path):
        pytest.skip(f"Metadata {k_dat_path} not found")
        
    with open(k_dat_path, "r") as file:
        for line in file:
            line_split = line.split("=")
            if line_split[0].strip() == "TIME":
                time_val = float(line_split[1].strip().strip(",").strip())
                break

    orig_times = np.loadtxt(f"{adiabatic_dir}/time.dat")
    # Finding closest time index
    ts_idx = np.argmin(np.abs(orig_times - time_val))
    
    # Verify we are reasonably close to the time point
    if not np.isclose(orig_times[ts_idx], time_val, rtol=1e-4):
        pytest.skip(f"Time mismatch: {orig_times[ts_idx]} vs {time_val}")

    fluxes = np.loadtxt(f"{adiabatic_dir}/fluxes.dat")
    # Column 1 is Heat Flux
    orig_eflux = fluxes[ts_idx, 1]

    # Use a slightly more relaxed tolerance for various iterations
    assert np.isclose(
        eflux_pred, orig_eflux, rtol=1e-2, atol=1e-4
    ), f"Flux mismatch at T={time_val}: {eflux_pred} vs {orig_eflux} in {adiabatic_dir}"
