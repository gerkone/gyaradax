import jax.numpy as jnp
import numpy as np
import os
import pytest
from gyaradax.integrals import get_integrals
from gyaradax.utils import load_gkw_k_dump, K_files


def test_flux_integral_shapes(adiabatic_geom, adiabatic_shape):
    geom = adiabatic_geom
    # Ensure adiabatic flag is set for consistent physics
    geom["adiabatic"] = jnp.array(1.0, dtype=jnp.float64)

    df = jnp.zeros(adiabatic_shape, dtype=jnp.complex128)
    phi, (pflux, eflux, vflux) = get_integrals(df, geom)

    ns, nkx, nky = adiabatic_shape[2:]
    assert phi.shape == (ns, nkx, nky)
    assert pflux.shape == ()
    assert eflux.shape == ()
    assert vflux.shape == ()


@pytest.mark.parametrize("idx", [10, 50, 100])
def test_flux_integral_real_data_parity(
    adiabatic_dir, adiabatic_geom, adiabatic_shape, idx
):
    geom = adiabatic_geom
    geom["adiabatic"] = jnp.array(1.0, dtype=jnp.float64)

    ks = K_files(adiabatic_dir)
    if idx >= len(ks):
        pytest.skip(f"Index {idx} out of range for {adiabatic_dir}")

    k_file = ks[idx]
    df = load_gkw_k_dump(os.path.join(adiabatic_dir, k_file), adiabatic_shape)

    phi_pred, (pflux_pred, eflux_pred, vflux_pred) = get_integrals(df, geom)

    # get the exact timestamp for this K file from its metadata
    time_val = None
    k_dat_path = os.path.join(adiabatic_dir, f"{k_file}.dat")
    if not os.path.exists(k_dat_path):
        pytest.skip(f"Metadata {k_dat_path} not found")

    with open(k_dat_path, "r") as file:
        for line in file:
            line_split = line.split("=")
            if line_split[0].strip() == "TIME":
                time_val = float(line_split[1].strip().strip(",").strip())
                break

    orig_times = np.loadtxt(os.path.join(adiabatic_dir, "time.dat"))
    ts_idx = np.argmin(np.abs(orig_times - time_val))

    if not np.isclose(orig_times[ts_idx], time_val, rtol=1e-4):
        pytest.skip(
            f"Time mismatch in reference data: {orig_times[ts_idx]} vs {time_val}"
        )

    fluxes = np.loadtxt(os.path.join(adiabatic_dir, "fluxes.dat"))
    # Column 1 is Heat Flux (eflux)
    orig_eflux = fluxes[ts_idx, 1]

    # Verify heat flux parity across iterations
    assert np.isclose(
        eflux_pred, orig_eflux, rtol=1e-2, atol=1e-4
    ), f"Flux mismatch at T={time_val}: {eflux_pred} vs {orig_eflux} in {adiabatic_dir}"
