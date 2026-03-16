import jax.numpy as jnp
import numpy as np
import os
import pytest
from gyaradax.integrals import (
    get_integrals,
    geom_tensors,
    calculate_phi_kinetic,
    calculate_fluxes_kinetic,
)
from gyaradax.utils import load_gkw_k_dump, load_gkw_dump, K_files


def test_flux_integral_shapes(adiabatic_geom, adiabatic_shape):
    geom = adiabatic_geom
    # Ensure adiabatic flag is set for consistent physics
    geom["adiabatic"] = jnp.array(1.0, dtype=jnp.float64)

    df = jnp.zeros(adiabatic_shape, dtype=jnp.complex128)
    phi, (pflux, eflux, vflux) = get_integrals(df, geom)

    ns, nkx, nky = adiabatic_shape[2:]
    assert phi.shape == (ns, nkx, nky)
    assert phi.dtype == jnp.complex128
    assert jnp.all(jnp.isfinite(phi))
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


# ── Kinetic electron integral tests ────────────────────────────────────────


def test_kinetic_geom_tensors_per_species(kinetic_geom):
    """geom_tensors must produce per-species Bessel and gamma arrays."""
    geom = kinetic_geom
    n_species = len(geom["mas"])
    assert n_species == 2, f"Expected 2 species, got {n_species}"

    # Compute tensors for each species independently
    bessel_list, gamma_list = [], []
    for isp in range(n_species):
        # Build a single-species geometry view
        sp_geom = dict(geom)
        for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
            sp_geom[k] = geom[k][isp : isp + 1]
        gt = geom_tensors(sp_geom)
        bessel_list.append(gt["bessel"])
        gamma_list.append(gt["gamma"])

    # Ion and electron Bessel functions must differ (different mass/charge)
    ion_bessel = bessel_list[0]
    elec_bessel = bessel_list[1]
    assert not jnp.allclose(
        ion_bessel, elec_bessel, atol=1e-6
    ), "Ion and electron Bessel J0 should differ due to mass/charge"

    # Electron gamma should be much closer to 1 than ion gamma
    # (smaller FLR due to low mass), though not exactly 1 at high k_perp
    elec_gamma = gamma_list[1]
    ion_gamma = gamma_list[0]
    elec_dev = float(jnp.max(jnp.abs(elec_gamma - 1.0)))
    ion_dev = float(jnp.max(jnp.abs(ion_gamma - 1.0)))
    assert elec_dev < ion_dev, (
        f"Electron FLR deviation ({elec_dev:.4f}) should be smaller "
        f"than ion ({ion_dev:.4f})"
    )


def test_kinetic_k_dump_loading(kinetic_dir, kinetic_shape):
    """Verify multi-species K-dump loading produces correct shapes."""
    ks = K_files(kinetic_dir)
    if len(ks) == 0:
        pytest.skip(f"No K files found in {kinetic_dir}")

    k_file = ks[0]
    n_species = 2
    df, info = load_gkw_dump(
        os.path.join(kinetic_dir, k_file), kinetic_shape, n_species=n_species
    )

    nvpar, nmu, ns, nkx, nky = kinetic_shape
    assert df.shape == (
        n_species,
        nvpar,
        nmu,
        ns,
        nkx,
        nky,
    ), f"Expected (2, {nvpar}, {nmu}, {ns}, {nkx}, {nky}), got {df.shape}"
    assert df.dtype == jnp.complex128

    # Both species should contain non-trivial data
    for isp in range(n_species):
        sp_norm = float(jnp.linalg.norm(df[isp]))
        assert sp_norm > 0, f"Species {isp} distribution is all zeros"

    assert info["time"] > 0, "Metadata time should be positive"


def test_kinetic_flux_shapes_per_species(kinetic_geom, kinetic_shape):
    """Flux integrals produce correct shapes when called per-species."""
    n_species = len(kinetic_geom["mas"])
    ns, nkx, nky = kinetic_shape[2:]

    for isp in range(n_species):
        sp_geom = dict(kinetic_geom)
        for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
            sp_geom[k] = kinetic_geom[k][isp : isp + 1]

        df_sp = jnp.zeros(kinetic_shape, dtype=jnp.complex128)
        phi_sp, (pflux_sp, eflux_sp, vflux_sp) = get_integrals(df_sp, sp_geom)

        assert phi_sp.shape == (ns, nkx, nky)
        assert pflux_sp.shape == ()
        assert eflux_sp.shape == ()
        assert vflux_sp.shape == ()


def test_kinetic_flux_species_differ(kinetic_dir, kinetic_geom, kinetic_shape):
    """Per-species flux calculations produce different results for ions vs electrons.

    Uses a shared (incorrect single-species) phi as a smoke test that the
    per-species weighting (mass, charge, vthrat) is applied distinctly.
    The absolute values are not validated here — that requires the full
    multi-species phi solver (Phase 5).
    """
    from gyaradax.integrals import calculate_fluxes

    n_species = 2
    ks = K_files(kinetic_dir)
    if len(ks) == 0:
        pytest.skip(f"No K files in {kinetic_dir}")

    k_file = ks[0]
    df_full = load_gkw_k_dump(
        os.path.join(kinetic_dir, k_file), kinetic_shape, n_species=n_species
    )

    # Use ion-only phi as a shared reference potential (not physically correct,
    # but sufficient to test that per-species flux weighting differs)
    sp0_geom = dict(kinetic_geom)
    for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
        sp0_geom[k] = kinetic_geom[k][0:1]
    from gyaradax.integrals import calculate_phi

    phi_shared = calculate_phi(geom_tensors(sp0_geom), df_full[0])

    efluxes = []
    for isp in range(n_species):
        sp_geom = dict(kinetic_geom)
        for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
            sp_geom[k] = kinetic_geom[k][isp : isp + 1]
        gt = geom_tensors(sp_geom)
        _, eflux, _ = calculate_fluxes(gt, df_full[isp], phi_shared)
        efluxes.append(float(eflux))

    # Ion and electron fluxes should not be identical
    assert not np.isclose(
        efluxes[0], efluxes[1], rtol=1e-3
    ), f"Ion and electron eflux should differ: {efluxes[0]:.6e} vs {efluxes[1]:.6e}"


@pytest.mark.parametrize("idx", [10, 50])
def test_kinetic_flux_integral_per_species_parity(
    kinetic_dir, kinetic_geom, kinetic_shape, idx
):
    """Verify per-species flux integrals match GKW reference for kinetic electrons.

    Uses the multi-species phi solver (kinetic quasineutrality) to compute
    the correct shared potential, then computes per-species fluxes.
    """
    geom = kinetic_geom
    n_species = 2
    ks = K_files(kinetic_dir)
    if idx >= len(ks):
        pytest.skip(f"Index {idx} out of range for {kinetic_dir}")

    k_file = ks[idx]
    df_full = load_gkw_k_dump(
        os.path.join(kinetic_dir, k_file), kinetic_shape, n_species=n_species
    )

    # Get timestamp from metadata
    k_dat_path = os.path.join(kinetic_dir, f"{k_file}.dat")
    if not os.path.exists(k_dat_path):
        pytest.skip(f"Metadata {k_dat_path} not found")

    time_val = None
    with open(k_dat_path, "r") as file:
        for line in file:
            parts = line.split("=")
            if parts[0].strip() == "TIME":
                time_val = float(parts[1].strip().strip(",").strip())
                break

    orig_times = np.loadtxt(os.path.join(kinetic_dir, "time.dat"))
    ts_idx = np.argmin(np.abs(orig_times - time_val))
    if not np.isclose(orig_times[ts_idx], time_val, rtol=1e-4):
        pytest.skip(f"Time mismatch: {orig_times[ts_idx]} vs {time_val}")

    fluxes_ref = np.loadtxt(os.path.join(kinetic_dir, "fluxes.dat"))
    assert (
        fluxes_ref.shape[1] == 6
    ), f"Expected 6 flux columns, got {fluxes_ref.shape[1]}"

    phi = calculate_phi_kinetic(geom, df_full)

    # phi sanity checks
    ns, nkx, nky = kinetic_shape[2:]
    assert phi.shape == (ns, nkx, nky)
    assert jnp.all(jnp.isfinite(phi))
    assert float(jnp.max(jnp.abs(phi))) > 0, "phi should be non-trivial"

    per_sp_fluxes = calculate_fluxes_kinetic(geom, df_full, phi)

    for isp in range(n_species):
        # Reference columns: [pflux_i, eflux_i, vflux_i, pflux_e, eflux_e, vflux_e]
        col_offset = isp * 3
        ref_eflux = fluxes_ref[ts_idx, col_offset + 1]
        sp_name = "ion" if isp == 0 else "electron"
        pred_eflux = float(per_sp_fluxes[isp, 1])

        assert np.isclose(pred_eflux, ref_eflux, rtol=1e-2, atol=1e-4), (
            f"{sp_name} eflux mismatch at T={time_val}: "
            f"{pred_eflux:.6e} vs {ref_eflux:.6e} in {kinetic_dir}"
        )
