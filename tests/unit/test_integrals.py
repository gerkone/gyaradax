import jax.numpy as jnp
import numpy as np
import os
import pytest
from conftest import read_dump_time
from gyaradax.integrals import (
    get_integrals,
    geom_tensors,
    calculate_phi,
    calculate_phi_kinetic,
    calculate_fluxes,
    calculate_fluxes_kinetic,
)
from gyaradax.utils import load_gkw_k_dump, load_gkw_dump, K_files


def test_flux_integral_shapes(adiabatic_geom, adiabatic_shape):
    geom = adiabatic_geom
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
        pytest.skip(f"index {idx} out of range for {adiabatic_dir}")

    k_file = ks[idx]
    df = load_gkw_k_dump(os.path.join(adiabatic_dir, k_file), adiabatic_shape)
    phi_pred, (pflux_pred, eflux_pred, vflux_pred) = get_integrals(df, geom)

    k_dat_path = os.path.join(adiabatic_dir, f"{k_file}.dat")
    if not os.path.exists(k_dat_path):
        pytest.skip(f"metadata {k_dat_path} not found")
    time_val = read_dump_time(k_dat_path)

    orig_times = np.loadtxt(os.path.join(adiabatic_dir, "time.dat"))
    ts_idx = np.argmin(np.abs(orig_times - time_val))
    if not np.isclose(orig_times[ts_idx], time_val, rtol=1e-4):
        pytest.skip(f"time mismatch: {orig_times[ts_idx]} vs {time_val}")

    fluxes = np.loadtxt(os.path.join(adiabatic_dir, "fluxes.dat"))
    ref_eflux = fluxes[ts_idx, 1]

    assert np.isclose(
        eflux_pred, ref_eflux, rtol=1e-2, atol=1e-4
    ), f"eflux mismatch at T={time_val}: {eflux_pred} vs {ref_eflux}"


def test_kinetic_geom_tensors_per_species(kinetic_geom):
    """per-species bessel and gamma must differ between ions and electrons."""
    geom = kinetic_geom
    n_species = len(geom["mas"])
    assert n_species == 2

    bessel_list, gamma_list = [], []
    for isp in range(n_species):
        sp_geom = dict(geom)
        for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
            sp_geom[k] = geom[k][isp : isp + 1]
        gt = geom_tensors(sp_geom)
        bessel_list.append(gt["bessel"])
        gamma_list.append(gt["gamma"])

    assert not jnp.allclose(bessel_list[0], bessel_list[1], atol=1e-6)

    elec_dev = float(jnp.max(jnp.abs(gamma_list[1] - 1.0)))
    ion_dev = float(jnp.max(jnp.abs(gamma_list[0] - 1.0)))
    assert elec_dev < ion_dev


def test_kinetic_k_dump_loading(kinetic_dir, kinetic_shape):
    ks = K_files(kinetic_dir)
    if len(ks) == 0:
        pytest.skip(f"no K files in {kinetic_dir}")

    df, info = load_gkw_dump(
        os.path.join(kinetic_dir, ks[0]), kinetic_shape, n_species=2
    )

    nvpar, nmu, ns, nkx, nky = kinetic_shape
    assert df.shape == (2, nvpar, nmu, ns, nkx, nky)
    assert df.dtype == jnp.complex128
    for isp in range(2):
        assert float(jnp.linalg.norm(df[isp])) > 0
    assert info["time"] > 0


def test_kinetic_flux_shapes_per_species(kinetic_geom, kinetic_shape):
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


def test_kinetic_flux_species_differ(kinetic_dir, kinetic_geom, kinetic_shape):
    """ion and electron fluxes must differ given the same phi."""
    ks = K_files(kinetic_dir)
    if len(ks) == 0:
        pytest.skip(f"no K files in {kinetic_dir}")

    df_full = load_gkw_k_dump(
        os.path.join(kinetic_dir, ks[0]), kinetic_shape, n_species=2
    )

    sp0_geom = dict(kinetic_geom)
    for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
        sp0_geom[k] = kinetic_geom[k][0:1]
    phi_shared = calculate_phi(geom_tensors(sp0_geom), df_full[0])

    efluxes = []
    for isp in range(2):
        sp_geom = dict(kinetic_geom)
        for k in ("mas", "tmp", "de", "signz", "vthrat", "rlt", "rln"):
            sp_geom[k] = kinetic_geom[k][isp : isp + 1]
        gt = geom_tensors(sp_geom)
        _, eflux, _ = calculate_fluxes(gt, df_full[isp], phi_shared)
        efluxes.append(float(eflux))

    assert not np.isclose(efluxes[0], efluxes[1], rtol=1e-3)


@pytest.mark.parametrize("idx", [10, 50])
def test_kinetic_flux_integral_per_species_parity(
    kinetic_dir, kinetic_geom, kinetic_shape, idx
):
    """per-species eflux matches gkw reference using kinetic quasineutrality."""
    ks = K_files(kinetic_dir)
    if idx >= len(ks):
        pytest.skip(f"index {idx} out of range")

    k_file = ks[idx]
    df_full = load_gkw_k_dump(
        os.path.join(kinetic_dir, k_file), kinetic_shape, n_species=2
    )

    k_dat_path = os.path.join(kinetic_dir, f"{k_file}.dat")
    if not os.path.exists(k_dat_path):
        pytest.skip(f"metadata not found: {k_dat_path}")
    time_val = read_dump_time(k_dat_path)

    orig_times = np.loadtxt(os.path.join(kinetic_dir, "time.dat"))
    ts_idx = np.argmin(np.abs(orig_times - time_val))
    if not np.isclose(orig_times[ts_idx], time_val, rtol=1e-4):
        pytest.skip(f"time mismatch: {orig_times[ts_idx]} vs {time_val}")

    fluxes_ref = np.loadtxt(os.path.join(kinetic_dir, "fluxes.dat"))
    assert fluxes_ref.shape[1] == 6

    phi = calculate_phi_kinetic(kinetic_geom, df_full)

    ns, nkx, nky = kinetic_shape[2:]
    assert phi.shape == (ns, nkx, nky)
    assert jnp.all(jnp.isfinite(phi))
    assert float(jnp.max(jnp.abs(phi))) > 0

    per_sp_fluxes = calculate_fluxes_kinetic(kinetic_geom, df_full, phi)

    for isp in range(2):
        col_offset = isp * 3
        ref_eflux = fluxes_ref[ts_idx, col_offset + 1]
        sp_name = "ion" if isp == 0 else "electron"
        pred_eflux = float(per_sp_fluxes[isp, 1])

        assert np.isclose(pred_eflux, ref_eflux, rtol=1e-2, atol=1e-4), (
            f"{sp_name} eflux mismatch at T={time_val}: "
            f"{pred_eflux:.6e} vs {ref_eflux:.6e}"
        )


@pytest.mark.parametrize("idx", [10, 50, 100])
def test_spectrum_parity_with_gkw(adiabatic_dir, adiabatic_geom, adiabatic_shape, idx):
    """ky_spec and kx_spec from phi must match GKW's kxspec/kyspec diagnostics.

    GKW conventions:
      ky_spec[ky] = ds * sum_{s,kx} |phi(s,kx,ky)|^2   (per-mode density)
      kx_spec[kx] = ds * sum_{s,ky} P(ky) * |phi|^2     (P=1 for ky=0, 2 for ky>0)
    """
    geom = adiabatic_geom
    geom["adiabatic"] = jnp.array(1.0, dtype=jnp.float64)

    ks = K_files(adiabatic_dir)
    if idx >= len(ks):
        pytest.skip(f"index {idx} out of range for {adiabatic_dir}")

    ref_kx_path = os.path.join(adiabatic_dir, "kxspec")
    ref_ky_path = os.path.join(adiabatic_dir, "kyspec")
    if not os.path.exists(ref_kx_path) or not os.path.exists(ref_ky_path):
        pytest.skip(f"kxspec/kyspec not found in {adiabatic_dir}")

    k_file = ks[idx]
    k_dat_path = os.path.join(adiabatic_dir, f"{k_file}.dat")
    if not os.path.exists(k_dat_path):
        pytest.skip(f"metadata {k_dat_path} not found")
    time_val = read_dump_time(k_dat_path)

    orig_times = np.loadtxt(os.path.join(adiabatic_dir, "time.dat"))
    ts_idx = np.argmin(np.abs(orig_times - time_val))
    if not np.isclose(orig_times[ts_idx], time_val, rtol=1e-4):
        pytest.skip(f"time mismatch: {orig_times[ts_idx]} vs {time_val}")

    df = load_gkw_k_dump(os.path.join(adiabatic_dir, k_file), adiabatic_shape)
    phi = calculate_phi(geom_tensors(geom), df)

    ds = float(jnp.asarray(geom["ints"])[0])
    nky = adiabatic_shape[-1]
    parseval_ky = jnp.array([1.0] + [2.0] * (nky - 1))
    phi_sq = jnp.abs(phi) ** 2

    pred_ky = np.array(jnp.sum(ds * phi_sq, axis=(0, 1)))
    pred_kx = np.array(jnp.sum(ds * phi_sq * parseval_ky[None, None, :], axis=(0, 2)))

    ref_ky = np.loadtxt(ref_ky_path)[ts_idx]
    ref_kx = np.loadtxt(ref_kx_path)[ts_idx]

    np.testing.assert_allclose(pred_ky, ref_ky, rtol=1e-4, atol=1e-10)
    # kx_spec sums over ky with Parseval weights; the lowest-kx edge mode
    # can differ by ~1-2% due to GKW diagnostic timing vs K-dump timing
    np.testing.assert_allclose(pred_kx, ref_kx, rtol=2e-2, atol=1e-6)
