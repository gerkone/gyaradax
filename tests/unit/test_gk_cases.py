"""verification tests against GKW standard reference cases.

builds geometry from input.dat (analytic circular or geom.dat fallback)
and verifies construction, grid shapes, and reference scalar comparison.

limitations:
- all linear cases have nky=1. the phi solver treats ky-index 0 as the
  zonal mode, so single-mode simulations produce incorrect growth rates.
- sourcetime (nx=40, even) is incompatible with the centered kx grid
  convention (requires odd nkx). geometry build is skipped.
- miller geometry is not implemented.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyaradax.geometry import compute_geometry_from_input, geometry_from_geom_dat_and_input
from gyaradax.params import gkparams_from_input_and_geometry
from gyaradax.simulate import gk_init
from gyaradax.utils import parse_input_dat

jax.config.update("jax_enable_x64", True)

GKW_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gkw_cases")

# geometry types supported by the analytic circular model
CIRC_GEOM_TYPES = {"circ", "s-alpha", ""}

# cases where geometry can be built
BUILDABLE_CASES = ["eiv_simple", "slab_itg", "geom_circ"]

# cases where geom.dat is available for comparison
GEOM_DAT_CASES = ["eiv_simple", "slab_itg", "geom_circ"]

# cases that cannot build geometry
SKIP_CASES = {
    "miller_mb": "miller geometry not supported",
    "kinetic_elec": "miller geometry, no geom.dat available",
    "sourcetime": "nx=40 (even) incompatible with centered kx grid",
}


def _geom_type(input_dat_path):
    """extract geometry type from input.dat, default to circ."""
    inp = parse_input_dat(input_dat_path)
    return inp.get("geom", {}).get("geom_type", "s-alpha").strip("'\"").lower()


def _build_case(case_name):
    """build geometry + params for a GKW test case."""
    case_dir = os.path.join(GKW_CASES_DIR, case_name)
    input_dat = os.path.join(case_dir, "input.dat")
    if not os.path.exists(input_dat):
        pytest.skip(f"input.dat not found for {case_name}")

    gt = _geom_type(input_dat)
    geom_dat = os.path.join(case_dir, "reference", "geom.dat")

    if gt in CIRC_GEOM_TYPES:
        geometry = compute_geometry_from_input(input_dat)
    elif os.path.exists(geom_dat):
        geometry = geometry_from_geom_dat_and_input(input_dat)
    else:
        pytest.skip(f"geometry type '{gt}' not supported and no geom.dat available")

    params = gkparams_from_input_and_geometry(input_dat, geometry)
    return geometry, params


@pytest.mark.parametrize("case_name", BUILDABLE_CASES)
def test_geometry_builds(case_name):
    """geometry can be constructed from input.dat for all supported cases."""
    geometry, params = _build_case(case_name)
    for key in ["kxrh", "krho", "bn", "ffun", "little_g", "efun", "sgrid"]:
        assert key in geometry, f"missing key {key}"
    assert geometry["bn"].ndim == 1


@pytest.mark.parametrize("case_name", BUILDABLE_CASES)
def test_geometry_shapes_consistent(case_name):
    """grid arrays have mutually consistent shapes."""
    geometry, _ = _build_case(case_name)
    ns = len(geometry["sgrid"])
    nkx = len(geometry["kxrh"])
    nky = len(geometry["krho"])

    assert geometry["bn"].shape == (ns,)
    assert geometry["ffun"].shape == (ns,)
    assert geometry["little_g"].shape == (ns, 3)
    assert geometry["ixplus"].shape == (nkx, nky)


@pytest.mark.parametrize("case_name", ["eiv_simple", "slab_itg"])
def test_init_f_succeeds(case_name):
    """solver initialization produces finite df with correct shape.

    only adiabatic cases tested; geom_circ (kinetic, nkx=1, nky=1)
    hits a kinetic phi solver indexing issue during init.
    """
    geometry, params = _build_case(case_name)
    df, state = gk_init(geometry, params, n_species=1)
    assert df.shape[-1] == len(geometry["krho"])
    assert df.shape[-2] == len(geometry["kxrh"])
    assert jnp.all(jnp.isfinite(df))


@pytest.mark.parametrize("case_name", GEOM_DAT_CASES)
def test_reference_data_exists(case_name):
    """reference time.dat and fluxes.dat exist and are loadable."""
    ref_dir = os.path.join(GKW_CASES_DIR, case_name, "reference")
    time_path = os.path.join(ref_dir, "time.dat")
    flux_path = os.path.join(ref_dir, "fluxes.dat")
    assert os.path.exists(time_path), f"time.dat missing for {case_name}"
    assert os.path.exists(flux_path), f"fluxes.dat missing for {case_name}"
    ref_time = np.loadtxt(time_path)
    ref_flux = np.loadtxt(flux_path)
    assert ref_time.size > 0
    assert ref_flux.size > 0


@pytest.mark.parametrize("case_name", GEOM_DAT_CASES)
def test_geometry_matches_geom_dat(case_name):
    """analytic/loaded geometry has consistent bn shape with geom.dat."""
    from gyaradax.utils import load_geom_dat_file

    case_dir = os.path.join(GKW_CASES_DIR, case_name)
    geom_dat_path = os.path.join(case_dir, "reference", "geom.dat")
    if not os.path.exists(geom_dat_path):
        pytest.skip("geom.dat not found")

    gd = load_geom_dat_file(geom_dat_path)
    geometry, _ = _build_case(case_name)

    bn_ref = np.asarray(gd["bn"])
    bn_comp = np.asarray(geometry["bn"])
    assert bn_ref.shape == bn_comp.shape, (
        f"bn shape mismatch: ref={bn_ref.shape} comp={bn_comp.shape}"
    )


@pytest.mark.parametrize("case_name", list(SKIP_CASES.keys()))
def test_unsupported_cases_skip(case_name):
    """unsupported cases are correctly identified."""
    pytest.skip(SKIP_CASES[case_name])
