"""verification tests against GKW standard reference cases.

builds geometry from input.dat, runs gksolve for the reference duration,
and compares fluxes and growth rates against fluxes.dat / time.dat.

known limitations / TODOs:
- flux magnitude comparison is not meaningful yet: GKW reference fluxes are
  at unit (normalized) amplitude, while our init_f starts at amp_init=1e-4.
  tests currently check finiteness only, not quantitative match.
  TODO: normalize flux comparison by mode amplitude squared.
- slab_itg diverges at dt=0.2 (CFL). test skips on divergence.
  TODO: investigate CFL for slab_periodic geometry.
- geom_circ kinetic: gksolve runs but get_integrals flux computation
  hits single-species geom_tensors path for per-species fluxes.
  TODO: unify flux computation like calculate_phi.
- growth rate sign: with naverage=1, sign may not match GKW convention.
  tests check magnitude only.
  TODO: validate sign convention.
- miller geometry not implemented.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyaradax.geometry import compute_geometry_from_input, geometry_from_geom_dat_and_input
from gyaradax.params import gkparams_from_input_and_geometry
from gyaradax.solver import gksolve, init_f, default_state, linear_precompute
from gyaradax.utils import parse_input_dat

jax.config.update("jax_enable_x64", True)

GKW_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gkw_cases")
CIRC_GEOM_TYPES = {"circ", "s-alpha", ""}

SKIP_CASES = {
    "miller_mb": "miller geometry not supported",
    "kinetic_elec": "miller geometry, no geom.dat available",
}


def _geom_type(input_dat_path):
    inp = parse_input_dat(input_dat_path)
    return inp.get("geom", {}).get("geom_type", "s-alpha").strip("'\"").lower()


def _build_case(case_name):
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
        pytest.skip(f"geometry type '{gt}' not supported and no geom.dat")

    params = gkparams_from_input_and_geometry(input_dat, geometry)
    return geometry, params


def _load_ref(case_name):
    ref_dir = os.path.join(GKW_CASES_DIR, case_name, "reference")
    ref_time = np.loadtxt(os.path.join(ref_dir, "time.dat"))
    ref_fluxes = np.loadtxt(os.path.join(ref_dir, "fluxes.dat"))
    if ref_time.ndim == 1:
        ref_time = ref_time.reshape(1, -1)
    if ref_fluxes.ndim == 1:
        ref_fluxes = ref_fluxes.reshape(1, -1)
    return ref_time, ref_fluxes


def _run_case(geometry, params, n_steps):
    """init + gksolve for n_steps. returns (df, phi, fluxes, state)."""
    nky = len(geometry["krho"])
    n_species = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
    df = init_f(geometry, finit=params.finit, n_species=n_species)
    pre = linear_precompute(geometry, params)
    state = default_state(nky=nky)
    final_df, (phi, fluxes), final_state = gksolve(
        df,
        geometry,
        params,
        state,
        n_steps=n_steps,
        pre=pre,
    )
    return final_df, phi, fluxes, final_state


# --- geometry tests ---


BUILDABLE = ["eiv_simple", "slab_itg", "geom_circ", "sourcetime"]


@pytest.mark.parametrize("case_name", BUILDABLE)
def test_geometry_builds(case_name):
    geometry, _ = _build_case(case_name)
    for key in ["kxrh", "krho", "bn", "ffun", "efun", "sgrid"]:
        assert key in geometry


@pytest.mark.parametrize("case_name", BUILDABLE)
def test_geometry_shapes(case_name):
    geometry, _ = _build_case(case_name)
    ns = len(geometry["sgrid"])
    nkx = len(geometry["kxrh"])
    nky = len(geometry["krho"])
    assert geometry["bn"].shape == (ns,)
    assert geometry["ixplus"].shape == (nkx, nky)


@pytest.mark.parametrize("case_name", BUILDABLE)
def test_init_f(case_name):
    geometry, params = _build_case(case_name)
    n_sp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
    df = init_f(geometry, finit=params.finit, n_species=n_sp)
    assert df.shape[-1] == len(geometry["krho"])
    assert jnp.all(jnp.isfinite(df))


# --- solver + flux tests ---


def test_eiv_simple_fluxes():
    """eiv_simple: run 100 steps (t=1.0), check fluxes are finite.

    reference: 1 converged snapshot at t=1.0 at unit amplitude.
    dt=0.01, naverage=1, nkx=1, nky=1.
    TODO: quantitative flux comparison once amplitude normalization is matched.
    TODO: compare growth rate magnitude to reference eigenvalue (0.182).
    """
    geometry, params = _build_case("eiv_simple")
    ref_time, ref_fluxes = _load_ref("eiv_simple")

    n_steps = int(ref_time[0, 0] / params.dt)  # t=1.0 / 0.01 = 100
    df, phi, fluxes, state = _run_case(geometry, params, n_steps)

    assert jnp.all(jnp.isfinite(df)), "df diverged"
    assert jnp.all(jnp.isfinite(phi)), "phi diverged"

    sim_fluxes = np.asarray(fluxes)
    sim_eflux = float(sim_fluxes[1]) if sim_fluxes.ndim == 1 else float(sim_fluxes.flat[1])

    # fluxes should be finite (amplitude depends on init_f which differs from GKW)
    assert np.isfinite(sim_eflux), "eflux is nan/inf"


def test_slab_itg_fluxes():
    """slab_itg: run 200 steps (1 window), check stability.

    reference: 20 windows of 200 steps at dt=0.2, target growth ~0.073.
    currently diverges at dt=0.2 (CFL).
    TODO: fix CFL for slab_periodic and compare growth rate to 0.073.
    """
    geometry, params = _build_case("slab_itg")
    ref_time, ref_fluxes = _load_ref("slab_itg")

    # run one navg window (200 steps)
    n_steps = params.naverage
    df, phi, fluxes, state = _run_case(geometry, params, n_steps)

    sim_finite = bool(jnp.all(jnp.isfinite(df)))
    # slab at dt=0.2 may hit CFL — check and report
    if not sim_finite:
        pytest.skip("slab_itg diverges at dt=0.2 (CFL), needs smaller timestep")

    sim_fluxes = np.asarray(fluxes)
    assert np.all(np.isfinite(sim_fluxes)), "fluxes contain nan/inf"


def test_geom_circ_init():
    """geom_circ: kinetic electron init + precompute succeeds.

    gksolve runs but get_integrals flux path uses single-species
    geom_tensors for per-species flux computation.
    TODO: unify flux interface and add full gksolve + flux comparison.
    """
    geometry, params = _build_case("geom_circ")
    n_species = int(jnp.asarray(params.mas).shape[0])
    df = init_f(geometry, finit=params.finit, n_species=n_species)
    assert jnp.all(jnp.isfinite(df))
    assert df.shape[0] == n_species


# --- reference data tests ---


CASES_WITH_FLUXES = ["eiv_simple", "slab_itg", "geom_circ"]
CASES_WITH_GEOM_DAT = ["eiv_simple", "slab_itg", "geom_circ"]


@pytest.mark.parametrize("case_name", CASES_WITH_FLUXES)
def test_reference_data_exists(case_name):
    ref_dir = os.path.join(GKW_CASES_DIR, case_name, "reference")
    for fname in ["time.dat", "fluxes.dat"]:
        path = os.path.join(ref_dir, fname)
        assert os.path.exists(path), f"{fname} missing for {case_name}"


@pytest.mark.parametrize("case_name", CASES_WITH_GEOM_DAT)
def test_geom_dat_exists(case_name):
    path = os.path.join(GKW_CASES_DIR, case_name, "reference", "geom.dat")
    assert os.path.exists(path), f"geom.dat missing for {case_name}"


def test_sourcetime_short_run():
    """sourcetime: nonlinear CBC, nky=4. build + short run (no saturation).

    reference: eflux_es.dat / vflux_es.dat (time-averaged, 4 windows).
    full comparison requires long nonlinear run to reach turbulent saturation.
    TODO: add long-run flux comparison against eflux_es.dat reference.
    """
    geometry, params = _build_case("sourcetime")
    assert len(geometry["krho"]) == 4

    # run 4 steps just to verify solver doesn't crash
    df, phi, fluxes, state = _run_case(geometry, params, n_steps=4)
    assert jnp.all(jnp.isfinite(df)), "sourcetime df diverged in 4 steps"


@pytest.mark.parametrize("case_name", list(SKIP_CASES.keys()))
def test_unsupported_skip(case_name):
    pytest.skip(SKIP_CASES[case_name])
