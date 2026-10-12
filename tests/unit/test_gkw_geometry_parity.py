"""Geometry and linear physics against GKW runs outside the campaign regime.

Each directory under tests/data/gkw_cases/geometry_parity holds a small GKW run: negative
shear, signB = -1, s-alpha, Miller with shape derivatives and a one-mode ballooning run.
The geometry built from input.dat alone and the one of the converted YAML config must
match what GKW wrote, the connectivity must match kx_connect.dat, and negative shear
also checks the linear growth rates GKW found.
"""

import os
import shutil

import numpy as np
import pytest
from omegaconf import OmegaConf

from conftest import HAS_CUDA  # type: ignore[import-not-found]

from gyaradax.geometry import compute_geometry_from_config, compute_geometry_from_input
from gyaradax.io.gkw import gkw_to_yaml, load_geometry

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "gkw_cases", "geometry_parity")
CASES = ("negshat", "signb", "salpha", "miller", "ballooning")
KEYS = ("bn", "ffun", "gfun", "efun", "dfun", "hfun", "ifun", "little_g", "bt_frac", "rfun", "kxrh",
        "krho", "kthnorm", "sgrid", "ints", "vpgr", "mugr", "intvp", "intmu", "ixplus", "ixminus",
        "pos_par_grid_class", "s_shift", "kx_shift", "valid_shift")
# gkw growth.dat at t = 24 (ky index 1, 2); signB = -1 equals the signB = +1 eigenvalue
GKW_GROWTH = {"negshat": (0.132900, 0.300130), "signb": (0.302349, 0.513717)}


def _rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float).reshape(np.shape(a))
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _assert_matches(geom, ref, tol):
    for key in KEYS:
        if key in geom and key in ref:
            assert _rel(geom[key], ref[key]) < tol, f"{key}: rel err {_rel(geom[key], ref[key]):.2e}"


def _finite_ky_steps(geom):
    """kx index steps ixplus - ix over the connected finite-ky modes."""
    ixp = np.asarray(geom["ixplus"])
    ix = np.arange(ixp.shape[0])[:, None]
    connected = (ixp >= 0) & (np.arange(ixp.shape[1])[None, :] != int(geom["iyzero"]))
    return (ixp - ix)[connected]


def test_cases_cover_their_regime(tmp_path):
    geoms = {}
    for case in CASES:
        shutil.copy(os.path.join(ROOT, case, "input.dat"), tmp_path)
        geoms[case] = compute_geometry_from_input(str(tmp_path / "input.dat"))
    # negative shear and signB = -1 run the field line towards lower kx, positive shear upwards
    assert float(geoms["negshat"]["shat"]) < 0 and np.all(_finite_ky_steps(geoms["negshat"]) < 0)
    assert float(geoms["signb"]["signB"]) == -1.0 and np.all(_finite_ky_steps(geoms["signb"]) < 0)
    for case in ("salpha", "miller"):
        assert float(geoms[case]["shat"]) > 0 and np.all(_finite_ky_steps(geoms[case]) > 0)
    # s-alpha neglects finite epsilon: R = 1, G from the numerical derivative of B
    sa = geoms["salpha"]
    assert np.all(np.asarray(sa["rfun"]) == 1.0)
    bn, ds = np.asarray(sa["bn"]), float(sa["sgr_dist"])
    dbn = (np.roll(bn, 2) - 8 * np.roll(bn, 1) + 8 * np.roll(bn, -1) - np.roll(bn, -2)) / (12 * ds)
    np.testing.assert_allclose(np.asarray(sa["gfun"]), np.asarray(sa["ffun"]) * dbn / bn, rtol=1e-12)
    # miller: the psi-psi metric varies along s and enters k_perp
    gpp = np.asarray(geoms["miller"]["little_g"])[:, 2]
    assert np.ptp(gpp) > 1e-2 and not np.allclose(gpp, 1.0)
    # ballooning: one kx mode over three poloidal turns
    assert np.asarray(geoms["ballooning"]["kxrh"]).size == 1 and np.ptp(np.asarray(geoms["ballooning"]["sgrid"])) > 2


@pytest.mark.parametrize("case", CASES)
def test_loader_connectivity_matches_kx_connect(case):
    d = os.path.join(ROOT, case)
    if not os.path.exists(os.path.join(d, "kx_connect.dat")):
        pytest.skip("single kx mode")
    ref = load_geometry(d)
    rows = [l.split() for l in open(os.path.join(d, "kx_connect.dat")) if l.strip() and l.split()[0].isdigit()]
    for r in rows:
        imod, ix = int(r[0]) - 1, int(r[2]) - 1
        assert (int(ref["ixplus"][ix, imod]), int(ref["ixminus"][ix, imod])) == (int(r[3]) - 1, int(r[4]) - 1)


@pytest.mark.parametrize("case", CASES)
def test_geometry_from_input_matches_gkw(case, tmp_path):
    shutil.copy(os.path.join(ROOT, case, "input.dat"), tmp_path)
    _assert_matches(compute_geometry_from_input(str(tmp_path / "input.dat")), load_geometry(os.path.join(ROOT, case)), 1e-4)


@pytest.mark.parametrize("case", CASES)
def test_converted_config_geometry_matches_gkw(case, tmp_path):
    out = str(tmp_path / "case.yaml")
    gkw_to_yaml(os.path.join(ROOT, case), out)
    _assert_matches(compute_geometry_from_config(OmegaConf.load(out)), load_geometry(os.path.join(ROOT, case)), 1e-4)


@pytest.mark.parametrize("backend", ["jax", "cuda"])
@pytest.mark.parametrize("case", list(GKW_GROWTH))
def test_linear_growth_matches_gkw(case, backend):
    if backend == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    from gyaradax.solver.driver import gk_from_gkw_dir, gk_run

    df, geom, params, state, pre = gk_from_gkw_dir(os.path.join(ROOT, case), backend=backend)
    text = open(os.path.join(ROOT, case, "input.dat")).read()
    ntime = int(text.split("ntime = ")[1].split()[0])
    for _ in range(ntime):
        df, _, _, state = gk_run(df, geom, params, state, params.naverage, pre=pre)
    np.testing.assert_allclose(np.asarray(state.last_growth_rate)[1:3], GKW_GROWTH[case], rtol=1e-4)
