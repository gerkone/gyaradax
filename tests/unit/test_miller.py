"""Miller geometry parity test: match GKW geom.dat to <1e-4."""
import os

import numpy as np
import pytest

from gyaradax.geometry import compute_geometry
from gyaradax.utils import load_geom_dat_file


GKW_CASE = os.path.join(
    os.path.dirname(__file__), "..", "data", "gkw_cases", "miller_ref"
)


def _rel_max(a, b):
    """Max |a-b| normalised by max |b|, robust to exact zeros in b."""
    scale = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(np.asarray(a).ravel() - np.asarray(b).ravel())) / scale)


def _miller_geom():
    # parameters from tests/data/gkw_cases/miller_ref/input.dat
    return compute_geometry(
        q=2.0, shat=1.0, eps=0.16,
        ns=105, nkx=1, nky=1, nvpar=16, nmu=8,
        vpar_max=3.0, nperiod=3, krhomax=0.3,
        signB=-1.0, Rref=100.0,
        geom_type="miller",
        kappa=1.4, delta=-0.3, square=0.2, Zmil=0.1,
        dRmil=-0.22, dZmil=-0.2,
        skappa=0.4, sdelta=0.8, ssquare=0.4,
        gradp=-0.2, gradp_type="alpha",
    )


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GKW_CASE, "geom.dat")),
    reason="GKW miller_ref/geom.dat not available",
)
def test_miller_matches_gkw_geom_dat():
    """Every scalar/array in GKW geom.dat matches gyaradax Miller to <1e-4."""
    gd = load_geom_dat_file(os.path.join(GKW_CASE, "geom.dat"))
    g = _miller_geom()

    dfun = np.asarray(g["dfun"])
    hfun = np.asarray(g["hfun"])
    ifun = np.asarray(g["ifun"])
    efun3 = np.asarray(g["efun_3x3"])

    checks = {
        "bn_G": np.asarray(g["bn"]),
        "F": np.asarray(g["ffun"]),
        "G": np.asarray(g["gfun"]),
        "Bt_frac": np.asarray(g["bt_frac"]),
        "R": np.asarray(g["rfun"]),
        "D_eps": dfun[:, 0], "D_zeta": dfun[:, 1], "D_s": dfun[:, 2],
        "H_eps": hfun[:, 0], "H_zeta": hfun[:, 1], "H_s": hfun[:, 2],
        "I_eps": ifun[:, 0], "I_zeta": ifun[:, 1], "I_s": ifun[:, 2],
        "E_eps_zeta": -np.asarray(g["efun"]),
        "E_eps_s": efun3[:, 0, 2],
        "E_zeta_s": efun3[:, 1, 2],
        "J": np.asarray(g["jfun"]),
        "K": np.asarray(g["kfun"]),
        "kthnorm": np.asarray(g["kthnorm"]),
        "R0": np.asarray(g["R0"]),
    }

    for name, gyr in checks.items():
        if name not in gd:
            continue
        err = _rel_max(gyr, np.asarray(gd[name]))
        assert err < 1e-4, f"{name}: rel err {err:.4e} > 1e-4"
