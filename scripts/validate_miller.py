"""Validate gyaradax Miller geometry against GKW's geom.dat reference.

Runs `compute_geometry(geom_type='miller', ...)` with the shape parameters
from `tests/data/gkw_cases/miller_ref/input.dat`, then compares every
geom.dat field to the GKW reference. Reports the max relative error per
field and flags anything above 1e-4.
"""
import os
import numpy as np
import jax.numpy as jnp

from gyaradax.geometry import compute_geometry
from gyaradax.utils import load_geom_dat_file


CASE = os.path.join(os.path.dirname(__file__), "..",
                    "tests", "data", "gkw_cases", "miller_ref")


def rel_err(a, b):
    """Max-normalised relative error: |a-b| / max(|b|)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        if a.size == 1 and b.size >= 1:
            a = np.full_like(b, float(a))
        elif b.size == 1 and a.size >= 1:
            b = np.full_like(a, float(b))
        else:
            return float("nan"), float("nan")
    scale = max(np.max(np.abs(b)), 1e-30)
    diff = np.abs(a - b) / scale
    return float(np.max(diff)), float(np.mean(diff))


def main():
    gd = load_geom_dat_file(os.path.join(CASE, "geom.dat"))

    # shape parameters from the GKW input (&GEOM namelist)
    geom = compute_geometry(
        q=2.0, shat=1.0, eps=0.16,
        ns=105, nkx=1, nky=1, nvpar=16, nmu=8,
        vpar_max=3.0, nperiod=3, krhomax=0.3,
        signB=-1.0,
        Rref=100.0,
        geom_type="miller",
        kappa=1.4, delta=-0.3, square=0.2, Zmil=0.1,
        dRmil=-0.22, dZmil=-0.2,
        skappa=0.4, sdelta=0.8, ssquare=0.4,
        gradp=-0.2, gradp_type="alpha",
    )

    # fields in geom.dat → gyaradax dict keys
    checks = [
        ("bn_G", "bn"),
        ("F", "ffun"),
        ("G", "gfun"),
        ("Bt_frac", "bt_frac"),
        ("R", "rfun"),
        ("D_eps", None),
        ("D_zeta", None),
        ("D_s", None),
        ("E_eps_zeta", None),
        ("E_eps_s", None),
        ("E_zeta_s", None),
        ("H_eps", None),
        ("H_zeta", None),
        ("H_s", None),
        ("I_eps", None),
        ("I_zeta", None),
        ("I_s", None),
        ("J", "jfun"),
        ("K", "kfun"),
        ("kthnorm", "kthnorm"),
        ("R0", "R0"),
    ]

    print(f"{'field':>12s}  {'max_rel':>12s}  {'mean_rel':>12s}  ok")
    all_ok = True
    for field, key in checks:
        if field not in gd:
            continue
        gk = np.asarray(gd[field])
        if field in ("D_eps", "D_zeta", "D_s"):
            j = {"D_eps": 0, "D_zeta": 1, "D_s": 2}[field]
            ga = np.asarray(geom["dfun"])[:, j]
        elif field in ("H_eps", "H_zeta", "H_s"):
            j = {"H_eps": 0, "H_zeta": 1, "H_s": 2}[field]
            ga = np.asarray(geom["hfun"])[:, j]
        elif field in ("I_eps", "I_zeta", "I_s"):
            j = {"I_eps": 0, "I_zeta": 1, "I_s": 2}[field]
            ga = np.asarray(geom["ifun"])[:, j]
        elif field == "E_eps_zeta":
            # geom.dat stores signed E_{eps,zeta}; gyaradax stores -efun[:,0,1]
            # as 'efun' for the flux calculation. Sign-flip back here.
            ga = -np.asarray(geom["efun"])
        elif field in ("E_eps_s", "E_zeta_s"):
            j = {"E_eps_s": (0, 2), "E_zeta_s": (1, 2)}[field]
            ga = np.asarray(geom["efun_3x3"])[:, j[0], j[1]]
        else:
            ga = np.asarray(geom[key])
        mx, mn = rel_err(ga, gk)
        ok = "✓" if mx < 1e-4 else "✗"
        if mx >= 1e-4:
            all_ok = False
        print(f"{field:>12s}  {mx:>12.4e}  {mn:>12.4e}  {ok}")
    print()
    print("PASS" if all_ok else "FAIL (some fields > 1e-4)")


if __name__ == "__main__":
    main()
