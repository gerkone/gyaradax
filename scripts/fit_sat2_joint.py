"""Joint (cn, flux_exponent) fit of the sat2 rule on the summary tables, Qi only.

One pair for the adiabatic and kinetic sets together, least squares on log(Q_rule / Qi_avg)
over the train split, scored on the test split. Rows: saturated, with QL weights; kinetic
rows collisionless unless --collisional.

    python scripts/fit_sat2_joint.py [--weight all|chain0] [--collisional] [--out fit.json]
"""

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from gyaradax.quasilinear.rules import CALIBRATION, dlnpdr_tglf, ql_flux_sat2
from gyaradax.quasilinear.table_loader import rule_inputs

SUMMARY = "/restricteddata/ukaea/gyrokinetics/preprocessed/summary"


def load(kind, weight, collisional):
    df = pd.read_parquet(f"{SUMMARY}/{kind}.parquet")
    df = df[(df["run_class"] == "saturated") & df["Qi_avg"].notna() & (df["Qi_avg"] > 0)]
    if kind == "kinetic" and not collisional:
        df = df[df["collisionality"].fillna(0.0) == 0.0]
    rows = []
    for _, r in df.iterrows():
        inp = rule_inputs(r.to_dict(), "Qi", chain0_weight=weight == "chain0")
        if inp is None:
            continue
        ky, g, w, _kp = inp
        adiabatic = kind == "adiabatic"
        rlt = r["rlt_i"] if adiabatic else np.array([r["rlt_i"], r["rlt_e"]])
        rln = r["rln"] if adiabatic else np.array([r["rln"], r["rln"]])
        dlnpdr = float(dlnpdr_tglf(rlt, rln, te_ti=r["te_ti"], adiabatic=adiabatic))
        # the rules take gyaradax's krho = lin_ky / kthnorm
        rows.append(dict(kind=kind, split=r["split"], Q=float(r["Qi_avg"]), ky=ky / r["kthnorm"], g=g, w=w,
                         kthnorm=float(r["kthnorm"]), dlnpdr=dlnpdr))
    return rows


def rule(r, cn, p):
    n = r["ky"].size
    return float(ql_flux_sat2(
        jnp.asarray(r["g"]), None, jnp.ones((1, n)), jnp.asarray(r["w"])[None, :],
        jnp.asarray(r["ky"]), jnp.zeros(1), None, None, cn=cn, dlnpdr=r["dlnpdr"],
        kthnorm=r["kthnorm"], flux_exponent=p,
    ))


def score(rows, cn, p):
    pred = np.array([rule(r, cn, p) for r in rows])
    q = np.array([r["Q"] for r in rows])
    return {"n": len(rows), "rmse": float(np.sqrt(np.mean((pred - q) ** 2))),
            "med_rel": float(np.median(np.abs(pred / q - 1)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="all", choices=["all", "chain0"])
    parser.add_argument("--collisional", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load("adiabatic", args.weight, False) + load("kinetic", args.weight, args.collisional)
    train = [r for r in rows if r["split"] == "train"]
    test = {k: [r for r in rows if r["split"] == "test" and r["kind"] == k] for k in ("adiabatic", "kinetic")}

    def resid(theta):
        return np.array([np.log(max(rule(r, np.exp(theta[0]), theta[1]), 1e-30) / r["Q"]) for r in train])

    fit = least_squares(resid, [np.log(0.7), 0.6])
    cn, p = float(np.exp(fit.x[0])), float(fit.x[1])
    stored = CALIBRATION["sat2"]
    report = {
        "args": vars(args),
        "n_train": {k: sum(r["kind"] == k for r in train) for k in ("adiabatic", "kinetic")},
        "fit": {"cn": cn, "flux_exponent": p, **{k: score(v, cn, p) for k, v in test.items()}},
        "stored": {**stored, **{k: score(v, stored["cn"], stored["flux_exponent"]) for k, v in test.items()}},
    }
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
