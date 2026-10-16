"""Export the parquet rows usable for SAT2 calibration to a JSON the TGLF sweep reads."""
import json
import sys

import numpy as np
import pandas as pd

D = "/restricteddata/ukaea/gyrokinetics/preprocessed/summary/"
COLS = ("name", "split", "lin_code", "rlt_i", "rlt_e", "rln", "te_ti", "q", "shat", "eps",
        "dlnpdr_tglf", "Qi_avg", "Qe_avg", "Gamma_avg")


def rows(name, kinetic):
    df = pd.read_parquet(D + name + ".parquet")
    has_w = df.lin_weight_Qi.apply(lambda v: v is not None and len(np.atleast_1d(v)) > 0)
    df = df[has_w & df.Qi_avg.notna() & (df.run_class == "saturated")]
    out = []
    for _, r in df.iterrows():
        rec = {c: (None if c not in r or pd.isna(r[c]) else
                   (r[c] if isinstance(r[c], str) else float(r[c]))) for c in COLS}
        rec["kinetic"] = kinetic
        rec["lin_ky"] = np.asarray(r.lin_ky, dtype=float).tolist()
        rec["lin_gamma"] = np.asarray(r.lin_gamma, dtype=float).tolist()
        kw = r.get("lin_kx_width")
        rec["kx_width"] = (None if kw is None or len(np.atleast_1d(kw)) == 0
                           else np.asarray(kw, dtype=float).tolist())
        for ch in ("Qi", "Qe", "Gamma"):
            w = r.get(f"lin_weight_{ch}")
            rec[f"w_{ch}"] = (None if w is None or len(np.atleast_1d(w)) == 0
                              else np.asarray(w, dtype=float).tolist())
        out.append(rec)
    return out


if __name__ == "__main__":
    data = {"adiabatic": rows("adiabatic", False), "kinetic": rows("kinetic", True)}
    out = sys.argv[1] if len(sys.argv) > 1 else "sat2_points.json"
    with open(out, "w") as fh:
        json.dump(data, fh)
    for k, v in data.items():
        sp = {}
        for r in v:
            sp[r["split"]] = sp.get(r["split"], 0) + 1
        print(f"{k}: {len(v)} rows  splits={sp}")
    print("wrote", out)
