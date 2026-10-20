"""Joint (cn, flux_exponent) fit of the sat2 rule on the summary tables, Qi only.

The rule is `ql_flux_sat2` itself, as the TORAX plugin evaluates it, with each row's own gq.
Rows: saturated, settled QL weights; kinetic rows collisionless unless --collisional. Test
rows: 10% per log-flux decile of each set. One (cn, p) pair for both sets: p on a 0.01 grid,
cn the log-mean ratio on the train rows, minimising log10 RMSE.

    python scripts/fit_sat2.py [--gate-sharpness 20] [--collisional] [--out fit.json]
"""

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

from gyaradax.quasilinear.rules import CALIBRATION, ql_flux_sat2

SUMMARY = "/restricteddata/ukaea/gyrokinetics/preprocessed/summary"
GQ_DEFAULT = 0.826


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    if m.sum() < 2:
        return dict(n=int(m.sum()))
    y, p = y[m], p[m]
    lr = np.log10(p / y)
    return dict(n=int(m.sum()),
                rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                rmse_log10=float(np.sqrt(np.mean(lr**2))),
                med_rel=float(np.median(np.abs(p / y - 1))),
                r2_log=float(1 - np.sum((np.log(p) - np.log(y)) ** 2)
                             / np.sum((np.log(y) - np.log(y).mean()) ** 2)),
                bias_log10=float(np.mean(lr)))


def stratified_test(y, frac=0.10, nbin=10, seed=0):
    """Deterministic ~frac test mask, stratified over deciles of log(flux).

    The parquet's own adiabatic test split is a purpose-built neurips26/OOD
    holdout spanning only 3x in flux against 1450x in train, so R^2 on it is
    meaningless. This gives a representative one.
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    mask = np.zeros(y.size, bool)
    ly = np.full(y.size, np.nan)
    ly[ok] = np.log10(y[ok])
    edges = np.nanquantile(ly[ok], np.linspace(0, 1, nbin + 1))
    edges[-1] += 1e-9
    rng = np.random.default_rng(seed)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where(ok & (ly >= lo) & (ly < hi))[0]
        if idx.size == 0:
            continue
        k = max(1, int(round(frac * idx.size)))
        mask[rng.choice(idx, size=min(k, idx.size), replace=False)] = True
    return mask


def load(kind, collisional):
    df = pd.read_parquet(f"{SUMMARY}/{kind}.parquet")
    rows = []
    for _, r in df.iterrows():
        if r.run_class != "saturated" or not np.isfinite(r.get("Qi_avg", np.nan)):
            continue
        if r.get("lin_weight_settled") is False:
            continue
        if kind == "kinetic" and not collisional and (r.get("collisionality") or 0.0) > 0.0:
            continue
        w = r.lin_weight_Qi
        if w is None or np.size(w) == 0:
            continue
        ky, g, w = (np.asarray(x, float) for x in (r.lin_ky, r.lin_gamma, w))
        n = min(ky.size, g.size, w.size)
        ky, g, w = ky[:n], g[:n], w[:n]
        m = np.isfinite(ky) & (ky > 0) & np.isfinite(g) & np.isfinite(w)
        if m.sum() < 4 or not (g[m] > 0).any():
            continue
        o = np.argsort(ky[m])
        gq = r.get("lin_gq")
        gq = GQ_DEFAULT if gq is None or not np.isfinite(gq) else float(gq)
        # lin_ky is k_theta*rho; with kthnorm = 1 the rule's physical axis is lin_ky itself
        rows.append(dict(ky=ky[m][o], g=np.maximum(g[m], 0.0)[o], w=w[m][o],
                         dlnpdr=float(r.dlnpdr_tglf), Q=float(r.Qi_avg), gq=gq))
    return rows


def predict(rows, cn, p, gate_sharpness):
    return np.array([
        float(ql_flux_sat2(
            jnp.asarray(r["g"]), None, jnp.ones((1, r["ky"].size)), jnp.asarray(r["w"])[None, :],
            jnp.asarray(r["ky"]), jnp.zeros(1), None, None, cn=cn, dlnpdr=r["dlnpdr"],
            kthnorm=1.0, grad_r0=1.0, gq=r["gq"], flux_exponent=p, gate_sharpness=gate_sharpness,
        ))
        for r in rows
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-sharpness", type=float, default=20.0)
    parser.add_argument("--collisional", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sets = {k: load(k, args.collisional) for k in ("adiabatic", "kinetic")}
    y = {k: np.array([r["Q"] for r in v]) for k, v in sets.items()}
    best = None
    for p in np.arange(0.30, 1.35, 0.01):
        x = {k: predict(v, 1.0, p, args.gate_sharpness) for k, v in sets.items()}
        ok = {k: np.isfinite(x[k]) & (x[k] > 0) & (y[k] > 0) for k in sets}
        test = {k: stratified_test(np.where(ok[k], y[k], np.nan)) & ok[k] for k in sets}
        train = {k: ok[k] & ~test[k] for k in sets}
        xj = np.concatenate([x[k][train[k]] for k in sets])
        yj = np.concatenate([y[k][train[k]] for k in sets])
        cn = float(np.exp(np.mean(np.log(yj) - np.log(xj))))
        err = float(np.sqrt(np.mean((np.log10(cn * xj) - np.log10(yj)) ** 2)))
        if best is None or err < best[0]:
            best = (err, float(p), cn, x, test, train)
    err, p, cn, x, test, train = best
    stored = CALIBRATION["sat2"]
    xs = {k: predict(v, stored["cn"], stored["flux_exponent"], args.gate_sharpness) for k, v in sets.items()}
    report = {
        "args": vars(args),
        "n": {k: {"train": int(train[k].sum()), "test": int(test[k].sum())} for k in sets},
        "fit": {"cn": cn, "flux_exponent": p, "train_rmse_log10": err,
                **{k: metrics(y[k][test[k]], cn * x[k][test[k]]) for k in sets}},
        "stored": {**stored, **{k: metrics(y[k][test[k]], xs[k][test[k]]) for k in sets}},
    }
    print(json.dumps(report, indent=1, default=float))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
