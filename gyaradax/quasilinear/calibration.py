"""sat2 calibration on the summary tables.

`fit_sat2` fits the one (cn, flux_exponent) pair of `rules.ql_flux_sat2` for the adiabatic and
kinetic sets together; the result goes into `rules.CALIBRATION`.
"""

import numpy as np

SUMMARY = "/restricteddata/ukaea/gyrokinetics/preprocessed/summary"
GQ_DEFAULT = 0.826


def summary_rows(kind, collisional=False, summary=SUMMARY):
    """sat2 inputs and Qi of the saturated rows with settled QL weights in one summary table."""
    import pandas as pd

    rows = []
    for _, r in pd.read_parquet(f"{summary}/{kind}.parquet").iterrows():
        if r.run_class != "saturated" or not np.isfinite(r.get("Qi_avg", np.nan)):
            continue
        if r.get("lin_weight_settled") is False:
            continue
        if kind == "kinetic" and not collisional and (r.get("collisionality") or 0.0) > 0.0:
            continue
        if r.lin_weight_Qi is None or np.size(r.lin_weight_Qi) == 0:
            continue
        ky, g, w = (np.asarray(x, float) for x in (r.lin_ky, r.lin_gamma, r.lin_weight_Qi))
        n = min(ky.size, g.size, w.size)
        ky, g, w = ky[:n], g[:n], w[:n]
        m = np.isfinite(ky) & (ky > 0) & np.isfinite(g) & np.isfinite(w)
        if m.sum() < 4 or not (g[m] > 0).any():
            continue
        o = np.argsort(ky[m])
        gq = r.get("lin_gq")
        gq = GQ_DEFAULT if gq is None or not np.isfinite(gq) else float(gq)
        # lin_ky is k_theta*rho, the rule's physical axis when kthnorm = 1
        rows.append(dict(ky=ky[m][o], g=np.maximum(g[m], 0.0)[o], w=w[m][o],
                         dlnpdr=float(r.dlnpdr_tglf), Q=float(r.Qi_avg), gq=gq))
    return rows


def sat2_predict(rows, cn, flux_exponent, gate_sharpness=20.0):
    import jax.numpy as jnp

    from .rules import ql_flux_sat2

    return np.array([
        float(ql_flux_sat2(
            jnp.asarray(r["g"]), None, jnp.ones((1, r["ky"].size)), jnp.asarray(r["w"])[None, :],
            jnp.asarray(r["ky"]), jnp.zeros(1), None, None, cn=cn, dlnpdr=r["dlnpdr"],
            kthnorm=1.0, grad_r0=1.0, gq=r["gq"], flux_exponent=flux_exponent,
            gate_sharpness=gate_sharpness,
        ))
        for r in rows
    ])


def stratified_test(y, frac=0.10, nbin=10, seed=0):
    """Deterministic test mask: a fraction `frac` of each decile of log(y)."""
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
        if idx.size:
            k = max(1, int(round(frac * idx.size)))
            mask[rng.choice(idx, size=min(k, idx.size), replace=False)] = True
    return mask


def flux_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    y, p = y[m], p[m]
    return dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean((p - y) ** 2))), mean_error=float(np.mean(p - y)),
                rmse_log10=float(np.sqrt(np.mean(np.log10(p / y) ** 2))),
                med_rel=float(np.median(np.abs(p / y - 1))))


def fit_sat2(summary=SUMMARY, gate_sharpness=20.0, collisional=False):
    """Joint (cn, flux_exponent) of ql_flux_sat2 for the adiabatic and kinetic tables, Qi only.

    Test rows are 10% of each log-flux decile of each set; p runs over a 0.01 grid, cn is the
    log-mean ratio over the train rows, and the pair with the lowest train log10 RMSE wins.
    Returns the pair and the test metrics per set.
    """
    sets = {k: summary_rows(k, collisional, summary) for k in ("adiabatic", "kinetic")}
    y = {k: np.array([r["Q"] for r in v]) for k, v in sets.items()}
    best = None
    for p in np.arange(0.30, 1.35, 0.01):
        x = {k: sat2_predict(v, 1.0, p, gate_sharpness) for k, v in sets.items()}
        ok = {k: np.isfinite(x[k]) & (x[k] > 0) & (y[k] > 0) for k in sets}
        test = {k: stratified_test(np.where(ok[k], y[k], np.nan)) & ok[k] for k in sets}
        train = {k: ok[k] & ~test[k] for k in sets}
        xj = np.concatenate([x[k][train[k]] for k in sets])
        yj = np.concatenate([y[k][train[k]] for k in sets])
        cn = float(np.exp(np.mean(np.log(yj) - np.log(xj))))
        err = float(np.sqrt(np.mean((np.log10(cn * xj) - np.log10(yj)) ** 2)))
        if best is None or err < best[0]:
            best = (err, float(p), cn, x, test, train)
    _, p, cn, x, test, train = best
    return dict(
        cn=cn,
        flux_exponent=round(p, 2),
        n={k: {"train": int(train[k].sum()), "test": int(test[k].sum())} for k in sets},
        test={k: flux_metrics(y[k][test[k]], cn * x[k][test[k]]) for k in sets},
    )
