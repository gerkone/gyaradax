"""Fit the single SAT2 amplitude cn and score it against TGLF-SAT2.

gyx-sat2 here uses gyaradax's gamma(ky) and W(ky) with TGLF's MODELLED
kx_width, because no kx-resolved linear |phi|^2 is stored. The shipped rule
measures kx_width from the eigenmode, so this cn is for comparison, not for
the plugin.
"""
import json
import sys

import numpy as np

B0, B1 = 0.76, 1.22
KYMIN_C = 0.173 * np.sqrt(2.0)      # active for ALPHA_ZF < 0
KYCUT_C = 0.8
CZ2, CKY = 1.05, 3.0
SQCKY = np.sqrt(CKY)
RMAJ = 3.0                           # R/a used in the TGLF sweep


def kthnorm(q, eps):
    gzz = (1 / (2 * np.pi * (1 + eps))) ** 2 * (1 + (1 - eps**2) * (q / eps) ** 2)
    return np.sqrt(gzz)


def zonal_mixing(ky, gam, kycut, kymin):
    """TGLF get_zonal_mixing: capped, floored, quadratically refined peak."""
    rat = np.where(ky > 0, gam / np.maximum(ky, 1e-30), 0.0)
    nxt = np.concatenate([ky[1:], ky[-1:]])
    win = (nxt >= kymin) & (ky <= kycut)
    if not win.any():
        win = ky > 0
    j = int(np.argmax(np.where(win, rat, -np.inf)))
    j1 = int(np.max(np.where(win)[0]))
    n = ky.size
    if not (0 < j < j1):
        k, g = max(ky[j], kymin), gam[j]
        if ky[j] < kymin and n > 1:
            g = gam[0] + (gam[1] - gam[0]) / max(ky[1] - ky[0], 1e-30) * (kymin - ky[0])
        return max(k, 1e-30), g / max(k, 1e-30), j
    jm = min(max(j, 1), n - 2)
    f0, f1, f2 = rat[jm - 1], rat[jm], rat[jm + 1]
    k0, k2 = ky[jm - 1], ky[jm + 1]
    dky = max(k2 - k0, 1e-30)
    x1 = (ky[jm] - k0) / dky
    b = (f1 - f0 * (1 - x1 * x1) - f2 * x1 * x1) / max(x1 - x1 * x1, 1e-30)
    c = f2 - f0 - b
    quad = lambda x: f0 + b * x + c * x * x
    xmin = (kymin - k0) / dky if k0 < kymin else 0.0
    xmax = np.clip(-b / (2 * c) if abs(c) > 1e-30 else 1e30, 0.0, 1.0)
    if f0 >= f1:
        k = kymin if k0 < kymin else k0
        g = quad(xmin) * kymin if k0 < kymin else f0 * k0
    elif xmax <= xmin:
        k = kymin if xmin > 0 else k0
        g = quad(xmin) * kymin if xmin > 0 else f0 * k0
    else:
        k = k0 + dky * xmax
        g = quad(xmax) * k
    k = max(k, 1e-30)
    return k, g / k, j


def gamma_kymix(ky, gam, kymax, vzf, j):
    n = ky.size
    g = np.where(ky < kymax, gam, vzf * kymax + np.maximum(gam - CZ2 * vzf * ky, 0.0))
    out = g.copy()
    if j + 2 >= n:
        return out
    ky0 = ky[:, None]
    k1, k2 = ky[None, :-1], ky[None, 1:]
    g1, g2 = g[None, :-1], g[None, 1:]
    with np.errstate(all="ignore"):
        mix1 = ky0 * (np.arctan(SQCKY * (k2 / ky0 - 1)) - np.arctan(SQCKY * (k1 / ky0 - 1)))
        delta = (g2 - g1) / np.maximum(k2 - k1, 1e-30)
        mix2 = ky0 * mix1 + (ky0**2 / (2 * SQCKY)) * (
            np.log(CKY * (k2 - ky0) ** 2 + ky0**2) - np.log(CKY * (k1 - ky0) ** 2 + ky0**2))
        terms = (g1 - k1 * delta) * mix1 + delta * mix2
        idx = np.arange(n - 1)[None, :]
        ave = np.sum(np.where(idx >= j + 1, terms, 0.0), axis=1)
        lo, hi = ky[min(j + 1, n - 1)], ky[-1]
        norm = ky * (np.arctan(SQCKY * (hi / np.maximum(ky, 1e-30) - 1))
                     - np.arctan(SQCKY * (lo / np.maximum(ky, 1e-30) - 1)))
        mixed = np.where(np.abs(norm) > 1e-30, ave / np.where(np.abs(norm) > 1e-30, norm, 1), 0.0)
    sel = np.arange(n) >= j + 2
    out[sel] = mixed[sel]
    return out


def predict(row, geo, kx_width_model="tglf"):
    """Unscaled SAT2 flux: sum_ky W(ky) * intensity(ky). cn multiplies this."""
    ky = np.asarray(row["lin_ky"], float)
    gam = np.asarray(row["lin_gamma"], float)
    w = np.asarray(row["w_Qi"], float)
    # the table is in one convention (ky_convention == "krho", max krhomax),
    # which is already the k_theta*rho axis SAT2's thresholds are stated in.
    # no transform here. at inference the rule receives gyaradax's internal
    # krho = k_theta*rho / kthnorm and applies kthnorm itself, landing on the
    # same axis, so fit and inference see identical wavenumbers.
    m = np.isfinite(ky) & np.isfinite(gam) & np.isfinite(w) & (ky > 0)
    if m.sum() < 3:
        return np.nan
    ky, gam, w = ky[m], np.maximum(gam[m], 0.0), w[m]
    o = np.argsort(ky)
    ky, gam, w = ky[o], gam[o], w[o]
    if not (gam > 0).any():
        return np.nan

    gr, bg = geo["grad_r0"], geo["b_geo0"]
    kymax, vzf, j = zonal_mixing(ky, gam, KYCUT_C * gr, KYMIN_C * gr)
    gmix = gamma_kymix(ky, gam, kymax, vzf, j)

    kycut = B0 * kymax
    gq = bg / gr
    if kx_width_model == "measured":
        kw = row.get("kx_width")
        if kw is None:
            return np.nan
        kxw = np.asarray(kw, float)[m][o]
        live = np.isfinite(kxw) & (kxw > 0)
        if live.sum() < 3:
            return np.nan
        ky, gmix, w, kxw = ky[live], gmix[live], w[live], kxw[live]
    else:
        kxw = np.where(ky < kycut, kycut / gr, kycut / gr + B1 * (ky - kycut) * gq)
    d1 = (geo["bt0"] / bg) ** 4 / gr
    d2 = 1.0 / gq**2
    sgeo = np.where(ky < kycut, geo["sat_geo0"] * d1 * geo["sat_geo1"],
                    geo["sat_geo0"] * (d1 * geo["sat_geo1"] * kycut
                                       + (ky - kycut) * d2 * geo["sat_geo2"])
                    / np.maximum(ky, 1e-30))
    cnorm = 12.0 / max(float(row["dlnpdr_tglf"]), 4.0)
    inten = (1.0 / kymax) * cnorm * (gmix / (kxw * ky)) ** 2 * sgeo
    return float(np.sum(w * inten))


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


if __name__ == "__main__":
    SPLIT = sys.argv[4] if len(sys.argv) > 4 else "parquet"
    pts = json.load(open(sys.argv[1]))
    swp = json.load(open(sys.argv[2]))
    report = {}
    for ds in ("adiabatic", "kinetic"):
        rows = pts[ds]
        geos = {g["name"]: g for g in swp[ds] if "error" not in g}
        X, Y, S, QT, NM = [], [], [], [], []
        for r in rows:
            g = geos.get(r["name"])
            if g is None:
                continue
            x = predict(r, g)
            X.append(x); Y.append(r["Qi_avg"]); S.append(r["split"])
            te = float(r["te_ti"]) if r["te_ti"] else 1.0
            # Q_gkw = Q_tglf / (2 sqrt2 (Ti/Te)^2.5 (a/R)^2), a/R = 1/RMAJ
            QT.append(g["Qi_tglf"] * RMAJ**2 / (2 * np.sqrt(2.0)) * te**2.5)
            NM.append(r["name"])
        X, Y, QT = np.array(X), np.array(Y), np.array(QT)
        S = np.array(S)
        valid = np.isfinite(X) & (X > 0) & (Y > 0)
        if SPLIT == "stratified":
            te_m = stratified_test(np.where(valid, Y, np.nan)) & valid
            tr = valid & ~te_m
        else:
            tr = (S == "train") & valid
            te_m = (S == "test") & valid
        cn_log = float(np.exp(np.mean(np.log(Y[tr]) - np.log(X[tr]))))
        cn_lsq = float(np.sum(Y[tr] * X[tr]) / np.sum(X[tr] ** 2))
        # TGLF returns exactly 0 at a few stable points; those must not enter
        # the geometric-mean rescale or they dominate it
        qt_ok = tr & (QT > 0)
        n_tglf_zero = int((tr & ~(QT > 0)).sum())
        k_tglf = float(np.exp(np.mean(np.log(Y[qt_ok]) - np.log(QT[qt_ok]))))
        report[ds] = dict(
            n_train=int(tr.sum()), n_test=int(te_m.sum()),
            cn_log=cn_log, cn_lsq=cn_lsq, tglf_rescale=k_tglf,
            n_tglf_zero=n_tglf_zero,
            gyx_test=metrics(Y[te_m], cn_log * X[te_m]),
            gyx_test_lsqcn=metrics(Y[te_m], cn_lsq * X[te_m]),
            gyx_train=metrics(Y[tr], cn_log * X[tr]),
            tglf_test_raw=metrics(Y[te_m], QT[te_m]),
            tglf_test_rescaled=metrics(Y[te_m], k_tglf * QT[te_m]),
            tglf_train_raw=metrics(Y[tr], QT[tr]),
        )
    print(json.dumps(report, indent=1))
    json.dump(report, open(sys.argv[3], "w"), indent=1)
