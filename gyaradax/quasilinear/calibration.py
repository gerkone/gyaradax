"""Saturation-rule calibrations: the bundled JSON records and the fits that write them.

A calibration holds the rule it belongs to, the fitted `parameters` (always `cn`, plus e.g. sat2's
`flux_exponent`), the rule `options` held fixed during the fit, the `electrons` models it was
fitted on, and the `fit` method and date. The bundled ones live in `weights/`, one JSON each.
`fit_canonical` and `fit_sat2` fit one parameter set of their rule for the adiabatic and kinetic
sets together, and return the record with its test and cross-validation metrics alongside.
"""

import datetime
import functools
import json
import pathlib

import numpy as np

_DIR = pathlib.Path(__file__).parent / "weights"

# the calibration `auto` resolves to, per rule
DEFAULTS = {"canonical": "canonical_v2", "sat2": "sat2_v1"}


def names():
    return sorted(p.stem for p in _DIR.glob("*.json"))


@functools.lru_cache(maxsize=16)
def load(spec, rule=None):
    """Calibration from a bundled name, a JSON path, or `auto` (the rule's default); None if empty."""
    if not spec:
        return None
    if spec == "auto":
        spec = DEFAULTS.get(rule)
        if spec is None:
            return None
    path = _DIR / f"{spec}.json"
    with open(path if path.exists() else spec) as f:
        cal = json.load(f)
    if rule is not None and cal["rule"] != rule:
        raise ValueError(f"calibration {spec!r} is for rule {cal['rule']!r}, not {rule!r}")
    return cal


def rule_kwargs(cal):
    """Keyword arguments of the rule besides cn: the fitted parameters and the fixed options."""
    return {**cal.get("options", {}), **{k: v for k, v in cal["parameters"].items() if k != "cn"}}


def save(cal, path):
    with open(path, "w") as f:
        json.dump(cal, f, indent=1)
        f.write("\n")


SETS = ("adiabatic", "kinetic")


def summary_rows(kind, summary, collisional=False):
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
        if not (np.isfinite(r.get("lin_gq", np.nan)) and np.isfinite(r.get("lin_grad_r0", np.nan))):
            continue
        ky, g, w = (np.asarray(x, float) for x in (r.lin_ky, r.lin_gamma, r.lin_weight_Qi))
        n = min(ky.size, g.size, w.size)
        ky, g, w = ky[:n], g[:n], w[:n]
        m = np.isfinite(ky) & (ky > 0) & np.isfinite(g) & np.isfinite(w)
        if m.sum() < 4 or not (g[m] > 0).any():
            continue
        o = np.argsort(ky[m])
        # lin_ky is k_theta*rho, the rule's physical axis when kthnorm = 1
        rows.append(dict(ky=ky[m][o], g=np.maximum(g[m], 0.0)[o], w=w[m][o], Q=float(r.Qi_avg),
                         dlnpdr=float(r.dlnpdr_tglf), gq=float(r.lin_gq),
                         grad_r0=float(r.lin_grad_r0), row=r))
    return rows


def sat2_predict(rows, cn, options):
    import jax.numpy as jnp

    from .rules import ql_flux_sat2

    return np.array([
        float(ql_flux_sat2(
            jnp.asarray(r["g"]), None, jnp.ones((1, r["ky"].size)), jnp.asarray(r["w"])[None, :],
            jnp.asarray(r["ky"]), jnp.zeros(1), None, None, cn=cn, dlnpdr=r["dlnpdr"],
            kthnorm=1.0, grad_r0=r["grad_r0"], gq=r["gq"], **options,
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
    return dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                mean_error=float(np.mean(p - y)),
                rmse_log10=float(np.sqrt(np.mean(np.log10(p / y) ** 2))),
                med_rel=float(np.median(np.abs(p / y - 1))))


def _best(x_by_p, y, mask, exponents):
    """(p index, cn) minimising log10 RMSE over `mask`, cn the log-mean ratio there."""
    best = None
    for i in range(len(exponents)):
        x = x_by_p[i]
        ok = mask & np.isfinite(x) & (x > 0) & (y > 0)
        cn = float(np.exp(np.mean(np.log(y[ok]) - np.log(x[ok]))))
        err = float(np.sqrt(np.mean((np.log10(cn * x[ok]) - np.log10(y[ok])) ** 2)))
        if best is None or err < best[0]:
            best = (err, i, cn)
    return best[1], best[2]


def canonical_inputs(r):
    """ql_flux inputs of one summary row, from its raw linear-eigenmode columns."""
    from gyaradax.geometry.grids import _build_mode_label

    from .rules import kx_chain0

    def arr(key):
        return np.asarray(r[key], float).reshape(np.asarray(r[f"{key}_shape"]).astype(int))

    phi = arr("lin_phi")
    phi2 = phi[0] ** 2 + phi[1] ** 2
    ints, kxrh = arr("lin_ints"), arr("lin_kxrh")
    nkx, nky = phi2.shape[1:]
    gamma = np.asarray(r.lin_gamma, float)[:nky]
    labels = _build_mode_label(nkx, nky, int(r.lin_ikxspace))
    return dict(
        growth_rate=np.where(np.isfinite(gamma), gamma, 0.0),
        phi2=phi2,
        phi2_kxy=np.sum(phi2 * ints[:, None, None], axis=0),
        flux_kxy=arr("lin_flux_kxy"),
        # little_g carries kthnorm^2, so the rule's krho is lin_ky / kthnorm
        krho=np.asarray(r.lin_ky, float)[:nky] / float(r.kthnorm),
        kxrh=kxrh,
        little_g=arr("lin_little_g"),
        ds=float(np.mean(ints)),
        kx_chain=kx_chain0(labels, kxrh),
    )


def canonical_predict(rows, cn):
    import jax.numpy as jnp

    from .rules import ql_flux

    return np.array([
        float(ql_flux(cn=cn, **{k: jnp.asarray(v) for k, v in canonical_inputs(r["row"]).items()}))
        for r in rows
    ])


def _fit(x_by_p, y, kind, exponents, folds):
    """(p index, cn) on 90% of each set, with the test and `folds`-fold cross-validation metrics."""
    test = np.zeros(y.size, bool)
    for k in SETS:
        sel = kind == k
        test[sel] = stratified_test(y[sel])
    i, cn = _best(x_by_p, y, ~test, exponents)
    pred = cn * x_by_p[i]

    fold = np.random.default_rng(0).permutation(y.size) % folds
    cv = np.full(y.size, np.nan)
    for f in range(folds):
        j, c = _best(x_by_p, y, fold != f, exponents)
        cv[fold == f] = c * x_by_p[j][fold == f]

    metrics = dict(
        n_train={k: int(((kind == k) & ~test).sum()) for k in SETS},
        test={k: flux_metrics(y[(kind == k) & test], pred[(kind == k) & test]) for k in SETS},
        cross_validation={k: flux_metrics(y[kind == k], cv[kind == k]) for k in SETS},
    )
    return i, cn, metrics


def _record(rule, parameters, options, method):
    return dict(
        rule=rule,
        parameters=parameters,
        options=options,
        electrons=list(SETS),
        fit=dict(method=f"gyaradax.quasilinear.calibration.{method}",
                 date=datetime.date.today().isoformat()),
    )


def _sets(summary, collisional):
    sets = {k: summary_rows(k, summary, collisional) for k in SETS}
    kind = np.concatenate([[k] * len(v) for k, v in sets.items()])
    rows = [r for k in SETS for r in sets[k]]
    return rows, kind, np.array([r["Q"] for r in rows])


def fit_canonical(summary, collisional=False, folds=10):
    """(record, metrics) of ql_flux fitted jointly on the adiabatic and kinetic tables, Qi only.

    One cn, the log-mean ratio over 90% of each set (the test rows are 10% of each log-flux
    decile), with <k_perp^2> on the kx = 0 chain as `point_eval` evaluates it.
    """
    rows, kind, y = _sets(summary, collisional)
    x = canonical_predict(rows, 1.0)
    _, cn, metrics = _fit(x[None], y, kind, [1.0], folds)
    return _record("canonical", dict(cn=cn), {}, "fit_canonical"), metrics


def fit_sat2(summary, options=None, collisional=False, exponents=None, folds=10):
    """(record, metrics) of ql_flux_sat2 fitted on the adiabatic and kinetic tables, Qi only.

    p over `exponents`, cn the log-mean ratio over 90% of each set (the test rows are 10% of each
    log-flux decile), minimising log10 RMSE.
    """
    options = dict(options or {"gate_sharpness": 20.0, "kx_width_model": "tglf"})
    exponents = np.arange(0.30, 1.35, 0.01) if exponents is None else np.asarray(exponents)
    exponents = np.round(exponents, 4)
    rows, kind, y = _sets(summary, collisional)
    x_by_p = np.array([sat2_predict(rows, 1.0, {**options, "flux_exponent": p}) for p in exponents])
    i, cn, metrics = _fit(x_by_p, y, kind, exponents, folds)
    parameters = dict(cn=cn, flux_exponent=float(exponents[i]))
    return _record("sat2", parameters, options, "fit_sat2"), metrics
