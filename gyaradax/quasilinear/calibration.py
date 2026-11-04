"""Saturation-rule calibrations: the bundled JSON records and the fits that write them.

A calibration holds the rule it belongs to, the fitted `parameters` (always `cn`, plus e.g. sat2's
`flux_exponent`), the rule `options` held fixed during the fit, the `electrons` models it was
fitted on, and the `fit` method and date. The bundled ones live in `weights/`, one JSON each.
`fit_canonical` and `fit_sat2` fit one parameter set of their rule on every summary table
(one per electron model) together, and return the record with its metrics alongside.

The summary tables are per-run parquets of nonlinear gyrokinetic runs: inputs, the linear run
(``lin_*``), and the nonlinear time averages, with ``test_set`` naming the held-out rows.
"""

import datetime
import functools
import json
import pathlib

import numpy as np

_DIR = pathlib.Path(__file__).parent / "weights"

# the calibration `auto` resolves to, per rule
DEFAULTS = {"canonical": "canonical_v2", "sat2": "sat2_v2"}


def names():
    return sorted(p.stem for p in _DIR.glob("*.json"))


@functools.lru_cache(maxsize=16)
def load(spec, rule=None):
    """A calibration record: a bundled name, a JSON path, or ``"auto"`` for the rule's default.

    Args:
        spec: Name, path or ``"auto"``.
        rule: Rule whose default ``"auto"`` picks.

    Returns:
        The record dict, or None when ``spec`` is empty.
    """
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


# summary schema: per flux channel, the nonlinear target, the linear weight, and the
# (species, channel) of the per-mode linear flux in lin_flux_kxy_sp
CHANNELS = {
    "Qi": ("Qi_fluxavg", "lin_weight_Qi", (0, 1)),
    "Qe": ("Qe_fluxavg", "lin_weight_Qe", (1, 1)),
    "Gamma": ("Gamma_fluxavg", "lin_weight_Gamma", (1, 0)),
}
TRAIN = "train"


def tables(summary):
    """The summary tables in a directory, by name (the file stem, e.g. ``adiabatic``)."""
    import pandas as pd

    return {p.stem: pd.read_parquet(p) for p in sorted(pathlib.Path(summary).glob("*.parquet"))}


def _finite(r, key):
    v = r.get(key)
    return v is not None and np.isfinite(v)


def summary_rows(summary, channels=("Qi", "Qe"), collisional=True, kinds=None):
    """Fit rows of the summary tables: one per run and flux channel the run has.

    A run counts when its linear growth rate and the channel's weight exist. Stable runs
    (no ``saturation_time``) are kept with their flux, so metrics can check the rule
    predicts little there; the fits themselves only use positive fluxes.

    Args:
        summary: Directory of the summary tables.
        channels: Flux channels to take, keys of :data:`CHANNELS`.
        collisional: Keep collisional runs.
        kinds: Table names to read; all when None.

    Returns:
        List of row dicts: ``kind``, ``channel``, ``split`` (``test_set`` or ``"train"``),
        ``stable``, ``Q``, the sat2 inputs ``ky, g, w, dlnpdr, gq, grad_r0`` and the table ``row``.
    """
    import pandas as pd

    rows = []
    for kind, table in tables(summary).items():
        if kinds is not None and kind not in kinds:
            continue
        for _, r in table.iterrows():
            if not collisional and (r.get("collisionality") or 0.0) > 0.0:
                continue
            if r.get("lin_gamma") is None or not all(_finite(r, k) for k in ("lin_gq", "lin_grad_r0", "dlnpdr_tglf")):
                continue
            split = r.get("test_set")
            for ch in channels:
                target, weight, _ = CHANNELS[ch]
                if target not in r or weight not in r or not _finite(r, target) or r[weight] is None:
                    continue
                ky, g, w = (np.asarray(x, float) for x in (r.lin_ky, r.lin_gamma, r[weight]))
                n = min(ky.size, g.size, w.size)
                ky, g, w = ky[:n], g[:n], w[:n]
                m = np.isfinite(ky) & (ky > 0) & np.isfinite(g) & np.isfinite(w)
                if m.sum() < 4:
                    continue
                o = np.argsort(ky[m])
                # lin_ky is k_theta rho, the rule's physical axis when kthnorm = 1
                rows.append(dict(
                    kind=kind, channel=ch, split=split if isinstance(split, str) else TRAIN,
                    stable=bool(pd.isna(r.get("saturation_time"))), Q=float(r[target]),
                    ky=ky[m][o], g=np.maximum(g[m], 0.0)[o], w=w[m][o], dlnpdr=float(r.dlnpdr_tglf),
                    gq=float(r.lin_gq), grad_r0=float(r.lin_grad_r0), row=r,
                ))
    return rows


def sat2_predict(rows, cn, options):
    """:func:`~gyaradax.quasilinear.rules.ql_flux_sat2` evaluated on summary rows.

    Args:
        rows: Rows from :func:`summary_rows`.
        cn: Amplitude calibration.
        options: Further keyword arguments of the rule.

    Returns:
        Predicted flux per row.
    """
    import jax
    import jax.numpy as jnp

    from .rules import ql_flux_sat2

    def one(g, w, ky, dlnpdr, grad_r0, gq):
        return ql_flux_sat2(g, None, jnp.ones((1, ky.size)), w[None, :], ky, jnp.zeros(1), None, None,
                            cn=cn, dlnpdr=dlnpdr, kthnorm=1.0, grad_r0=grad_r0, gq=gq, **options)

    batched = jax.vmap(one)
    out = np.zeros(len(rows))
    by_n = {}
    for i, r in enumerate(rows):
        by_n.setdefault(r["ky"].size, []).append(i)
    # rows sharing a ky grid run as one batch
    for idx in by_n.values():
        sel = [rows[i] for i in idx]
        args = [jnp.asarray(np.stack([r[k] for r in sel])) for k in ("g", "w", "ky")]
        args += [jnp.asarray([r[k] for r in sel]) for k in ("dlnpdr", "grad_r0", "gq")]
        out[idx] = np.asarray(batched(*args))
    return out


def flux_metrics(y, p):
    """Error metrics of predicted against true fluxes, over the rows where both are positive.

    Args:
        y: True fluxes.
        p: Predicted fluxes.

    Returns:
        Dict with ``n``, ``rmse``, ``mean_error``, ``rmse_log10`` and ``med_rel``.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    y, p = y[m], p[m]
    if not m.any():
        return dict(n=0)
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


def canonical_inputs(r, channel="Qi"):
    """ql_flux inputs of one summary row and flux channel, from its raw linear-eigenmode columns."""
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
    if channel == "Qi" and r.get("lin_flux_kxy") is not None:
        flux = arr("lin_flux_kxy")
    else:
        sp, c = CHANNELS[channel][2]
        flux = arr("lin_flux_kxy_sp")[sp, c]
    return dict(
        growth_rate=np.where(np.isfinite(gamma), gamma, 0.0),
        phi2=phi2,
        phi2_kxy=np.sum(phi2 * ints[:, None, None], axis=0),
        flux_kxy=flux,
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
        float(ql_flux(cn=cn, **{k: jnp.asarray(v) for k, v in canonical_inputs(r["row"], r["channel"]).items()}))
        for r in rows
    ])


def _groups(rows):
    return sorted({(r["kind"], r["channel"]) for r in rows})


def _fit(x_by_p, rows, exponents, folds):
    """(p index, cn) on the training rows, with cross-validation and held-out metrics.

    The folds split runs, so the channels of one run stay together. Each held-out split
    (``test_set``) is scored with the fit on all training rows. ``stable_over_1`` is the
    fraction of stable runs predicted above one gyro-Bohm unit.
    """
    y = np.array([r["Q"] for r in rows])
    split = np.array([r["split"] for r in rows])
    stable = np.array([r["stable"] for r in rows])
    train = (split == TRAIN) & ~stable
    i, cn = _best(x_by_p, y, train, exponents)
    pred = cn * x_by_p[i]

    runs = np.array([f'{r["kind"]}/{r["row"]["name"]}' for r in rows])
    uniq = np.unique(runs[split == TRAIN])
    fold_of = dict(zip(uniq, np.random.default_rng(0).permutation(uniq.size) % folds))
    fold = np.array([fold_of.get(n, -1) for n in runs])
    cv = np.full(y.size, np.nan)
    for f in range(folds):
        j, c = _best(x_by_p, y, train & (fold != f), exponents)
        sel = (split == TRAIN) & (fold == f)
        cv[sel] = c * x_by_p[j][sel]

    def by_group(p, mask):
        out = {}
        for kind, ch in _groups(rows):
            g = mask & np.array([(r["kind"], r["channel"]) == (kind, ch) for r in rows])
            if not g.any():
                continue
            m = flux_metrics(y[g & ~stable], p[g & ~stable])
            if (g & stable).any():
                m["stable_over_1"] = float(np.mean(p[g & stable] > 1.0))
            out[f"{kind}/{ch}"] = m
        return out

    metrics = dict(
        n_train={f"{k}/{c}": int((train & np.array([(r["kind"], r["channel"]) == (k, c) for r in rows])).sum())
                 for k, c in _groups(rows)},
        cross_validation=by_group(cv, split == TRAIN),
        **{s: by_group(pred, split == s) for s in sorted(set(split) - {TRAIN})},
    )
    return i, cn, metrics


def _record(rule, parameters, options, method, rows):
    return dict(
        rule=rule,
        parameters=parameters,
        options=options,
        electrons=sorted({r["kind"] for r in rows if r["split"] == TRAIN}),
        channels=sorted({r["channel"] for r in rows if r["split"] == TRAIN}),
        fit=dict(method=f"gyaradax.quasilinear.calibration.{method}",
                 date=datetime.date.today().isoformat()),
    )


def fit_canonical(summary, channels=("Qi", "Qe"), collisional=True, folds=10):
    """(record, metrics) of ql_flux fitted on every summary table together.

    One cn, the log-mean ratio over the training rows, with <k_perp^2> on the kx = 0 chain
    as `point_eval` evaluates it.
    """
    rows = summary_rows(summary, channels, collisional)
    x = canonical_predict(rows, 1.0)
    _, cn, metrics = _fit(x[None], rows, [1.0], folds)
    return _record("canonical", dict(cn=cn), {}, "fit_canonical", rows), metrics


def fit_sat2(summary, options=None, channels=("Qi", "Qe"), collisional=True, exponents=None, folds=10):
    """(record, metrics) of ql_flux_sat2 fitted on every summary table together.

    p over `exponents`, cn the log-mean ratio over the training rows, minimising log10 RMSE.
    """
    options = dict(options or {"gate_sharpness": 20.0, "kx_width_model": "tglf"})
    exponents = np.arange(0.30, 1.35, 0.01) if exponents is None else np.asarray(exponents)
    exponents = np.round(exponents, 4)
    rows = summary_rows(summary, channels, collisional)
    x_by_p = np.array([sat2_predict(rows, 1.0, {**options, "flux_exponent": p}) for p in exponents])
    i, cn, metrics = _fit(x_by_p, rows, exponents, folds)
    parameters = dict(cn=cn, flux_exponent=float(exponents[i]))
    return _record("sat2", parameters, options, "fit_sat2", rows), metrics
