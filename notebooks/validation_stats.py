"""Statistical validation of gyaradax against the reference runs of gerkone/gyaradax-tests.

Provides quantitative metrics for spectra, growth rates, flux averages,
and stationarity to complement the qualitative visual comparisons. References
are the ``nonlinear.json`` / ``linear.json`` diagnostics of the dataset, runs are
gyaradax output directories.
"""

import json
import os

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Stationarity tests (on potential energy or flux time series)
# ---------------------------------------------------------------------------


def integrated_autocorrelation_time(x, max_lag=None):
    """Estimate integrated autocorrelation time tau_int.

    tau_int = 1 + 2 * sum_{k=1}^{M} acf(k)

    where M is truncated when acf drops below a threshold or at max_lag.
    Returns tau_int (>= 1). The effective sample size is N / (2 * tau_int).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 10:
        return 1.0

    if max_lag is None:
        max_lag = min(n // 3, 500)

    xm = x - np.mean(x)
    var = np.var(x)
    if var < 1e-30:
        return 1.0

    tau = 1.0
    for k in range(1, max_lag + 1):
        acf_k = np.mean(xm[: n - k] * xm[k:]) / var
        if acf_k < 0.05:
            break
        tau += 2.0 * acf_k

    return max(tau, 1.0)


def standard_error(x):
    """Standard error of the mean with autocorrelation correction.

    Returns (mean, se, tau_int, n_eff).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if n > 1 else 0.0

    if n < 2 or sigma < 1e-30:
        return mu, 0.0, 1.0, float(n)

    tau = integrated_autocorrelation_time(x)
    n_eff = n / (2.0 * tau)
    se = sigma / np.sqrt(max(n_eff, 1.0))
    return mu, float(se), float(tau), float(n_eff)


def stationarity_test(x):
    """Test stationarity using ADF and KPSS.

    Returns dict with adf_stat, adf_pval, kpss_stat, kpss_pval, is_stationary.
    A series is deemed stationary if ADF rejects (p < 0.05) AND KPSS does not reject.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 20:
        return {
            "adf_stat": np.nan,
            "adf_pval": np.nan,
            "kpss_stat": np.nan,
            "kpss_pval": np.nan,
            "is_stationary": False,
        }

    try:
        from statsmodels.tsa.stattools import adfuller, kpss as kpss_test
    except ImportError:
        return {
            "adf_stat": np.nan,
            "adf_pval": np.nan,
            "kpss_stat": np.nan,
            "kpss_pval": np.nan,
            "is_stationary": None,
            "note": "statsmodels not installed; install for ADF/KPSS tests",
        }

    try:
        adf_result = adfuller(x, autolag="AIC")
        adf_stat, adf_p = float(adf_result[0]), float(adf_result[1])
    except Exception:
        adf_stat, adf_p = np.nan, np.nan

    try:
        kpss_result = kpss_test(x, regression="c", nlags="auto")
        kpss_stat, kpss_p = float(kpss_result[0]), float(kpss_result[1])
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan

    # stationary if ADF rejects non-stationarity AND KPSS does not reject stationarity
    is_stationary = not np.isnan(adf_p) and adf_p < 0.05 and not np.isnan(kpss_p) and kpss_p > 0.05

    return {
        "adf_stat": adf_stat,
        "adf_pval": adf_p,
        "kpss_stat": kpss_stat,
        "kpss_pval": kpss_p,
        "is_stationary": is_stationary,
    }


# ---------------------------------------------------------------------------
# Spectra metrics
# ---------------------------------------------------------------------------


def relative_l2(sim, ref):
    """Relative L2 error: ||sim - ref||_2 / ||ref||_2."""
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-30:
        return np.nan
    return float(np.linalg.norm(sim - ref) / ref_norm)


def log_relative_l2(sim, ref, floor=1e-30):
    """Relative L2 error in log10-space, appropriate for log-scale quantities.

    ||log10(sim) - log10(ref)||_2 / ||log10(ref)||_2

    Spectra span orders of magnitude, so linear L2 is dominated by the peak.
    Log-space L2 weights all decades equally, matching visual perception on
    log-scale plots.
    """
    s = np.maximum(np.asarray(sim, dtype=np.float64), floor)
    r = np.maximum(np.asarray(ref, dtype=np.float64), floor)
    log_s = np.log10(s)
    log_r = np.log10(r)
    ref_norm = np.linalg.norm(log_r)
    if ref_norm < 1e-30:
        return np.nan
    return float(np.linalg.norm(log_s - log_r) / ref_norm)


def pearson_corr(sim, ref):
    """Pearson correlation between two spectral vectors.

    Returns (r, p_value).
    """
    if len(sim) < 3 or np.std(sim) < 1e-30 or np.std(ref) < 1e-30:
        return np.nan, np.nan
    r, p = stats.pearsonr(sim, ref)
    return float(r), float(p)


def ks_test_spectra(sim_spec, ref_spec):
    """KS statistic on normalized spectra (treated as discrete CDFs).

    Normalizes both spectra to sum to 1, constructs CDFs, and returns
    the max absolute CDF difference. No p-value: with discrete k-grids
    the asymptotic KS p-value is not meaningful; the statistic itself
    is a [0,1] distance metric (0 = identical shapes).

    Returns ks_statistic (float).
    """
    s = np.asarray(sim_spec, dtype=np.float64).copy()
    r = np.asarray(ref_spec, dtype=np.float64).copy()
    s = np.maximum(s, 0.0)
    r = np.maximum(r, 0.0)

    s_sum, r_sum = s.sum(), r.sum()
    if s_sum < 1e-30 or r_sum < 1e-30:
        return np.nan

    cdf_sim = np.cumsum(s / s_sum)
    cdf_ref = np.cumsum(r / r_sum)
    return float(np.max(np.abs(cdf_sim - cdf_ref)))


# ---------------------------------------------------------------------------
# Growth rate metrics
# ---------------------------------------------------------------------------


def growth_rate_stats(
    sim_growth_2d,
    ref_growth_2d,
    sim_ky_spec=None,
    ref_ky_spec=None,
    avg_window_sim=80,
    avg_window_ref=240,
):
    """Compute mean +/- std of time-averaged growth rate profiles.

    Scalar summaries are energy-weighted over ky modes (excluding zonal ky=0)
    to avoid noisy high-ky modes near the noise floor dominating the average.

    Parameters
    ----------
    sim_growth_2d : (n_runs, n_time, nky) or (n_time, nky) array
        Simulation growth rates.
    ref_growth_2d : (n_time, nky) array or None
        Reference growth rates.
    sim_ky_spec : (nky,) array or None
        Time-averaged ky spectrum from sim, used as energy weights.
    ref_ky_spec : (nky,) array or None
        Time-averaged ky spectrum from ref, used as energy weights.
    avg_window_sim, avg_window_ref : int
        Number of trailing windows to average over.

    Returns
    -------
    dict with sim_mean, sim_std (per-ky), ref_mean, ref_std (per-ky),
    and energy-weighted scalar summaries.
    """
    sim = np.asarray(sim_growth_2d)
    if sim.ndim == 2:
        sim = sim[np.newaxis]  # (1, n_time, nky)

    n_runs, n_time, nky = sim.shape
    w_sim = min(avg_window_sim, n_time)

    # per-run time average -> (n_runs, nky), then stats across runs
    per_run = np.mean(sim[:, -w_sim:, :], axis=1)  # (n_runs, nky)
    sim_mean = np.mean(per_run, axis=0)  # (nky,)
    sim_std = np.std(per_run, axis=0) if n_runs > 1 else np.std(sim[0, -w_sim:, :], axis=0)

    # energy-weighted scalar summary (skip ky=0 zonal mode)
    def _weighted_scalar(profile, weights):
        w = np.asarray(weights).copy()
        w[0] = 0.0  # exclude zonal mode
        w_sum = w.sum()
        if w_sum < 1e-30:
            return float(np.mean(profile[1:])), float(np.mean(np.abs(profile[1:])))
        return float(np.average(profile, weights=w)), float(np.average(np.abs(profile), weights=w))

    sim_weights = sim_ky_spec if sim_ky_spec is not None else np.ones(nky)
    sim_scalar, sim_scalar_abs = _weighted_scalar(sim_mean, sim_weights)
    sim_std_scalar, _ = _weighted_scalar(sim_std, sim_weights)

    result = {
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "sim_scalar_mean": sim_scalar,
        "sim_scalar_std": sim_std_scalar,
    }

    if ref_growth_2d is not None:
        ref = np.asarray(ref_growth_2d)
        w_ref = min(avg_window_ref, ref.shape[0])
        ref_avg = ref[-w_ref:]
        ref_mean = np.mean(ref_avg, axis=0)
        ref_std = np.std(ref_avg, axis=0)

        ref_weights = ref_ky_spec if ref_ky_spec is not None else np.ones(ref_mean.shape[0])
        ref_scalar, _ = _weighted_scalar(ref_mean, ref_weights)
        ref_std_scalar, _ = _weighted_scalar(ref_std, ref_weights)

        result.update(
            {
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "ref_scalar_mean": ref_scalar,
                "ref_scalar_std": ref_std_scalar,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Flux comparison (with autocorrelation-corrected standard error)
# ---------------------------------------------------------------------------


def flux_comparison(sim_ts, ref_ts):
    """Compare flux time series using autocorrelation-corrected standard error.

    Per time series: compute mu +/- SE (accounting for autocorrelation via
    integrated autocorrelation time). Then Z-test on |mu_sim - mu_ref|.

    Parameters
    ----------
    sim_ts : 1D array — flux time series from simulation (last avg_sim windows).
    ref_ts : 1D array — flux time series from reference (last avg_ref windows).

    Returns
    -------
    dict with sim_mean, sim_se, ref_mean, ref_se, z_score, z_pval, rel_diff.
    """
    sim_mu, sim_se, sim_tau, sim_neff = standard_error(sim_ts)
    ref_mu, ref_se, ref_tau, ref_neff = standard_error(ref_ts)

    # SE of the difference and Z-test
    se_diff = float(np.sqrt(sim_se**2 + ref_se**2))
    if se_diff > 1e-30:
        z = abs(sim_mu - ref_mu) / se_diff
        z_p = float(2.0 * (1.0 - stats.norm.cdf(z)))  # two-sided
    else:
        z, z_p = 0.0, 1.0

    # relative difference
    denom = max(abs(ref_mu), 1e-30)
    rel_diff = abs(sim_mu - ref_mu) / denom

    return {
        "sim_mean": sim_mu,
        "sim_se": sim_se,
        "ref_mean": ref_mu,
        "ref_se": ref_se,
        "se_diff": se_diff,
        "z_score": float(z),
        "z_pval": float(z_p),
        "rel_diff": float(rel_diff),
    }


# ---------------------------------------------------------------------------
# Per-configuration comparison (master function)
# ---------------------------------------------------------------------------


FLUX_NAMES = ("particle", "heat", "momentum")
FLUX_KEYS = ("pflux", "eflux", "vflux")


def reference_cases(root):
    """(label, directory) of every case of the dataset under root that has JSON diagnostics."""
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    cases = [(f"{kind}/{name}", os.path.join(root, kind, name)) for kind, v in manifest["cases"].items() for name in v]
    index = os.path.join(root, "long_horizon", "index.json")
    if os.path.exists(index):
        cases += [(f"long_horizon/{n}", os.path.join(root, "long_horizon", n)) for n in json.load(open(index))["cases"]]
    return [(label, d) for label, d in cases if os.path.isdir(d)]


def load_reference(case_dir, kind="nonlinear"):
    path = os.path.join(case_dir, f"{kind}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def load_run(out_dir):
    """Time traces of a gyaradax output directory, fluxes as (time, species, 3)."""
    try:
        f = np.load(os.path.join(out_dir, "fluxes.npz"))
    except FileNotFoundError:
        return None
    fluxes = f["fluxes"]
    run = {"time": f["time"], "fluxes": fluxes[:, None, :] if fluxes.ndim == 2 else fluxes}
    for name, key in (("kyspec", "ky_spec"), ("kxspec", "kx_spec"), ("growth", "growth")):
        try:
            run[key] = np.load(os.path.join(out_dir, f"{name}.npz"))[key]
        except (FileNotFoundError, KeyError):
            pass
    return run


def run_window(run_time, ref):
    """The reference window, or the same fraction at the end of a run that does not reach it."""
    t0, t1 = ref["average"]["window"]
    if run_time[-1] >= t1 - 1e-6:
        return t0, t1
    frac = (t1 - t0) / (ref["time"][-1] - ref["time"][0])
    return run_time[-1] - frac * (run_time[-1] - run_time[0]), run_time[-1]


def _in(t, window):
    return (t >= window[0] - 1e-9) & (t <= window[1] + 1e-9)


def compare_config(sim_dirs, ref, species=0):
    """Run all statistical comparisons for one configuration group.

    Parameters
    ----------
    sim_dirs : list of str
        Paths to gyaradax output directories (one or more runs of the same config).
    ref : dict
        Reference ``nonlinear.json`` of the case.
    species : int
        Species index of the flux comparison.

    Returns
    -------
    dict with all comparison results, or None when no run reaches the reference window.
    """
    t_start = ref["average"]["window"][0]
    runs = [r for r in (load_run(d) for d in sim_dirs) if r is not None and r["time"][-1] >= t_start]
    if not runs:
        return None
    rt = np.asarray(ref["time"])
    rwin = ref["average"]["window"]
    rmask = _in(rt, rwin)
    results = {}

    if "ky_spec" in runs[0]:
        results["stationarity"] = stationarity_test(np.sum(runs[0]["ky_spec"], axis=1))

    def sim_avg(key):
        per_run = [np.mean(r[key][_in(r["time"], run_window(r["time"], ref))], axis=0) for r in runs if key in r]
        return np.mean(per_run, axis=0) if per_run else None

    sim_ky_avg = sim_avg("ky_spec")
    sim_kx_avg = sim_avg("kx_spec")
    ref_ky_spec = np.mean(np.asarray(ref["ky_spectrum"])[rmask], axis=0)
    ref_kx_spec = np.mean(np.asarray(ref["kx_spectrum"])[rmask], axis=0)
    for key, sim_spec, ref_spec in (("ky_spec", sim_ky_avg, ref_ky_spec), ("kx_spec", sim_kx_avg, ref_kx_spec)):
        if sim_spec is not None and len(sim_spec) == len(ref_spec):
            pr_r, pr_p = pearson_corr(sim_spec, ref_spec)
            results[key] = {
                "ks_stat": ks_test_spectra(sim_spec, ref_spec),
                "pearson_r": pr_r,
                "pearson_p": pr_p,
                "log_rel_l2": log_relative_l2(sim_spec, ref_spec),
            }

    growth_runs = [r["growth"][_in(r["time"], run_window(r["time"], ref))] for r in runs if "growth" in r]
    if growth_runs and "growth" in ref:
        n = min(len(g) for g in growth_runs)
        ref_growth = np.asarray(ref["growth"])[rmask]
        results["growth"] = growth_rate_stats(
            np.stack([g[-n:] for g in growth_runs]),
            ref_growth,
            sim_ky_spec=sim_ky_avg,
            ref_ky_spec=ref_ky_spec,
            avg_window_sim=n,
            avg_window_ref=len(ref_growth),
        )

    sp = ref["species"][species]
    sim_windows = np.concatenate([r["fluxes"][_in(r["time"], run_window(r["time"], ref)), species] for r in runs])
    for i, (name, key) in enumerate(zip(FLUX_NAMES, FLUX_KEYS)):
        ref_ts = np.asarray(ref["fluxes"][sp][name])[rmask]
        # zero by construction, e.g. the particle flux with adiabatic electrons
        if np.abs(ref_ts).max() <= 1e-12:
            continue
        results[key] = flux_comparison(sim_windows[:, i], ref_ts)
    return results


def compare_linear(sim_dir, ref, min_fraction=0.25):
    """Growth rates at the end of a linear run against the converged reference, per ky.

    The summary covers the ky whose reference growth rate exceeds ``min_fraction`` of
    the maximum; marginal and damped ky converge slowly and are less well defined.
    """
    try:
        g = np.load(os.path.join(sim_dir, "growth.npz"))
    except FileNotFoundError:
        return None
    sim = np.asarray(g["growth"])[-1]
    gamma = np.asarray(ref["gamma"])
    mask = gamma > min_fraction * gamma.max()
    rel = np.abs(sim - gamma) / np.maximum(np.abs(gamma), 1e-12)
    return {
        "ky": np.asarray(ref["ky"]),
        "sim_gamma": sim,
        "ref_gamma": gamma,
        "rel_diff": rel,
        "max_rel_diff": float(rel[mask].max()) if mask.any() else float("nan"),
        "t_end": float(g["time"][-1]),
    }


# ---------------------------------------------------------------------------
# Aggregation across configurations
# ---------------------------------------------------------------------------


def aggregate_results(all_config_results):
    """Aggregate per-config results across all trajectories.

    Parameters
    ----------
    all_config_results : dict[str, dict]
        Mapping config_name -> compare_config() output.

    Returns
    -------
    dict with mean +/- std of each metric across configs.
    """
    valid = {k: v for k, v in all_config_results.items() if v is not None}
    if not valid:
        return {}

    agg = {}

    # stationarity
    stat_vals = [v["stationarity"] for v in valid.values() if "stationarity" in v]
    if stat_vals:
        n_stationary = sum(1 for s in stat_vals if s.get("is_stationary"))
        n_tested = sum(1 for s in stat_vals if s.get("is_stationary") is not None)
        agg["stationarity"] = {"n_stationary": n_stationary, "n_tested": n_tested}

    # spectra metrics
    for spec_key in ("ky_spec", "kx_spec"):
        vals = [v[spec_key] for v in valid.values() if spec_key in v]
        if not vals:
            continue
        agg[spec_key] = {}
        for metric in ("ks_stat", "pearson_r", "pearson_p", "log_rel_l2"):
            arr = np.array([v[metric] for v in vals if not np.isnan(v.get(metric, np.nan))])
            if len(arr) > 0:
                agg[spec_key][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "n": len(arr),
                }

    # growth rate
    growth_vals = [v["growth"] for v in valid.values() if "growth" in v]
    if growth_vals:
        sim_means = [g["sim_scalar_mean"] for g in growth_vals]
        sim_stds = [g["sim_scalar_std"] for g in growth_vals]
        agg["growth"] = {
            "sim_mean": {"mean": float(np.mean(sim_means)), "std": float(np.std(sim_means))},
            "sim_std": {"mean": float(np.mean(sim_stds)), "std": float(np.std(sim_stds))},
        }
        ref_means = [g["ref_scalar_mean"] for g in growth_vals if "ref_scalar_mean" in g]
        ref_stds = [g["ref_scalar_std"] for g in growth_vals if "ref_scalar_std" in g]
        if ref_means:
            agg["growth"]["ref_mean"] = {
                "mean": float(np.mean(ref_means)),
                "std": float(np.std(ref_means)),
            }
            agg["growth"]["ref_std"] = {
                "mean": float(np.mean(ref_stds)),
                "std": float(np.std(ref_stds)),
            }

    # flux metrics — aggregate mu +/- SE and Z-test results
    for flux_key in ("pflux", "eflux", "vflux"):
        vals = [v[flux_key] for v in valid.values() if flux_key in v]
        if not vals:
            continue
        agg[flux_key] = {}
        for metric in (
            "sim_mean",
            "sim_se",
            "ref_mean",
            "ref_se",
            "se_diff",
            "z_score",
            "z_pval",
            "rel_diff",
        ):
            arr = np.array([v[metric] for v in vals])
            agg[flux_key][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
            }

    agg["n_configs"] = len(valid)
    return agg


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def _fmt(mean, std, precision=3):
    """Format mean +/- std as LaTeX string."""
    if np.isnan(mean):
        return "--"
    if np.isnan(std) or std < 10 ** (-precision - 1):
        return f"${mean:.{precision}f}$"
    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def _fmt_p(mean, std):
    """Format p-value."""
    if np.isnan(mean):
        return "--"
    if mean < 0.001:
        return "$< 0.001$"
    return _fmt(mean, std, precision=2)


def results_to_latex(aggregated, per_config=None):
    """Generate a LaTeX table from aggregated results.

    Transposed layout: rows = quantity categories, columns = metrics.
    """
    n = aggregated.get("n_configs", 0)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Statistical validation: gyaradax vs reference (aggregated over ${n}$ trajectories)}}"
    )
    lines.append(r"\label{tab:validation_stats}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")

    # ---- Stationarity ----
    st = aggregated.get("stationarity", {})
    if st:
        ns, nt = st["n_stationary"], st["n_tested"]
        lines.append(rf"\multicolumn{{5}}{{l}}{{Stationarity (ADF+KPSS): {ns}/{nt} stationary}} \\")
        lines.append(r"\midrule")

    # ---- Spectra section ----
    lines.append(r" & KS stat & Pearson $r$ ($p$) & Log rel.\ $L_2$ \\")
    lines.append(r"\midrule")

    for spec_key, label in [("ky_spec", r"$k_y$ spectrum"), ("kx_spec", r"$k_x$ spectrum")]:
        d = aggregated.get(spec_key, {})
        if not d:
            lines.append(rf"{label} & -- & -- & -- \\")
            continue

        ks = d.get("ks_stat", {})
        pr = d.get("pearson_r", {})
        pr_p = d.get("pearson_p", {})
        l2 = d.get("log_rel_l2", {})

        ks_str = _fmt(ks.get("mean", np.nan), ks.get("std", np.nan))
        pr_str = _fmt(pr.get("mean", np.nan), pr.get("std", np.nan))
        pr_p_str = _fmt_p(pr_p.get("mean", np.nan), pr_p.get("std", np.nan))
        l2_str = _fmt(l2.get("mean", np.nan), l2.get("std", np.nan))

        lines.append(rf"{label} & {ks_str} & {pr_str} ({pr_p_str}) & {l2_str} \\")

    # ---- Growth rate section ----
    lines.append(r"\midrule")
    lines.append(
        r" & $\bar{\gamma}_{\mathrm{sim}} \pm \sigma$ & $\bar{\gamma}_{\mathrm{ref}} \pm \sigma$ & \\"
    )
    lines.append(r"\midrule")

    gd = aggregated.get("growth", {})
    if gd:
        sm = gd.get("sim_mean", {})
        ss = gd.get("sim_std", {})
        rm = gd.get("ref_mean", {})
        rs = gd.get("ref_std", {})
        sim_str = _fmt(sm.get("mean", np.nan), sm.get("std", np.nan), 4)
        sim_std_str = _fmt(ss.get("mean", np.nan), ss.get("std", np.nan), 4)
        ref_str = _fmt(rm.get("mean", np.nan), rm.get("std", np.nan), 4) if rm else "--"
        ref_std_str = _fmt(rs.get("mean", np.nan), rs.get("std", np.nan), 4) if rs else "--"
        lines.append(rf"Growth rate & {sim_str} ({sim_std_str}) & {ref_str} ({ref_std_str}) & \\")
    else:
        lines.append(r"Growth rate & -- & -- & \\")

    # ---- Flux section ----
    lines.append(r"\midrule")
    lines.append(
        r" & $\mu_{\mathrm{sim}} \pm \mathrm{SE}$ & $\mu_{\mathrm{ref}} \pm \mathrm{SE}$ & $\mathrm{SE}_{\Delta}$ & rel.\ diff \\"
    )
    lines.append(r"\midrule")

    flux_labels = {"pflux": r"$\Gamma_p$", "eflux": r"$Q$", "vflux": r"$\Pi$"}
    for fk, fl in flux_labels.items():
        fd = aggregated.get(fk, {})
        if not fd:
            lines.append(rf"{fl} & -- & -- & -- & -- \\")
            continue

        sm = fd.get("sim_mean", {})
        sse = fd.get("sim_se", {})
        rm = fd.get("ref_mean", {})
        rse = fd.get("ref_se", {})
        sed = fd.get("se_diff", {})
        rd = fd.get("rel_diff", {})

        sim_str = _fmt(sm.get("mean", np.nan), sm.get("std", np.nan), 4)
        sim_se_str = _fmt(sse.get("mean", np.nan), sse.get("std", np.nan), 4)
        ref_str = _fmt(rm.get("mean", np.nan), rm.get("std", np.nan), 4)
        ref_se_str = _fmt(rse.get("mean", np.nan), rse.get("std", np.nan), 4)
        sed_str = _fmt(sed.get("mean", np.nan), sed.get("std", np.nan), 4)
        rd_str = _fmt(rd.get("mean", np.nan), rd.get("std", np.nan), 4)

        lines.append(
            rf"{fl} & {sim_str} ($\pm${sim_se_str}) & {ref_str} ($\pm${ref_se_str}) & {sed_str} & {rd_str} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------


def _mfmt(mean, std, precision=3):
    """Format mean +/- std for markdown."""
    if np.isnan(mean):
        return "--"
    if np.isnan(std) or std < 10 ** (-precision - 1):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def _mfmt_p(mean, std):
    """Format p-value for markdown."""
    if np.isnan(mean):
        return "--"
    if mean < 0.001:
        return "< 0.001"
    return _mfmt(mean, std, precision=2)


def results_to_markdown(aggregated, per_config=None):
    """Generate a markdown table from aggregated results."""
    n = aggregated.get("n_configs", 0)

    lines = []
    lines.append(
        f"**Statistical validation: gyaradax vs reference** (aggregated over {n} trajectories)\n"
    )

    # ---- Stationarity ----
    st = aggregated.get("stationarity", {})
    if st:
        ns, nt = st["n_stationary"], st["n_tested"]
        lines.append(f"**Stationarity (ADF+KPSS):** {ns}/{nt} trajectories stationary\n")

    # ---- Spectra ----
    lines.append("| | KS stat | Pearson r (p) | Log rel. L2 |")
    lines.append("|---|---|---|---|")

    for spec_key, label in [("ky_spec", "ky spectrum"), ("kx_spec", "kx spectrum")]:
        d = aggregated.get(spec_key, {})
        if not d:
            lines.append(f"| {label} | -- | -- | -- |")
            continue
        ks = _mfmt(
            d.get("ks_stat", {}).get("mean", np.nan), d.get("ks_stat", {}).get("std", np.nan)
        )
        pr = _mfmt(
            d.get("pearson_r", {}).get("mean", np.nan), d.get("pearson_r", {}).get("std", np.nan)
        )
        pr_p = _mfmt_p(
            d.get("pearson_p", {}).get("mean", np.nan), d.get("pearson_p", {}).get("std", np.nan)
        )
        l2 = _mfmt(
            d.get("log_rel_l2", {}).get("mean", np.nan), d.get("log_rel_l2", {}).get("std", np.nan)
        )
        lines.append(f"| {label} | {ks} | {pr} ({pr_p}) | {l2} |")

    lines.append("")

    # ---- Growth rate ----
    lines.append("| | gamma_sim +/- std | gamma_ref +/- std |")
    lines.append("|---|---|---|")
    gd = aggregated.get("growth", {})
    if gd:
        sm = gd.get("sim_mean", {})
        ss = gd.get("sim_std", {})
        rm = gd.get("ref_mean", {})
        rs = gd.get("ref_std", {})
        sim_str = f"{_mfmt(sm.get('mean', np.nan), sm.get('std', np.nan), 4)} ({_mfmt(ss.get('mean', np.nan), ss.get('std', np.nan), 4)})"
        ref_str = (
            f"{_mfmt(rm.get('mean', np.nan), rm.get('std', np.nan), 4)} ({_mfmt(rs.get('mean', np.nan), rs.get('std', np.nan), 4)})"
            if rm
            else "--"
        )
        lines.append(f"| Growth rate | {sim_str} | {ref_str} |")
    else:
        lines.append("| Growth rate | -- | -- |")

    lines.append("")

    # ---- Fluxes ----
    lines.append("| | sim mean +/- SE | ref mean +/- SE | SE(diff) | rel. diff |")
    lines.append("|---|---|---|---|---|")
    flux_labels = {"pflux": "Gamma_p", "eflux": "Q", "vflux": "Pi"}
    for fk, fl in flux_labels.items():
        fd = aggregated.get(fk, {})
        if not fd:
            lines.append(f"| {fl} | -- | -- | -- | -- |")
            continue
        sm = _mfmt(
            fd.get("sim_mean", {}).get("mean", np.nan), fd.get("sim_mean", {}).get("std", np.nan), 4
        )
        sse = _mfmt(
            fd.get("sim_se", {}).get("mean", np.nan), fd.get("sim_se", {}).get("std", np.nan), 4
        )
        rm = _mfmt(
            fd.get("ref_mean", {}).get("mean", np.nan), fd.get("ref_mean", {}).get("std", np.nan), 4
        )
        rse = _mfmt(
            fd.get("ref_se", {}).get("mean", np.nan), fd.get("ref_se", {}).get("std", np.nan), 4
        )
        sed = _mfmt(
            fd.get("se_diff", {}).get("mean", np.nan), fd.get("se_diff", {}).get("std", np.nan), 4
        )
        rd = _mfmt(
            fd.get("rel_diff", {}).get("mean", np.nan), fd.get("rel_diff", {}).get("std", np.nan), 4
        )
        lines.append(f"| {fl} | {sm} (+/-{sse}) | {rm} (+/-{rse}) | {sed} | {rd} |")

    return "\n".join(lines)
