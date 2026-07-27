"""Compare gyaradax linear-IVP output against GKW on matched configs.

Produces two grid plots (growth rate and QL energy-flux transport weight) and
prints per-trajectory correlations. Data paths are for the neurips26 rebuttal
set on this system; edit RAW / iters to point elsewhere. See LINEAR_BUG.md.

Usage: python linear_gkw_compare.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "/restricteddata/ukaea/gyrokinetics/raw/neurips26_rebuttal"
OUT = "/system/user/galletti/git/neural-gyrokinetics-gitlab/rebuttal/results"
# gyaradax runs live in extra_iteration_<n><GY_SUF>; GKW in extra_iteration_<n>_Lin_gkw
GY_SUF = "_Lin_gy"
iters = ["0", "11", "14", "15", "16", "17", "19", "20", "22", "23", "24", "25"]


def load(n):
    gy = np.load(f"{RAW}/extra_iteration_{n}{GY_SUF}/gyaradax_linear.npz")
    g = f"{RAW}/extra_iteration_{n}_Lin_gkw"
    gam_gy = np.asarray(gy["gamma"])
    gam_gkw = np.loadtxt(f"{g}/growth.dat")[-1]
    ef = np.asarray(gy["eflux_kxy"]).sum(0)
    p2 = np.asarray(gy["phi2_kxy"]).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_gy = np.where(p2 > 0, ef / p2, 0.0)
    ef_gkw = np.loadtxt(f"{g}/eflux_spectra.dat")[-1]
    ks = np.loadtxt(f"{g}/kyspec")[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d_gkw = np.where(ks > 0, ef_gkw / ks, 0.0)
    return gam_gy, gam_gkw, d_gy, d_gkw


def grid(kind, ylabel, title, fname):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    corrs = []
    for ax, n in zip(axes.ravel(), iters):
        gam_gy, gam_gkw, d_gy, d_gkw = load(n)
        k = np.arange(len(gam_gy))
        if kind == "gamma":
            a, b = gam_gy, gam_gkw
            c = np.corrcoef(a, b)[0, 1]
        else:
            a = d_gy / (np.max(np.abs(d_gy)) or 1)
            b = d_gkw / (np.max(np.abs(d_gkw)) or 1)
            m = (np.abs(d_gy) > 0) & (np.abs(d_gkw) > 0)
            c = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan
        ax.plot(k, b, "o-", ms=3, color="#c44", label="GKW", lw=1)
        ax.plot(k, a, "-", color="#248", label="gyaradax", lw=1.6)
        corrs.append(c)
        ax.set_title(f"iter {n}  (r={c:.3f})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("ky mode index")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(f"{title}   (mean r = {np.nanmean(corrs):.3f}, N={len(iters)})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/{fname}", dpi=110, bbox_inches="tight")
    print(f"saved {OUT}/{fname}  mean_r={np.nanmean(corrs):.4f}")
    return np.array(corrs)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cg = grid("gamma", r"$\gamma(k_y)$", "Linear growth rates: gyaradax vs GKW", "gy_gkw_growth_grid.png")
    cs = grid("flux", r"eflux / $|\phi|^2$ (norm.)", "QL energy-flux transport weight: gyaradax vs GKW", "gy_gkw_flux_spectra_grid.png")
    for n, a, b in zip(iters, cg, cs):
        print(f"  iter {n}: gamma r={a:.3f}  flux r={b:.3f}")
