"""Production-case figures of the docs: ``python make_production.py REFERENCE_ROOT RUNS_DIR``.

REFERENCE_ROOT holds the gerkone/gyaradax-tests diagnostics (``--include "*.json"`` is enough),
RUNS_DIR the ``gyaradax run`` output directories laid out as in ``notebooks/validation.ipynb``.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gyaradax.diagnostics.plot_utils import JAX_COLORS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "notebooks"))
from validation_stats import compare_config, compare_linear, load_run, run_window  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
NONLINEAR = [
    ("long_horizon/adiabatic_itg_high_shear", "long_horizon/adiabatic_itg_high_shear/nonlinear"),
    ("long_horizon/adiabatic_itg_high_q", "long_horizon/adiabatic_itg_high_q/nonlinear"),
    ("nonlinear/itg_weak_drive", "nonlinear/itg_weak_drive"),
]
LINEAR = [
    ("long_horizon/adiabatic_itg_high_shear", "long_horizon/adiabatic_itg_high_shear/linear"),
    ("long_horizon/adiabatic_itg_high_q", "long_horizon/adiabatic_itg_high_q/linear"),
    ("linear/itg_strong_low_shear", "linear/itg_strong_low_shear"),
]
GYX = JAX_COLORS["cyan"]


def reference(root, label, kind):
    return json.load(open(os.path.join(root, label, f"{kind}.json")))


def main(root, runs):
    fig_f, ax_f = plt.subplots(1, 3, figsize=(13, 3.2))
    fig_s, ax_s = plt.subplots(2, 3, figsize=(13, 6))
    for i, (label, run_dir) in enumerate(NONLINEAR):
        ref = reference(root, label, "nonlinear")
        run = load_run(os.path.join(runs, run_dir))
        t0, t1 = ref["average"]["window"]
        name = label.split("/")[1]
        ax = ax_f[i]
        ax.axvspan(t0, t1, color="0.92", zorder=-1)
        ax.plot(ref["time"], ref["fluxes"]["ion"]["heat"], color="k", lw=0.8, label="GKW")
        ax.plot(run["time"], run["fluxes"][:, 0, 1], color=GYX, lw=0.8, label="gyaradax")
        ax.set(xlabel=r"$t$ [$R/v_{th}$]", title=name)
        if i == 0:
            ax.set_ylabel("ion heat flux")
            ax.legend(frameon=False, fontsize=9)
        rt = np.asarray(ref["time"])
        rmask = (rt >= t0) & (rt <= t1)
        s0, s1 = run_window(run["time"], ref)
        smask = (run["time"] >= s0) & (run["time"] <= s1)
        for row, (key, rkey, k, xl) in enumerate(
            (("ky_spec", "ky_spectrum", "ky", r"$k_y\rho_i$"), ("kx_spec", "kx_spectrum", "kx", r"$k_x\rho_i$"))
        ):
            ax = ax_s[row, i]
            kk = np.asarray(ref[k])
            ax.semilogy(kk, np.asarray(ref[rkey])[rmask].mean(0), "k.-", ms=3, lw=0.8, label="GKW")
            ax.semilogy(kk, run[key][smask].mean(0), ".-", color=GYX, ms=3, lw=0.8, label="gyaradax")
            ax.set_xlabel(xl)
            if row == 0:
                ax.set_title(name)
            if i == 0:
                ax.set_ylabel(r"$\langle|\phi|^2\rangle$")
        r = compare_config([os.path.join(runs, run_dir)], ref)
        q, p = r["eflux"], r["vflux"]
        print(
            f"| `{label}` | {q['sim_mean']:.1f} ± {q['sim_se']:.1f} | {q['ref_mean']:.1f} ± {q['ref_se']:.1f} "
            f"| {q['z_score']:.2f} | {p['z_score']:.2f} | {r['ky_spec']['ks_stat']:.3f} / {r['ky_spec']['pearson_r']:.3f} "
            f"| {r['kx_spec']['ks_stat']:.3f} / {r['kx_spec']['pearson_r']:.3f} |"
        )
    ax_s[0, 0].legend(frameon=False, fontsize=9)
    fig_f.tight_layout()
    fig_f.savefig(os.path.join(OUT, "fluxes.png"), dpi=130, bbox_inches="tight")
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(OUT, "spectra.png"), dpi=130, bbox_inches="tight")

    fig_l, ax_l = plt.subplots(1, 3, figsize=(13, 3.2))
    for i, (label, run_dir) in enumerate(LINEAR):
        ref = reference(root, label, "linear")
        res = compare_linear(os.path.join(runs, run_dir), ref)
        ax = ax_l[i]
        ax.axhline(0, color="0.6", lw=0.5)
        ax.plot(res["ky"], res["ref_gamma"], "k.-", ms=4, lw=0.8, label="GKW")
        ax.plot(res["ky"], res["sim_gamma"], "o", color=GYX, ms=3, mfc="none", label="gyaradax")
        ax.set(xlabel=r"$k_y\rho_i$", title=label.split("/")[1])
        if i == 0:
            ax.set_ylabel(r"$\gamma$ [$v_{th}/R$]")
            ax.legend(frameon=False, fontsize=9)
        print(f"linear `{label}`: max relative difference {res['max_rel_diff']:.1e} at t = {res['t_end']:.0f}")
    fig_l.tight_layout()
    fig_l.savefig(os.path.join(OUT, "linear.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
