"""Compare a gyaradax run directory with a reference run: flux traces, saturated spectra and averaged fluxes.

Usage:
    python scripts/compare_runs.py RUN [REF] [--label new] [--ref-label master] [--out figs] [--name case] [--window 0.3]

Writes ``NAME_fluxes.png`` (new solid, reference dashed, one column per species) and ``NAME_spectra.png`` (ky and kx
spectra averaged over the window, spread as a band), and prints the heat and particle fluxes averaged over the last
``--window`` of the shorter run with autocorrelation-corrected errors, their difference in units of the combined error,
and the drift between the two halves of the window. Without REF only the run itself is plotted.
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gyaradax.diagnostics.plot_utils import plot_flux_trace, plot_spectra  # noqa: E402

CHANNELS = ("Particle", "Heat", "Momentum")


def load(run):
    out = {k: np.load(os.path.join(run, f"{k}.npz")) for k in ("fluxes", "kyspec", "kxspec")}
    with open(os.path.join(run, "geometry.pkl"), "rb") as fh:
        out["geometry"] = pickle.load(fh)
    f = np.asarray(out["fluxes"]["fluxes"], float)
    out["t"], out["f"] = np.ravel(out["fluxes"]["time"]).astype(float), f.reshape(len(f), -1, 3)
    return out


def mean_err(t, x, t0, t1):
    v = x[(t >= t0) & (t <= t1)]
    d, n = v - v.mean(), len(v)
    if n < 4 or not d.any():
        return v.mean(), 0.0
    ac = np.correlate(d, d, "full")[n - 1 :] / (d.var() * np.arange(n, 0, -1))
    cut = np.argmax(ac < 0) if (ac < 0).any() else n
    return v.mean(), v.std() * np.sqrt((1 + 2 * ac[1:cut].sum()) / n)


def drift(t, x, t0, t1):
    v = x[(t >= t0) & (t <= t1)]
    a, b = np.array_split(v, 2)
    return (b.mean() - a.mean()) / abs(v.mean()) if v.mean() else np.nan


def spectra(d, t0, t1):
    s = []
    for k, key in (("kyspec", "ky_spec"), ("kxspec", "kx_spec")):
        w = (np.ravel(d[k]["time"]) >= t0) & (np.ravel(d[k]["time"]) <= t1)
        x = np.asarray(d[k][key])[w]
        s += [x.mean(0), x.std(0)]
    return s


def flat(f, nsp):
    return np.concatenate([f[:, s, :].T for s in range(nsp)])


def relabel(fig, names):
    for ax in fig.axes:
        for text in ax.get_legend().get_texts() if ax.get_legend() else []:
            text.set_text(names.get(text.get_text(), text.get_text()))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run")
    p.add_argument("ref", nargs="?")
    p.add_argument("--label", default="new")
    p.add_argument("--ref-label", default="reference")
    p.add_argument(
        "--name",
        default=None,
        help="figure name prefix and title (default: the run directory name)",
    )
    p.add_argument("--out", default=".")
    p.add_argument(
        "--window",
        type=float,
        default=0.3,
        help="averaging window, fraction of the shorter run (default 0.3)",
    )
    a = p.parse_args()

    name = a.name or os.path.basename(os.path.normpath(a.run))
    names = {"gyaradax": a.label, "GKW": a.ref_label}
    run, ref = load(a.run), load(a.ref) if a.ref else None
    nsp = run["f"].shape[1]
    t1 = min(run["t"][-1], ref["t"][-1]) if ref else run["t"][-1]
    t0 = (1 - a.window) * t1
    os.makedirs(a.out, exist_ok=True)

    fig = plot_flux_trace(
        run["t"],
        flat(run["f"], nsp),
        labels=list(CHANNELS),
        n_species=nsp,
        title=name,
        species_labels=["ion", "electron"][:nsp] if nsp == 2 else None,
        ref_time=ref["t"] if ref else None,
        ref_fluxes=flat(ref["f"], nsp) if ref else None,
    )
    for ax in fig.axes:
        ax.axvspan(t0, t1, color="0.92", zorder=-1)
    relabel(fig, names)
    fig.savefig(os.path.join(a.out, f"{name}_fluxes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    geom = run["geometry"]
    ky = np.asarray(geom["krho"]) * float(np.asarray(geom.get("kthnorm", 1.0)))
    kx = np.asarray(geom["kxrh"])
    order = np.argsort(kx)
    s = spectra(run, t0, t1)
    r = spectra(ref, t0, t1) if ref else [None] * 4
    fig = plot_spectra(
        kx[order],
        ky,
        ky_spec=s[0],
        ky_spec_std=s[1],
        kx_spec=s[2][order],
        kx_spec_std=s[3][order],
        ref_ky_spec=r[0],
        ref_ky_spec_std=r[1],
        ref_kx_spec=None if r[2] is None else r[2][order],
        ref_kx_spec_std=None if r[3] is None else r[3][order],
        title=f"{name}: spectra, t {t0:.0f}-{t1:.0f}",
    )
    relabel(fig, names)
    fig.savefig(os.path.join(a.out, f"{name}_spectra.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"{name}: window t {t0:.1f}-{t1:.1f}")
    for sp in range(nsp):
        for ch in (1, 0):
            label = f"{['ion', 'electron'][sp] if nsp == 2 else 'species ' + str(sp)} {CHANNELS[ch].lower()}"
            m, e = mean_err(run["t"], run["f"][:, sp, ch], t0, t1)
            line = f"  {label:18s} {a.label} {m:.4g} ± {e:.2g} (drift {drift(run['t'], run['f'][:, sp, ch], t0, t1):+.1%})"
            if ref:
                mr, er = mean_err(ref["t"], ref["f"][:, sp, ch], t0, t1)
                z = (m - mr) / np.hypot(e, er) if e or er else np.nan
                rel = (m - mr) / abs(mr) if mr else np.nan
                line += f"  {a.ref_label} {mr:.4g} ± {er:.2g}  diff {rel:+.1%} ({z:+.1f} sigma)"
            print(line)


if __name__ == "__main__":
    main()
