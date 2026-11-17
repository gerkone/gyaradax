"""Example figures of the docs: ``python make_examples.py NONLINEAR_RUN_DIR LINEAR_RUN_DIR``."""

import glob
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

from gyaradax.diagnostics.plot_utils import plot_flux_trace, plot_growth_rates, plot_nd, plot_spectra

OUT = os.path.dirname(os.path.abspath(__file__))


def main(run_dir, linear_dir):
    geometry = pickle.load(open(os.path.join(run_dir, "geometry.pkl"), "rb"))
    kx, ky = np.asarray(geometry["kxrh"]), np.asarray(geometry["krho"])
    fluxes = np.load(os.path.join(run_dir, "fluxes.npz"))
    tail = int(0.3 * len(fluxes["time"]))
    fig = plot_flux_trace(fluxes["time"], fluxes["fluxes"][:, 1:].T, labels=["Heat", "Momentum"],
                          show_average=True, avg_window=tail, title="")
    fig.savefig(os.path.join(OUT, "flux_trace.png"), dpi=130, bbox_inches="tight")

    kys, kxs = np.load(os.path.join(run_dir, "kyspec.npz")), np.load(os.path.join(run_dir, "kxspec.npz"))
    tail = slice(int(0.7 * len(kys["time"])), None)
    fig = plot_spectra(kx, ky, kx_spec=kxs["kx_spec"][tail].mean(0), ky_spec=kys["ky_spec"][tail].mean(0),
                       kx_spec_std=kxs["kx_spec"][tail].std(0), ky_spec_std=kys["ky_spec"][tail].std(0))
    fig.savefig(os.path.join(OUT, "spectra.png"), dpi=130, bbox_inches="tight")

    snapshot = np.load(sorted(glob.glob(os.path.join(run_dir, "step_*.npz")))[-1])
    fig = plot_nd(np.abs(snapshot["df"]), labels=["vpar", "mu", "s", "kx", "ky"])
    fig.savefig(os.path.join(OUT, "plot_nd.png"), dpi=110, bbox_inches="tight")

    ky_lin = np.asarray(pickle.load(open(os.path.join(linear_dir, "geometry.pkl"), "rb"))["krho"])
    growth = np.load(os.path.join(linear_dir, "growth.npz"))
    fig = plot_growth_rates(growth["time"], growth["growth"], ky_lin, title="")
    fig.savefig(os.path.join(OUT, "growth.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
