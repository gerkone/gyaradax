"""Compare velocity grids on the single-mode, three-turn 3559 CGYRO deck."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def settings(run):
    deck = {}
    for line in (run / "input.cgyro").read_text().splitlines():
        line = line.partition("#")[0].strip()
        if line:
            key, value = line.split("=", 1)
            deck[key.strip()] = value.strip()
    return deck


def profile(run, field, samples, cells):
    data = np.fromfile(run / f"bin.cgyro.{field}b", dtype=np.complex64)
    return data.reshape(samples, cells).astype(np.complex128)


def overlap(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("refinements", nargs="+", type=Path)
    args = parser.parse_args()
    baseline = settings(args.reference)
    factor = float(baseline["RMAJ"]) / np.sqrt(2)
    cells = int(baseline["N_THETA"]) * int(baseline["N_RADIAL"])
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["n_xi", "n_energy", "end_time_cgyro", "linear_converged",
                     "gamma", "omega", "gamma_phi_fit", "omega_phi_phase",
                     "phi_overlap_with_reference", "apar_overlap_with_reference"])
    for run in [args.reference, *args.refinements]:
        deck = settings(run)
        changes = {k for k in deck.keys() | baseline.keys() if deck.get(k) != baseline.get(k)}
        if changes - {"N_XI", "N_ENERGY"}:
            raise ValueError(f"{run} also changes {sorted(changes)}")
        t = np.loadtxt(run / "out.cgyro.time")[:, 0]
        freq = np.loadtxt(run / "out.cgyro.freq")
        if freq.shape != (len(t), 2) or not np.all(np.diff(t) > 0):
            raise ValueError(f"Inconsistent frequency and time records in {run}")
        phi = profile(run, "phi", len(t), cells)
        apar = profile(run, "apar", len(t), cells)
        if run == args.reference:
            ref_phi, ref_apar = phi[-1], apar[-1]
        tail = slice(int(0.7 * len(t)), None)
        omega, gamma = np.mean(freq[tail], axis=0)
        z = phi[tail]
        fit = np.polyfit(t[tail], np.log(np.linalg.norm(z, axis=1)), 1)[0]
        phase = np.angle(np.sum(z[:-1].conj() * z[1:], axis=1)) / np.diff(t[tail])
        converged = "Linear converged" in (run / "out.cgyro.info").read_text()
        writer.writerow([deck["N_XI"], deck["N_ENERGY"], t[-1], converged,
                         factor * gamma, -factor * omega, factor * fit, factor * phase.mean(),
                         overlap(ref_phi, phi[-1]), overlap(ref_apar, apar[-1])])


if __name__ == "__main__":
    main()
