---
name: gyaradax-plot
description: Plot and summarise gyaradax results (flux traces, averaged fluxes, ky/kx spectra, growth rates, distribution function, torus animations), alone or against a reference run. Use to look at a run, compare two runs or codes, or make figures for a PR or the docs.
---

# Plotting gyaradax results

References: `docs/getting_started/examples.md` (Plotting, Animations), `gyaradax/diagnostics/plot_utils.py`, `notebooks/validation.ipynb`.

## Runs against a reference

```python
import pickle
import numpy as np
from gyaradax.diagnostics.plot_utils import plot_run_fluxes, plot_run_spectra

runs = {"new": "runs/new", "master": "runs/master"}           # label -> run directory (a list pools realisations)
ref = "long_horizon/kinetic_itg_mid_flux/nonlinear.json"       # optional: a reference from huggingface.co/datasets/gerkone/gyaradax-tests, dashed
plot_run_fluxes(runs, ref, channels=("particle", "heat", "momentum")).savefig("case_fluxes.png")
plot_run_spectra(runs, ref).savefig("case_spectra.png")
```

- With a reference, the window and the wavenumbers come from its `average.window`, `ky` and `kx`.
- Without one, pass `window=(0.7 * t_end, t_end)` and the grid of the runs:
  `geo = pickle.load(open("runs/new/geometry.pkl", "rb"))`, `ky=geo["krho"] * geo["kthnorm"]`, `kx=geo["kxrh"]`.
- Numbers to go with the figures: `notebooks/validation_stats.py` (`standard_error`, `stationarity_test`, `flux_comparison`).

## Other plots

`gyaradax.diagnostics.plot_utils` also has:

- `plot_growth_rates(g["time"], g["growth"], ky=geometry["krho"])` for `growth.npz`; linear runs converge to the growth rates.
- `plot_nd(np.abs(df), labels=["vpar", "mu", "s", "kx", "ky"])` for a snapshot (`gyaradax.io.rundir.load_checkpoint`).
- Torus animation: `python scripts/animate_sim.py RUN_DIR -o torus.mp4 --fps 12` (run with `--save-dumps` for one snapshot per block).

## Rules

- Average over the last 30 % of a run and check the flux no longer drifts; a run whose flux still grows is not stationary.
- Turbulent runs decorrelate: compare averages and spectra, not trajectories.
- Figures for the docs go in `docs/_static/figs/<section>/` next to the script that makes them; figures for a PR go on the `pr-assets` branch; anything else stays out of the repository.
