---
name: gyaradax-plot
description: Plot and summarise gyaradax results (flux traces, averaged fluxes, ky/kx spectra, growth rates, distribution function, torus animations), alone or against a reference run. Use to look at a run, compare two runs or codes, or make figures for a PR or the docs.
---

# Plotting gyaradax results

References: `docs/getting_started/examples.md` (Plotting, Animations), `gyaradax/diagnostics/plot_utils.py`.

## A run against a reference

```bash
python scripts/compare_runs.py RUN REF --label "new" --ref-label "master" --out figs [--name case] [--window 0.3]
```

Writes `case_fluxes.png` (particle, heat, momentum per species; new solid, reference dashed, averaging window shaded) and `case_spectra.png` (ky and kx spectra averaged over the window, spread as a band), and prints the averaged fluxes with autocorrelation-corrected errors, their difference in sigma and the drift between the window's halves. Without REF it plots the run alone. These are the figures for PRs (see the `gyaradax-bench` skill).

## Other plots

```python
from gyaradax.diagnostics.plot_utils import plot_flux_trace, plot_spectra, plot_growth_rates, plot_nd
```

- `plot_growth_rates(g["time"], g["growth"], ky=geometry["krho"])` for `growth.npz`; linear runs converge to the growth rates.
- `plot_nd(np.abs(df), labels=["vpar", "mu", "s", "kx", "ky"])` for a snapshot (`gyaradax.io.rundir.load_checkpoint`).
- Torus animation: `python scripts/animate_sim.py RUN_DIR -o torus.mp4 --fps 12` (run with `--save-dumps` for one snapshot per block).

## Rules

- Average over the last 30 % of a run and check the flux no longer drifts; a run whose flux still grows is not stationary.
- Turbulent runs decorrelate: compare averages and spectra, not trajectories.
- Figures for the docs go in `docs/_static/figs/<section>/` next to the script that makes them; figures for a PR go on the `pr-assets` branch; anything else stays out of the repository.
