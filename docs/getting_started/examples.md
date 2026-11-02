# Examples and visualisation

## Notebooks

The notebooks in [`notebooks/`](https://github.com/gerkone/gyaradax/tree/master/notebooks)
reproduce the checks and experiments of {doc}`../verification/index`. Most need a
GPU; the ones comparing against GKW need the reference runs (`GKW_DATA_ROOT`).

| Notebook | Content |
|---|---|
| `analytical.ipynb` | Rosenbluth-Hinton zonal-flow residual and Cyclone Base Case growth rates |
| `kinetic_analytical.ipynb` | kinetic-electron benchmarks at $\beta = 0$ |
| `em_analytical.ipynb` | electromagnetic benchmarks: $\beta$ scan through the ITG to KBM transition and more |
| `gkw_cases.ipynb` | growth rates of GKW's standard test cases against the GKW reference |
| `inverse_problem.ipynb` | recovering $R/L_T$ from the potential by gradient descent through the solver |
| `sensitivity_analysis.ipynb` | $\partial\gamma(k_y)/\partial(R/L_T)$ by automatic differentiation |
| `sitl.ipynb` | a learned spectral correction trained inside the simulation loop |
| `validation.ipynb` | nonlinear fluxes and spectra against GKW trajectories |
| `em_rollout_flux_comparison.ipynb` | electromagnetic nonlinear fluxes against GKW |

## Plotting

{mod}`gyaradax.diagnostics.plot_utils` has the plotting functions used for the
figures in these pages: flux traces with GKW references
({func}`~gyaradax.diagnostics.plot_utils.plot_flux_trace`), time-averaged spectra
({func}`~gyaradax.diagnostics.plot_utils.plot_spectra`) and growth rates
({func}`~gyaradax.diagnostics.plot_utils.plot_growth_rates`). They read the
diagnostic histories of a run directory:

```python
import numpy as np
from gyaradax.diagnostics.plot_utils import plot_flux_trace

h = np.load("outputs_adiabatic_iteration_13/fluxes.npz")
plot_flux_trace(h["time"], h["fluxes"].T)   # rows: particle, heat, momentum
```

Restart snapshots hold the full state and load with
{func}`~gyaradax.io.rundir.load_checkpoint`:

```python
from gyaradax.io.rundir import load_checkpoint

snap = load_checkpoint("outputs_adiabatic_iteration_13/step_000400.npz")
df, phi = snap["df"], snap["phi"]
```

## Animations

`scripts/animate_sim.py` renders the potential of a run's snapshots on a torus,
as in the animation on the front page:

```bash
python scripts/animate_sim.py outputs_adiabatic_iteration_13/ -o torus.mp4 --fps 12
```

Run with `--save-dumps` to keep one snapshot per block for a smooth animation.
