# Examples and visualisation

## Example runs

The configs in [`configs/`](https://github.com/gerkone/gyaradax/tree/master/configs) are ready to run. Run times below are from `gyaradax bench` on one GPU (see
{doc}`../performance/index`).

**Adiabatic ITG turbulence.** `configs/iteration_13.yaml` is ion-temperature-gradient turbulence with adiabatic electrons in circular geometry: $q = 4.57$, $\hat s = 3.08$, $\varepsilon = 0.19$, $R/L_T = 10.2$, $R/L_n = 2.6$. The grid is $(n_{v_\parallel}, n_\mu, n_s, n_{k_x}, n_{k_y}) = (32, 8, 16, 85, 32)$ with a fixed `dt = 0.01`. It saturates after about t = 60, so run it for at least 12 000 steps:

```bash
gyaradax run configs/iteration_13.yaml --backend cuda --n-steps 12000
```

`configs/adiabatic_a.yaml` is the same case under another name, and `configs/adiabatic_b.yaml` a weaker-gradient one ($R/L_T = 7.4$, $q = 9.0$, $\hat s = 1.67$) started from `finit: sine`. Both share the grid, so they run batched in one process:

```bash
gyaradax run configs/adiabatic_a.yaml configs/adiabatic_b.yaml --backend cuda --output-dir scan
```

**Linear growth rates.** `configs/linear.yaml` is `iteration_13` with `non_linear: false`. Every `naverage = 40` steps the modes are renormalised and the per-ky growth rates are written to `growth.npz`. 3000 steps converge them:

```bash
gyaradax run configs/linear.yaml --backend cuda --n-steps 3000
```

**Kinetic electrons.** `configs/kinetic.yaml` evolves ions and electrons
($m_e/m_i = 2.72 \times 10^{-4}$, $T_e/T_i = 1.72$) on the larger grid
$(2, 32, 12, 18, 192, 32)$ that kinetic runs need. The timestep adapts to the CFL limit (`adaptive_dt: true`), which the fast electrons set near $2 \times 10^{-3}$. The 94 000 steps reach about t = 180. On a B300 this takes about 3.5 hours with the default kernels and 1.5 hours with the fused ones:

```bash
gyaradax run configs/kinetic.yaml --backend cuda --fused-linear --linear-arith ds --fused-bracket
```

**Kinetic electrons with collisions.** `configs/kinetic_collisions.yaml` adds the linearised Fokker-Planck operator with pitch-angle scattering, energy scattering and friction. The collision frequencies come from Coulomb logarithms at physical reference values: $R_\mathrm{ref} = 6.2$ m, $n_\mathrm{ref} = 10^{20}$ m$^{-3}$, $T_\mathrm{ref} = 12$ keV. The same command as above runs it.

**Electromagnetic turbulence.** `configs/nl_em_apar.yaml` is a small Cyclone-like case ($q = 1.4$, $\hat s = 0.78$, $R/L_T = 6.9$) with kinetic electrons and $A_\parallel$ at $\beta = 0.001$, on a 9 by 5 mode box for quick tests. `configs/nl_em_waltz_b005.yaml` and `nl_em_waltz_b01.yaml` are Waltz-style cases ($q = 2$, $\hat s = 1$, $R/L_T = 9$) at $\beta = 0.005$ and $0.01$ on a 55 by 8 box. Their `run.data_dir` points to GKW reference runs. Without them the geometry is computed from the config.

```bash
gyaradax run configs/nl_em_apar.yaml --backend cuda
```

**Sharded runs.** Any config runs on several GPUs. The layout follows the grid, here species first:

```bash
gyaradax run configs/kinetic.yaml --backend cuda --n-gpus 2
```

## Notebooks

The notebooks in [`notebooks/`](https://github.com/gerkone/gyaradax/tree/master/notebooks) reproduce the checks and experiments of {doc}`../validation/verification`. Most need a GPU. `validation.ipynb` compares runs with the JSON diagnostics of the reference dataset.

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

{mod}`gyaradax.diagnostics.plot_utils` reads the diagnostic histories of a run directory. The figures below come from `configs/iteration_13.yaml` run for 12 000 steps on one A100 with `--backend cuda` (about 4 minutes), and from `configs/linear.yaml` for 3000 steps. The script that makes them is `docs/_static/figs/examples/make_examples.py`.

```python
import numpy as np
from gyaradax.diagnostics.plot_utils import plot_flux_trace, plot_spectra

h = np.load("outputs_adiabatic_iteration_13/fluxes.npz")
plot_flux_trace(h["time"], h["fluxes"][:, 1:].T, labels=["Heat", "Momentum"],
                show_average=True, avg_window=20)
```

```{figure} ../_static/figs/examples/flux_trace.png
:width: 90%

Ion heat and momentum flux of the adiabatic ITG case, with the average over the
last 30 % of the run. With adiabatic electrons the particle flux vanishes.
```

{func}`~gyaradax.diagnostics.plot_utils.plot_spectra` takes the time-averaged spectra from `kyspec.npz` and `kxspec.npz`, with their spread as a band:

```{figure} ../_static/figs/examples/spectra.png
:width: 100%

ky and kx spectra of the potential, averaged over the last 30 % of the run.
```

{func}`~gyaradax.diagnostics.plot_utils.plot_growth_rates` shows the per-ky growth rates of `growth.npz`. In a linear run they converge to the linear growth rates:

```{figure} ../_static/figs/examples/growth.png
:width: 90%

Growth rates of the lowest ky modes of `configs/linear.yaml`.
```

{func}`~gyaradax.diagnostics.plot_utils.plot_nd` shows a 5D or 6D array as a grid of 2D panels, one per pair of dimensions, averaged over the others. Applied to the distribution function of a snapshot:

```python
from gyaradax.diagnostics.plot_utils import plot_nd
from gyaradax.io.rundir import load_checkpoint

snap = load_checkpoint("outputs_adiabatic_iteration_13/step_012400.npz")
plot_nd(np.abs(snap["df"]), labels=["vpar", "mu", "s", "kx", "ky"])
```

```{figure} ../_static/figs/examples/plot_nd.png
:width: 80%

Mean ``|δf|`` of the saturated adiabatic run over every pair of dimensions.
```

Restart snapshots hold the full state and load with
{func}`~gyaradax.io.rundir.load_checkpoint`:

```python
from gyaradax.io.rundir import load_checkpoint

snap = load_checkpoint("outputs_adiabatic_iteration_13/step_012400.npz")
df, phi = snap["df"], snap["phi"]
```

## Animations

`scripts/animate_sim.py` renders the potential of a run's snapshots on a torus, as in the animation on the front page:

```bash
python scripts/animate_sim.py outputs_adiabatic_iteration_13/ -o torus.mp4 --fps 12
```

Run with `--save-dumps` to keep one snapshot per block for a smooth animation.
