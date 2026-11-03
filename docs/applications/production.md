# Production cases

The reference dataset ([gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests)) holds whole GKW runs as JSON diagnostics, with `gyaradax` configs for the same parameters. This page runs those configs from scratch to the length of the GKW runs and compares the statistics. Turbulent runs of two codes separate after a few tens of $R/v_{th}$, so the comparison is between time averages over GKW's window, the last 30% of the run ($t$ = 224 to 320), not between trajectories.

## Setup

Four adiabatic-electron ITG cases on the (32, 8, 16, 85, 32) grid of the dataset, fixed step 0.01, 32 000 steps to $t$ = 320. One A100 per run, JAX backend, mixed precision. A nonlinear run takes 35 to 42 minutes, a linear one 7 minutes.

| case | dataset folder | runs | $q$ | $\hat s$ | $R/L_{T_i}$ | $R/L_n$ |
|---|---|---|---|---|---|---|
| `adiabatic_itg_high_shear` | `long_horizon/` | nonlinear, linear | 4.57 | 3.08 | 10.17 | 2.61 |
| `adiabatic_itg_high_q` | `long_horizon/` | nonlinear, linear | 8.95 | 1.26 | 10.43 | 0.74 |
| `itg_weak_drive` | `nonlinear/` | nonlinear | 8.99 | 1.67 | 7.37 | 1.59 |
| `itg_strong_low_shear` | `linear/` | linear | 8.06 | 0.57 | 10.91 | 0.94 |

The kinetic-electron cases of `long_horizon/` take 0.17 to 0.9 million steps each and are not on this page yet.

To reproduce, download the diagnostics and configs, run them, and compare with [`notebooks/validation.ipynb`](https://github.com/gerkone/gyaradax/blob/master/notebooks/validation.ipynb), which prints the command of every missing run:

```bash
ROOT=~/.cache/gyaradax/test_data RUNS=runs
hf download gerkone/gyaradax-tests --repo-type dataset --local-dir $ROOT --include "*.json" "*.yaml" "*input*.dat"
gyaradax run $ROOT/long_horizon/adiabatic_itg_high_shear/config_nonlinear.yaml --output-dir $RUNS/long_horizon/adiabatic_itg_high_shear/nonlinear
(cd $ROOT/nonlinear/itg_weak_drive && gyaradax run config.yaml --from-scratch --n-steps 32000 --output-dir $OLDPWD/$RUNS/nonlinear/itg_weak_drive)
```

The figures on this page come from `docs/_static/figs/production/make_production.py`.

## Nonlinear runs

```{figure} ../_static/figs/production/fluxes.png
:width: 100%

Ion heat flux of GKW and `gyaradax`. The shaded band is the averaging window.
```

Averages over the window, with autocorrelation-corrected standard errors. $z$ is the difference over the combined error, so values below 2 are consistent. KS and $r$ compare the time-averaged spectra (Kolmogorov-Smirnov statistic and Pearson correlation).

| case | $Q_i$ `gyaradax` | $Q_i$ GKW | $z(Q_i)$ | $z(\Pi_i)$ | $k_y$ KS / $r$ | $k_x$ KS / $r$ |
|---|---|---|---|---|---|---|
| `adiabatic_itg_high_shear` | 48.6 ± 2.8 | 50.3 ± 4.3 | 0.33 | 0.93 | 0.067 / 0.981 | 0.020 / 0.992 |
| `adiabatic_itg_high_q` | 128.2 ± 20.3 | 140.5 ± 13.1 | 0.51 | 0.69 | 0.083 / 0.997 | 0.118 / 0.917 |
| `itg_weak_drive` | 75.6 ± 8.4 | 66.6 ± 3.7 | 0.98 | 0.16 | 0.341 / 0.917 | 0.130 / 0.942 |

```{figure} ../_static/figs/production/spectra.png
:width: 100%

$k_y$ (top) and $k_x$ (bottom) spectra of $|\phi|^2$, averaged over the window.
```

The heat and momentum fluxes agree within their errors in all three cases, and so do the spectra away from the largest zonal scale. In `adiabatic_itg_high_q` and `itg_weak_drive` the zonal energy at the smallest nonzero $k_x$ keeps growing in GKW until $t$ = 320, while in `gyaradax` it levels off: for $t \geq 250$ it is 3.3 and 6 to 7 times lower. This sets the $k_y$ KS of `itg_weak_drive`, whose $k_y = 0$ point carries it. The other $k_x$ and the non-zonal spectrum agree. With one run per code, it is open whether this is the spread of a slowly evolving mode or a difference in how the box-scale zonal mode is damped.

## Linear runs

```{figure} ../_static/figs/production/linear.png
:width: 100%

Growth rates at $t$ = 320 against GKW.
```

Over the $k_y$ with a growth rate above a quarter of the maximum, the largest relative difference is 1.1e-4 (`adiabatic_itg_high_shear`), 1.9e-5 (`adiabatic_itg_high_q`) and 6.3e-6 (`itg_strong_low_shear`).
