# Production cases

Long nonlinear runs and converged linear runs of the reference dataset ([gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests)): adiabatic-electron cases rerun from scratch against GKW, and kinetic-electron `gyaradax` runs as stored in the dataset. Every case has a config in [`configs/reference/`](https://github.com/gerkone/gyaradax/tree/master/configs/reference), under `adiabatic/` or `kinetic/` and `nonlinear/` or `linear/`, with the name used on this page:

```bash
gyaradax run configs/reference/adiabatic/nonlinear/itg_high_shear.yaml
```

Averages are over the last 30% of each run, with autocorrelation-corrected errors. [`notebooks/validation.ipynb`](https://github.com/gerkone/gyaradax/blob/master/notebooks/validation.ipynb) makes the comparison, `docs/_static/figs/production/make_production.py` the figures.

## Parameters

| parameter | what it sets |
|---|---|
| $q$ | safety factor: the connection length $qR$ along the field line |
| $\hat s$ | magnetic shear: how fast the field-line pitch changes with radius; sets the $k_x$ coupling along the field line |
| $R/L_{T_i}$, $R/L_{T_e}$ | temperature gradients: drive ion-temperature-gradient (ITG) and electron modes (TEM) |
| $R/L_n$ | density gradient: raises the ITG threshold with adiabatic electrons, drives trapped-electron modes with kinetic ones |
| $T_e/T_i$ | temperature ratio |
| $\varepsilon$ | inverse aspect ratio, the trapped fraction: 0.19 for the adiabatic cases, 0.13 to 0.15 for the kinetic ones |
| collisions | linearised Fokker-Planck operator, frequency from the reference values `rref`, `nref`, `tref` |

## Adiabatic electrons, against GKW

| setting | value | why |
|---|---|---|
| grid $(n_{v_\parallel}, n_\mu, n_s, n_{k_x}, n_{k_y})$ | (32, 8, 16, 85, 32), $k_y\rho_i \leq 1.4$, `ikxspace` 5 | the grid and box of the GKW runs |
| time step | 0.01, fixed | GKW's step |
| dissipation | `disp_par` 1, `disp_vp` 0.2, `disp_x` = `disp_y` = 0.1 | GKW's inputs |
| initial condition | GKW's (`cosine2` or `sine`) | both codes start from the same state |
| length | 32 000 steps, $t$ = 320 | the length of the GKW runs |
| backend | JAX, mixed precision, one A100 | 35 to 42 min per nonlinear run, 7 min per linear run |

The linear runs use the same inputs with the nonlinear term off.

| case | $q$ | $\hat s$ | $R/L_{T_i}$ | $R/L_n$ | runs |
|---|---|---|---|---|---|
| `itg_high_shear` | 4.57 | 3.08 | 10.17 | 2.61 | nonlinear, linear |
| `itg_high_q` | 8.95 | 1.26 | 10.43 | 0.74 | nonlinear, linear |
| `itg_weak_drive` | 8.99 | 1.67 | 7.37 | 1.59 | nonlinear |
| `itg_strong_low_shear` | 8.06 | 0.57 | 10.91 | 0.94 | linear |

- `itg_high_shear`: strong ITG drive against a steep density gradient and high shear. The lowest flux of the set, in bursts.
- `itg_high_q`: long connection length and flat density leave the ITG drive unopposed. The largest growth and flux, strong zonal flows.
- `itg_weak_drive`: the lowest temperature gradient, closest to threshold. Zonal flows dominate the $k_y$ spectrum.
- `itg_strong_low_shear`: low shear and a steep temperature gradient. The fastest linear growth ($\gamma_{max}$ = 0.65).

| case | $Q_i$ `gyaradax` | $Q_i$ GKW | $z(Q_i)$ | $z(\Pi_i)$ | $k_y$ KS / $r$ | $k_x$ KS / $r$ |
|---|---|---|---|---|---|---|
| `itg_high_shear` | 48.6 ± 2.8 | 50.3 ± 4.3 | 0.33 | 0.93 | 0.067 / 0.981 | 0.020 / 0.992 |
| `itg_high_q` | 128.2 ± 20.3 | 140.5 ± 13.1 | 0.51 | 0.69 | 0.083 / 0.997 | 0.118 / 0.917 |
| `itg_weak_drive` | 75.6 ± 8.4 | 66.6 ± 3.7 | 0.98 | 0.16 | 0.341 / 0.917 | 0.130 / 0.942 |

$z$ is the difference over the combined error. KS and $r$ compare the time-averaged spectra. Linear growth rates agree within 1.1e-4, 1.9e-5 and 6.3e-6 (largest relative difference over the $k_y$ with $\gamma$ above a quarter of the maximum).

```{figure} ../_static/figs/production/fluxes.png
:width: 100%

Ion heat flux. Shaded: averaging window.
```

```{figure} ../_static/figs/production/spectra.png
:width: 100%

$k_y$ (top) and $k_x$ (bottom) spectra of $|\phi|^2$ over the window.
```

```{figure} ../_static/figs/production/linear.png
:width: 100%

Linear growth rates at $t$ = 320.
```

In `itg_high_q` and `itg_weak_drive` the zonal energy at the smallest nonzero $k_x$ keeps growing in GKW and levels off in `gyaradax`: 3.3 and 6 to 7 times lower for $t \geq 250$. It sets the $k_y$ KS of `itg_weak_drive`; the other $k_x$ and the fluxes agree.

## Kinetic electrons

| setting | value | why |
|---|---|---|
| grid $(n_{sp}, n_{v_\parallel}, n_\mu, n_s, n_{k_x}, n_{k_y})$ | (2, 32, 12, 18, 192, 32), $k_y\rho_i \leq 1.5$ | coarser $k_x$, $k_y$ grids did not reach a stationary state |
| `ikxspace` | 17, 23, 10, 2 for $\hat s$ = 2.67, 3.74, 1.69, 0.28 | scales with $\hat s$ to keep $k_x\rho_i \leq$ 4.6 to 5.6 and $\Delta k_x \approx 0.05$; smaller $k_x$ ranges diverged |
| electrons | $m_e/m_i$ = 2.72e-4 | deuterium mass ratio |
| time step | adaptive from 0.004, mean 0.0005 to 0.0018 | electron streaming sets the limit |
| collisions | on in three cases | one collisionless case for comparison |
| initial condition | `noise` | seeds every mode |
| backend | CUDA, mixed precision, `fused_linear` (`linear_arith: ds` on B300 for `itg_collisionless`) | production settings, see {doc}`../performance/index` |
| length | $t$ = 300 or 500 | until the last 30% is stationary: halves differ by 3% at most |
| linear runs | same config, nonlinear term off, $\Delta t$ at the linear stability limit | run until every $k_y$ settles: relative change below 1e-5 over 6 $R/v_{th}$ |

| case | $q$ | $\hat s$ | $R/L_{T_i}$ | $R/L_{T_e}$ | $R/L_n$ | $T_e/T_i$ | coll. |
|---|---|---|---|---|---|---|---|
| `tem_low_flux` | 2.92 | 2.67 | 1.85 | 9.18 | 4.82 | 1.35 | yes |
| `itg_mid_flux` | 2.27 | 3.74 | 8.82 | 2.25 | 0.40 | 1.59 | yes |
| `itg_collisionless` | 1.86 | 1.69 | 9.51 | 9.22 | 2.61 | 0.54 | no |
| `tem_high_flux` | 1.09 | 0.28 | 2.33 | 9.27 | 4.65 | 1.99 | yes |

- `tem_low_flux`: electron temperature and density gradients over a nearly flat ion temperature. A trapped-electron mode ($\omega < 0$); electron heat dominates, low flux.
- `itg_mid_flux`: ion gradient with a weak electron drive and flat density. ITG ($\omega > 0$); ion heat dominates.
- `itg_collisionless`: equal temperature gradients, cold electrons, no collisions. ITG below $k_y\rho_i \approx 0.8$ and an electron-direction branch above; inward particle flux.
- `tem_high_flux`: low $q$ and shear, steep electron gradients, hot electrons. The strongest trapped-electron mode and the largest fluxes.

Averaging windows: $t$ = 213 to 301 for `tem_low_flux`, 352 to 502 for the others. $t_{lin}$ is when the linear run converged.

| case | $Q_i$ | $Q_e$ | $\Gamma$ | $\gamma_{max}$ ($k_y\rho_i$) | $\omega$ | $t_{lin}$ |
|---|---|---|---|---|---|---|
| `tem_low_flux` | 0.40 ± 0.01 | 3.76 ± 0.11 | 0.22 | 0.170 (0.48) | −1.38 | 122 |
| `itg_mid_flux` | 8.60 ± 0.21 | 0.23 ± 0.01 | 0.14 | 0.101 (0.39) | +1.31 | 102 |
| `itg_collisionless` | 16.5 ± 0.2 | 3.37 ± 0.05 | −0.15 | 0.157 (0.48) | +0.80 | 355 |
| `tem_high_flux` | 24.9 ± 0.3 | 69.3 ± 0.7 | 13.8 | 1.154 (0.82) | −0.85 | 81 |

```{figure} ../_static/figs/production/kinetic.png
:width: 100%

Heat fluxes (top, shaded: averaging window) and $k_y$ spectrum over the window (bottom).
```

```{figure} ../_static/figs/production/kinetic_linear.png
:width: 100%

Converged linear growth rates and frequencies. Positive $\omega$ is the ion diamagnetic direction.
```
