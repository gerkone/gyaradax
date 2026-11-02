# Verification

The results behind {doc}`development`. Figures marked "paper" are from
[the paper](https://arxiv.org/abs/2604.06085). They were made with the code of
April 2026, which was electrostatic and collisionless. The electromagnetic and
collisional checks came later.

## Analytical benchmarks

**Rosenbluth-Hinton zonal-flow residual.** A radial ($k_y = 0$) perturbation
excites a geodesic acoustic mode that Landau-damps to a residual
$\phi(\infty)/\phi(0) = 1/(1 + q^2\Theta/\varepsilon^2)$
([Rosenbluth and Hinton 1998](https://doi.org/10.1103/PhysRevLett.80.724), with the
finite-$\varepsilon$ form $\Theta = 1.6\varepsilon^{3/2} + 0.5\varepsilon^2 + 0.36\varepsilon^{5/2}$
of Xiao and Catto 2006). With GKW's benchmark parameters (`zonal01`: $q = 1.3$,
$\hat s = 0.16$, $\varepsilon = 0.05$, s-α, $n_s = n_{v_\parallel} = 128$,
$n_\mu = 16$) the residual converges to 0.0711, the predicted value to 0.1 %, and
follows the prediction across $\varepsilon$.

```{figure} ../_static/figs/paper/rh_zonal_flow.png
:width: 90%
:class: paper-fig

Zonal potential over time at $\varepsilon = 0.05$ (a) and residual against
$\varepsilon$ (b). Paper.
```

**Cyclone Base Case.** The standard linear ITG benchmark
([Dimits et al. 2000](https://doi.org/10.1063/1.873896)): adiabatic electrons,
$q = 1.4$, $\hat s = 0.78$, $\varepsilon = 0.19$, $R/L_n = 2.2$, s-α geometry,
GKW's `cyclone/linear` resolution ($n_s = 160$ over 9 poloidal turns,
$n_{v_\parallel} = 64$, $n_\mu = 16$). Growth rates match GKW, run with the same
parameters, within 1 % over $k_\theta\rho$ and $R/L_T$ ($\gamma = 0.179$ at
$k_\theta\rho = 0.5$, $R/L_T = 6.9$).

```{figure} ../_static/figs/paper/cbc_linear_itg.png
:width: 90%
:class: paper-fig

Growth rate against $k_\theta\rho$ at $R/L_T = 6.9$ (a) and against $R/L_T$ at
$k_\theta\rho = 0.5$ (b). Paper.
```

These two cases are unit tests (`tests/unit/test_gk_cases.py`) and are shown in
`notebooks/analytical.ipynb`.

## Linear runs against GKW

- **Short trajectories.** Starting from the same GKW dump, the distribution after
  80 linear steps agrees with GKW's to 9e-6 (relative L2, adiabatic), and after
  300 steps to 8e-7 (ions) and 2e-6 (electrons) for kinetic electrons.
- **Electromagnetic.** Linear growth rates match GKW to 0.02-0.1 % over
  $\beta = 0.001$-$0.01$ at $k_\theta\rho = 0.4$, through the ITG to KBM
  transition (at $\beta = 0.01$: $\gamma = 0.9236$ against GKW's 0.9238), with and
  without $B_\parallel$, and to 0.3 % with rotation.
- **Collisions.** The collision increment of one step (the difference between a
  step with and without collisions) agrees with GKW to 3e-8 relative for ions
  and electrons, and the Coulomb logarithms and collision frequencies agree to
  1e-13 for five reference sets.
- **Eigenvalue solver.** The dominant eigenvalue of the linear operator matches
  the growth rate and frequency of the initial-value run. The solver also
  returns the subdominant modes.

```{figure} ../figs/eigensolver_spectrum.png
:width: 95%

Growth rate and frequency from the linear initial-value run and from the
eigensolver, adiabatic electrons, 16 x 85 x 32.
```

```{figure} ../figs/eigensolver_structure.png
:width: 95%

Dominant and subdominant eigenvalues against the initial-value growth rate, for
electrostatic adiabatic, kinetic-electron and electromagnetic cases.
```

## Nonlinear runs against GKW

Nonlinear runs are compared through their statistics: time-averaged fluxes,
$k_y$ and $k_x$ spectra of $|\phi|^2$, and growth rates. On 46 adiabatic ITG
equilibria (grid 32 x 8 x 16 x 85 x 32, three runs per code from the same initial
state), the grand-mean heat flux is 90.9 for `gyaradax` and 91.3 for GKW, and the
spectra agree as follows (paper):

| | KS statistic | Pearson $r$ | log relative L2 |
|---|---|---|---|
| $k_y$ spectrum | 0.073 ± 0.049 | 0.990 ± 0.018 | 0.073 ± 0.042 |
| $k_x$ spectrum | 0.055 ± 0.035 | 0.952 ± 0.068 | 0.076 ± 0.042 |

```{figure} ../_static/figs/paper/fluxes_iteration_13.png
:width: 100%
:class: paper-fig

Heat and momentum flux of one adiabatic case, three runs per code. The codes
coincide in the linear phase and separate when turbulence sets in. The averages
agree. Paper.
```

```{figure} ../_static/figs/paper/spectra_iteration_13.png
:width: 100%
:class: paper-fig

Time-averaged $k_y$ and $k_x$ spectra of the same case. Paper.
```

```{figure} ../_static/figs/paper/fluxes_iteration_200.png
:width: 100%
:class: paper-fig

A second adiabatic case: heat and momentum flux. Paper.
```

```{figure} ../_static/figs/paper/spectra_iteration_200.png
:width: 100%
:class: paper-fig

Its time-averaged spectra. Paper.
```

```{figure} ../_static/figs/paper/kinetic_fluxes_v3_kiteration_991_half_rlt.png
:width: 100%
:class: paper-fig

Kinetic electrons: ion and electron fluxes. Paper.
```

```{figure} ../_static/figs/paper/kinetic_spectra_v3_kiteration_991_half_rlt.png
:width: 100%
:class: paper-fig

Kinetic electrons: time-averaged $k_y$ and $k_x$ spectra. Paper.
```

:::{dropdown} All 46 adiabatic cases
```{image} ../_static/figs/paper/flux_overview_grid.png
:width: 100%
:class: paper-fig
```
```{image} ../_static/figs/paper/spectra_overview_grid.png
:width: 100%
:class: paper-fig
```
:::

Electromagnetic nonlinear runs with kinetic electrons on the Cyclone geometry
give ion and electron heat fluxes within 4 % of GKW at $\beta = 0.001$ (ratios
1.04 and 1.02) and within 5 % at $\beta = 0.01$ (0.97 and 0.95), with
$k_y$-spectrum Pearson correlation 1.00.

## Differentiable experiments

**Inverse problem.** The temperature gradient is recovered from a target
potential: a 400-step linear run at $R/L_T = 6.9$ ($q = 1.4$, $\hat s = 0.78$,
$\varepsilon = 0.19$, grid 24 x 8 x 16 x 9 x 8) gives $\phi^*$, and Adam on the
mean squared potential mismatch, with gradients by reverse-mode AD through all
400 steps, goes from $R/L_T = 10$ to 6.908. The loss is smooth with a single
minimum. `notebooks/inverse_problem.ipynb`.

```{figure} ../_static/figs/paper/inverse_spectrum.png
:width: 60%
:class: paper-fig

Initial, target and recovered $k_y$ spectra (top) and $R/L_T$ over the
iterations (bottom). Paper.
```

```{figure} ../_static/figs/paper/inv_landscape.png
:width: 50%
:class: paper-fig

Loss and its AD gradient over $R/L_T$, with the optimiser's path. Paper.
```

**Sensitivity.** `jax.jacrev` gives $\partial\gamma(k_y)/\partial(R/L_T)$ for the
whole spectrum in $n_{k_y}$ backward passes, here over 15 values of
$R/L_T \in [3, 10]$ on the same grid. `notebooks/sensitivity_analysis.ipynb`.

```{figure} ../_static/figs/paper/sensitivity_gamma.png
:width: 70%
:class: paper-fig

Growth rate $\gamma(k_y, R/L_T)$. Paper.
```

```{figure} ../_static/figs/paper/sensitivity_dgamma.png
:width: 70%
:class: paper-fig

Its sensitivity $\partial\gamma/\partial(R/L_T)$. Paper.
```

## Checking code changes

Every optimisation went through the loop of {doc}`development`.

**Per evaluation, CUDA against JAX** (relative L2 on the production grids, worst
species, H100, the same on B300):

| case | linear RHS (FP64) | bracket FP64 | bracket mixed precision |
|---|---|---|---|
| ES adiabatic | 1.5e-16 | 1.1e-15 | 6.2e-7 |
| ES kinetic | 1.5e-16 | 1.6e-15 | 8.4e-7 |
| kinetic $A_\parallel$ | 1.5e-16 | 1.6e-15 | 8.5e-7 |
| kinetic $A_\parallel + B_\parallel$ | 2.0e-16 | 1.6e-15 | 8.2e-7 |
| adiabatic + $A_\parallel$ | 1.5e-16 | 1.1e-15 | 6.2e-7 |
| conservative dissipation | 3.0e-16 | 2.2e-15 | 1.3e-6 |

`fused_linear` and `linear_arith: ds` agree with the default kernel to about
1e-14 per RHS evaluation.

**Over long runs, new code against old.** Changes that should not alter the
numbers are checked bitwise: saturated turbulent runs, adiabatic and kinetic, are
restarted from a snapshot and stepped with both versions, and the distribution
functions must be identical at the end (thousands to tens of thousands of steps).
The flux and $k_x$-spectrum diagnostics are not used for this, because their GPU
reductions differ at 1e-14 between two runs of the same code. Changes that are
not bitwise, such as a different FFT length or a new kernel arithmetic, must
reproduce the time-averaged fluxes of the old code over long runs, for adiabatic
and kinetic electrons. The pull requests on
[GitHub](https://github.com/gerkone/gyaradax/pulls?q=is%3Apr) record these checks
with their figures.
