# Configuration

A run config is a YAML file with the sections below. Names follow GKW's
`input.dat` where there is an equivalent; `gyaradax convert` writes a complete
config from a GKW run directory. Every key maps to a field of
{class}`~gyaradax.core.params.GKParams` or of the geometry, whose docstrings give
the full list and defaults.

```yaml
run:
  name: iteration_13
  data_dir: null           # GKW run directory: geometry from its geom.dat, resume from its K-files
solver:
  dt: 0.01
  n_steps: 400
  naverage: 40             # steps per normalisation / growth-rate window
  dump_interval: 3         # checkpoint every dump_interval x naverage steps
  non_linear: true
  adaptive_dt: false       # CFL-adaptive dt (nonlinear runs)
  finit: cosine2
  disp_par: 1.0
  disp_vp: 0.2
  disp_x: 0.1
  disp_y: 0.1
  backend: auto
physics:
  rlt: 10.17               # one value per species for kinetic runs
  rln: 2.61
  mas: 1.0
  tmp: 1.0
  de: 1.0
  signz: 1.0
geometry:
  geometry_model: circ     # circ, s-alpha or miller
  q: 4.57
  shat: 3.08
  eps: 0.19
grid:
  nvpar: 32
  nmu: 8
  ns: 16
  nkx: 85
  nky: 32
  adiabatic_electrons: true
  vpar_max: 3.0
  nperiod: 1
  krhomax: 1.4
  ikxspace: 5
```

## Sections

`run`
: `name` (names the output directory), `data_dir`, and optionally `block_size`,
  `snapshot_every`, `save_dumps` (the CLI flags of the same names override them).

`solver`
: Time stepping (`dt`, `n_steps`, `naverage`, `adaptive_dt`, `fac_dtim_est`,
  `fac_dtim_nl`), initial condition (`finit`, `amp_init`), dissipation
  (`disp_par`, `disp_vp`, `disp_x`, `disp_y`, `idisp`, `disp_par_conserve`,
  `disp_par_conserve_kycut`), the electromagnetic switches `nlapar` and `nlbpar`,
  and the kernel choices `backend`, `mixed_precision`, `fused_linear`,
  `linear_arith`, `fused_bracket`, `use_z2z`.

`physics`
: Species: `rlt`, `rln`, `mas`, `tmp`, `de`, `signz` and optionally `vthrat`
  (default $\sqrt{T/m}$). Scalars for one species, lists for several. Also
  `beta`, `betaprime` and `drift_gradp_type` for electromagnetic runs.

`geometry`
: `geometry_model` and the local equilibrium `q`, `shat`, `eps`, plus `signB`,
  `Rref`. Miller geometry adds `kappa`, `delta`, `square`, `skappa`, `sdelta`,
  `ssquare`, `Zmil`, `dRmil`, `dZmil`.

`grid`
: Resolution `nvpar`, `nmu`, `ns`, `nkx`, `nky`; `vpar_max` (velocity box, in
  thermal velocities), `nperiod` (poloidal turns of the field line), `krhomax`
  (largest $k_\theta\rho$), `ikxspace` (kx modes between connected ones), and
  `adiabatic_electrons`.

`collisions`
: Present to switch collisions on: `pitch_angle`, `en_scatter`, `friction`,
  `freq` with `freq_override: true`, or the reference values `rref` [m],
  `nref` [10^19 m^-3], `tref` [keV] for the Coulomb-log frequencies;
  `mass_conserve`, `mom_conservation`, `ene_conservation`.

`sharding`
: `n_gpus`, or `n_gpus_sp`, `n_gpus_vp`, `n_gpus_mu`.

`debug`
: `telemetry: true`, `profile: true`.

## Choosing the grid

- `nkx` and `ikxspace` fix the radial box. The kx spacing follows the magnetic
  shear so that modes `ikxspace` apart connect after one poloidal turn; each
  connected chain holds about `nkx / ikxspace` modes. Kinetic runs need a larger
  `nkx` than adiabatic ones: `configs/kinetic.yaml` uses 192, `configs/iteration_13.yaml` 85.
- `ns` points cover `2 nperiod - 1` poloidal turns. Linear benchmarks need
  enough of them to resolve Landau damping: the Cyclone Base Case uses
  `ns = 160` with `nperiod = 5`; at `ns = 16` its growth rates come out too high.
- The FFT lengths of the nonlinear term follow from `nkx` and `nky`, see
  {doc}`../performance/index` for sizes that run fastest.
