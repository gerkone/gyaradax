# Configuration

A run config is a YAML file with the sections below, in this order. Names follow
GKW's `input.dat` where there is an equivalent; `gyaradax convert` writes a
complete config from a GKW run directory. Every key maps to a field of
{class}`~gyaradax.core.params.GKParams` or of the geometry, whose docstrings give
the full list and defaults.

```yaml
run:
  name: iteration_13
  data_dir: null           # GKW run directory: geometry from its geom.dat, resume from its K-files
solver:
  dt: 0.01                 # time stepping
  n_steps: 400
  naverage: 40             # steps per normalisation / growth-rate window
  dump_interval: 3         # checkpoint every dump_interval x naverage steps
  non_linear: true
  adaptive_dt: false       # CFL-adaptive dt (nonlinear runs)
  finit: cosine2           # initial condition
  amp_init: 1.0e-4
  disp_par: 1.0            # dissipation
  disp_vp: 0.2
  disp_x: 0.1
  disp_y: 0.1
  nlapar: false            # electromagnetic fields
  nlbpar: false
  backend: auto            # backend and kernels
  mixed_precision: true
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
# optional sections
# collisions:
#   freq_override: true
#   freq: 0.01
# sharding:
#   n_gpus: 4
# debug:
#   telemetry: true
```

## `run`

| Key | Default | Meaning |
|---|---|---|
| `name` | | names the output directory `outputs_<adiabatic or kinetic>_<name>` |
| `data_dir` | none | GKW run directory: geometry from its `geom.dat`, resume from its K-files |
| `block_size` | `dump_interval * naverage` | steps per block. Diagnostics are written every block |
| `snapshot_every` | 1 | write the restart snapshot every this many blocks |
| `save_dumps` | false | keep every snapshot instead of only the latest |

The command-line flags of the same names (`--block-size`, `--snapshot-every`,
`--save-dumps`) override them.

## `solver`

### Time stepping

| Key | Default | Meaning |
|---|---|---|
| `dt` | 0.01 | timestep. With `adaptive_dt` the starting value and the upper bound |
| `n_steps` | | total steps of the run |
| `naverage` | 40 | steps per normalisation and growth-rate window |
| `dump_interval` | | blocks are `dump_interval * naverage` steps long |
| `non_linear` | false | include the ExB nonlinearity |
| `adaptive_dt` | false | adapt dt to the CFL limit every step (nonlinear runs) |
| `fac_dtim_est` | 0.95 | safety factor of the linear CFL estimate |
| `fac_dtim_nl` | 0.95 | safety factor of the nonlinear CFL estimate |

### Initial condition

| Key | Default | Meaning |
|---|---|---|
| `finit` | `cosine2` | `cosine2`, `cosine`, `cosine3`, `sine`, `noise`, `gnoise` or `zonal` |
| `amp_init` | 1e-4 | initial amplitude |

### Dissipation

| Key | Default | Meaning |
|---|---|---|
| `disp_par` | 1.0 | parallel (s) dissipation |
| `disp_vp` | 0.2 | parallel-velocity dissipation |
| `disp_x`, `disp_y` | 0.1 | radial and binormal hyper-dissipation |
| `idisp` | 2 | velocity scaling of the dissipation, as GKW |
| `disp_par_conserve` | 0 | conservative parallel dissipation (1, 2 or 3, see {class}`~gyaradax.core.params.GKParams`) |
| `disp_par_conserve_kycut` | 0 | apply it only up to this $k_\theta\rho$ |

### Electromagnetic fields

| Key | Default | Meaning |
|---|---|---|
| `nlapar` | false | evolve $A_\parallel$ |
| `nlbpar` | false | evolve $B_\parallel$ |

The plasma $\beta$ is set in `physics`.

### Backend and kernels

See {doc}`../performance/index` for when to use them.

| Key | Default | Meaning |
|---|---|---|
| `backend` | `jax` | `jax`, `cuda`, or `auto` (CUDA when available) |
| `mixed_precision` | true | FP32 FFTs in the nonlinear term |
| `use_z2z` | false | Z2Z FFTs in the JAX bracket |
| `fused_linear` | false | CUDA: one fused kernel per RK4 stage (electrostatic) |
| `linear_arith` | `fp64` | CUDA: `ds` for double-single linear arithmetic on weak-FP64 GPUs |
| `fused_bracket` | false | CUDA: bracket packing inside the cuFFT callbacks |

## `physics`

| Key | Meaning |
|---|---|
| `rlt`, `rln` | gradients $R/L_T$, $R/L_n$ |
| `mas`, `tmp`, `de`, `signz` | mass, temperature, density and charge, relative to the reference |
| `vthrat` | thermal-velocity ratio, default $\sqrt{T/m}$ |
| `dgrid`, `tgrid` | grid normalisation of the Maxwellian, usually 1 |
| `beta` | reference plasma $\beta$ (electromagnetic runs) |
| `betaprime` | pressure-gradient correction of the curvature drift |
| `drift_gradp_type` | `full_drift` or `curv_only`, as GKW |

Species values are scalars for one species and lists (ions first) for several.

Rigid toroidal rotation (`vcor`, `uprim`) is read from GKW input files only.

## `geometry`

| Key | Meaning |
|---|---|
| `geometry_model` | `circ` (default), `s-alpha` or `miller` |
| `q`, `shat`, `eps` | safety factor, magnetic shear, inverse aspect ratio |
| `signB`, `Rref` | sign of the field, reference major radius |
| `kappa`, `delta`, `square`, `skappa`, `sdelta`, `ssquare`, `Zmil`, `dRmil`, `dZmil` | Miller shape |

Configs written by `gyaradax convert` also store derived values (`kthnorm`,
`dvp`, `sgr_dist`, `kxmax`, `kymax`). They are recomputed when absent.

## `grid`

| Key | Meaning |
|---|---|
| `nvpar`, `nmu`, `ns`, `nkx`, `nky` | resolution |
| `adiabatic_electrons` | adiabatic (true) or kinetic electrons |
| `vpar_max` | velocity box, in thermal velocities |
| `nperiod` | the field line covers `2 nperiod - 1` poloidal turns |
| `krhomax` | largest $k_\theta\rho$ |
| `ikxspace` | sets the kx spacing from the shear |

### Choosing the grid

- `nkx` and `ikxspace` fix the radial box. The kx spacing follows the magnetic
  shear, so that at the `iy`-th ky the modes `(2 nperiod - 1) ikxspace iy` cells
  apart connect through the ends of the field line. Kinetic runs need a larger
  `nkx` than adiabatic ones: `configs/kinetic.yaml` uses 192, `configs/iteration_13.yaml` 85.
- `ns` points cover `2 nperiod - 1` poloidal turns. Linear benchmarks need
  enough of them to resolve Landau damping: the Cyclone Base Case uses
  `ns = 160` with `nperiod = 5`. At `ns = 16` its growth rates come out too high.
- The FFT lengths of the nonlinear term follow from `nkx` and `nky`, see
  {doc}`../performance/index` for sizes that run fastest.

## `collisions`

The section switches collisions on (`enabled: true` by default once present).
The defaults are GKW's.

| Key | Default | Meaning |
|---|---|---|
| `pitch_angle`, `en_scatter`, `friction` | true | the three parts of the operator |
| `freq_override` | false | use `freq` instead of the Coulomb-log frequencies |
| `freq` | 0 | ion-ion collision frequency, with `freq_override` |
| `rref`, `nref`, `tref` | 1 | references of the Coulomb logs: m, 10^19 m^-3, keV |
| `mass_conserve` | false | zero flux through the velocity boundaries |
| `mom_conservation`, `ene_conservation` | false | momentum and energy conservation (Xu) |

## `sharding`

| Key | Meaning |
|---|---|
| `n_gpus` | number of GPUs, layout chosen from the grid |
| `n_gpus_sp`, `n_gpus_vp`, `n_gpus_mu` | explicit layout over species, $v_\parallel$ and $\mu$ |

## `debug`

| Key | Meaning |
|---|---|
| `telemetry` | write `telemetry.jsonl` |
| `profile` | trace one block with the JAX profiler |
