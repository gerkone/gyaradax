# gyaradax — Agent Guide

A JAX reimplementation of the GKW Fortran gyrokinetic solver. Supports
adiabatic and kinetic electrons, nonlinear ExB advection, adaptive CFL.

## File map

```
gyaradax/
  solver.py      — RK4 integrator, linear RHS (Terms I–VIII), CFL, gksolve
  integrals.py   — phi solve (adiabatic + kinetic), flux integrals, Bessel functions
  geometry.py    — circular geometry model, metric tensors, drift tensors
  params.py      — GKParams dataclass, YAML/input.dat loading
  simulate.py    — high-level entry points (gksimulate, gk_run_batched)
  utils.py       — GKW I/O (K-dumps, geom.dat, input.dat parsing)
  plot_utils.py  — publication-quality plotting functions

scripts/
  run.py         — main entry for running simulations (adiabatic + kinetic)
  gkw_to_yaml.py — convert GKW run directories to YAML configs

tests/unit/      — pytest suite (~160 tests)
gkw_ref/         — GKW Fortran source and manual (read-only reference)
configs/sweep/   — YAML configs for validation trajectories
docs/NOTES.md    — detailed physics notes and validation results
```

## Call graph

```
gksimulate / gk_run_batched          ← entry points (simulate.py)
  └─ gksolve                         ← multi-step driver via jax.lax.scan (solver.py)
       ├─ gkstep_single              ← one RK4 step
       │    ├─ _compute_phi          ← field solve
       │    │    ├─ _phi_adiabatic   ← adiabatic: quasineutrality + zonal FSA correction
       │    │    └─ _phi_kinetic     ← kinetic: multi-species Poisson
       │    ├─ _compute_linear_rhs   ← Terms I, II, IV, V, VII, VIII + dissipation
       │    │    └─ _linear_rhs_core ← inner RHS (vmapped over species for kinetic)
       │    └─ _compute_nonlinear_rhs← Term III: pseudospectral ExB (vmapped over species)
       │         └─ nonlinear_term_iii ← 2D FFT Poisson bracket per s-slice
       └─ estimate_timestep          ← adaptive CFL (nonlinear + von Neumann + field)
            ├─ estimate_nl_timestep   ← max|∇φ| from dealiased FFT
            └─ estimate_linear_timestep ← streaming + trapping + dissipation + field CFL

linear_precompute                    ← one-time setup of all static coefficients
  ├─ _precompute_shared              ← stencils, mode connectivity, FFT metadata
  ├─ _compute_species_coeffs         ← per-species: Bessel, Maxwellian, drifts, drives
  ├─ _fuse_stencils                  ← merge streaming + dissipation into 9-point stencil
  └─ precompute_phi_kinetic          ← static arrays for kinetic field solve

get_integrals                        ← diagnostics at block boundaries
  ├─ calculate_phi                   ← dispatches adiabatic/kinetic
  ├─ calculate_fluxes                ← adiabatic: (pflux, eflux, vflux)
  └─ calculate_fluxes_kinetic        ← kinetic: per-species (nsp, 3) array

compute_geometry                     ← build geometry dict from equilibrium params
  ├─ _parallel_sgrid                 ← field-line coordinate grid
  ├─ _calc_geom_tensors              ← E, D, H, I tensors (ExB, curvature, Coriolis, centrifugal)
  └─ _build_mode_connectivity        ← kx mode labels and parallel boundary maps
```

## GKW ↔ gyaradax mapping

| gyaradax | GKW subroutine | Fortran file |
|----------|----------------|--------------|
| `_phi_adiabatic` | `calculate_fields` + `poisson_zf` | `fields.F90`, `linear_terms.f90` |
| `_linear_rhs_core` | `calc_linear_terms` | `linear_terms.f90` |
| `nonlinear_term_iii` | `calculate_nonlinear` | `non_linear_terms.F90` |
| `estimate_linear_timestep` | `get_estimated_timestep` | `matdat.F90` |
| `init_f` | `init_dist` | `init.f90` |
| `compute_geometry` | `geom_circ` | `geom.f90` |
| `gkstep_single` | `rk4` | `exp_integration.F90` |

## Key concepts

- **Species**: ions (signz=+1) and optionally kinetic electrons (signz=−1).
  Adiabatic electrons use `_phi_adiabatic` with zonal FSA correction.
  Kinetic electrons vmap the linear/nonlinear RHS over species.
- **Terms I–VIII**: GKW numbering for the gyrokinetic equation RHS.
  Term VI (neoclassical/rotation) is not implemented.
  `drive_scale` controls Terms V and VIII jointly — do NOT set to 0.
- **CFL**: adaptive dt from von Neumann analysis + nonlinear ExB velocity.
  For kinetic electrons, the field CFL (electron Alfvén frequency) dominates.
- **Grid**: 5D `(vpar, mu, s, kx, ky)` for adiabatic; 6D `(species, ...)` for kinetic.

## Running tests

```bash
python -m pytest tests/ -x -q
```

GPU required. Set `CUDA_VISIBLE_DEVICES=N` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Running simulations

```bash
# adiabatic
python -u scripts/run.py configs/sweep/iteration_13.yaml --device=N

# kinetic electrons
python -u scripts/run.py configs/kinetic.yaml --kinetic --device=N
```

Add `--from-scratch` to cold-start instead of resuming from K-files.
Add `--block-size=300` for faster checkpoint cadence.

## Common pitfalls

- **JIT caching**: after editing `gyaradax/`, always test in a fresh Python
  process. `importlib.reload` does NOT clear JAX's compilation cache.
- **drive_scale=0 kills Term VIII**: disables the drift-field coupling
  needed for GAM oscillations and correct turbulence dynamics.
- **RH test needs specific params**: rlt=rln=0, all disp=0, drive_scale=1.0.
- **mixed_precision**: defaults to True in run.py. NL FFTs use FP32, linear
  terms and field solver use FP64.

## Skills

| command | description |
|---------|-------------|
| `/run-sim` | Run a simulation from a YAML config |
| `/run-gkw` | Run the GKW Fortran reference and set up input.dat |
| `/run-tests` | Run the pytest suite on a free GPU |
| `/validate` | Compare output dir against GKW reference |
| `/compare-gkw` | Compare a gyaradax function against its GKW Fortran equivalent |
