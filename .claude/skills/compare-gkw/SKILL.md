---
name: compare-gkw
description: Compare a gyaradax function against its GKW Fortran equivalent
disable-model-invocation: true
argument-hint: <function-name>
---

Look up the GKW Fortran equivalent of a gyaradax function and compare implementations.

## Mapping

| gyaradax | GKW Fortran | file |
|----------|-------------|------|
| `_phi_adiabatic` | `calculate_fields` + `poisson_zf` | `fields.F90`, `linear_terms.f90` |
| `_phi_kinetic` | `calculate_fields` | `fields.F90` |
| `_linear_rhs_core` | `calc_linear_terms` | `linear_terms.f90` |
| `nonlinear_term_iii` | `calculate_nonlinear` | `non_linear_terms.F90` |
| `estimate_linear_timestep` | `get_estimated_timestep` | `matdat.F90` |
| `estimate_nl_timestep` | lines 1530–1777 | `non_linear_terms.F90` |
| `init_f` | `init_dist` | `init.f90` |
| `compute_geometry` | `geom_circ` | `geom.f90` |
| `gkstep_single` | `rk4` | `exp_integration.F90` |
| `mode_amplitude` | `diagnos_growth_freq` | `diagnos.f90` |

Read both the gyaradax function and the corresponding GKW subroutine from `gkw_ref/src/`.
Report differences in: formula, sign conventions, normalization, array ordering, boundary handling.
Cross-reference `docs/NOTES.md` for the physics equations (Terms I–VIII).
