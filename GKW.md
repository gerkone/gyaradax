# GKW -> JAX Port Notes (Linear V1, Adiabatic Electrons)

## 1) Scope and locked decisions

- Physics scope: electrostatic, adiabatic-electron case only.
- Solver-space ordering: `(vpar, mu, s, kx, ky)` (authoritative naming: `kx`, not `x`).
- Target grid: `(32, 8, 16, 85, 32)`.
- Time step: `dt = 0.01`.
- Precision: fp64 only (`jax_enable_x64=True`, `complex128/float64` arrays).
- Integration scope for V1: linear-only update path.
- Baseline data:
  - Primary linear diagnostics baseline: `/restricteddata/ukaea/gyrokinetics/raw/iteration_13_Lin`.
  - Secondary/debug baseline with 5D dumps: `/restricteddata/ukaea/gyrokinetics/raw/iteration_13`.

## 2) Relevant GKW sources for this stage

### Time integration and update loop

- `gkw_ref/src/gkw.f90`
  - Main loop over large time steps.
  - Calls `advance_large_step_explicit` for `method='EXP'`.
  - Calls `normalise_after_timestep()`.

- `gkw_ref/src/exp_integration.F90`
  - `advance_large_step_explicit`: loops small steps (`naverage`), updates `time`, updates fields/cons.
  - `rk2`, `rk4`, `rk3`, `rkc_2` explicit schemes.
  - `calculate_rhs`: canonical RHS assembly order.

### Linear operators / field coupling

- `gkw_ref/src/linear_terms.f90`
  - `calc_linear_terms`: builds linear matrices and field-coupling matrices.
  - Field split:
    - `poisson_int`
    - `poisson_dia`
    - `poisson_zf` (adiabatic + zonal correction)

- `gkw_ref/src/fields.F90`
  - `calculate_fields`: applies field solve and zonal adiabatic correction.
  - `calculate_cons`: collision-conservation fields (inactive in this configuration).

### Configuration / species model

- `gkw_ref/src/control.f90`
  - `method`, `meth`, `dtim`, `naverage`, dissipation controls.
  - `zonal_adiabatic`, normalization controls.

- `gkw_ref/src/components.f90`
  - `adiabatic_electrons`, species setup, quasi-neutrality checks.

### Growth-rate diagnostics

- `gkw_ref/src/diagnos_growth_freq.f90`
  - `calc_amplitudes`, `diagnostic_growth_rate`, mode-label mapping.
  - Writes:
    - `growth.dat` (kx=0 slice vs ky),
    - `growth_rates_all_modes` (flattened global mode labels),
    - `frequencies.dat`,
    - `frequencies_all_modes`.

## 3) Manual references used

- `gkw_ref/manual/practise.tex`
  - Complete set of equations and adiabatic Poisson form (`Poisson-adiabatic`).

- `gkw_ref/manual/implementation.tex`
  - Explicit integration overview.
  - Mapping from equation terms to code routines.
  - Poisson splitting details and matrix interpretation.

- `gkw_ref/manual/diagnostics.tex`
  - Growth/frequency output definitions and file conventions.

- `gkw_ref/manual/buildandrun.tex`
  - Input options for adiabatic electrons, `dtim`, and output behavior.

## 4) Data observations for baselines

### iteration_13 (nonlinear run with 5D dumps)

- `input.dat` key controls:
  - `non_linear = .true.`
  - `method='EXP'`, `meth=2`, `dtim=0.01`, `naverage=40`
  - `adiabatic_electrons = .true.`
  - `zonal_adiabatic = .true.`

- Contains K-like distribution dumps (`K*` + numeric files), plus diagnostics.

### iteration_13_Lin (linear diagnostics baseline)

- `input.dat` key controls:
  - `non_linear = .false.`
  - `method='EXP'`, `meth=2`, `dtim=0.01`, `naverage=40`
  - `adiabatic_electrons = .true.`
  - `zonal_adiabatic = .true.`
  - `normalize_per_toroidal_mode = .true.`

- No 5D `K*` or numeric per-step distribution dumps.
- Diagnostics available:
  - `Poten*` (periodic field dumps),
  - `growth.dat`,
  - `growth_rates_all_modes`,
  - `frequencies.dat`,
  - `frequencies_all_modes`,
  - `time.dat` (legacy 3-column format),
  - `mode_label`.

### Exact mapping confirmed in data

- For both `iteration_13` and `iteration_13_Lin`:
  - `growth.dat` is exactly the kx=0 slice of `growth_rates_all_modes` using `mode_label`.
  - `frequencies.dat` is exactly the kx=0 slice of `frequencies_all_modes` using `mode_label`.

This is the diagnostic identity used for growth/frequency validation tests in V1.

## 5) JAX implementation status for this stage

### Existing (verified before solver work)

- `jax_integrals.py`
  - `get_integrals(df, geometry) -> (phi, (pflux, eflux, vflux))`
  - fp64-enabled and jit-compatible.

- `jax_geometry.py`
  - Loads geometry/data into JAX arrays used by integrals and solver.

### Added for linear V1

- `gksolver.py`
  - `gksolve(prev_df, geometry, params, state)`
    - Required interface: returns `next_df, (phi, fluxes)`.
  - `gksolve_with_state(...)`
    - Same physics step, additionally returns explicit diagnostic state.
  - `GKParams`, `GKState`, `default_state`.
  - Utility functions:
    - `kx0_mode_columns`
    - `project_all_modes_to_kx0`

## 6) Linear V1 update model implemented

- Time integrator:
  - One explicit RK4 small step per call (`dt=0.01` default).

- RHS structure (linear-only V1):
  - Spectral damping term in `(kx, ky)` using configured dissipation coefficients.
  - Electrostatic drive coupling through `phi` from `get_integrals`.
  - Uses geometry tensors (`vpgr`, `mugr`, `bn`, `rln`, `rlt`) for velocity-space drive weighting.

- Post-step normalization:
  - Always applied.
  - Per-`ky` mode normalization based on `phi` amplitude.
  - Explicit state keeps cumulative normalization factor and window growth tracker.

Notes:
- This V1 is intentionally linear and reduced-order relative to full GKW matrix assembly.
- It is designed as a pure, differentiable, jit-compatible stepping core for incremental extension.

## 7) Tests added (TDD stage artifact)

- `test_gksolver_linear.py` includes:
  - API contract + shape/dtype checks (`(32, 8, 16, 85, 32)` and scalar flux outputs).
  - JIT-compatibility checks for stateful stepping function.
  - Zero-input invariance.
  - Determinism and finiteness checks.
  - Exact growth/frequency diagnostic mapping checks:
    - `growth_rates_all_modes` -> `growth.dat` via `mode_label` at `kx=0`.
    - `frequencies_all_modes` -> `frequencies.dat` via same mapping.
  - `_Lin` dataset diagnostics-only assertion (no `K*`/numeric 5D dumps) plus `Poten*` cadence check.

## 8) Known limitations / next extension points

- Full linear operator parity with all GKW terms I, II, IV, V, VII, VIII is not complete yet.
- Nonlinear term III is intentionally not included in this V1.
- No collision/neoclassical/electromagnetic terms in this stage.
- Next steps should expand RHS term-by-term against `linear_terms.f90` while preserving:
  - pure function semantics,
  - jit compatibility,
  - fp64 strictness,
  - no output-forcing/normalization sweeps.

## 9) Phase 1 completion notes (init + geometry parity)

- `jax_geometry.py` now loads additional geometry/state needed for active linear ES terms:
  - `gfun <- geom.dat:G`
  - `dfun <- (D_eps, D_zeta, D_s)`
  - `hfun <- (H_eps, H_zeta, H_s)` and `ifun <- (I_eps, I_zeta, I_s)` (for upcoming drift parity)
  - `sgrid` and `sgr_dist`
  - `mode_label` (loaded from `mode_label`)
  - `kxmax`, `kymax`

- Spectral parallel-connectivity metadata is now derived directly from `mode_label`:
  - `ixplus[kx,ky]`, `ixminus[kx,ky]` with `-1` denoting open boundary.
  - `ixzero`, `iyzero` from nearest-zero entries of `kxrh`, `krho`.
  - `pos_par_grid_class[s,kx,ky]` in `{ -2, -1, 0, 1, 2 }`, matching GKW open-boundary stencil classes for term-I / term-VII differencing.

- Initialization helper added in `gksolver.py`:
  - `init_df_cosine2(...)` is a JAX port of `init_fdis` branch `finit='cosine2'` for this run scope.
  - Implements zonal suppression for mode-box (`ky=0` seeded to zero when `nky>1`).
  - Optional startup normalization reproduces GKW per-toroidal-mode behavior:
    - uses `phi`-based mode amplitude `sqrt(ds * sum_{s,kx}|phi|^2)`,
    - tiny amplitudes guarded to factor 1,
    - applies one global startup rescaling of `df` per `ky`.

- Phase-1 tests added to `test_gksolver_linear.py`:
  - geometry/connectivity key and shape checks;
  - `init_df_cosine2` contract + zonal suppression check;
  - startup normalization check (`|phi|` mode amplitude equals 1 for non-zonal modes).

## 10) Phase 2 completion notes (active linear ES terms)

- `gksolver.py` now implements the active linear electrostatic spectral terms used by the
  `_Lin` setup, with state ordering `(vpar, mu, s, kx, ky)`:
  - Term I: `vpar_grd_df` with open parallel boundaries and `mode_label`-driven `kx` connectivity.
  - Parallel dissipation (`disp_par`) with `idisp` branch handling (`idisp=2` active for this run).
  - Term IV: `dfdvp_trap`.
  - `dfdvp_dissipation` (`disp_vp`) with zero-outside-`vpar` boundaries.
  - Term II: `vdgradf` from drift dot wavevector.
  - `hyper_disp_perp` spectral branch from `disp_x`, `disp_y`, `kxmax`, `kymax`.
  - Term V: `ve_grad_fm` electrostatic branch.
  - Term VIII: `vd_grad_phi_fm` electrostatic branch.
  - Term VII: `vpar_grd_phi` electrostatic branch.

- Stencil parity details:
  - Uses fourth-order differential coefficients from `linear_terms.f90::differential_scheme`.
  - Open-boundary `s` stencils use precomputed shift maps:
    - `s_shift`, `kx_shift`, `valid_shift` (shape `[9, ns, nkx, nky]`).
  - `vpar` stencils use zero-outside-grid behavior consistent with `elem_is_on_vpar_grid`.

- Geometry/runtime data additions in `jax_geometry.py`:
  - `vpgr_rms`, `mugr_rms`, `dvp`.
  - Precomputed parallel shift maps (`s_shift`, `kx_shift`, `valid_shift`).

- Normalization cadence fix:
  - `gksolve_with_state` now applies per-`ky` normalization only at large-step boundaries
    (`step % naverage == 0`), matching GKW’s `normalise_after_timestep` cadence.
  - No normalization on intermediate small steps.

- Validation snapshot (`iteration_13_Lin`, `DM2 -> 80 small steps -> FDS`):
  - `rel_l2(df_pred, FDS) = 8.9098e-06`
  - `rel_l2(phi_pred, phi_FDS) = 7.6035e-06`
  - Flux comparison:
    - predicted `(p,e,v) = (8.88e-16, 38.2988647344, 9.2704e-14)`
    - reference `(p,e,v) = (8.88e-16, 38.2987731059, 9.3925e-14)`
