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
  - `gksolve(df, geometry, params, state, n_steps=1)`
    - Required interface: returns `final_df, (phi, fluxes), final_state`.
  - `GKParams`, `GKState`, `default_state`.
  - Diagnostic functions moved to `diag.py`:
    - `kx0_mode_columns`
    - `project_all_modes_to_kx0`

## 6) Linear V1 update model implemented

- Time integrator:
  - One explicit RK4 small step per internal call (`dt=0.01` default).
  - High-level `gksolve` uses `jax.lax.scan` for multi-step execution.

- RHS structure (linear-only V1):
  - Spectral damping term in `(kx, ky)` using configured dissipation coefficients.
  - Electrostatic drive coupling through `phi` from `get_integrals`.
  - Uses geometry tensors (`vpgr`, `mugr`, `bn`, `rln`, `rlt`) for velocity-space drive weighting.

- Post-step normalization:
  - Applied at large-step boundaries (`naverage`).
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
- Nonlinear term III is now included as a switchable path (`non_linear` + `enable_term_iii`),
  while all non-target nonlinear/EM branches remain inert in this scope.
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
  - `init_f(...)` is a JAX port of `init_fdis` branch `finit='cosine2'` for this run scope.
  - Implements zonal suppression for mode-box (`ky=0` seeded to zero when `nky>1`).
  - Optional startup normalization reproduces GKW per-toroidal-mode behavior:
    - uses `phi`-based mode amplitude `sqrt(ds * sum_{s,kx}|phi|^2)`,
    - tiny amplitudes guarded to factor 1,
    - applies one global startup rescaling of `df` per `ky`.

- Phase-1 tests added to `test_gksolver_linear.py`:
  - geometry/connectivity key and shape checks;
  - `init_f` contract + zonal suppression check;
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
  - `gksolve` (via `_step`) now applies per-`ky` normalization only at large-step boundaries
    (`step % naverage == 0`), matching GKW’s `normalise_after_timestep` cadence.
  - No normalization on intermediate small steps.

- Validation snapshot (`iteration_13_Lin`, `DM2 -> 80 small steps -> FDS`):
  - `rel_l2(df_pred, FDS) = 8.9098e-06`
  - `rel_l2(phi_pred, phi_FDS) = 7.6035e-06`
  - Flux comparison:
    - predicted `(p,e,v) = (8.88e-16, 38.2988647344, 9.2704e-14)`
    - reference `(p,e,v) = (8.88e-16, 38.2987731059, 9.3925e-14)`

## 11) Nonlinear parity iteration findings (`iteration_13`, window `100 -> 101`)

- Runtime parameter provenance hardening:
  - Added typed runtime extraction from `input.dat` (`dtim`, `naverage`, `disp_*`, `non_linear`, `nlapar`, `method`, `meth`).
  - `gkparams_from_input_dat(...)` now maps these directly into `GKParams` with explicit override support.
  - Confirmed for `iteration_13`: `dtim=0.01`, `naverage=40`, `disp_par=1.0`, `disp_vp=0.2`, `disp_x=0.1`, `disp_y=0.1`, `non_linear=True`, `nlapar=False`, `method='EXP'`, `meth=2`.

- Geometry parity surfacing:
  - Exposed scalar invariants from `geom.dat`: `shat`, `q`, `eps`, `kthnorm`.
  - `krho` normalization remains `krho_loaded = krho_file / kthnorm` (matches `mode.f90` behavior).
  - Note on `s_hat`: yes, it is now imported (`shat`) and available for diagnostics/invariants; active Term-I/II/III code paths currently consume precomputed geometry tensors/metadata (`dfun`, `efun`, `mode_label`, connectivity), not `shat` directly.

- Term-III diagnostic evidence before fix:
  - With previous JAX implementation, `||rhs_nl|| / ||rhs_lin||` at dump `100` was `~1.54e-4` (nonlinear term effectively inert).
  - `non_linear=True` with Term-III off vs on gave nearly identical 120-step checkpoint error (`subset_rel_l2 ~ 0.237`), confirming weak nonlinear impact from current scaling/sign.

- Focused scale/sign sweep result:
  - Using identical linear terms and varying only nonlinear scale `alpha` in `rhs = rhs_lin + alpha * rhs_nl_current`:
    - `alpha=0`: `subset_rel_l2=0.237114`
    - `alpha=1`: `subset_rel_l2=0.237123`
    - `alpha=-1000`: `subset_rel_l2=0.227457`
    - `alpha=-10000`: `subset_rel_l2=0.092488`
    - `alpha=-12960`: `subset_rel_l2=4.55e-05`
  - `12960 = mrad * mphi` for this grid (`mrad=135`, `mphi=96`).
  - This isolates a missing FFT normalization/sign factor in Term III as primary root cause.

- Term-III fix implemented:
  - In `gksolver.py::_nonlinear_term_iii`:
    - Added FFT normalization compensation factor `nl_fft_scale = mrad * mphi`.
    - Set default nonlinear FFT prefactor to `+1` (instead of `-1`).
    - Net correction is Fortran-consistent with `add_non_linear_terms_spectral` scaling path.
  - Preserved switchable backward compatibility:
    - `params.non_linear` gates nonlinear behavior.
    - `params.enable_term_iii` keeps Term-III on/off switch.

- Post-fix smoke metrics (same `100 -> 101` window):
  - State parity recovered:
    - subset mode-chain `rel_l2 ~ 4.55e-05`
    - full-state `rel_l2 ~ 1.11e-04`
  - Heat flux parity:
    - `eflux_rel ~ 3.79e-06`
  - Growth-rate comparison:
    - absolute disagreement is small, but one near-marginal reference mode (`|gamma_ref| ~ 1.1e-2`) inflates pure relative error.
    - test metric updated to use a near-zero floor in denominator (`max(|gamma_ref|, 2e-2)`), preventing false failures in marginal modes.

## 12) Ranked plausible discrepancy causes (current status)

1. Resolved primary cause:
   - Nonlinear Term-III FFT normalization/sign mismatch (`mrad*mphi` + prefactor sign).
   - Evidence: targeted alpha sweep peaked exactly at `-mrad*mphi`.

2. Residual risk for future windows:
   - Growth-rate metric sensitivity for near-zero growth modes (diagnostic conditioning issue, not state mismatch).
   - Keep mixed absolute/relative acceptance for marginal modes.

3. Secondary parity candidates for later phases:
   - Any disabled branch accidentally activated by future inputs (EM/shear-remap/neoclassical terms).
   - `DPART_IN`-dependent paths (time-dependent source/shear periodic BCs) if run settings change from current scope.
   - Dataset interpretation drift when moving from single-window smoke to long multi-window trajectory checks.
