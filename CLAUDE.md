# gyaradax — Agent Guide

A JAX reimplementation of the GKW Fortran gyrokinetic solver. Supports
adiabatic and kinetic electrons, nonlinear ExB advection, adaptive CFL,
electromagnetic A_parallel (shear Alfvén via Ampere's law, mixed variable g)
and B_parallel (magnetic compression via coupled Poisson-Bpar solve),
with an optional CUDA backend for fused stencil and cuFFT kernels.

## File map

```
gyaradax/
  solver.py      — RK4 integrator, linear RHS (Terms I–VIII), CFL, gksolve
  integrals.py   — phi solve (adiabatic + kinetic), Ampere solve (A_par), Bpar coupled solve, flux integrals, EM flux diagnostics
  geometry.py    — circular/s-alpha geometry model, metric tensors, drift tensors
  params.py      — GKParams dataclass (JAX pytree), YAML/input.dat loading
  simulate.py    — high-level entry points (gksimulate, gk_run, gk_run_batched)
  diag.py        — diagnostics: growth rates, spectra, term_iii_rhs
  types.py       — GKPre (precomputed coeffs pytree), GKState (diagnostic state)
  stencils.py    — 4th-order FD stencil coefficients (parallel + velocity)
  bootstrap.py   — bootstrap utilities
  utils.py       — GKW I/O (K-dumps, geom.dat, input.dat parsing)
  plot_utils.py  — publication-quality plotting functions

  backends/
    __init__.py  — create_ops(): backend dispatch (auto/jax/cuda)
    ops.py       — SolverOps ABC: linear_rhs, nonlinear_term_iii
    _jax.py      — pure JAX backend: stencils, R2C/Z2Z FFT bracket
    _cuda.py     — CUDA FFI backend: fused kernels via libgyaradax_cuda.so
    cuda_kernels/
      CMakeLists.txt — build system (NVCC + cuFFT + LTO callbacks)
      kernels/       — .cu files: stencils, brackets, linear_rhs_fused
      lto_callbacks/ — cuFFT LTO fatbin callbacks (FP32 cast, store)

scripts/
  run.py          — main entry for running simulations (adiabatic + kinetic)
  animate_sim.py  — torus visualization (mp4/gif/html)
  gkw_to_yaml.py  — convert GKW run directories to YAML configs
  solver_benchmark.py — performance benchmarking

tests/
  conftest.py    — fixtures, helpers, backend registry (JAX_BACKENDS, CUDA_BACKENDS, ALL_BACKENDS)
  unit/          — pytest suite
configs/         — YAML configs for simulations and validation sweeps
gkw_ref/         — GKW Fortran source and manual (read-only reference)
docs/NOTES.md    — detailed physics notes and validation results
```

## Call graph

```
gksimulate / gk_run_batched          <- entry points (simulate.py)
  +- gksolve                         <- multi-step driver via jax.lax.scan (solver.py)
       +- gkstep_single              <- one RK4 step
       |    +- _compute_fields       <- field solve (phi + optional A_par)
       |    |    +- _phi_adiabatic   <- adiabatic: quasineutrality + zonal FSA correction
       |    |    +- _phi_kinetic     <- kinetic: multi-species Poisson
       |    |    +- calculate_apar   <- EM: Ampere's law for A_parallel (when nlapar=True)
       |    |    +- g_to_f           <- EM: mixed variable g -> physical f (when nlapar=True)
       |    +- ops.linear_rhs        <- Terms I, II, IV, V, VII, VIII + dissipation (backend dispatch)
       |    |    +- _linear_rhs_core <- inner RHS (JAX backend, 5D/6D, uses chi=phi-2*v_R*v_par*A_par)
       |    +- ops.nonlinear_term_iii<- Term III: pseudospectral ExB (backend dispatch)
       |         +- _nonlinear_term_iii_core <- 2D FFT Poisson bracket per s-slice (JAX backend)
       +- estimate_timestep          <- adaptive CFL (nonlinear + von Neumann + field)
            +- estimate_nl_timestep   <- max|grad(phi)| from dealiased FFT
            +- estimate_linear_timestep <- streaming + trapping + dissipation + field CFL

linear_precompute                    <- one-time setup of all static coefficients
  +- _precompute_shared              <- stencils, mode connectivity, FFT metadata
  +- _compute_species_coeffs         <- per-species: Bessel, Maxwellian, drifts, drives
  +- _fuse_stencils                  <- merge streaming + dissipation into 9-point stencil
  +- precompute_phi_kinetic          <- static arrays for kinetic field solve
  +- precompute_apar                 <- EM: Ampere weights, g2f factor, chi factor (when nlapar=True)

get_integrals                        <- diagnostics at block boundaries
  +- calculate_phi                   <- dispatches adiabatic/kinetic
  +- calculate_fluxes                <- adiabatic: (pflux, eflux, vflux)
  +- calculate_fluxes_kinetic        <- kinetic: per-species (nsp, 3) array
  +- calculate_em_fluxes             <- EM: magnetic flutter pflux/eflux from A_par

compute_geometry                     <- build geometry dict from equilibrium params
  +- _parallel_sgrid                 <- field-line coordinate grid
  +- _calc_geom_tensors              <- E, D, H, I tensors (ExB, curvature, Coriolis, centrifugal)
  +- _build_mode_connectivity        <- kx mode labels and parallel boundary maps
```

## Backend dispatch

```
create_ops(pre, backend="auto", use_z2z=False, mixed_precision=True)
  backend="jax"  -> JAXOps   (pure JAX, R2C or Z2Z FFTs)
  backend="cuda" -> CUDAOps  (FFI kernels, Z2Z only)
  backend="auto" -> CUDAOps if GPU + libgyaradax_cuda.so, else JAXOps
```

SolverOps interface: `linear_rhs()`, `nonlinear_term_iii()`.
5D input (adiabatic), 6D input (kinetic: vmap over species in JAX, loop in CUDA).

## GKW <-> gyaradax mapping

| gyaradax | GKW subroutine | Fortran file |
|----------|----------------|--------------|
| `_phi_adiabatic` | `calculate_fields` + `poisson_zf` | `fields.F90`, `linear_terms.f90` |
| `_linear_rhs_core` (JAX) | `calc_linear_terms` | `linear_terms.f90` |
| `_nonlinear_term_iii_core` (JAX) | `calculate_nonlinear` | `non_linear_terms.F90` |
| `calculate_apar` | `ampere_int` + `ampere_dia` | `linear_terms.f90` |
| `g_to_f` / `f_to_g` | `g2f_correct` | `linear_terms.f90` |
| `_compute_fields` | `calculate_fields` (full EM) | `fields.F90` |
| `estimate_linear_timestep` | `get_estimated_timestep` | `matdat.F90` |
| `init_f` | `init_dist` | `init.f90` |
| `compute_geometry` | `geom_circ` | `geom.f90` |
| `gkstep_single` | `rk4` | `exp_integration.F90` |

## Key concepts

- **Species**: ions (signz=+1) and optionally kinetic electrons (signz=-1).
  Adiabatic electrons use `_phi_adiabatic` with zonal FSA correction.
  Kinetic electrons vmap the linear/nonlinear RHS over species.
- **Terms I-VIII**: GKW numbering for the gyrokinetic equation RHS.
  Term VI (neoclassical/rotation) is not implemented.
  `drive_scale` controls Terms V and VIII jointly -- do NOT set to 0.
- **Electromagnetic (A_parallel)**: Enabled by `nlapar=True, beta>0` in GKParams.
  Evolves the mixed variable g = f + (2Z/T)*v_R*v_par*J0*A_par*F_M.
  Field solve: self-consistent phi + A_par (Ampere's law with g2f correction).
  RHS uses generalized potential chi = phi - 2*v_R*v_par*A_par in drive terms.
  B_parallel not yet implemented (Phase 2).
- **CFL**: adaptive dt from von Neumann analysis + nonlinear ExB velocity.
  For kinetic electrons, the field CFL (electron Alfven frequency) dominates.
  With finite beta, the Alfven CFL is tighter: includes beta in field period.
- **Grid**: 5D `(vpar, mu, s, kx, ky)` for adiabatic; 6D `(species, ...)` for kinetic.
- **Backends**: JAX (default, differentiable, R2C/Z2Z), CUDA (fused kernels, Z2Z only, ~10x NL speedup).

## Running tests

```bash
python -m pytest tests/ -x -q
```

GPU required. Set `CUDA_VISIBLE_DEVICES=N` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Backend registry in `tests/conftest.py`:
- `JAX_BACKENDS`: 4 configs (R2C/Z2Z x FP64/MP)
- `CUDA_BACKENDS`: 2 configs (Z2Z x FP64/MP)
- `ALL_BACKENDS`: auto-detects CUDA availability, includes CUDA only when `is_available()` is True

Run only CUDA backend tests:
```bash
python -m pytest tests/ -x -q -k "cuda"
```

## Running simulations

```bash
# adiabatic
python -u scripts/run.py configs/iteration_13.yaml --device=N

# kinetic electrons
python -u scripts/run.py configs/kinetic.yaml --kinetic --device=N
```

Add `--from-scratch` to cold-start instead of resuming from K-files.
Add `--block-size=300` for faster checkpoint cadence.
Add `--backend=cuda` to force CUDA backend.

## Building the CUDA backend

From `gyaradax/backends/cuda_kernels/`:
```bash
mkdir -p _build && cd _build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cmake --install .
```

Requires CUDA Toolkit >= 13.1, compute capability >= 80.
Pip-installed cuFFT/nvJitLink (`nvidia-cufft-cu12`, `nvidia-nvjitlink-cu12`)
are auto-detected by CMake — look for `CUDA::cufft from pip:` in configure output.

## Common pitfalls

- **JIT caching**: after editing `gyaradax/`, always test in a fresh Python
  process. `importlib.reload` does NOT clear JAX's compilation cache.
- **drive_scale=0 kills Term VIII**: disables the drift-field coupling
  needed for GAM oscillations and correct turbulence dynamics.
- **RH test needs specific params**: rlt=rln=0, all disp=0, drive_scale=1.0.
- **mixed_precision**: defaults to True in run.py. NL FFTs use FP32, linear
  terms and field solver use FP64.
- **CUDA backend**: Z2Z only (use_z2z flag ignored). FFI custom calls are not
  AD-differentiable; gradient tests use JAX backend for nonlinear path.

## Skills

| command | description |
|---------|-------------|
| `/run-sim` | Run a simulation from a YAML config |
| `/run-gkw` | Run the GKW Fortran reference and set up input.dat |
| `/run-tests` | Run the pytest suite on a free GPU |
| `/validate` | Compare output dir against GKW reference |
| `/compare-gkw` | Compare a gyaradax function against its GKW Fortran equivalent |
