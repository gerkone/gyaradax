# gyaradax — Agent Guide

A JAX reimplementation of the GKW Fortran gyrokinetic solver. Supports
adiabatic and kinetic electrons, nonlinear ExB advection, adaptive CFL,
electromagnetic A_parallel (shear Alfvén via Ampere's law, mixed variable g)
and B_parallel (magnetic compression via coupled Poisson-Bpar solve),
linearized Fokker-Planck collision operator (pitch-angle, energy, friction;
adiabatic + single ion MVP), and an optional CUDA backend for fused
stencil and cuFFT kernels (electrostatic and electromagnetic).

## File map

```
gyaradax/
  solver.py      — RK4 step (gkstep_single) and multi-step driver (gksolve)
  fields.py      — field solve wrapper (_compute_fields: phi, A_par, B_par), g <-> f transforms
  integrals.py   — phi solve (adiabatic + kinetic), Ampere solve (A_par), Bpar coupled solve, flux integrals, EM flux diagnostics
  precompute.py  — linear_precompute: stencil class tables, species coefficients, EM factors
  cfl.py         — adaptive CFL timestep estimates
  collisions.py  — Fokker-Planck stencil precompute + apply (9-point in (vpar, mu))
  geometry/      — geometry models (circular, s-alpha, Miller, loaded geom.dat), metric and drift tensors
  params.py      — GKParams dataclass (JAX pytree), YAML/input.dat loading
  state.py       — GKPre (precomputed coeffs pytree), GKState (diagnostic state)
  stencils.py    — 4th-order FD stencil coefficients (parallel + velocity)
  simulate.py    — high-level entry points (gksimulate, gk_run, gk_run_batched)
  sharding.py    — device mesh, sharded precompute / init, shard_map helpers for the kernels
  eigenvalue.py  — linear eigenvalue solver (GKW eiv_integration)
  quasilinear/   — quasilinear transport: linear harvest, saturation rules, calibration
  diag.py        — diagnostics: growth rates, spectra, term_iii_rhs
  utils.py       — GKW I/O (K-dumps, geom.dat, input.dat parsing)
  plot_utils.py  — publication-quality plotting functions
  cli.py         — `gyaradax` console script: run/bench/info (see docs/CLI.md)

  backends/
    __init__.py  — create_ops(): backend dispatch (auto/jax/cuda)
    ops.py       — SolverOps ABC: compute_fields, linear_rhs(_from_g), nonlinear_term_iii
    _jax.py      — pure JAX backend: stencils, R2C/Z2Z FFT bracket
    _cuda.py     — CUDA FFI backend: fused kernels via libgyaradax_cuda.so
    cuda_kernels/
      CMakeLists.txt — build system (NVCC + cuFFT + LTO callbacks, optional cuFFTDx)
      kernels/       — .cu files: linear_rhs_fused, field_moments, v5 / v6 Poisson brackets
      lto_callbacks/ — cuFFT LTO load / store callbacks

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
       |    +- ops.linear_rhs_from_g <- Terms I, II, IV, V, VII, VIII, X, XI + dissipation on f = g_to_f(g)
       |    |    +- _linear_rhs_core <- inner RHS (JAX backend, 5D/6D, uses chi=phi-2*v_R*v_par*A_par)
       |    |    +- linear_rhs_fused <- CUDA backend: all species in one launch, g -> f formed in-kernel
       |    +- ops.nonlinear_term_iii<- Term III: pseudospectral ExB (backend dispatch)
       |         +- _nonlinear_term_iii_core <- 2D FFT Poisson bracket per s-slice (JAX backend)
       +- estimate_timestep          <- adaptive CFL (nonlinear + von Neumann + field)
            +- estimate_nl_timestep   <- max|grad(phi)| from dealiased FFT
            +- estimate_linear_timestep <- streaming + trapping + dissipation + field CFL

linear_precompute                    <- one-time setup of all static coefficients
  +- _precompute_shared              <- stencils, mode connectivity, FFT metadata
  +- _compute_species_coeffs         <- per-species: Bessel, Maxwellian, drifts, drives
  +- _stencil_class_tables          <- fused streaming + dissipation stencils per (class, shift)
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

SolverOps interface: `linear_rhs()`, `linear_rhs_from_g()`, `nonlinear_term_iii()`
(`apar`/`bpar` kwargs build the EM potential chi).
5D input (adiabatic), 6D input (kinetic: vmap over species in JAX, one batched launch in CUDA).

## GKW <-> gyaradax mapping

| gyaradax | GKW subroutine | Fortran file |
|----------|----------------|--------------|
| `_phi_adiabatic` | `calculate_fields` + `poisson_zf` | `fields.F90`, `linear_terms.f90` |
| `_linear_rhs_core` (JAX) | `calc_linear_terms` | `linear_terms.f90` |
| `_nonlinear_term_iii_core` (JAX) | `calculate_nonlinear` | `non_linear_terms.F90` |
| `calculate_apar` | `ampere_int` + `ampere_dia` | `linear_terms.f90` |
| `g_to_f` / `f_to_g` | `g2f_correct` | `linear_terms.f90` |
| `_compute_fields` | `calculate_fields` (full EM) | `fields.F90` |
| `precompute_collisions` / `collision_rhs` | `collision_differential_numu` | `collisionop.f90` |
| `estimate_linear_timestep` | `get_estimated_timestep` | `matdat.F90` |
| `init_f` | `init_dist` | `init.f90` |
| `compute_geometry` | `geom_circ` | `geom.f90` |
| `gkstep_single` | `rk4` | `exp_integration.F90` |

## Key concepts

- **Species**: ions (signz=+1) and optionally kinetic electrons (signz=-1).
  Adiabatic electrons use `_phi_adiabatic` with zonal FSA correction.
  Kinetic electrons vmap the linear/nonlinear RHS over species (JAX); CUDA runs all
  species in one launch.
- **Terms I-VIII**: GKW numbering for the gyrokinetic equation RHS.
  Term VI (neoclassical/rotation) is not implemented.
  `drive_scale` controls Terms V and VIII jointly -- do NOT set to 0.
- **Collisions**: Enabled by `collisions=True, coll_freq>0` in GKParams.
  Linearized Fokker-Planck via 9-point `(vpar, mu)` stencil precomputed
  in `gyaradax/collisions.py` and applied as an additive RHS in
  `_linear_rhs_core`. MVP scope: adiabatic electrons + 1 kinetic ion,
  `freq_override=True` only, `mass_conserve=True`, no momentum/energy
  conservation corrections. Kinetic-electron path raises clearly.
  Flags `coll_pitch_angle`, `coll_en_scatter`, `coll_friction` toggle
  the three pieces independently.
- **Electromagnetic (A_parallel)**: Enabled by `nlapar=True, beta>0` in GKParams.
  Evolves the mixed variable g = f + (2Z/T)*v_R*v_par*J0*A_par*F_M.
  Field solve: self-consistent phi + A_par (Ampere's law with g2f correction).
  RHS uses generalized potential chi = phi - 2*v_R*v_par*A_par in drive terms.
- **Electromagnetic (B_parallel)**: `nlbpar=True` adds magnetic compression via the
  coupled Poisson-B_par solve; chi gains bpar_chi_factor * B_par (terms X, XI).
- **CFL**: adaptive dt from von Neumann analysis + nonlinear ExB velocity.
  For kinetic electrons, the field CFL (electron Alfven frequency) dominates.
  With finite beta, the Alfven CFL is tighter: includes beta in field period.
- **Grid**: 5D `(vpar, mu, s, kx, ky)` for adiabatic; 6D `(species, ...)` for kinetic.
- **Backends**: JAX (default, differentiable, R2C/Z2Z), CUDA (fused kernels, Z2Z only; 2.8-4.4x faster
  steps than JAX on H100, ES and EM incl. A_par/B_par, conservative parallel dissipation and
  collisions — see docs/NOTES.md §10.14).

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

Use the `gyaradax` console script (implemented in `gyaradax/cli.py`; see
`docs/CLI.md`). Adiabatic vs kinetic, EM terms and sharding are all read from
the config — there is no `--kinetic` flag any more.

```bash
gyaradax run configs/iteration_13.yaml --device=N     # adiabatic
gyaradax run configs/kinetic.yaml --device=N          # kinetic (auto-detected)
gyaradax run configs/nl_em_apar.yaml --n-gpus-mu=4    # EM, sharded over 4 GPUs
gyaradax bench configs/adiabatic_a.yaml --backend=cuda
gyaradax info
```

Add `--from-scratch` to cold-start instead of resuming from K-files.
Add `--block-size=300` for faster checkpoint cadence.
Add `--backend=cuda` to force CUDA backend.
`scripts/run.py` is a back-compat shim that forwards to `gyaradax run`.

## Building the CUDA backend

The base package depends on CPU-compatible `jax`. Install the CUDA extra before building or using the CUDA backend:
```bash
pip install -e ".[cuda13]"
```

From `gyaradax/backends/cuda_kernels/`:
```bash
mkdir -p _build && cd _build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cmake --install .
```

Requires CUDA Toolkit >= 13.1, compute capability >= 80.
One library can serve several GPU generations, e.g. `-DGPU_ARCHITECTURES="90;103"`
(H100 + B300).

On older toolkits, override the two architecture lists — `GPU_ARCHITECTURES`
(the kernels) and `LTO_ARCHITECTURES` (the cuFFT LTO callbacks). `compute_100`
needs CUDA >= 12.8, so e.g. on a CUDA 12.6 / GH200 system:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_ARCHITECTURES=90 -DLTO_ARCHITECTURES="80;90"
```
`scripts/make_cuda_root.sh` builds a merged toolkit root when nvcc and cuFFT
live in separate trees (e.g. the NVIDIA HPC SDK).
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
- **Parallel stencils**: both backends read the per-class stencil tables
  (`s_upar_tab`, `s_t7_tab`, `s_disp_par_tab`, indexed by `par_stencil_class`);
  the full 9 x 6D stencil arrays are no longer stored. The CUDA kernel also reads
  the EM velocity factors (`apar_chi_vfac`, `g2f_vfac`) from `linear_precompute`.
- **Large grids (JAX)**: the bracket loops over species / vpar chunks once one
  real-space intermediate would exceed `_NL_CHUNK_BYTES` (1 GiB); smaller grids
  run the single batched path.

## Multi-GPU Sharding

Velocity-space grid parallelism is supported via JAX GSPMD. Set via params:

```python
params = GKParams(
    n_gpus_sp=1,   # Shard species axis (for kinetic multi-species)
    n_gpus_vp=2,   # Shard vparallel axis
    n_gpus_mu=1,   # Shard mu axis
    ...
)
```

Or via YAML config:

```yaml
sharding:
  n_gpus_sp: 1
  n_gpus_vp: 2
  n_gpus_mu: 1
```

When `n_gpus_sp * n_gpus_vp * n_gpus_mu > 1`, the following automatically use
sharding:

- `init_f()` - Creates df already distributed across devices (no single-GPU OOM)
- `linear_precompute()` - Precomputes coefficients sharded
- `gksolve()` - Runs simulation with sharded arrays

The mesh is built automatically from available GPUs. Arrays are sharded as:
- 5D df (vpar, mu, s, kx, ky): ("vp", "mu", None, None, None)
- 6D df (sp, vpar, mu, s, kx, ky): ("sp", "vp", "mu", None, None, None)
- Fields (phi, A_par): Replicated across all devices

**Note**: Field solves require all-reduce operations over velocity axes,
which can limit scaling for small grids. Sharding is most beneficial for
large grids (≥128×32 velocity space) that don't fit on a single GPU.

`gksolve` passes the mesh to `create_ops(mesh=...)`. GSPMD cannot partition
the FFI kernels or the bracket's FFTs (it all-gathers their operands), so the
bracket (both backends), the CUDA linear / field-moment kernels and the JAX
vpar stencil run on the local (sp, vp, mu) blocks via `sharding.velocity_map`
(`shard_map`). Sharding vpar exchanges the two vpar planes next to each shard
edge per RK stage (`sharding.vpar_halo_planes`); species and mu need no halo.
All three axes run at about the same speed. Multi-GPU needs NCCL
(`nvidia-nccl-cu13`, in the `cuda13` extra). See docs/NOTES.md §10.14.

## Skills

| command | description |
|---------|-------------|
| `/run-sim` | Run a simulation from a YAML config |
| `/run-gkw` | Run the GKW Fortran reference and set up input.dat |
| `/run-tests` | Run the pytest suite on a free GPU |
| `/validate` | Compare output dir against GKW reference |
| `/compare-gkw` | Compare a gyaradax function against its GKW Fortran equivalent |
