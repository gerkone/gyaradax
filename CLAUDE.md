# gyaradax

A JAX reimplementation of the GKW gyrokinetic solver (flux tube, adiabatic or kinetic electrons, electrostatic or electromagnetic, collisions), with an optional CUDA backend. Start from `README.md` and `docs/index.md`.

## Where things are

| Path | What |
|---|---|
| `gyaradax/core/params.py` | `GKParams`: every run parameter, YAML and input.dat loading |
| `gyaradax/geometry/` | circular, s-alpha, Miller and loaded geometries; kx connectivity |
| `gyaradax/operators/` | precompute, stencils, field solves (Poisson, Ampere, Bpar), collisions |
| `gyaradax/solver/` | RK4 step and driver (`gk_init`, `gk_run`, `gksimulate`), CFL, eigenvalue solver |
| `gyaradax/backends/` | `create_ops`: JAX backend, CUDA FFI backend, kernels in `cuda_kernels/` |
| `gyaradax/diagnostics/` | fluxes, spectra, growth rates, telemetry, `plot_utils` |
| `gyaradax/io/` | GKW files, run directories and snapshots |
| `gyaradax/sharding.py` | multi-GPU mesh and `shard_map` wrappers |
| `gyaradax/quasilinear/`, `gyaradax/transport/` | quasilinear rules and the TORAX coupling |
| `gyaradax/cli.py` | the `gyaradax` command |
| `configs/` | ready configs: `adiabatic/`, `kinetic/`, `em/` |

GKW counterparts (GKW source files): `gkstep_single` = `rk4` (exp_integration.F90), `_linear_rhs_core` = `calc_linear_terms` (linear_terms.f90), `_nonlinear_term_iii_core` = `calculate_nonlinear` (non_linear_terms.F90), `_compute_fields` = `calculate_fields` (fields.F90), `g_to_f` = `g2f_correct`, collisions = `collisionop.f90`, `compute_geometry` = `geom.f90`, `estimate_linear_timestep` = `get_estimated_timestep` (matdat.F90), `init_f` = `init_fdisi` (init.f90).

## Docs to read before changing things

- Equations, terms I-XI, grids, boundary conditions, differences from GKW: `docs/numerics/index.md`.
- Config keys: `docs/getting_started/configuration.md`. Commands and run directories: `docs/getting_started/running.md`.
- Kernels, precision switches, throughput, multi-GPU: `docs/performance/index.md`.
- Validation against GKW and theory, how the kernels were built: `docs/validation/`.
- Building the CUDA backend: `docs/getting_started/installation.md`.
- Tests, reference data and parity: `CONTRIBUTING.md`. GKW reference runs and long-horizon references: [gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests) on Hugging Face.

## Pitfalls

- After editing `gyaradax/`, test in a fresh Python process; JAX keeps compiled functions.
- `drive_scale = 0` removes Term VIII (needed for GAMs and correct turbulence); never set it.
- Rosenbluth-Hinton test: `rlt = rln = 0`, all `disp_* = 0`, `drive_scale = 1`.
- The CUDA kernels are not differentiable; gradients use `backend: jax`.
- Mixed precision (the default) runs the nonlinear FFTs in FP32; linear terms and field solves stay FP64.
- Tests and runs need a free GPU: `CUDA_VISIBLE_DEVICES=N XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Skills

`.claude/skills/`: `gyaradax-cli` (configure, run, resume), `gyaradax-plot` (figures, run against reference), `gyaradax-bench` (benchmarks and kernel changes), `gyaradax-datagen` (campaigns of many runs: scheduling over machines, campaign page).

## How to contribute

- **Comments:** one line, lowercase, only on lines that are hard to read. No history, measurements or reasons in code; those go in the docs or the PR.
- **Docstrings:** Google style, what the module, class or function is or does. None on trivial functions or functions under 5 lines.
- **Prose** (docs, PRs, commits): dry and simple, moderate sentence length, no em dashes. Markdown: one line per paragraph.
- **Simple and explicit:** no hidden switches, special cases or thresholds that change results; new options default to the old behaviour and are named in the docs.
- **Numerics changes:** bitwise identical to master and faster is enough; anything else needs long-run statistics against master, adiabatic and kinetic (`gyaradax-bench` skill). Check on the production grids and label any other grid as a test grid.
- **Scope:** one change per PR, with tests; keep local tooling (campaign scripts, harnesses, logs) out of the repository.
