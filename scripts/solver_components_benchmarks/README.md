# solver_components_benchmarks

Isolated benchmarks for gyaradax solver components. Each benchmark loads real solver state
from `iteration_13.yaml`, runs JIT-compiled, and validates output against saved baselines.

## Files

| File | Description |
|------|-------------|
| `bench_apply_parallel.py` | 9-point parallel streaming stencil (`_apply_parallel`) |
| `bench_apply_vpar.py` | 5-point velocity-space stencil (`_apply_vpar`) |
| `bench_linear_rhs.py` | Full linear RHS operator (Terms I, II, IV, V, VII, VIII) |
| `bench_nonlinear.py` | Nonlinear ExB advection (Term III) via pseudospectral FFT |
| `bench_phi_solve.py` | Field solve: quasineutrality (adiabatic) / Poisson (kinetic) |
| `bench_pack_spectrum.py` | Half-spectrum pack/unpack utilities for FFT indexing |
| `bench_rk4_step.py` | Single RK4 integration step (linear + nonlinear phases) |
| `bench_rk4_scan.py` | Multi-step fused RK4 via `jax.lax.scan` |
| `common.py` | Shared utilities: `load_setup`, `BenchTimer`, accuracy checks |
| `generate_baselines.py` | Generate reference baselines (run once after code changes) |
| `run_all.py` | Run all benchmarks sequentially |

## Usage

```bash
# Generate baselines (run once after code changes)
PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
  python solver_components_benchmarks/generate_baselines.py --device <N>

# Run all benchmarks
PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
  python solver_components_benchmarks/run_all.py --device <N> [--mp]

# Run a single component
PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
  python solver_components_benchmarks/bench_nonlinear.py --device <N> [--mp] [--z2z]
```

## CLI Reference

```bash
# Component benchmarks (apply_parallel, apply_vpar, linear_rhs, nonlinear, phi_solve, pack_spectrum)
python bench_<component>.py --device N [--config PATH] [--mp]

# RK4 benchmarks
python bench_rk4_step.py --device N [--config PATH] [--mp] [--backend {jax,cuda}]
python bench_rk4_scan.py --device N [--config PATH] [--mp] [--backend {jax,cuda}] [--nsteps N] [--sweep]

# Nonlinear-specific
python bench_nonlinear.py --device N [--config PATH] [--mp] [--z2z]  # tests R2C + Z2Z
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device N` | 1 | GPU device index (sets `CUDA_VISIBLE_DEVICES`) |
| `--config PATH` | `configs/iteration_13.yaml` | Solver config |
| `--mp` | off | Mixed precision (FP32 forward FFTs) |
| `--backend {jax,cuda}` | both | Run only specified backend |
| `--z2z` | off | Test Z2Z FFT mode (nonlinear only) |
| `--nsteps N` | 50 | Number of RK4 steps (`bench_rk4_scan.py` only) |
| `--sweep` | off | Sweep over N=[1,5,10,25,50,100,200] (`bench_rk4_scan.py` only) |

## Output per benchmark

- **timing** — mean ± std ms/call over 15 trials (3 warmup)
- **bandwidth** — achieved GB/s and % of device peak
- **roofline** — arithmetic intensity and % of BW-limited roofline
- **accuracy** — rel_l2 vs saved baseline; passes if < 1e-10

## Baselines

`baselines/` contains one `.npz` per component with reference inputs and outputs.
Regenerate via `generate_baselines.py` after solver implementation changes.
