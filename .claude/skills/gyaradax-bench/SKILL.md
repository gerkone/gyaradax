---
name: gyaradax-bench
description: Benchmark and validate a change to the gyaradax solver or its CUDA kernels (throughput, per-kernel accuracy, bitwise trajectory checks, long-run statistics against master). Use for any optimisation, kernel or numerics change, and before opening its PR.
---

# Benchmarking and kernel development

References: `docs/performance/index.md` (configs, kernel options, throughput tables), `docs/validation/development.md` (the optimisation loop), `CONTRIBUTING.md` (showing parity), `notebooks/validation.ipynb` with `notebooks/validation_stats.py` (long-run comparison).

**The rule: bitwise identical and faster is enough. Anything not bitwise identical needs the long-run checks below.**

## 1. Reference

Build the base (master, or the PR's base) in its own worktree and keep it untouched. Every comparison is new code against that build, CUDA against CUDA (JAX only for gradients or per-kernel reference values).

## 2. Throughput

```bash
gyaradax bench configs/adiabatic/nonlinear/itg_high_shear.yaml --device N --backend cuda --steps 100 --blocks 5 [kernel flags]
gyaradax bench configs/kinetic/nonlinear/benchmark.yaml        --device N --backend cuda --steps 100 --blocks 5 [kernel flags]
```

- Only these production grids (adiabatic 32x8x16x85x32, kinetic 2x32x12x18x192x32); speedups on smaller grids do not carry over.
- An exclusive GPU, old and new alternated, steady-state median steps/s and peak memory. Name the GPU; a switch that helps one GPU type and hurts another stays off by default.

## 3. Per-kernel accuracy

Compare each changed kernel with the JAX reference on the production grids: about 1e-16 for the FP64 linear RHS, 1e-15 for the FP64 bracket, 1e-6 for the mixed-precision bracket.

## 4. Bitwise trajectory check

Restart a saturated adiabatic and a saturated kinetic run from a snapshot and step the old and the new build for thousands of steps (fixed and adaptive dt, FP64 and mixed precision); the distribution functions must be bitwise identical. Set `XLA_FLAGS=--xla_gpu_autotune_level=0` on both sides, since autotuning changes the last bit even between identical builds. Passing steps 2 and 4: done.

## 5. Long-run statistics (anything not bitwise)

- Run the long-horizon cases of `gerkone/gyaradax-tests` (`long_horizon/`: adiabatic and kinetic, nonlinear and linear, configs in `configs/`) with the new code to saturation, and compare with `notebooks/validation.ipynb`: heat flux per species averaged over the last 30 % with autocorrelation-corrected errors, stationarity, ky and kx spectra, converged growth rates. Also restart from a saturated reference state and compare averages.
- Report differences in units of the combined error. Trajectories decorrelate after a few tens of R/v_th; that is expected, a shift of the averages is not.
- Linear-only changes: converged growth rates, frequencies and quasilinear weights against converged references.

## 6. The PR

Give the steps/s table (GPU, grid, old and new, speedup), the accuracy numbers and, for non-bitwise changes, the long-run table. Attach figures for every long-run case: flux traces and saturated ky/kx spectra from `python scripts/compare_runs.py NEW_RUN REF_RUN --label NEW --ref-label master` (the `gyaradax-plot` skill). Push the PNGs to the `pr-assets` branch and link them as `https://github.com/gerkone/gyaradax/blob/pr-assets/NAME.png?raw=true`; do not commit them to the code branch.
