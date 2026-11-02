# Performance

```{toctree}
:hidden:

development
```

## Recommended settings

| Use | Settings |
|---|---|
| production nonlinear runs | `--backend cuda`, mixed precision (the default), `--fused-linear` |
| on B300 (weak FP64) | additionally `--linear-arith ds --fused-bracket` |
| gradients, bitwise references | `backend: jax` (the CUDA kernels are not differentiable) |
| grids that do not fit one GPU | `--n-gpus N`; close to the limit, `--mem-fraction 0.95` |

Every switch keeps the physics: the kernels are checked against the JAX
reference per evaluation, and against previous versions over long runs
({doc}`development`). `gyaradax info` shows which kernels the installed library
provides, and `gyaradax bench CONFIG` measures a config on your GPU.

## Throughput

Steps per second, mixed precision, `gyaradax bench` steady state (median of the
timed blocks), mean of two interleaved rounds on one B300, current code:

| grid | CUDA | `--fused-linear --linear-arith ds` | + `--fused-bracket` |
|---|---|---|---|
| adiabatic, 32 x 8 x 16 x 85 x 32 | 85.3 | 137.5 | 136.7 |
| kinetic, 2 x 32 x 12 x 18 x 192 x 32 | 7.36 | 14.98 | 17.18 |

The adiabatic grid is the one of the paper, where GKW ran 5.75 steps/s on a
128-core AMD EPYC 9754 (64 MPI processes) and the JAX backend 30.9 steps/s on a
B300 (paper Table 2). The kinetic grid is the one of `configs/kinetic.yaml`.

CUDA against JAX, ms per step, H100 NVL, mixed precision (October 2026 code;
JAX on its default R2C path):

| case | grid | JAX | CUDA | speed-up |
|---|---|---|---|---|
| ES adiabatic | 32 x 8 x 16 x 85 x 32 | 33.9 | 11.3 | 3.0x |
| ES kinetic | 2 x 32 x 8 x 16 x 85 x 32 | 66.2 | 24.4 | 2.7x |
| kinetic $A_\parallel$ | 2 x 32 x 8 x 16 x 85 x 32 | 103.9 | 31.2 | 3.3x |
| kinetic $A_\parallel + B_\parallel$ | 2 x 32 x 8 x 16 x 85 x 32 | 105.6 | 31.6 | 3.3x |
| adiabatic + $A_\parallel$ | 32 x 8 x 16 x 85 x 32 | 51.1 | 13.5 | 3.8x |

## Where the time goes

Each RK4 stage solves the fields (velocity moments of the distribution), applies
the linear operator (parallel and $v_\parallel$ stencils, drifts, drives,
dissipation, collisions) and, in nonlinear runs, the Poisson bracket. On the
kinetic 85 x 32 grid on an H100 one stage takes about 1.3 ms for the linear RHS,
2.8-3.0 ms for the bracket and 0.5 ms for the electromagnetic field moments.

- **Linear RHS.** One fused kernel per stage reads the stencil coefficients from
  small per-class tables and runs all species in one launch. It is FP64-bound,
  which is why GPUs with weak FP64 (B300) gain from `linear_arith: ds`, which
  evaluates it in double-single arithmetic (pairs of FP32 numbers, about 48 bits)
  on the FP32 units; `fused_linear` also folds the collision stencil and the RK
  update into the kernel.
- **Bracket.** The ExB term needs two inverse 2D FFTs (the two derivatives of each
  field packed into one complex transform) and one forward FFT per stage, in FP32
  with mixed precision. The CUDA bracket fuses the spectral derivatives, the
  gyro-average and the product into [cuFFT LTO callbacks](https://docs.nvidia.com/cuda/cufft/),
  so no intermediate array goes through memory. The v6 version also splits the 2D
  transforms into 1D passes, skipping the zero padding in ky, with cuFFTDx row
  kernels that do the inverse row FFT, the product and the forward row FFT in one
  kernel.

```{figure} ../_static/figs/paper/cufft_lto_fusion_paper.png
:width: 70%
:class: paper-fig

The bracket without (left) and with (right) LTO callbacks; from the paper.
```

```{figure} ../_static/figs/paper/stencil_fusion_v2.png
:width: 70%
:class: paper-fig

The linear RHS as one fused kernel; from the paper.
```

## Grid sizes and FFT lengths

The bracket runs on the dealiased real-space grid. Its lengths are GKW's:

- along y, `mphi` starts at `3 nky - 2`, rounded up to an even length with prime
  factors at most 7, or to a power of two if one lies within 16;
- along x, `mrad` starts at `ceil(1.5 (nkx + 1)) + 1`, rounded up to a length with
  prime factors at most 7, or to a power of two within 8.

| nky | 8 | 16 | 24 | 32 | 40 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| mphi | 32 | 64 | 70 | 96 | 128 | 144 | 192 | 288 | 384 |

| nkx | 43 | 55 | 63 | 85 | 95 | 127 | 191 | 255 |
|---|---|---|---|---|---|---|---|---|
| mrad | 70 | 90 | 98 | 135 | 147 | 196 | 294 | 392 |

Hints:

- Powers of two and lengths of few prime factors run fastest in cuFFT. `nky = 40`
  gives `mphi = 128` for little more than `nky = 32` costs.
- The CUDA v6 bracket has a cuFFTDx row kernel for every `nky` up to
  `BRACKET_V6_NKY_MAX` (build option, default 192). Lengths cuFFTDx can only run
  with a workspace (70, 90, 98, 140, ...) are replaced by the next built length
  that is still alias free, so only round-off changes (`nky = 24` runs at 80).
- Along x the v6 column passes pick the smallest alias-free length that cuFFT runs
  as a single kernel (the GKW length 294 at `nkx = 191` needs three passes).
  `GYARADAX_BRACKET_MRAD=native|auto|N` overrides the choice.
- v6 needs an even number of distribution planes per device; otherwise, and for
  any grid the build does not cover, the bracket runs v5 and warns once with the
  reason. `GYARADAX_BRACKET=v5` selects v5 quietly.
- `fused_linear` and `linear_arith: ds` need `ns <= 576` and fewer than 2^31
  points per device, and are electrostatic only (no conservative dissipation);
  otherwise they fall back to the default kernel with a one-time warning. With the
  momentum or energy conservation corrections on, the collisions run outside
  the fused kernel.

## Memory

The fused stencils are stored as per-class tables rather than full 6D arrays,
and each RK stage's RHS is freed before the next. Peak device memory of one
compiled step on a 2 x 64 x 16 x 32 x 85 x 64 grid with $A_\parallel$ (one df is
5.3 GiB), mixed precision: 62.7 GiB with JAX, 40.0 GiB with CUDA plus about
13 GiB of cuFFT workspace that the bracket allocates outside XLA's pool.

The CLI runs with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. On grids near the
device limit the pool can fragment during setup, so that a large allocation fails
although the compiled peak fits; `--mem-fraction 0.95` or
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` avoid it, and so does sharding.

## Multi-GPU

The distribution function is sharded over species, $v_\parallel$ and $\mu$
(`jax.sharding`). The kernels and the bracket run on each device's local block
through `shard_map`; only the field moments are summed across devices, plus a
two-plane $v_\parallel$ halo per stage when $v_\parallel$ is split (and one plane
for the collision stencil). Species and $\mu$ need no halo. All three axes run
at about the same speed.

On two B300 a 2 x 64 x 16 x 32 x 85 x 64 grid with $A_\parallel + B_\parallel$
goes from 593 to 292 ms per step (CUDA, `mu = 2`). Small grids gain less: the
per-step reductions and kernel launches weigh more. Sharded runs match a single
device at round-off, not bitwise, because the field moments are summed per shard.
Shard over GPUs that are connected by NVLink; on machines whose GPUs only
connect through the host, check that NCCL collectives between the GPUs give
correct sums before trusting a sharded run.
