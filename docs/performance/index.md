# Performance

## Recommended settings

| Use | Settings |
|---|---|
| production nonlinear runs | `--backend cuda`, mixed precision (the default), `--fused-linear` |
| on B300 (weak FP64) | additionally `--linear-arith ds --fused-bracket` |
| gradients, bitwise references | `backend: jax` (the CUDA kernels are not differentiable) |
| grids that do not fit one GPU | `--n-gpus N`; close to the limit, `--mem-fraction 0.95` |

Every switch keeps the physics: the kernels are checked against the JAX reference per evaluation, and against previous versions over long runs
({doc}`../validation/development`). `gyaradax info` shows which kernels the installed library
provides, and `gyaradax bench CONFIG` measures a config on your GPU.

## Throughput

Both grids are the production grids of the configs in the repository, as $(n_{sp}, n_{v_\parallel}, n_\mu, n_s, n_{k_x}, n_{k_y})$:

- adiabatic: 32, 8, 16, 85, 32 (`configs/iteration_13.yaml`),
- kinetic: 2, 32, 12, 18, 192, 32 (`configs/kinetic.yaml`).

The columns are the backends and kernel options of the `solver` section:

- **JAX**: `backend: jax`, the pure JAX solver.
- **CUDA v6**: `backend: cuda` with the default kernels and the v6 bracket.
- **fused_lin**: `fused_linear: true`. One CUDA kernel per RK4 stage for the electrostatic linear RHS, with the collision stencil and the RK update.
- **ds**: `linear_arith: ds`. The linear RHS in double-single arithmetic, pairs of FP32 numbers with about 48 bits, for GPUs with weak FP64.
- **fused_nl**: `fused_bracket: true`. The mixed-precision bracket packs and unpacks its data inside the cuFFT callbacks. Bitwise identical results on B300, also with collisions.

Steps per second in mixed precision, from `gyaradax bench` (median of four timed blocks of 30 steps, after compilation). The peak device memory during the run, from `nvidia-smi`, is in brackets in GB.

B300:

| grid | JAX | CUDA v6 | + fused_lin + ds | + fused_lin + ds + fused_nl |
|---|---|---|---|---|
| adiabatic | 31.2 (5.9) | 85.5 (2.6) | **137.8** (3.2) | 136.8 (3.2) |
| kinetic | 4.25 (25.4) | 8.99 (20.2) | 16.6 (20.2) | **20.2** (20.2) |

H100:

| grid | JAX | CUDA v6 | + fused_lin | + fused_lin + ds | + fused_lin + ds + fused_nl |
|---|---|---|---|---|---|
| adiabatic | 31.7 (5.8) | 90.1 (2.5) | **124.3** (3.6) | 120.5 (3.1) | 99.2 (3.1) |
| kinetic | 4.29 (25.3) | 9.61 (20.1) | **15.8** (20.1) | 15.5 (20.1) | 13.5 (20.1) |

On H100, `ds` makes no difference because its FP64 units are fast, and `fused_nl` is slower.

## Other codes

The same two cases in other codes, on the same GPUs:

- **`gyaradax` (`gyx`)**: `gyx, best` is the fastest setting per GPU from the tables above, fused_lin + ds + fused_nl on B300 and fused_lin on H100. `gyx, v6` is CUDA v6 with the default kernels. Mixed precision. Velocity grids of $(n_{v_\parallel}, n_\mu)$ = (32, 8) adiabatic and (32, 12) kinetic. RK4, fixed step $0.01\,R/v_{th}$ adiabatic, adaptive step kinetic.
- **[GKX](https://github.com/uwplasma/gkx) (FP32)**: version 2.8.0 in its default single precision, with its default Hermite-Laguerre velocity space, $(N_l, N_m)$ = (4, 8) adiabatic and (8, 16) kinetic, so 32 and 128 moments. RK3 with its adaptive CFL step.
- **[CGYRO](https://github.com/gafusion/gacode)**: gacode commit 1df9cdc, built with NVHPC 26.3 (OpenACC and cuFFT), one MPI rank per GPU. FP64, default velocity space $(n_\xi, n_{energy})$ = (16, 8), so 128 points. RK4 with a fixed step: $0.01\,R/v_{th}$ adiabatic, about $0.001\,R/v_{th}$ kinetic (a step 2.5 times larger exceeds its error limit). `COLLISION_MODEL=5`, its cheapest collision step, at zero collision frequency. 84 radial modes on the adiabatic grid. The B300 has a small fraction of the H100's FP64 throughput, and CGYRO runs slower on it than on the H100.
- **GKW\***: nonlinear GKW runs on the same grids, on CPU nodes: 64 Ice Lake cores (adiabatic) and 96 to 128 cores (kinetic). FP64, RK4, circular geometry.

The GPU runs use s-alpha geometry with the $q$, $\hat s$, $\epsilon$ and gradients of the production configs. They are electrostatic and collisionless, on the same $(n_s, n_{k_x}, n_{k_y})$ and the same box. Nothing is printed or written during the timed steps. Each code keeps its own time stepper and dissipation, so the step sizes differ. The tables therefore report simulated time per wall-clock hour, in $[R/v_{th}]$/h, with the step rate in brackets for comparison with the throughput tables above.

Simulated time is measured in $R/v_{th}$, the time a thermal ion with $v_{th} = \sqrt{2T/m}$ takes to cross one major radius $R$ (GKW units, other codes converted). A rate of 1000 $[R/v_{th}]$/h means one hour of wall time advances the simulation by $1000\,R/v_{th}$. Higher is better.

- adiabatic: one run over $t \approx 50\,R/v_{th}$, through the linear phase into saturation.
- kinetic: the mean over five ion temperature gradients, 0.6 to 1.5 times the production $R/L_{T_i}$, each over about $0.1\,R/v_{th}$ from the same start. The adaptive steps change through a run, so other phases give other rates.

B300:

| code | adiabatic | kinetic |
|---|---|---|
| `gyx`, best | **4959.4** $[R/v_{th}]$/h (137.8 steps/s) | **86.0** $[R/v_{th}]$/h (19.8 steps/s) |
| `gyx`, v6 | 3093.5 $[R/v_{th}]$/h (85.9 steps/s) | 40.3 $[R/v_{th}]$/h (9.2 steps/s) |
| GKX (FP32) | 3979.4 $[R/v_{th}]$/h (359.2 steps/s) | 8.2 $[R/v_{th}]$/h (17.0 steps/s) |
| CGYRO | 507.2 $[R/v_{th}]$/h (14.1 steps/s) | 11.3 $[R/v_{th}]$/h (2.9 steps/s) |
| GKW\* | 205.2 $[R/v_{th}]$/h (5.7 steps/s) | 5.0 to 9.0 $[R/v_{th}]$/h (0.8 to 1.0 steps/s) |

H100:

| code | adiabatic | kinetic |
|---|---|---|
| `gyx`, best | **4507.2** $[R/v_{th}]$/h (125.2 steps/s) | **69.5** $[R/v_{th}]$/h (15.9 steps/s) |
| `gyx`, v6 | 3246.8 $[R/v_{th}]$/h (90.2 steps/s) | 42.8 $[R/v_{th}]$/h (9.8 steps/s) |
| GKX (FP32) | 3220.6 $[R/v_{th}]$/h (290.7 steps/s) | 6.0 $[R/v_{th}]$/h (12.5 steps/s) |
| CGYRO | 1338.1 $[R/v_{th}]$/h (37.2 steps/s) | 30.4 $[R/v_{th}]$/h (7.9 steps/s) |
| GKW\* | 205.2 $[R/v_{th}]$/h (5.7 steps/s) | 5.0 to 9.0 $[R/v_{th}]$/h (0.8 to 1.0 steps/s) |

\* CPU runs, the same rows in both tables. Comparisons between GPUs and CPUs depend on the hardware chosen and can be unfair. The kinetic GKW wall times come from file times and include start-up and output.

## Where the time goes

Each RK4 stage solves the fields (velocity moments of the distribution), applies the linear operator (stencils, drifts, drives, dissipation, collisions) and, in nonlinear runs, the Poisson bracket. On H100 the bracket is the largest cost of a nonlinear step. The linear RHS is FP64-bound, so it takes a larger share on GPUs with weak FP64 such as the B300. The next section shows how each kernel was sped up.

`gyaradax bench CONFIG --profile` prints the GPU time per kernel of a timed block, and `gyaradax run ... --profile` writes the same breakdown to `profile_summary.json` in the run directory.

## Optimisations

The speed of `gyaradax` comes from a stack of changes, each building on the previous one. The stages below were measured one at a time on one B300 with `gyaradax bench` (median of four timed blocks of 30 steps, after compilation), on the production grids above. Every stage keeps all the changes above it and adds one. The total is about 13x on the adiabatic grid and 17x on the kinetic grid.

```{figure} ../_static/figs/perf/stages.png
:width: 100%
:class: paper-fig

Speed-up over JAX in FP64. Each bar starts where the previous stage ended,
and its label is the gain over that stage. Gray marks a stage that is slower
than the one before.
```

| # | stage | setting | adiabatic | kinetic |
|---|---|---|---|---|
| 1 | JAX, FP64 | `mixed_precision: false` | 10.3 steps/s | 1.18 steps/s |
| 2 | + mixed precision | `mixed_precision: true` | 31.2 steps/s | 4.22 steps/s |
| 3 | + Z2Z packing | `use_z2z: true` | 47.1 steps/s | 5.24 steps/s |
| 4 | + CUDA, v5 bracket | `backend: cuda`, `GYARADAX_BRACKET=v5` | 71.3 steps/s | 7.98 steps/s |
| 5 | + v6 bracket | `GYARADAX_BRACKET_MRAD=native` | 81.4 steps/s | 5.50 steps/s |
| 6 | + kx padding | default v6 | 85.6 steps/s | 9.35 steps/s |
| 7 | + fused linear stage | `fused_linear: true` | 100.5 steps/s | 13.0 steps/s |
| 8 | + double-single | `linear_arith: ds` | **137.7** steps/s | 17.8 steps/s |
| 9 | + fused bracket packing | `fused_bracket: true` | 136.8 steps/s | **20.2** steps/s |

Nonlinear RK4 steps per second on the adiabatic grid (32, 8, 16, 85, 32) and the kinetic grid (2, 32, 12, 18, 192, 32). Row 5 runs v6 at GKW's kx length. On the kinetic grid that length (294) needs three cuFFT passes, so v6 is slower than v5 until row 6 pads it to a single-pass length. Every stage keeps the results within round-off of the stage before, checked as described in
{doc}`../validation/development`.

### Mixed precision

The nonlinear bracket runs its FFTs in FP32. The linear terms, the field solve and the accumulated RHS stay in FP64. FP32 halves the memory traffic of the bracket.

### Z2Z packing

The bracket needs the x and y derivatives of the potential and of the distribution. The two derivatives of one field are real in real space, so they fit in the real and imaginary parts of one complex inverse FFT. A stage then needs two inverse transforms instead of four. The gyro-averaged potential is not exactly Hermitian at $k_y = 0$, so that column is symmetrised first, otherwise the two channels leak into each other.

### CUDA kernels and the v5 bracket

The CUDA backend replaces the XLA code with three kinds of kernels:

- **Linear RHS.** One kernel applies the parallel and $v_\parallel$ stencils, the drifts, the drives and the dissipation in one pass, for all species in one launch. The stencil coefficients come from small per-class tables.
- **Field moments.** The velocity moments behind the field solve, up to two per pass, with deterministic chunked sums.
- **Bracket (v5).** cuFFT with LTO callbacks. The scatter onto the dealiased grid, the spectral derivatives and the gyro-average run inside the load callback of the inverse FFT, and the product inside the forward FFT. The electromagnetic potential is separable in $v_\parallel$, so only two potential planes per $(\mu, s)$ are transformed.

```{figure} ../_static/figs/paper/stencil_fusion_v2.png
:width: 60%
:class: paper-fig

The linear RHS as one fused kernel.
```

```{figure} ../_static/figs/paper/cufft_lto_fusion_paper.png
:width: 60%
:class: paper-fig

The bracket as separate passes (left) and fused into cuFFT callbacks (right).
```

### v6 bracket and kx padding

The 2D transforms of v5 include the zero padding in $k_y$. The v6 bracket splits them into 1D passes:

1. a cuFFT column pass along $k_x$ on the retained $k_y$ columns only,
2. one cuFFTDx kernel per row that does the inverse row FFT, the product with the potential row and the forward row FFT,
3. a column pass back on the kept columns.

The row kernels are compiled for fixed lengths (`BRACKET_V6_NKY_MAX`, see
{doc}`../getting_started/installation`). The column passes then use the smallest alias-free length that cuFFT runs as one kernel, instead of GKW's length: kx padding.

```{figure} ../_static/figs/diagrams/bracket_v6.png
:width: 65%

The v5 and v6 brackets.
```

### Fused linear stage

`fused_linear` runs a whole electrostatic RK4 stage in one kernel: the linear RHS reads $v_\parallel$ planes from a ring in shared memory, adds the collision stencil and writes the RK update from registers, so the stage's RHS is never stored.

```{figure} ../_static/figs/diagrams/fused_linear.png
:width: 65%

One RK4 stage with the default kernels and with `fused_linear`.
```

### Double-single arithmetic

The linear RHS is limited by FP64 throughput: about 60 ps per grid point on H100 and 250 ps on B300, whose FP64 rate is far lower. `linear_arith: ds` evaluates it with pairs of FP32 numbers (about 48 bits) on the FP32 units. It agrees with FP64 to about 1e-14 per evaluation. On H100 it makes no difference. FP64 emulation on tensor cores or the integer units (Ozaki schemes) was measured as well and is slower for this operator, see `docs/cuda_experiments/ozaki/`.

### Fused bracket packing

In mixed precision v6 packs the distribution and the potential into FP32 before the FFTs and unpacks the result after, in two kernels over the whole distribution. `fused_bracket` moves both into the cuFFT load and store callbacks, as the FP64 path already does. Results are bitwise identical on B300, where it is faster. On H100 and A100 it is slower.

### Tried and dropped

- A $v_\parallel$-fastest block order: more register spills, slower.
- Staged v6 row kernels and hand-written FFTs: slower than cuFFT and cuFFTDx.
- FP64 emulation with Ozaki schemes: slower than double-single.

Lessons that carried over between kernels are collected in `docs/cuda_experiments/README.md`: check register counts and spills with `cuobjdump --dump-resource-usage`, probe cuFFT callbacks once on a NaN buffer, and time with CUPTI, taking minima of interleaved runs.

## Grid sizes and FFT lengths

The bracket runs on the dealiased real-space grid. Its lengths are GKW's:

- along y, `mphi` starts at `3 nky - 2`, rounded up to an even length with prime factors at most 7, or to a power of two if one lies within 16.
- along x, `mrad` starts at `ceil(1.5 (nkx + 1)) + 1`, rounded up to a length with prime factors at most 7, or to a power of two within 8.

| nky | 8 | 16 | 24 | 32 | 40 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| mphi | 32 | 64 | 70 | 96 | 128 | 144 | 192 | 288 | 384 |

| nkx | 43 | 55 | 63 | 85 | 95 | 127 | 191 | 255 |
|---|---|---|---|---|---|---|---|---|
| mrad | 70 | 90 | 98 | 135 | 147 | 196 | 294 | 392 |

Hints:

- Powers of two and lengths of few prime factors run fastest in cuFFT. `nky = 40` gives `mphi = 128` for little more than `nky = 32` costs.
- The CUDA v6 bracket has a cuFFTDx row kernel for every `nky` up to `BRACKET_V6_NKY_MAX` (build option, default 192). Lengths cuFFTDx can only run with a workspace (70, 90, 98, 140, ...) are replaced by the next built length that is still alias free, so only round-off changes (`nky = 24` runs at 80).
- Along x the v6 column passes pick the smallest alias-free length that cuFFT runs as a single kernel (the GKW length 294 at `nkx = 191` needs three passes). `GYARADAX_BRACKET_MRAD=native|auto|N` overrides the choice.
- v6 needs an even number of distribution planes per device. Otherwise, and for any grid the build does not cover, the bracket runs v5 and warns once with the reason. `GYARADAX_BRACKET=v5` selects v5 quietly.
- `fused_linear` and `linear_arith: ds` need `ns <= 576` and fewer than 2^31 points per device, and do not cover conservative dissipation. Otherwise they fall back to the default kernel with a one-time warning. Electromagnetic runs always use the default kernel. With the momentum or energy conservation corrections on, the collisions run outside the fused kernel.

## Memory

The fused stencils are stored as per-class tables rather than full 6D arrays, and each RK stage's RHS is freed before the next. Peak device memory of one compiled step on a 2, 64, 16, 32, 85, 64 grid with $A_\parallel$ (one df is 5.3 GiB), mixed precision: 62.7 GiB with JAX, 40.0 GiB with CUDA plus about 13 GiB of cuFFT workspace that the bracket allocates outside XLA's pool.

The CLI runs with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. On grids near the device limit the pool can fragment during setup, so that a large allocation fails although the compiled peak fits. `--mem-fraction 0.95` or `XLA_PYTHON_CLIENT_ALLOCATOR=platform` avoid it, and so does sharding.

## Multi-GPU

The distribution function is sharded over species, $v_\parallel$ and $\mu$
(`jax.sharding`). The kernels and the bracket run on each device's local block
through `shard_map`. Only the field moments are summed across devices, plus a two-plane $v_\parallel$ halo per stage when $v_\parallel$ is split (and one plane for the collision stencil). Species and $\mu$ need no halo. All three axes run at about the same speed.

On two B300 a 2, 64, 16, 32, 85, 64 grid with $A_\parallel + B_\parallel$ goes from 593 to 292 ms per step (CUDA, `mu = 2`). Small grids gain less: the per-step reductions and kernel launches weigh more. Sharded runs match a single device at round-off, not bitwise, because the field moments are summed per shard. Shard over GPUs that are connected by NVLink. On machines whose GPUs only connect through the host, check that NCCL collectives between the GPUs give correct sums before trusting a sharded run.
