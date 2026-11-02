# Ozaki-scheme FP64 emulation for the linear RHS (parked)

Can FP64 emulation on tensor cores or the integer pipe beat native FP64 for the
gyaradax linear operator on B300? Short answer: not for this operator. Every
variant below was measured; all are dominated by double-single FP32 arithmetic
(see `docs/cuda_experiments/double_single/` on this branch), which reaches 5.3x
at ~1e-14. Kept here so the negative result and its causes are not rediscovered.

All timings: one B300 SXM6 (scorpion), production grid
(nsp, nv, nmu, ns, nkx, nky) = (2, 32, 12, 18, 191, 32), 84.5M complex outputs.

## Why this was tried

B300 (Blackwell Ultra) cut FP64 to ~1.1 TFLOP/s. The linear RHS is 10.44 GFLOP
per call (XLA cost analysis, 123.5 FLOP/element) and on B300 is FP64-compute
bound: 14.78 ms, achieving 0.71 TFLOP/s, slower than an A100 (8.99 ms) despite
4x the bandwidth. The memory-bound field solve, by contrast, is 3.1x faster on
B300. The Ozaki scheme trades FP64 work for low-precision work. Motivating
papers: Matsuoka, arXiv:2606.06510 (Part 1, the TME model) and
arXiv:2606.23698 (Part 2, FFT).

## Measured hardware rates on B300

| unit | rate | vs FP64 |
|---|---|---|
| FP64 (DGEMM / DFMA) | 1.1 TFLOP/s | 1x |
| FP32 SIMT FMA | 36.7 T FMA/s | 73x |
| INT32 SIMT multiply-add | 19.2 T/s | 38x |
| INT64 SIMT multiply-add | 4.8 T/s | 9.5x |
| TF32 tensor | 952 TFLOP/s | 865x |
| BF16 tensor | 1718 TFLOP/s | 1560x |
| FP8 e4m3 tensor | 3227 TFLOP/s | 3065x |
| INT8 tensor | 146 TOPS | 132x |

INT8 tensor cores are 22x slower than FP8 on B300 (the 5th-generation
integer MMA path is not exposed on sm_103a), so INT8 Ozaki is the wrong
substrate here. `tensor_core_rates.py`, `pint.cu`.

## What was tried, in order

**1. Slice accuracy, INT8 (`slice_accuracy_int8.py`).** Ozaki-I on the real
parallel stencil with physically evolved df: k = 8 INT8 slices reach FP64
roundoff (7.6e-16). Median cancellation in the stencil is only 2.9x, so
evolved data needs the same k as random data.

**2. Slice accuracy, FP8 (`slice_accuracy_fp8.py`).** With exact accumulation
FP8 needs k = 12 (78 triangular products) for FP64. With the FP32
accumulation tensor cores actually perform, it plateaus at 3.4e-9 for every k:
e4m3's exponent range lets slice products overflow FP32's 24-bit mantissa.
Naive FP8 Ozaki cannot reach FP64.

**3. Unfused FP8 Ozaki in JAX (`unfused_fp8_jax.py`).** First runs gave 5.4e-2
error at every k. Cause: XLA's default `--xla_allow_excess_precision=true`
elides the f64 -> f8 -> f64 round trip inside a fused computation, so every
residual after the first slice is exactly zero. With the flag off: 6.2e-8
(dense 18x18 GEMM sums 18 products per output in FP32), and 5-20x slower than
native because each slice pair round-trips the 1.2 GB accumulator through HBM.

**4. GEMMul8 Ozaki-II (`gemmul8_stencil_bench.cu`).** RIKEN's library, FP8
backend, 12 moduli. Reaches FP64 exactly on our shape (3.4e-16). But:

| shape | native DGEMM | Ozaki-II | |
|---|---|---|---|
| square control 8192^3 | 1045 ms | 16.3 ms | 64x faster |
| our stencil 32x32 @ 32x4.7M | 9.53 ms | 104.2 ms | 11x slower |

Phase profile on our shape: scaling 49%, mod reduction 22%, FP8 GEMMs 26%,
CRT reconstruction 2%. The loss is conversion (unfused, per output, paid per
modulus), not the CRT "integer-epilogue wall" of Part 2. The ~700x swing
between shapes is contraction length: conversion scales with the elements
touched, useful work with K. The stencil contracts over K = 18.

**5. Fused SIMT Ozaki-II, one stencil (`stencilA.cu`).** CRT on the integer
pipe, int32 residues mod 14-bit primes (a 5-term sum of products fits int32,
so one reduction per output), coefficients deconstructed once on the host,
df deconstructed in-kernel with a tile-wide scale, Garner reconstruction.

| r | B | first version | FP64-free | vs native 1.515 ms | max err |
|---|---|---|---|---|---|
| 5 | 31 | 8.32 ms | 4.22 ms | 0.36x | 2.8e-9 |
| 8 | 53 | 10.64 ms | 7.21 ms | 0.21x | 7.2e-16 |

Accurate (r = 8 is FP64-level) but always slower: a single 5-tap stencil
cannot amortize deconstruction, Garner and reconstruction.

**6. Tap sweep, the 5D-union idea (`stencilB_sweep.cu`).** If every term of
the operator shares one deconstruction, the conversion cost is amortized
over more taps. Same harness, T taps per output on one shared tile:

| T | native | Ozaki r=5 (1e-9) | | Ozaki r=8 (FP64) | |
|---|---|---|---|---|---|
| 5 | 1.49 | 5.81 | 0.26x | 9.09 | 0.16x |
| 10 | 2.90 | 6.64 | 0.44x | 10.46 | 0.28x |
| 20 | 5.71 | 9.00 | 0.63x | 13.99 | 0.41x |
| 30 | 8.56 | 10.54 | 0.81x | 16.37 | 0.52x |
| 45 | 12.75 | 14.70 | 0.87x | 21.85 | 0.58x |
| 60 | 17.97 | 17.56 | 1.02x | 26.80 | 0.67x |

The fixed cost does amortize, but the per-tap cost (0.214 ms/tap for r = 5
against native 0.298) caps the asymptote at 1.39x for r = 5 and 0.93x for
r = 8, at any T. Our operator is ~30 taps. The per-tap cost is 5x above the
INT32 arithmetic floor, i.e. load-bound (2R scalar residue loads per tap); a
vectorized-load version was the next step and was not attempted once
double-single came in at 5.3x.

## Why it was parked

Substrate against substrate on the same algorithm, double-single beats native
FP64 by 2.2-2.7x at ~1e-14, while the best Ozaki variant is 0.81x at the looser
1e-9 tolerance with a 1.39x ceiling. Double-single needs no moduli and no CRT.

## Lessons

- **Disable XLA excess precision for any error-free transformation.**
  `--xla_allow_excess_precision=false`, or the splitting silently degenerates.
- **Enable x64 before creating arrays.** A standalone script without it turned
  float64 into float32 and reported an impossible 915 TFLOP/s of "FP64".
- **FP8 tensor cores accumulate in FP32**, which caps slice-based (Ozaki-I)
  emulation at ~1e-9. Exact FP64 needs integer residues (Ozaki-II).
- **On B300 the emulation must be completely FP64-free.** `ilogb`, `ldexp`,
  `__double2ll_rn` and F2F conversions all run on the FP64 pipe; any of them in
  the conversion path costs more than the stencil being replaced. Extract
  exponent and mantissa from the raw bits instead (`stencilA.cu`).
- **Centre CRT results in the digit domain.** Computing `S - M` in FP64 for a
  negative result near M ~ 2^126 is a catastrophic cancellation; sum the
  complement digits instead.
- **Low contraction length is the enemy.** Part 1's conversion-bound threshold
  OI* applies directly; GEMM-shaped is not enough, K must be large.
- **Check registers before concluding.** The sweep first failed to launch
  past T = 10 ("too many resources"); restructure, do not shrink the problem.

## Running

All sources are self-contained. CUDA files build with the conda toolchain:

    ENV=/system/apps/userenv/galletti/gyaradax_env
    $ENV/bin/nvcc -std=c++17 -O3 -arch=sm_103 \
        -ccbin $ENV/bin/x86_64-conda-linux-gnu-c++ stencilB_sweep.cu -o _build/sweep

`gemmul8_stencil_bench.cu` needs GEMMul8 (github.com/RIKEN-RCCS/GEMMul8) built
with `BACKEND=cuda OPS=gemm GPU_ARCH=103 PROFILE=on`. Python scripts take
`PYTHONPATH` set to the repository root.
