# CUDA Experiments

This directory is a small active harness for trying CUDA kernels before they are
promoted to the production backend in `gyaradax/backends/cuda_kernels/`.

Historical prototypes, binaries, cuFFT examples, HLO dumps, and optimization
notes were intentionally removed from this directory. Use git history if an old
experiment is needed.

## How the production kernels evolved

The production kernels in `gyaradax/backends/cuda_kernels/` started as experiments here (the
prototypes are in git history before `7b49a2e`). Times are per evaluation on an H100 for the
2 x 32 x 8 x 16 x 85 x 32 grid, mixed precision; see `docs/NOTES.md` §10.14 and §13 for details.

| kernel | before (master) | now | what changed |
|---|---|---|---|
| `linear_rhs_fused.cu` | electrostatic only, one launch per species, streams a 9 x 6D stencil array; 1.78 ms (JAX 5.34 ms) | 1.27 ms ES, 1.36 ms A_par | per-class stencil tables, all species in one launch, EM (f = g + g2f A_par in-kernel, B_par through psi), conservative dissipation, ky tiles for ns * nky > 1024, HALO variants for vpar shards; ES bitwise identical to master |
| `cufft_graph_bracket_*.cu` (v5) | z2z 2-for-1 inverse + R2C forward with cuFFT LTO callbacks; 5.0 ms (JAX 10.2 ms) | 4.46 ms | EM via the separable potential (two planes per (sp, mu, s)); plan-time NaN probe for cuFFT skipping the FP32 load callback (master returned a zero bracket on power-of-two batches); explicit pack kept when faster |
| `cufft_bracket_v6.cu` (new) | - | 2.8 ms ES, 3.0 ms A_par, 3.32 ms FP64 | 1D column passes on the retained ky columns only, cuFFTDx row kernels fusing inverse FFT, bracket and forward FFT; needs `nvidia-mathdx`, falls back to v5 |
| `field_moments.cu` (new) | JAX field solve, 1.03 ms | 0.50 ms | up to two velocity moments per pass, g -> f in-kernel, deterministic chunked sums |
| `linear_rhs_vtiled.cu` | unreachable, incorrect | removed | - |

Around the kernels, the backend now runs them on the local blocks of a sharded df
(`sharding.velocity_map`) instead of letting GSPMD all-gather their operands.

Lessons worth keeping for new experiments:

- **Check registers and spills, not just timings.** Small source changes moved a kernel from 64 to
  76 registers and halved its occupancy (2.08 against 1.27 ms). Compare
  `cuobjdump --dump-resource-usage` across builds before and after a change.
- **Do not trust cuFFT callbacks blindly.** cuFFT can skip an LTO load callback without an error;
  probe the plan once on a NaN-filled buffer.
- **Know the hardware's FP64 rate.** The linear kernel is FP64 bound: ~60 ps per point on H100,
  ~250 ps on B300.
- **Time with CUPTI and minima.** On shared GPUs, take the minimum over interleaved repeats;
  compare outputs bitwise against a frozen copy of the previous library.

## Contract

- Keep experiments self-contained under `docs/cuda_experiments/`.
- Add CUDA sources under `kernels/*.cu`.
- Build only into `_build/`; do not write generated libraries or binaries into
  the source directory.
- Compare numerical behavior against a JAX reference before promoting code.
- Once an experiment becomes production code, move the minimal implementation to
  `gyaradax/backends/cuda_kernels/` and add backend parity tests there.

## Build

From this directory:

```bash
mkdir -p _build
cd _build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

CMake auto-detects the active Python executable, `jaxlib` version, and JAX FFI
include directory using `python3`, matching the production CUDA build style.

The shared library is written to:

```text
docs/cuda_experiments/_build/libgyaradax_cuda_experiments.so
```

## Compare against JAX

Use the lightweight harness as a starting point:

```bash
python compare_against_jax.py --library _build/libgyaradax_cuda_experiments.so
```

The checked-in script is intentionally conservative: it verifies that the
experiment library exists and computes a deterministic JAX reference problem.
Extend it for a specific kernel only after the kernel's C ABI or JAX FFI contract
is clear.

## Layout

```text
docs/cuda_experiments/
  README.md
  CMakeLists.txt
  compare_against_jax.py
  kernels/
    README.md
    example_kernel.cu
```
