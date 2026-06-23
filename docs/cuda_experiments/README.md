# CUDA Experiments

This directory is a small active harness for trying CUDA kernels before they are
promoted to the production backend in `gyaradax/backends/cuda_kernels/`.

Historical prototypes, binaries, cuFFT examples, HLO dumps, and optimization
notes were intentionally removed from this directory. Use git history if an old
experiment is needed.

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
