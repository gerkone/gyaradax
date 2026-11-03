# Installation

`gyaradax` needs Python 3.12 or newer. The base package runs on CPU JAX. GPUs need the CUDA 13 extra, which adds the CUDA build of JAX, cuFFTDx (`nvidia-mathdx`) and NCCL.

```bash
git clone https://github.com/gerkone/gyaradax.git
cd gyaradax
conda env create -f environment.yml    # optional: Python, CUDA toolkit, cmake, compilers
conda activate gyaradax_env
pip install -e ".[cuda13]"             # or pip install -e . for CPU only
```

The install provides the `gyaradax` command. `gyaradax info` lists the devices JAX sees and which backends are available. Add the `dev` extra for the test tools, and `docs` to build this documentation (`sphinx-build docs docs/_build`).

## CUDA backend

The CUDA backend is optional. It replaces the linear RHS, the field moments and the nonlinear bracket with fused kernels called through the [XLA FFI](https://docs.jax.dev/en/latest/ffi.html). It needs the CUDA toolkit 13.1 or newer and a GPU of compute capability 8.0 or newer (A100, H100, B200 / B300, ...). Build it once from `gyaradax/backends/cuda_kernels/`:

```bash
cd gyaradax/backends/cuda_kernels
mkdir -p _build && cd _build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
cmake --install .
```

The default backend stays `jax`. Select the kernels with `--backend cuda`, or with `backend: auto` (in the config or on the command line), which picks CUDA whenever the library and a GPU are present and warns when it falls back to JAX on a GPU. The build prints compiler warnings from the jaxlib headers
(`-Wreturn-type`); they are expected.

Build options:

`GPU_ARCHITECTURES`
: Kernel architectures. The default, `native`, builds for the GPU of the build machine only. A checkout shared by different nodes needs them listed, e.g. `-DGPU_ARCHITECTURES="80;90;103"` for A100, H100 and B300.

`LTO_ARCHITECTURES`
: Architectures of the cuFFT LTO callbacks, default `80;90;100`. On a toolkit older than 12.8 drop `100`.

`BRACKET_V6_NKY_MAX`
: Largest `nky` with row kernels for the faster v6 bracket (default 192). The v6 bracket uses [cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/), whose FFT lengths are fixed at compile time, so the build generates one kernel per dealiased row length. It needs `nvidia-mathdx`, which the `cuda13` extra installs. Each length adds about 3 s of build time, and the default build takes about 2 minutes.

Grids the build does not cover (a larger `nky`, an architecture not built, a library without cuFFTDx) run the v5 bracket and log a warning naming the reason and the rebuild that fixes it. `gyaradax info` prints what the installed library provides, e.g.

```text
bracket     : v6 (cuFFTDx, SM 80, 103; v5 with a warning elsewhere)
v6 sizes    : 4, 8, 16, 32, 36, 40, 64, 80, ..., 512, 576, 1024 (nky <= 342)
```

## Multi-GPU

Sharded runs need NCCL (`nvidia-nccl-cu13`, part of the `cuda13` extra). `gyaradax info` reports whether it is installed. See {doc}`running`.

## Tests

```bash
pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/ -x -q --download-test-data
```

Most tests compare against GKW reference runs on Hugging Face ([gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests)), which `--download-test-data` fetches into `$GYARADAX_TEST_DATA` (default `~/.cache/gyaradax/test_data`). Without it, those tests skip. {doc}`../contributing` has the other ways to enable the download and what the tests check.
