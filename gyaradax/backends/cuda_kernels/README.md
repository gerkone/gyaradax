# Gyradax CUDA Backend Kernels

This directory holds the CUDA kernels of the linear RHS, the EM field moments and the Poisson bracket, used via JAX FFI.

## Prerequisites
- **CUDA Toolkit**: NVCC and cuFFT. Tested for cudatoolkit >=13.1

- **Python**: An environment with `jaxlib` installed (for FFI headers). The base `gyaradax` install depends on CPU-compatible `jax`; install the CUDA extra when building or using these kernels:
  ```bash
  pip install -e ".[cuda13]"
  ```
  Make sure the installed JAX/JAXLIB CUDA version is compatible with your local CUDA toolkit. See https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder
  The default install of JAX does not use your system's cudatoolkit, but rather installs its own. This can lead to version mismatches.

## Building the Library

The kernels are compiled into a shared library (`libgyaradax_cuda.so`). 

### One-liner to Build
From this directory:
```bash
mkdir -p _build && cd _build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(nproc) && cmake --install . && cd ..
```

### Manual Steps
1. **Create Build Directory**:
   ```bash
   mkdir -p _build && cd _build
   ```

2. **Configure with CMake**:
   Make sure your target Python environment is active so CMake can find the correct JAX headers.
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
   By default (`GPU_ARCHITECTURES=native`) the kernels are compiled for the GPU of the build
   machine only: building on an H100 node gives sm_90, on a B300 node sm_103, with no flags. For a
   library that runs on several GPU types (a checkout shared by H100 and B300 nodes), or when the
   build node has no GPU, list the architectures:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_ARCHITECTURES="90;103"
   ```
   The cuFFT LTO callbacks use `LTO_ARCHITECTURES` (default `80;90;100`, enough for A100, H100 and
   Blackwell); the cuFFTDx row kernels follow `GPU_ARCHITECTURES`.
   Need compute capability >= 80.
   cmake prints the detected compute capability, jaxlib version, and cudatoolkit. Check that these are correct before proceeding.
   Kernels were tuned on sm_90 (H100) and sm_103 (B300).
   

3. **Build**:
   ```bash
   cmake --build . -j$(nproc)
   ```

4. **Install**:
   This command copies the library into the parent directory, where it is found by `_cuda.py`.
   ```bash
   cmake --install .
   ```

## Optional: cuFFTDx (v6 Poisson bracket)
With the header-only `nvidia-mathdx` package installed (`pip install nvidia-mathdx`), CMake
also builds the v6 Poisson bracket (`kernels/cufft_bracket_v6.cu`): cuFFT column passes on the
retained ky columns plus cuFFTDx row kernels that fuse the inverse row FFT, the bracket and the
forward row FFT. CMake prints `cuFFTDx: ... (v6 bracket for SM ...)`; point it at the headers with
`-DCUFFTDX_INCLUDE_DIR=<.../nvidia/mathdx/include>` if Python cannot import `nvidia.mathdx`.
Without it, or for dealiased grids the row kernels are not instantiated for, the v5 pipeline runs.
Set `GYARADAX_BRACKET=v5` to force the v5 pipeline (bitwise reproduction of earlier runs).

## Files
- `CMakeLists.txt`: Build system configuration.
- `kernels/linear_rhs_fused.cu`: fused linear RHS (ES and EM: A_par, B_par, conservative parallel dissipation), all species in one launch; ky tiles for ns * nky > 1024 and HALO variants that read the vpar neighbours of a vpar shard from halo buffers.
- `kernels/field_moments.cu`: velocity moments of the kinetic EM field solve (A_par, phi, B_par).
- `kernels/cufft_graph_bracket_true_fp32.cu`, `kernels/cufft_graph_bracket_fp64.cu`: v5 cuFFT Poisson bracket (mixed precision / FP64).
- `kernels/cufft_bracket_v6.cu`: v6 Poisson bracket (needs cuFFTDx, see above).
- `kernels/bracket_v5_pack_select.cuh`: explicit pack kernel and the plan-time C2C path selection of the v5 bracket.
- `lto_callbacks/`: cuFFT LTO load/store callbacks and the shared pack routines.
- `kernels/apply_*.cu`: standalone stencil kernels (benchmarks).
