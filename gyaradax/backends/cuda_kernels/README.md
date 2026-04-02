# Gyradax CUDA Backend Kernels

This directory the CUDA kernels for stencils and Poisson brackets, used via JAX FFI.

## Prerequisites
- **CUDA Toolkit**: NVCC and cuFFT. Tested for cudatoolkit >=13.1

- **Python**: A environment with `jaxlib` installed (for FFI headers). Make sure you have a jax version installed that is compatible with your cudatoolkit version. See https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder
The default install of jax does not use your system's cudatoolkit, but rather installs its own. This can lead to version mismatches. 

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
   To use a different GPU architecture, use the -DGPU_ARCHITECTURES="<arch>" flag. For example, to use Ampere (80), 
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_ARCHITECTURES="80"
   ```
   Need compute capability >= 80.
   cmake prints the detected compute capability, jaxlib version, and cudatoolkit. Check that these are correct before proceeding.
   Kernels were tuned for sm_103. 
   

3. **Build**:
   ```bash
   cmake --build . -j$(nproc)
   ```

4. **Install**:
   This command copies the library into the parent directory, where it is found by `_cuda.py`.
   ```bash
   cmake --install .
   ```

## Files
- `CMakeLists.txt`: Build system configuration.
- `kernels/cufft_bracket.cu`: Consolidated Graph-based Poisson brackets (Mixed Precision, FP64 Z2Z, and FP64 Direct). `_cuda.py` by default uses the mixed precision version.
- `kernels/*.cu`: Stencil and linear RHS fused kernels.
