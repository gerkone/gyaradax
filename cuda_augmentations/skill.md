# Skill: Compiling and Integrating cuFFT LTO Callbacks with JAX FFI

This skill documents how to bridge cuFFT's Link-Time Optimization (LTO) callbacks with the JAX Typed FFI system. It builds upon the core LTO knowledge in [cuFFT LTO Example Skill](file:///system/user/volkmann/gyrokinetics-jax/cuda_augmentations/cuFFT_LTO_example/skill.md).

---

## 1. Prerequisites and Environment

The JAX FFI headers are non-standard and located within `jaxlib`. For the current `jax_env` (A100 MIG), use these paths:

| Item | Value |
|------|-------|
| JAX FFI Include | `/system/apps/userenv/volkmann/jax_env/lib/python3.14/site-packages/jaxlib/include` |
| C++ Standard | **C++17** (Required for JAX FFI's `if constexpr` and `std::is_same_v` usage) |
| Target Arch | `sm_80` / `compute_80` |

---

## 2. Key Architectural Fix: LTO-IR Lifetime

The most common cause of `CUFFT_INTERNAL_ERROR (5)` during `cufftMakePlan*` is a **Use-After-Free** of the LTO-IR (fatbin) buffer.

> [!IMPORTANT]
> The buffer passed to `cufftXtSetJITCallback` **MUST** remain valid until `cufftMakePlan*` completes. 
> If using `posix_memalign` to stage the fatbin, do NOT call `free()` until after the plan is fully created.
> 
> **Safest Pattern**: Embed the fatbin as a static `long long` array using `bin2c` and pass its address directly.

---

## 3. CMake Configuration Pattern

Use this template to build the JAX FFI shared library with LTO support:

```cmake
# Add shared library target
add_library(lto_bracket SHARED 
            cufft_lto_bracket.cu 
            ${CMAKE_BINARY_DIR}/bracket_load_cb_fatbin.h)

# 1. Bumper C++ and CUDA standards to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 2. Add JAX FFI Include Path
set(JAX_INCLUDE_DIR "/system/apps/userenv/volkmann/jax_env/lib/python3.14/site-packages/jaxlib/include")

target_include_directories(lto_bracket PRIVATE 
                           ${JAX_INCLUDE_DIR}
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           ${CMAKE_BINARY_DIR})

# 3. Link required libraries
target_link_libraries(lto_bracket PRIVATE CUDA::cufft CUDA::cudart CUDA::nvJitLink)
```

---

## 4. JAX FFI Integration Flow

### Host Wrapper (CUDA)
The JAX FFI wrapper should manage the cuFFT plans. Note the persistence of `d_cb_info` (device memory for callback parameters) across the lifecycle of the plan.

```cpp
#include "xla/ffi/api/ffi.h"
#include "bracket_load_cb_fatbin.h" // Static array from bin2c

static cufftHandle lto_plan = 0;

xla_ffi::Error LtoFftBracketImpl(cudaStream_t stream, ...) {
    if (lto_plan == 0) {
        cufftCreate(&lto_plan);
        // REGISTER CALLBACK BEFORE MAKE PLAN
        cufftXtSetJITCallback(lto_plan, "my_cb_name", 
                               (void*)bracket_load_cb_fatbin, 
                               sizeof(bracket_load_cb_fatbin),
                               CUFFT_CB_LD_COMPLEX_DOUBLE, &d_ptr);
        cufftMakePlanMany(lto_plan, ...);
    }
    // ... execution ...
}
```

### Python Registration
Register the library as a standard JAX FFI target:

```python
import jax.ffi
import ctypes

_lib = ctypes.cdll.LoadLibrary("./liblto_bracket.so")
jax.ffi.register_ffi_target(
    "lto_fft_bracket_ffi",
    jax.ffi.pycapsule(_lib.lto_fft_bracket_ffi),
    platform="CUDA",
)
```

---

## 5. Performance Expectations on MIG

LTO is fully supported on **A100 MIG instances**, despite some warnings in documentation. 
By eliminating the materialization of intermediate arrays (e.g., gradients for spectral methods), we typically observe:
- **~15-20% speedup** over optimized native JAX.
- **~30-40% speedup** over non-LTO FFI implementations.
- Massive reduction in global HBM traffic (Visible in raw timings, though often hidden from `analyze_cost`).
