# Skill: Writing and Compiling cuFFT LTO Callbacks

Verified against CUDA 12.9, A100 (sm_80), jax_env.

---

## 1. What LTO Callbacks Do

cuFFT LTO callbacks fuse user-written `__device__` math directly into the FFT kernel via JIT link-time optimization. The callback runs **inside** the FFT kernel per element, reading or writing data without extra global-memory round-trips. Use them when you want to compute gradients, apply windowing, pack/unpack spectral arrays, or otherwise transform data in-flight during a cuFFT execute call.

---

## 2. Environment

| Item | Value |
|------|-------|
| nvcc | `/system/apps/userenv/volkmann/jax_env/bin/nvcc` |
| CUDA includes | `/system/apps/userenv/volkmann/jax_env/targets/x86_64-linux/include` |
| CUDA libs | `/system/apps/userenv/volkmann/jax_env/targets/x86_64-linux/lib` |
| bin2c | `/system/apps/userenv/volkmann/jax_env/bin/bin2c` |
| Target arch | `sm_80` / `compute_80` (A100) |
| Host compiler | `g++` (not `gcc` — C++ stdlib required) |
| Runtime lib path | `LD_LIBRARY_PATH=/system/apps/userenv/volkmann/jax_env/lib` |
| libcufft_static | **not available** — link against dynamic `-lcufft` only |

Activate with `source /system/apps/userenv/mambaforge/bashrc && mamba run -n jax_env ...`

---

## 3. Writing the Device Callback

The callback is a normal `__device__` function in a `.cu` file. You name it whatever you like — you pass the name as a string to the API. One file = one callback function per type.

### Load callback signature (intercepts cuFFT reading input)

```cpp
#include <cufftXt.h>

struct MyInfo {           // define in a shared header — must match host-side exactly
    const double2* data;
    int n;
    int flag;
};

__device__ cufftDoubleComplex my_load_cb(
    void             *dataIn,      // pointer to the FFT input buffer
    unsigned long long offset,     // flat element index
    void             *callerInfo,  // cast to your struct*
    void             *sharedMem)   // unused for simple callbacks
{
    const MyInfo* info = (const MyInfo*)callerInfo;
    // compute and return the value cuFFT should see at this index
    return info->data[offset];
}
```

### Store callback signature (intercepts cuFFT writing output)

```cpp
__device__ void my_store_cb(
    void             *dataOut,
    unsigned long long offset,
    cufftDoubleComplex element,    // value cuFFT computed
    void             *callerInfo,
    void             *sharedMem)
{
    double2* out = (double2*)dataOut;
    out[offset]  = element;
}
```

### Callback type table

| `cufftXtCallbackType` enum | Precision | Direction | Return / Element type |
|---------------------------|-----------|-----------|----------------------|
| `CUFFT_CB_LD_COMPLEX`       | single    | load      | `cufftComplex` |
| `CUFFT_CB_LD_COMPLEX_DOUBLE`| double    | load      | `cufftDoubleComplex` |
| `CUFFT_CB_LD_REAL`          | single    | load      | `cufftReal` |
| `CUFFT_CB_LD_REAL_DOUBLE`   | double    | load      | `cufftDoubleReal` |
| `CUFFT_CB_ST_COMPLEX`       | single    | store     | `cufftComplex` |
| `CUFFT_CB_ST_COMPLEX_DOUBLE`| double    | store     | `cufftDoubleComplex` |
| `CUFFT_CB_ST_REAL`          | single    | store     | `cufftReal` |
| `CUFFT_CB_ST_REAL_DOUBLE`   | double    | store     | `cufftDoubleReal` |

---

## 4. Compiling the Device Callback to LTO-IR

### Option A — Offline (nvcc → fatbin, then embed with xxd)

**Step 1: compile to fatbin**
```makefile
LTO_FLAGS := --generate-code arch=compute_80,code=lto_80 -fatbin

my_callback.fatbin: my_callback.cu
    nvcc $(LTO_FLAGS) -O3 $< -o $@
```

**Step 2: embed as a C byte array — two tools**

*Option A: `xxd -i` (used in this project's Makefile)*
```makefile
my_callback_fatbin.h: my_callback.fatbin
    xxd -i $< > $@
```
Generates a `unsigned char` array + `unsigned int` length:
```c
unsigned char my_callback_fatbin[]  = { 0x7f, 0x45, ... };
unsigned int  my_callback_fatbin_len = 1234;
```
**Requires the alignment fix** in §5 because `unsigned char` has 1-byte alignment.

*Option B: `bin2c` (NVIDIA CUDA SDK tool, used in the NVIDIA sample)*
```makefile
my_callback_fatbin.h: my_callback.fatbin
    bin2c --name my_callback --type longlong $< > $@
```
Generates a `long long` array, which carries 8-byte alignment — still below nvJitLink's 16-byte requirement, so **the alignment fix in §5 is still required**.

**Step 3: include in the host `.cu` file**
```cpp
#include "my_callback_fatbin.h"
// xxd:   provides my_callback_fatbin[]  (unsigned char) and my_callback_fatbin_len (unsigned int)
// bin2c: provides my_callback[]         (long long)     and sizeof(my_callback)
```

### Option B — NVRTC at runtime

Compile the `.cu` source string at runtime and retrieve LTO-IR bytes:

```cpp
#include <nvrtc.h>
#include <vector>

void compile_to_lto(std::vector<char>& lto_ir, const char* src_code, const char* name) {
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src_code, name, 0, nullptr, nullptr);

    const char* opts[] = {
        "-I/system/apps/userenv/volkmann/jax_env/targets/x86_64-linux/include",
        "-arch=compute_80",
        "--std=c++11",
        "--relocatable-device-code=true",
        "-default-device",
        "-dlto"
    };
    nvrtcCompileProgram(prog, 6, opts);

    size_t size;
    nvrtcGetLTOIRSize(prog, &size);
    lto_ir.resize(size);
    nvrtcGetLTOIR(prog, lto_ir.data());
    nvrtcDestroyProgram(&prog);
}
```
NVRTC output is already correctly aligned — no alignment fix needed (see §5 below).

---

## 5. Host-Side API

### Exact signature (CUDA 12.x)

```cpp
cufftResult cufftXtSetJITCallback(
    cufftHandle           plan,
    const char           *callback_name,     // exact name of your __device__ function
    const void           *lto_ir,            // fatbin bytes or NVRTC LTO-IR
    size_t                lto_ir_size,       // byte count
    cufftXtCallbackType   type,              // e.g. CUFFT_CB_LD_COMPLEX_DOUBLE
    void                **caller_info        // &d_my_info_ptr (device pointer, single-GPU)
);
```

> **Note:** `api_reference.md` in the repo shows an older 5-argument signature without `callback_name`. The 6-argument signature above is what CUDA 12.9 actually requires.

### Required execution order

```cpp
// 1. Allocate device CallerInfo first (needed before SetJITCallback)
MyInfo* d_info;
cudaMalloc(&d_info, sizeof(MyInfo));

// 2. Create plan handle
cufftHandle plan;
cufftCreate(&plan);

// 3. ⚠ ALIGNMENT FIX for xxd-embedded fatbins:
//    xxd -i produces a 1-byte-aligned array; nvJitLink requires 16-byte alignment.
//    Skipping this causes CUFFT_INTERNAL_ERROR (5) with no other diagnostic.
void* aligned_fatbin = nullptr;
posix_memalign(&aligned_fatbin, 16, (size_t)my_callback_fatbin_len);
memcpy(aligned_fatbin, my_callback_fatbin, (size_t)my_callback_fatbin_len);

// 4. Register callback — MUST be before cufftMakePlan*
void* d_info_void = (void*)d_info;
cufftResult res = cufftXtSetJITCallback(
    plan,
    "my_load_cb",                        // matches __device__ function name exactly
    aligned_fatbin,
    (size_t)my_callback_fatbin_len,
    CUFFT_CB_LD_COMPLEX_DOUBLE,
    &d_info_void                         // single-GPU: address of single device pointer
);
free(aligned_fatbin);
// check res != CUFFT_SUCCESS before proceeding

// 5. Make plan — JIT-linking happens here (50–500 ms first time)
size_t ws;
cufftMakePlanMany(plan, rank, n, ...);

// 6. Update callerInfo and execute
cudaMemcpy(d_info, &h_info, sizeof(MyInfo), cudaMemcpyHostToDevice);
cufftExecZ2D(plan, d_in, d_out);

// 7. Destroy
cufftDestroy(plan);
cudaFree(d_info);
```

> **NVRTC path**: pass `lto_ir.data()` and `lto_ir.size()` directly — no alignment fix needed.

---

## 6. Dynamic Behavior: Reuse One Plan with a Flag

Plan creation + JIT-linking is expensive. For multiple related operations (e.g. 4 gradient types), put a `gradient_type` enum inside `CallerInfo` and update it via stream-ordered async copy between each exec:

```cpp
struct CallbackInfo {
    const double2* input;
    int gradient_type;   // 0=phi_y, 1=f_x, 2=phi_x, 3=f_y
};

// Device callback switches on it:
__device__ cufftDoubleComplex my_load_cb(void* dataIn, unsigned long long idx,
                                          void* ci, void* sharedMem) {
    const CallbackInfo* info = (const CallbackInfo*)ci;
    switch (info->gradient_type) {
        case 0: return compute_phi_y(info, idx);
        case 1: return compute_f_x(info, idx);
        // ...
    }
}

// Host: update one field without rebuilding the plan
static const int grad_types[4] = {0, 1, 2, 3};
for (int gt = 0; gt < 4; ++gt) {
    cudaMemcpyAsync(
        (char*)d_cb_info + offsetof(CallbackInfo, gradient_type),
        &grad_types[gt], sizeof(int),
        cudaMemcpyHostToDevice, stream);
    cufftExecZ2D(plan, d_in, ws_ptrs[gt]);
}
```

---

## 7. Linking

### Makefile pattern for a shared library

```makefile
NVCC  := /system/apps/userenv/volkmann/jax_env/bin/nvcc
ARCH  := -arch=sm_80
OPT   := -O3 -lineinfo
CUDA_INC := /system/apps/userenv/volkmann/jax_env/targets/x86_64-linux/include
CUDA_LIB := /system/apps/userenv/volkmann/jax_env/targets/x86_64-linux/lib
LTO_FLAGS := --generate-code arch=compute_80,code=lto_80 -fatbin

my_callback.fatbin: my_callback.cu
    $(NVCC) $(LTO_FLAGS) $(OPT) $< -o $@

my_callback_fatbin.h: my_callback.fatbin
    xxd -i $< > $@

libmy.so: host.cu my_callback_fatbin.h
    $(NVCC) -shared -Xcompiler -fPIC $(ARCH) $(OPT) -I$(CUDA_INC) \
        host.cu -o $@ -lcufft

# For a standalone executable with g++ host compilation:
my_exe: host.o reference.o
    g++ -L$(CUDA_LIB) $^ -o $@ -lcufft -lcudart -lm
```

### Flags summary

| What | Flags |
|------|-------|
| Compile device callback to LTO-IR | `--generate-code arch=compute_80,code=lto_80 -fatbin` |
| Compile host `.cpp` with g++ | `g++ -I$(CUDA_INC) -c host.cpp -o host.o` |
| Link host executable | `g++ -L$(CUDA_LIB) ... -lcufft -lcudart -lm` |
| Link shared library (nvcc) | `nvcc -shared -Xcompiler -fPIC -arch=sm_80 ... -lcufft` |
| Runtime | `LD_LIBRARY_PATH=/system/apps/userenv/volkmann/jax_env/lib` |

---

## 8. Constraints and Pitfalls

| # | Rule |
|---|------|
| 1 | **SetJITCallback before MakePlan.** Calling `cufftMakePlan*` before `cufftXtSetJITCallback` silently produces a plan with no callback. |
| 2 | **Callbacks cannot be unset.** `cufftXtClearCallback` is not supported for LTO. To change a callback, destroy and recreate the plan. |
| 3 | **One callback function per type per compilation unit.** You cannot have two load-double-complex callbacks in the same `.cu` file. |
| 4 | **Fatbin alignment.** `xxd -i` output is 1-byte-aligned. `nvJitLink` requires 16-byte alignment. Always `posix_memalign` + `memcpy` before passing to `cufftXtSetJITCallback`. NVRTC output is pre-aligned. |
| 5 | **Do not use `cufftXtSetCallback` (legacy).** The old function-pointer API causes `CUFFT_INTERNAL_ERROR` in CUDA 12 and has no performance benefit. |
| 6 | **`libcufft_static` is not in jax_env.** Link with `-lcufft` (dynamic) only. Do not attempt `nvcc ... -lcufft_static -lculibos`. |
| 7 | **"TODO: nvvm input without LTO" error.** Means the compiler emitted standard NVVM instead of LTO-IR. Check that `code=lto_80` (not `sm_80`) is in the fatbin compile flags. |
| 8 | **`callback_name` must match exactly.** The string passed to `cufftXtSetJITCallback` must be the exact identifier of the `__device__` function in the compiled `.cu`. |

---

## 9. Reference Files

| File | Purpose |
|------|---------|
| `CUDALibrarySamples/cuFFT/lto_callback_window_1d/src/r2c_c2r_lto_callback_example.cpp` | NVIDIA host example: fatbin embed path |
| `CUDALibrarySamples/cuFFT/lto_callback_window_1d/src/r2c_c2r_lto_nvrtc_callback_example.cpp` | NVIDIA host example: NVRTC path |
| `CUDALibrarySamples/cuFFT/lto_callback_window_1d/src/nvrtc_helper.h` | NVRTC compile-to-LTO helper |
| `CUDALibrarySamples/cuFFT/lto_callback_window_1d/src/r2c_c2r_lto_callback_device.cu` | Device callback example (windowing) |
| `api_reference.md` | Callback type enums (note: API signature shown is outdated 5-arg form — see §5 above) |
