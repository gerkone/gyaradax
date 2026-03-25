**Role:** Expert HPC CUDA C++ and JAX engineer.

**Skill reference:** `cuda_augmentations/cuFFT_LTO_example/skill.md` — read it before writing any LTO callback code. It covers the correct `cufftXtSetJITCallback` 6-argument signature, fatbin alignment fix, build flags, pitfalls, and environment paths. Also read `cuda_augmentations/skill.md` for JAX FFI integration patterns and the fatbin alignment fix.

**Build system:** `cuda_augmentations/CMakeLists.txt` — use CMake, not Makefile. Build commands:
```bash
source /system/apps/userenv/mambaforge/bashrc
mamba run -n jax_env bash -c "cd cuda_augmentations/build && cmake .. && make -j"
```

**Benchmark command:**
```bash
mamba run -n jax_env python cuda_augmentations/jax_ffi_benchmark.py
```

---

# LTO Bracket Optimization Plan

## Baseline performance (A100 MIG 3g.20gb, batch=4096)

| Variant              | Time     | vs JAX FP64 |
|----------------------|----------|-------------|
| JAX FP64             | 31.6 ms  | baseline    |
| JAX mixed            | 29.0 ms  | −8%         |
| cuFFT FFI FP64       | 35.1 ms  | +11%        |
| **LTO cuFFT v0**     | **24.9 ms** | **−21%** |

## Grid dimensions (configs/iteration_13.yaml)

```
nkx=85  nky=32  nv=32  nmu=8  nspec=16  → batch = 4096
mrad=135  mphi=96  mphiw3=49
Sparsity: nkx×nky / (mrad×mphiw3) = 2720/6615 = 0.41
```

## Current LTO v0 data flow and HBM traffic

```
Step                             Kernel launches   HBM traffic
─────────────────────────────────────────────────────────────
Z2D ×4 (load CB reads df+phi)   4 launches        1.43 GB read (gather) + 1.70 GB write
Bracket kernel                   1 launch          1.70 GB read + 0.42 GB write
D2Z                              1 launch          0.42 GB read + 0.43 GB write
─────────────────────────────────────────────────────────────
Total                            6 launches        6.11 GB
Effective bandwidth at 24.9ms:   245 GB/s (31% of ~800 GB/s MIG peak)
```

Workspace: 5 real-space buffers × 424.7 MB = 2.12 GB + 433.5 MB dummy input = 2.55 GB total.

The dominant costs are:
1. **4 gather passes** over df+phi (1.43 GB) — these are random-access reads through `inverse_jind` indirection, which miss L2 because the packed arrays (356 MB) >> L2 cache (~13 MB on MIG 3g)
2. **Z2D writes + bracket reads** of the 4 real-space arrays (3.40 GB) — sequential but two full passes over the same data (write then read)
3. **Bracket write + D2Z read** of nl_real (0.85 GB) — another write-then-read materialization

--- [x] **Optimization Level 1: Bracket Fusion**
    - [x] Devise D2Z load callback signature for in-flight bracket.
    - [x] Update FFI wrapper to manage dual callbacks & workspaces.
    - [x] Bench: 24.96ms (v0) -> 22.66ms (v1). Save ~2.3ms.
    - [x] Verify: rel_l2 ≈ 2.88e-16.

--- [/] **Optimization Level 2: Two-for-one Z2Z**
    - [ ] Implement `bracket_z2z_load_cb.cu` with Hermitian extension.
    - [ ] Update D2Z fused callback to read `double2` workspace pairs.
    - [ ] Update host wrapper `cufft_lto_bracket.cu` with Z2Z plans & logic.
    - [ ] Bench: ~22.7ms (v1) -> ~19–20ms (v2).
    - [ ] Verify: rel_l2 ≈ 2.88e-16.

## Optimization Level 1 — Fuse bracket into D2Z load callback

**Idea:** Replace the separate bracket kernel with a `CUFFT_CB_LD_REAL_DOUBLE` load callback on the D2Z plan. The callback reads the 4 real-space gradient arrays at position `n`, computes `inv_n2 * (φ_y·f_x − φ_x·f_y)`, and returns the real value to cuFFT.

**What it eliminates:**
- The `lto_bracket_kernel` launch
- The `d_ws_nl` buffer (0.42 GB allocation)
- 1 HBM write (bracket output) + 1 HBM read (D2Z input) = **0.85 GB saved (14%)**

**Expected speedup:** ~10–12% → **~22 ms**

**Kernel launches:** 6 → 5

### Implementation

#### 1. New device callback: `bracket_d2z_load_cb.cu`

```cpp
#include <cufftXt.h>

struct BracketD2zInfo {
    const double* phi_y;     // [batch, mrad, mphi]
    const double* f_x;
    const double* phi_x;
    const double* f_y;
    double inv_n2;
};

__device__ cufftDoubleReal d_bracket_d2z_load(
    void*              dataIn,
    unsigned long long offset,
    void*              callerInfo,
    void*              sharedMem)
{
    const BracketD2zInfo* bi = (const BracketD2zInfo*)callerInfo;
    return bi->inv_n2 * (bi->phi_y[offset] * bi->f_x[offset]
                       - bi->phi_x[offset] * bi->f_y[offset]);
}
```

- Compile to `bracket_d2z_load_cb.fatbin` with same LTO flags
- Embed via `bin2c --name bracket_d2z_load_cb_fatbin --type longlong`

#### 2. Host changes in `cufft_lto_bracket.cu`

- Add `#include "bracket_d2z_load_cb_fatbin.h"`
- Allocate `BracketD2zInfo* d_bracket_info` on device
- On the D2Z plan: call `cufftXtSetJITCallback(lto_plan_d2z, "d_bracket_d2z_load", ..., CUFFT_CB_LD_REAL_DOUBLE, &d_bracket_info_void)` **before** `cufftMakePlanMany`
- Replace `cufftPlanMany` for D2Z with `cufftCreate` + `cufftXtSetJITCallback` + `cufftMakePlanMany`
- Before `cufftExecD2Z`: update `d_bracket_info` with the 4 workspace pointers + `inv_n2`
- Remove: `lto_bracket_kernel`, `d_ws_nl`

#### 3. CMake changes

Add the new fatbin build rule (same pattern as `bracket_load_cb`).

#### 4. Validation

Benchmark must still produce `rel_l2 ≈ 2.9e-16` vs `output_fp64`.

---

## Optimization Level 2 — Two-for-one Z2Z (halve FFT count)

**Idea:** The FFT is linear, so `IFFT(A + iB) = IFFT(A) + i·IFFT(B)`. Pack gradient pairs into complex signals and use Z2Z (complex-to-complex) instead of Z2D:
- Z2Z #0: `IFFT(i·ky·φ + i·(i·kx·f))` → `Re = φ_y`, `Im = f_x`
- Z2Z #1: `IFFT(i·kx·φ + i·(i·ky·f))` → `Re = φ_x`, `Im = f_y`

This halves the FFT launches from 4 → 2 and halves the callback gather passes over df+phi.

**What it eliminates:**
- 2 of 4 callback gather passes: **0.71 GB saved**
- Combined with Level 1: **1.57 GB total saved (26%)**

**Expected speedup:** ~20–24% → **~19–20 ms**

**Kernel launches:** 5 → 3 (with Level 1)

### Complications

Z2Z operates on the **full spectrum** `[mrad, mphi]` (not the half-spectrum `[mrad, mphiw3]`). The load callback must provide Hermitian-extended values for `j > mphi/2`:

```
For j in [0, mphi/2]: compute gradient directly (same as current Z2D callback)
For j in (mphi/2, mphi-1]:
    j' = mphi - j
    i' = (mrad - i_dense) % mrad
    A_conj = conj(gradient_A(i', j'))   // Hermitian extension of first field
    B_conj = conj(gradient_B(i', j'))   // Hermitian extension of second field
    return A_conj + i * B_conj          // combined two-for-one value
```

The conjugate half doubles callback computation (2 gradients per element), but only for ~half the positions. Net callback work: ~1.5× per Z2Z call, 2 calls → 3.0× total vs 4.0× for 4 Z2D = **25% less callback compute**.

### Implementation

#### 1. New device callback: `bracket_z2z_load_cb.cu`

New `CallbackInfoZ2Z` struct:
```cpp
struct CallbackInfoZ2Z {
    const double2* df_packed;     // [batch, nkx, nky]
    const double2* phi_packed;    // [batch, nkx, nky]
    const double*  kx;            // [nkx]
    const double*  ky;            // [nky]
    const int*     inverse_jind;  // [mrad]
    int mrad;
    int mphi;           // full size (not half!)
    int nkx;
    int nky;
    int pair_type;      // 0 = (phi_y, f_x), 1 = (phi_x, f_y)
};
```

Callback `d_z2z_load_cb_ptr`:
- Offset is flat into `[batch, mrad, mphi]` complex
- `j = local % mphi`
- If `j <= mphi/2`: compute as in current Z2D callback but return `A + i*B`
- If `j > mphi/2`: compute Hermitian conjugate pair at `(i', j')`, return `conj(A) + i*conj(B)`
- `pair_type` selects which gradient pair: 0 = (i·ky·φ, i·kx·f), 1 = (i·kx·φ, i·ky·f)

#### 2. Host changes

- Replace `lto_plan_z2d` (Z2D) with `lto_plan_z2z` (Z2Z)
- Z2Z output is `[batch, mrad, mphi]` complex → workspace is 2 complex arrays instead of 4 real
- Update `gradient_type` → `pair_type`, only 1 H2D copy between Z2Z calls instead of 3
- Bracket in D2Z load callback reads complex arrays: `inv_n2 * (Re(z0)*Im(z0) - Re(z1)*Im(z1))`

#### 3. Memory savings

Workspace drops from 5 × 424.7 MB = 2.12 GB to 2 × 849.3 MB = 1.70 GB. Net allocation is similar, but we avoid 2 kernel launches and 2 gather passes.

#### 4. Validation

Must produce `rel_l2 ≈ 2.9e-16`. The two-for-one trick is exact in IEEE 754 arithmetic.

---

## Optimization Level 3 — D2Z store callback (sparse output)

**Idea:** The D2Z currently writes a dense `[batch, mrad, mphiw3]` output (433 MB), of which only 41% is useful (the `nkx × nky` packed modes). A `CUFFT_CB_ST_COMPLEX_DOUBLE` store callback on D2Z can write directly to the packed `[batch, nkx, nky]` output, skipping 59% of writes.

**What it eliminates:**
- 59% of D2Z output bandwidth: **255 MB saved**
- Python-side `unpack_half_spectrum` call (a gather op on XLA side)
- Changes FFI output shape from `[batch, mrad, mphiw3]` to `[batch, nkx, nky]`

**Expected additional speedup:** ~2–4% (D2Z output is a small fraction of total traffic)

### Implementation

#### 1. New device callback: `bracket_d2z_store_cb.cu`

```cpp
struct D2zStoreInfo {
    double2*   packed_out;     // [batch, nkx, nky] — the actual FFI output
    const int* jind;           // [nkx] → dense kx index for each packed kx
    int mrad;
    int mphiw3;
    int nkx;
    int nky;
};

__device__ void d_d2z_store_cb(
    void*              dataOut,
    unsigned long long offset,
    cufftDoubleComplex element,
    void*              callerInfo,
    void*              sharedMem)
{
    const D2zStoreInfo* si = (const D2zStoreInfo*)callerInfo;
    long long dense_stride = (long long)si->mrad * si->mphiw3;
    long long batch_idx = (long long)offset / dense_stride;
    long long local     = (long long)offset % dense_stride;
    int i_dense = (int)(local / si->mphiw3);
    int j       = (int)(local % si->mphiw3);

    if (j >= si->nky) return;  // dealiased mode — skip

    // Linear search through jind to find packed index for this i_dense
    // (or use a reverse lookup table passed via callerInfo)
    // If i_dense is not in jind: skip
    // If found at i_packed: write to packed_out[batch_idx, i_packed, j]
}
```

Note: the reverse lookup requires `inverse_jind` (dense → packed), which is already available. Add it to the struct.

#### 2. Dual callback on D2Z plan

The D2Z plan needs BOTH a load callback (Level 1 bracket fusion) and a store callback. cuFFT supports registering multiple callback types on the same plan — call `cufftXtSetJITCallback` twice with different `cufftXtCallbackType` values before `cufftMakePlanMany`. Each callback must be in a separate `.cu` fatbin.

#### 3. Python changes

- FFI output shape changes from `(batch, mrad, mphiw3)` to `(batch, nkx, nky)`
- Remove `unpack_half_spectrum` call in `nonlinear_term_lto`
- Add `jind` as an additional FFI input (int32 array, shape `[nkx]`)

---

## Optimization Level 4 — CUDA Graphs

**Idea:** Capture the entire execution sequence (2 Z2Z + 1 D2Z, or whatever Level 1–3 produces) as a CUDA Graph. Replay the graph on subsequent calls, avoiding per-call host overhead (plan lookup, kernel launch API, stream synchronization).

**Expected speedup:** ~2–5% (reduces ~5 kernel launches of host overhead)

### Implementation

- On first call: `cudaStreamBeginCapture` → execute all kernels → `cudaStreamEndCapture` → `cudaGraphInstantiate`
- On subsequent calls: update input pointers via `cudaGraphExecKernelNodeSetParams` or use `cudaGraphExecUpdate`
- Requires careful handling: cuFFT internal workspace pointers must remain stable between replays (they do if the plan is reused)
- The CallbackInfo H2D copies and cuFFT exec calls are all on the same stream, so the graph captures the full dependency chain

### Risks

- cuFFT may use internal graph-incompatible operations
- Pointer update API may not cover all node types cuFFT uses internally
- Test feasibility with a minimal prototype before committing

---

## Optimization Level 5 — cuFFTDx fully fused kernel (stretch goal)

**Idea:** Use [cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/) (device-level FFT library) to embed the entire bracket computation in a single custom kernel:

1. Load `df[k]` and `phi[k]` once from HBM (packed spectral)
2. Compute 4 gradient spectral values
3. Run 4 independent 2D IFFTs via cuFFTDx (register + shared memory)
4. Multiply pointwise for Poisson bracket
5. Run 1 forward 2D FFT via cuFFTDx
6. Store result directly to packed output

**What it eliminates:** ALL intermediate HBM traffic. Reads `df + phi` exactly once (356 MB), writes output once (178 MB). Total: 534 MB vs current 6.11 GB = **91% reduction**.

**Expected speedup:** 3–5× over LTO v0 → potential **~5–7 ms**

### Complications

- cuFFTDx handles FFTs of **fixed, compile-time sizes**. Our sizes (mrad=135, mphi=96) must be known at compile time — feasible for this config but not general.
- 2D FFTs in cuFFTDx require careful tiling: 1D FFTs along mphi (size 96 = 2^5 × 3) in shared memory, then 1D FFTs along mrad (size 135 = 3^3 × 5) with transpose.
- Each thread block must hold enough shared memory for the FFT + working data. For FP64 complex: 135 × 96 × 16 = 207 KB per 2D slice — exceeds shared memory (164 KB on A100). Would need to tile the batch dimension and process sub-slices.
- Not available via conda — requires manual CUDA Toolkit integration.

This is a research-level optimization. Only pursue after Levels 1–3 are validated.

---

## Implementation order and testing strategy

```
Level 1 (bracket fusion)   ← start here, easiest win
  ↓ validate rel_l2 ≈ 2.9e-16, benchmark
Level 2 (two-for-one Z2Z)  ← biggest architectural change
  ↓ validate, benchmark
Level 3 (sparse output)    ← small incremental gain
  ↓ validate, benchmark
Level 4 (CUDA Graphs)      ← only if host overhead is visible in profiling
Level 5 (cuFFTDx)          ← stretch goal, research-level
```

Each level adds a new benchmark variant in `jax_ffi_benchmark.py`. Keep all previous variants for comparison. Use the naming convention `LTO v1`, `LTO v2`, etc.

For each level, create a **separate `.cu` host wrapper** (e.g., `cufft_lto_bracket_v1.cu`) rather than modifying the working v0. This keeps the baseline intact for A/B comparison.

---

## ⚠️ Fatbin alignment — root cause of CUFFT_INTERNAL_ERROR (5)
`xxd -i` generates `unsigned char[]` (1-byte aligned). nvJitLink internally casts the pointer to an
ELF/fatbin struct requiring ≥8-byte alignment, causing `CUFFT_INTERNAL_ERROR (5)` at `cufftMakePlan*`.

**Fix:** use `bin2c --type longlong` which generates `unsigned long long[]` (8-byte aligned). Pass the
array pointer directly to `cufftXtSetJITCallback` — no `posix_memalign` staging needed.

LTO callbacks work correctly on A100 MIG 3g.20gb with the bin2c-generated header.
