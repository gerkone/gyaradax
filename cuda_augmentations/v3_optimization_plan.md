# V3 Optimization Plan — Z2D×4 + D2Z Bracket Fusion

## Root cause (revised 2026-03-26)

**Original diagnosis was wrong.** The claim that Z2Z LTO load callbacks drop the
imaginary channel is disproven by a direct MRE:

```
MRE_callback_complex_drop/  — callback returns {1.0, 2.0} for every element
Result: DC mode = {1024.00, 2048.00}  ← both channels live
```

Z2Z `CUFFT_CB_LD_COMPLEX_DOUBLE` callbacks correctly propagate the imaginary
component on A100/CUDA 12.9. The `ws0.y ≈ 10⁻⁵¹` result from the diagnostic
came from something else — most likely a bug in the `bracket_d2z_load_v2_cb.cu`
callback or in how the diagnostic checked the output, not a cuFFT platform limit.

**Confirmed bug (still real):** `bracket_d2z_load_v2_cb.cu` computes `batch_idx`
but never uses it to apply `dum_s[batch_idx % nspec]`. This means every batch
element is weighted by an uninitialised/zero `dum_s`, producing near-zero output.
This alone is sufficient to explain the diagnostic result.

**Why Z2D×4 + D2Z is still the right V3 architecture** (motivation updated):
- The `dum_s` bug and any Hermitian formula issues in `bracket_z2z_load_cb.cu`
  would need their own debugging session; V3 avoids the complexity entirely.
- The Z2D×4 + fused D2Z path eliminates `d_nl` (0.85 GB HBM traffic saved) and
  is already proven correct via `bracket_d2z_load_cb.cu` which handles `dum_s`
  correctly.

---

## New V3 architecture

Keep the proven **4 × Z2D** approach from V1 but replace the explicit bracket kernel +
plain D2Z with a **single D2Z plan carrying the `bracket_d2z_load_cb` LTO load
callback**. `bracket_d2z_load_cb.cu` already exists, already includes `dum_s` support,
and its fatbin is already built by CMake.

### What gets eliminated vs V1

| Removed                          | Saving              |
|----------------------------------|---------------------|
| `lto_bracket_explicit_kernel`    | 1 kernel launch     |
| `d_nl` buffer                    | 0.42 GB allocation  |
| HBM write bracket → `d_nl`      | −0.42 GB traffic    |
| HBM read  `d_nl` → D2Z input    | −0.42 GB traffic    |
| **Total HBM saved**              | **−0.85 GB**        |

Expected wall-time: ~21–22 ms (from V1's 22.66 ms).

---

## Step-by-step implementation

### Step 1 — Replace the V3 state struct (`cufft_lto_bracket.cu`)

Remove all Z2Z fields:
- `p0, p1` (Z2Z plans), `d2z` (old D2Z plan)
- `d_cb0, d_cb1, d_d2z_cb` and their `_ptr` companions
- `d_ws0, d_ws1, d_d2z_in`

Replace with a struct mirroring V1 minus `d_nl`, plus a `BracketD2zInfo` device ptr.
`d_in` is kept for the Z2D dummy only — there is **no separate D2Z dummy buffer**
(see Step 3).

```cpp
static struct {
    cufftHandle z2d = 0, d2z = 0;
    int batch = -1;
    CallbackInfo*    d_cb      = nullptr;   // Z2D load callback info
    void*            d_cb_ptr  = nullptr;
    BracketD2zInfo*  d_d2z_cb  = nullptr;   // D2Z bracket-fusion callback info
    void*            d_d2z_ptr = nullptr;
    double*          d_dum     = nullptr;
    cufftDoubleComplex* d_in   = nullptr;   // Z2D dummy input (half-spectrum)
    double *d_py = nullptr, *d_fx = nullptr,
           *d_px = nullptr, *d_fy = nullptr;  // real-space workspaces (no d_nl)
} v3;
```

### Step 2 — Plan setup block (first call or batch change)

Teardown any existing resources **before** re-allocating (prevents VRAM leaks on batch
change):

```cpp
if (v3.batch != -1) {
    cufftDestroy(v3.z2d); cufftDestroy(v3.d2z);
    cudaFree(v3.d_cb);    cudaFree(v3.d_d2z_cb);
    cudaFree(v3.d_dum);   cudaFree(v3.d_in);
    cudaFree(v3.d_py);    cudaFree(v3.d_fx);
    cudaFree(v3.d_px);    cudaFree(v3.d_fy);
    v3.z2d = v3.d2z = 0;
    v3.d_cb = nullptr; v3.d_d2z_cb = nullptr;
    v3.d_dum = nullptr; v3.d_in = nullptr;
    v3.d_py = v3.d_fx = v3.d_px = v3.d_fy = nullptr;
}

// Allocate — no d_nl, no separate D2Z dummy
cudaMalloc(&v3.d_cb,     sizeof(CallbackInfo));
cudaMalloc(&v3.d_d2z_cb, sizeof(BracketD2zInfo));
cudaMalloc(&v3.d_dum,    dum_s.dimensions()[0] * 8);
cudaMalloc(&v3.d_in,     (size_t)batch * c_dist * 16);  // Z2D dummy only
cudaMalloc(&v3.d_py,     (size_t)batch * r_dist * 8);
cudaMalloc(&v3.d_fx,     (size_t)batch * r_dist * 8);
cudaMalloc(&v3.d_px,     (size_t)batch * r_dist * 8);
cudaMalloc(&v3.d_fy,     (size_t)batch * r_dist * 8);
v3.d_cb_ptr  = (void*)v3.d_cb;
v3.d_d2z_ptr = (void*)v3.d_d2z_cb;

// Z2D plan — identical to V1
cufftCreate(&v3.z2d);
cufftXtSetJITCallback(v3.z2d, "d_load_cb_ptr",
    (void*)bracket_load_cb_fatbin, sizeof(bracket_load_cb_fatbin),
    CUFFT_CB_LD_COMPLEX_DOUBLE, &v3.d_cb_ptr);
size_t ws = 0;
cufftXtMakePlanMany(v3.z2d, 2, n_ll,
    NULL, 1, c_dist, CUDA_C_64F,
    NULL, 1, r_dist, CUDA_R_64F,
    batch, &ws, CUDA_C_64F);

// D2Z plan with bracket-fusion LTO load callback
cufftCreate(&v3.d2z);
cufftXtSetJITCallback(v3.d2z, "d_bracket_d2z_load",
    (void*)bracket_d2z_load_cb_fatbin, sizeof(bracket_d2z_load_cb_fatbin),
    CUFFT_CB_LD_REAL_DOUBLE, &v3.d_d2z_ptr);
cufftXtMakePlanMany(v3.d2z, 2, n_ll,
    NULL, 1, r_dist, CUDA_R_64F,
    NULL, 1, c_dist, CUDA_C_64F,
    batch, &ws, CUDA_R_64F);

v3.batch = batch;
```

### Step 3 — Per-call execution block

The D2Z callback overrides every input read, so cuFFT ignores the `idata` pointer but
still validates it is non-null. Reuse `d_py` (already allocated, correct size) instead
of wasting VRAM on a dedicated dummy.

```cpp
cufftSetStream(v3.z2d, stream);
cufftSetStream(v3.d2z, stream);
cudaMemcpyAsync(v3.d_dum, dum_s.typed_data(), nspec * 8, cudaMemcpyHostToDevice, stream);

// 4 × Z2D — same H2D pattern as V1 (update gradient_type between calls)
CallbackInfo h_ci = { df, phi, kx, ky, jind, mrad, mphi_half, nkx, nky,
                      /*gradient_type=*/0, n_df_batches, n_phi_batches };
cudaMemcpyAsync(v3.d_cb, &h_ci, sizeof(CallbackInfo), cudaMemcpyHostToDevice, stream);
cufftExecZ2D(v3.z2d, v3.d_in, v3.d_py);   // phi_y
// ... update gradient_type 1,2,3 → d_fx, d_px, d_fy (same as V1)

// Populate D2Z bracket-fusion callback struct
BracketD2zInfo h_dci = { v3.d_py, v3.d_fx, v3.d_px, v3.d_fy,
                         v3.d_dum, nspec, mrad, mphi, /*scale=*/1.0 };
cudaMemcpyAsync(v3.d_d2z_cb, &h_dci, sizeof(BracketD2zInfo), cudaMemcpyHostToDevice, stream);

// D2Z with bracket fusion — pass d_py as the non-null dummy; callback overrides the read
cufftExecD2Z(v3.d2z, (double*)v3.d_py, (cufftDoubleComplex*)out->typed_data());
```

`scale = 1.0` because `apply_physics_wrapper` in Python applies `1/N²` normalisation
(same as V0/V1).

### Step 4 — `dum_s` arithmetic in `bracket_d2z_load_cb.cu`

The callback already exists with the correct pattern. Verify it reads:

```cpp
__device__ double d_bracket_d2z_load(..., unsigned long long offset, ...) {
    const BracketD2zInfo* ci = (const BracketD2zInfo*)callerInfo;
    size_t elements_per_batch = (size_t)ci->mrad * ci->mphi;
    int batch_idx   = (int)(offset / elements_per_batch);
    int species_idx = batch_idx % ci->nspec;
    double dum      = ci->dum_s[species_idx];

    return dum * (ci->py[offset] * ci->fx[offset]
                - ci->px[offset] * ci->fy[offset]);
}
```

`offset` is the flat real-element index into `[batch, mrad, mphi]`, so dividing by
`mrad * mphi` gives the correct batch index.

### Step 5 — Includes

Add at top of `cufft_lto_bracket.cu` if not already present:

```cpp
#include "bracket_d2z_load_cb_fatbin.h"
```

Also copy the `BracketD2zInfo` struct definition from `bracket_d2z_load_cb.cu` into
the host file so host and device layouts match exactly.

### Step 6 — Dead Z2Z files

`bracket_z2z_load_cb.cu` and `bracket_d2z_load_v2_cb.cu` remain in the repo and in
CMakeLists for reference. The V3 host code simply stops including their fatbins.

### Step 7 — Validation

```
rel_l2 ≈ 2.9e-16 vs output_fp64   (same threshold as V0/V1)
```

Run with:
```bash
mamba run -n jax_env python cuda_augmentations/jax_ffi_benchmark.py
```

---

## What this does NOT solve

- **Level 2 (two-for-one Z2Z):** halving FFT count from 4 → 2. The cuFFT platform
  limitation is **no longer the blocker** — Z2Z imaginary channel is confirmed working
  (see MRE). The remaining work is debugging `bracket_z2z_load_cb.cu`: fix the `dum_s`
  missing in `bracket_d2z_load_v2_cb.cu` and audit the Hermitian extension formula in
  `bracket_z2z_load_cb.cu` against the `ws0.y ≈ 10⁻⁵¹` diagnostic. Can be revisited
  as a follow-on once V3 is validated.

- Level 3 (sparse D2Z store), Level 4 (CUDA Graphs), Level 5 (cuFFTDx) are unaffected
  by this fix and remain on the roadmap as written in `tasks.md`.
