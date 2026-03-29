# LTO Bracket Optimization Plan

## Baseline performance (measured on full A100, batch=4096)

| Variant               | Time (ms) | Speedup vs FP64 | Rel L2     | Notes                          |
|-----------------------|-----------|-----------------|------------|--------------------------------|
| JAX FP64              | 29.34     | 1.00×           | 0          | reference                      |
| JAX Mixed             | 28.990    | 1.08×           | 3.05e-08   |                                |
| cuFFT FFI (std)       | 32.967    | 0.95×           | 5.98e-16   | slower than JAX due to FFI overhead |
| LTO cuFFT v0          | 24.464    | 1.28×           | 6.60e-16   | Z2D gather callbacks           |
| LTO cuFFT v1          | 24.497    | 1.27×           | 6.60e-16   | same as v0 (Level 1 negligible on A100) |
| **LTO cuFFT v2**      | **22.470**| **1.32×**       | 6.60e-16   | + D2Z bracket fusion (Level 1) |
| **LTO v3 (v1+store)** | **22.612**| **1.38×**       | 6.60e-16   | + sparse store CB (Level 3)    |
| **LTO vZ2Z**          | **19.780**| **1.48×**       | 2.58e-15   | + same-field Z2Z (Level 2)     |

## Grid dimensions (configs/iteration_13.yaml)

```
nkx=85  nky=32  nv=32  nmu=8  nspec=16  → batch = 4096
mrad=135  mphi=96  mphiw3=49
Sparsity: nkx×nky / (mrad×mphiw3) = 2720/6615 = 0.41
```

## Current vZ2Z data flow and HBM traffic

```
Step                             Kernel launches   HBM traffic
─────────────────────────────────────────────────────────────
Sym+Reduce                       1 launch          ~43 KB (negligible)
Z2Z #0 (ϕ, load CB gathers)     1 launch          178 MB read + 849 MB write
Z2Z #1 (f, load CB gathers)     1 launch          178 MB read + 849 MB write
D2Z (bracket load CB + sparse)   1 launch          1698 MB read + 178 MB write
─────────────────────────────────────────────────────────────
Total                            4 launches        3.93 GB
Effective bandwidth at 19.78ms:  199 GB/s (10% of ~2 TB/s A100 peak)
```

**Key observation:** External I/O (3.93 GB) at 19.78ms = 199 GB/s. A100 peak
is ~1555 GB/s. The FFTs' internal butterfly traffic dominates — cuFFT does
~10 internal passes over the data. External I/O is no longer the primary
bottleneck. This changes which optimizations matter.

Workspace: 2 complex buffers × 849 MB = 1.70 GB.

---

## [x] Optimization Level 1 — Fuse bracket into D2Z load callback

- [x] Bench: v2 = 22.470ms → **saves 1.8ms (7.3%)** vs v1.
- [x] Verify: rel_l2 ≈ 6.60e-16.

---

## [x] Optimization Level 3 — D2Z store callback (sparse output)

- [x] Bench: v3 = 22.612ms, gain ~0.09ms (noise). Worth keeping for correctness.

---

## [x] Optimization Level 2 — Same-field paired Z2Z (halve FFT count)

### ✅ VALIDATED — Python rel_l2 = 4.414e-16, CUDA rel_l2 = 2.58e-15

- [x] Step 2.0: Confirmed Z2Z LTO load callback invocation.
- [x] Step 2.1: Implemented symmetrize + reduce kernel.
- [x] Step 2.2: Implemented `bracket_z2z_load_cb.cu` with 2D Hermitian extension + j=0 symmetrization.
- [x] Step 2.3: Implemented `bracket_d2z_load_cb_v2.cu` (complex inputs, α/β rescaling).
- [x] Step 2.4: Host wrapper `cufft_lto_bracket_vz2z.cu`.
- [x] Step 2.5: Benchmark integration in `jax_ffi_benchmark.py`.
- [x] Step 2.6: Validated rel_l2 = 2.58e-15. Runtime = 19.78ms (1.48× vs JAX FP64).

### Core Design

Pack both spatial derivatives of the **same physical field** into one complex signal:

```
Z2Z #0:  Z_k = (ikx · ϕ_k)/α₀ + i·(iky · ϕ_k)/β₀   →  IFFT  →  Re = ∂xϕ/α₀,  Im = ∂yϕ/β₀
Z2Z #1:  Z_k = (ikx · f_k)/α₁ + i·(iky · f_k)/β₁    →  IFFT  →  Re = ∂xf/α₁,   Im = ∂yf/β₁
```

### Bugs Found and Fixed

**Bug 1 — Cross-field leakage (~1e-2):** Original packed (ϕ, f) cross-field.
|ϕ| ≫ |f| causes machine-epsilon leakage at catastrophic relative magnitude.
Fixed by same-field pairing.

**Bug 2 — Hermitian asymmetry at j=0 (~3.5e-4):** Gyro-averaging breaks
Hermitian symmetry at ky=0 column of phi (14% relative asymmetry). Invisible
to irfft2, fatal to Z2Z packing. Fixed by inline symmetrization in callback.

---

## [ ] Optimization Level 2a — Merge Z2Z Calls (Tier 1: cross 1.5×)

**Idea:** Merge the two separate Z2Z calls into a **single batched Z2Z with
batch=8192**. The first 4096 batches process ϕ, the next 4096 process f.
The load callback discriminates via batch index.

**What it gains:**
- One fewer kernel launch + one fewer cuFFT plan
- Better GPU occupancy from doubled batch parallelism
- cuFFT scheduler has more work to hide latency

**Expected speedup:** 0.3–0.8ms → **~19.0ms (1.54×)**

### Implementation

#### 1. Merged callback struct

```cpp
struct CallbackInfoZ2Z_Merged {
    const double2* phi_packed;   // field 0
    const double2* df_packed;    // field 1
    double alpha0, beta0;        // scale factors for ϕ
    double alpha1, beta1;        // scale factors for f
    int field_boundary;          // = 4096 (original batch size)
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    int mrad, mphi, nkx, nky;
};
```

#### 2. Callback discrimination logic

```cpp
__device__ cufftDoubleComplex d_z2z_merged_load(
    void* dataIn, unsigned long long offset,
    void* callerInfo, void* sharedMem)
{
    const auto* info = (const CallbackInfoZ2Z_Merged*)callerInfo;
    long long elems_per_batch = info->mrad * info->mphi;
    int batch_idx = (int)(offset / elems_per_batch);

    // Select field and scale factors based on batch index
    const double2* field;
    double alpha, beta;
    int local_batch;
    if (batch_idx < info->field_boundary) {
        field = info->phi_packed;
        alpha = info->alpha0;  beta = info->beta0;
        local_batch = batch_idx;
    } else {
        field = info->df_packed;
        alpha = info->alpha1;  beta = info->beta1;
        local_batch = batch_idx - info->field_boundary;
    }

    // Remap offset to local batch for field lookup
    long long local_offset = (long long)local_batch * elems_per_batch
                           + (offset % elems_per_batch);

    // ... rest of gather + Hermitian extension + j=0 symmetrize (unchanged)
}
```

#### 3. Workspace layout

Single contiguous buffer: `ws[8192, mrad, mphi]` complex.
- `ws[0..4095, :, :]` = ϕ gradients (∂xϕ/α₀ in real, ∂yϕ/β₀ in imag)
- `ws[4096..8191, :, :]` = f gradients (∂xf/α₁ in real, ∂yf/β₁ in imag)

#### 4. Updated D2Z bracket callback

```cpp
struct BracketD2zInfoMerged {
    const double2* ws;           // [8192, mrad, mphi] — merged workspace
    int phi_offset;              // = 0 (batch offset for ϕ gradients)
    int df_offset;               // = 4096 * mrad * mphi (batch offset for f)
    double alpha0, beta0;
    double alpha1, beta1;
    double inv_n2;
};

__device__ cufftDoubleReal d_bracket_d2z_load_merged(
    void* dataIn, unsigned long long offset,
    void* callerInfo, void* sharedMem)
{
    const auto* bi = (const BracketD2zInfoMerged*)callerInfo;

    // D2Z offset is into [4096, mrad, mphi] real.
    // Read phi gradients from ws[offset] and f gradients from ws[offset + df_offset]
    double2 z0 = bi->ws[offset + bi->phi_offset];
    double2 z1 = bi->ws[offset + bi->df_offset];

    double dxphi = z0.x * bi->alpha0;
    double dyphi = z0.y * bi->beta0;
    double dxf   = z1.x * bi->alpha1;
    double dyf   = z1.y * bi->beta1;

    return bi->inv_n2 * (dyphi * dxf - dxphi * dyf);
}
```

#### 5. Steps

- [ ] **Step 2a.1:** Create merged Z2Z load callback `bracket_z2z_merged_load_cb.cu`.
- [ ] **Step 2a.2:** Create merged D2Z bracket callback `bracket_d2z_load_cb_merged.cu`.
- [ ] **Step 2a.3:** New host wrapper `cufft_lto_bracket_vz2z_merged.cu`.
    - Single Z2Z plan with batch=8192.
    - Workspace: `[8192, mrad, mphi]` complex = 3.40 GB.
    - Execution: sym+reduce → Z2Z (batch=8192) → D2Z.
    - 3 kernel launches (down from 4).
- [ ] **Step 2a.4:** Benchmark variant `LTO vZ2Z-merged`.
- [ ] **Step 2a.5:** Validate rel_l2 ≈ 2.58e-15 vs fp64 baseline.

#### 6. Memory note

Workspace increases from 1.70 GB (2 × 849 MB) to 3.40 GB (1 × 8192 batches).
The buffers are contiguous, so this is a single allocation. Verify this fits
within A100 available memory after accounting for df/phi inputs and cuFFT
scratch space.

---

## [ ] Optimization Level 4 — CUDA Graphs (Tier 2: push toward 1.6×)

**Idea:** Capture the entire execution sequence as a CUDA Graph and replay it.

**What it gains:**
- Eliminates per-launch host overhead (~50μs × 3 launches = 150μs)
- Driver can optimize inter-kernel scheduling
- Zero code changes to callbacks

**Expected speedup:** 0.2–0.5ms → **~18.5ms (1.59×)**

### Implementation

```cpp
// First call: capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  compute_scale_factors<<<...>>>(phi, df, kx, ky, &scales);
  cufftExecZ2Z(merged_z2z_plan, ws_in, ws_out, CUFFT_INVERSE);
  cufftExecD2Z(d2z_plan, dummy_in, packed_out);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graph_exec, graph, 0);

// Subsequent calls: replay
cudaGraphLaunch(graph_exec, stream);
```

**Requirements:**
- All buffer pointers must remain stable between replays (they do with reused plans).
- CallbackInfo updates (scale factors change per call) can be handled by
  writing to device memory before graph launch — the graph captures the
  pointer, not the value.

### Steps

- [ ] **Step 4.1:** Prototype graph capture with the merged 3-kernel sequence.
- [ ] **Step 4.2:** Verify cuFFT kernels are graph-compatible (some cuFFT versions
      use graph-incompatible operations internally — test empirically).
- [ ] **Step 4.3:** Handle scale factor updates: write ScaleFactors to device
      memory before `cudaGraphLaunch`. The callback reads from the pointer
      (captured in the graph), which now points to updated values.
- [ ] **Step 4.4:** Benchmark variant `LTO vZ2Z-graph`.

### Risks

- cuFFT may use internal graph-incompatible operations.
- If graph capture fails, this level is skipped — no fallback needed,
  the non-graph path is the current working code.

---

## [ ] Optimization Level 4a — L2 Persistence Policy (Tier 2 addendum)

**Idea:** Hint the L2 cache controller to keep workspace data resident between
the Z2Z write and D2Z read. Zero code changes to kernels or callbacks.

**What it gains:**
- The tail batches of Z2Z are still warm in L2 when D2Z starts reading them.
- A100 has up to 40 MB of configurable persistent L2.

**Expected speedup:** 0.2–0.8ms (depends on temporal overlap).

### Implementation

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr = ws_buffer;
attr.accessPolicyWindow.num_bytes = min(ws_size, (size_t)prop.persistingL2CacheMaxSize);
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

### Steps

- [ ] **Step 4a.1:** Add L2 persistence hints for workspace buffers.
- [ ] **Step 4a.2:** Benchmark with and without — measure actual L2 hit rate
      via `ncu` profiler (`l2_hit_rate` metric).
- [ ] **Step 4a.3:** Can be combined with CUDA Graphs (set policy before capture).

---

## [ ] Optimization Level 5 — Tiled Execution with L2-Resident Workspace (Tier 3: reach 1.8×+)

**Idea:** Instead of processing all 8192 batches at once, tile into chunks of
~96 batches so the workspace per tile (~40 MB) fits in L2.

**What it gains:**
- The 1.70 GB workspace HBM read (D2Z side) becomes L2 hits
- Total external HBM traffic drops from 3.93 GB to ~2.23 GB
- At 199 GB/s effective: ~11ms theoretical floor

**Realistic estimate:** cuFFT efficiency drops with smaller batch sizes,
so **~15–16ms (1.8–1.9×)**.

### Design

```
for each tile of ~96 batches:
    Z2Z (batch=192, both fields merged)   → ws stays in L2
    D2Z (batch=96)                        → reads ws from L2, not HBM
```

4096 / 96 = 43 tiles × 2 launches = 86 kernel launches. This **requires**
CUDA Graphs to be practical — capture the tiled loop as a graph, replay it.

### HBM Traffic Analysis

```
Per tile:
  Z2Z load CB: read phi+df      2 × (96 × 85 × 32 × 16B) = 8.4 MB
  Z2Z store: write ws            192 × 135 × 96 × 16B = 39.8 MB → L2!
  D2Z load CB: read ws           96 × 135 × 96 × 32B = 39.8 MB → L2 hit!
  D2Z store CB: write output     96 × 85 × 32 × 16B = 4.2 MB

43 tiles total:
  Z2Z reads:   43 × 8.4 MB = 361 MB  (gather, same as current)
  Z2Z writes:  L2 (not HBM)
  D2Z reads:   L2 (not HBM)
  D2Z writes:  43 × 4.2 MB = 181 MB  (sparse output)
  Total HBM:   ~542 MB (vs 3.93 GB current = 7.2× reduction)
```

### Complications

- cuFFT plan efficiency drops significantly for small batch sizes.
  Need to benchmark the crossover point.
- 86 launches require CUDA Graph capture of the entire loop.
- Workspace per tile is exactly ~40 MB — tight fit for A100 L2.
  Profile with `ncu` to verify actual residency.
- The loop structure may not be capturable as a single CUDA Graph
  (variable kernel parameters per tile). May need one graph per tile
  with pointer arithmetic, or a single graph with fixed pointers
  and batch-offset logic in the callbacks.

### Steps

- [ ] **Step 5.1:** Benchmark cuFFT Z2Z throughput vs batch size (96, 192, 384, 768)
      to find the efficiency knee.
- [ ] **Step 5.2:** Prototype single-tile execution (batch=192 Z2Z + batch=96 D2Z)
      with L2 persistence. Measure D2Z L2 hit rate.
- [ ] **Step 5.3:** Implement tiled loop with CUDA Graph capture.
- [ ] **Step 5.4:** Full benchmark `LTO vZ2Z-tiled`.
- [ ] **Step 5.5:** Validate rel_l2 across all tiles.

---

## [ ] Optimization Level 6 — cuFFTDx fully fused kernel (stretch goal)

**Idea:** Use cuFFTDx to embed the entire bracket computation in a single
custom kernel: load spectral data, compute gradients, run IFFTs in
register/shared memory, multiply for bracket, run forward FFT, store output.

**HBM reduction caveat:** 2D IFFT on [135, 96] = 207 KB > 164 KB shared memory
limit. Row/column FFTs require global-memory transpose. Realistic HBM
reduction: ~40–60%.

**Expected speedup:** ~2–3× over LTO v0 → **~8–12 ms** best case.

Research-level optimization. Only pursue after Levels 2a–5 are exhausted.

---

## Implementation order and testing strategy

```
Level 1 (bracket fusion)        ✅ DONE — v2: 22.470ms, rel_l2=6.60e-16
Level 3 (sparse output)         ✅ DONE — v3: 22.612ms, rel_l2=6.60e-16
Level 2 (same-field Z2Z)        ✅ DONE — vZ2Z: 19.780ms, rel_l2=2.58e-15
  ↓
Level 2a (merge Z2Z calls)      ← NEXT (Tier 1: cross 1.5×)
  |   Merge 2× Z2Z → 1× batch=8192
  |   Target: ~19.0ms (1.54×)
  ↓
Level 4 (CUDA Graphs)           ← Tier 2: push toward 1.6×
  |   Capture reduce → Z2Z → D2Z as graph
  |   Target: ~18.5ms (1.59×)
  ↓
Level 4a (L2 persistence)       ← Tier 2 addendum
  |   Hint L2 to keep ws resident between Z2Z and D2Z
  |   Target: ~18.0ms (1.63×)
  ↓
Level 5 (tiled execution)       ← Tier 3: reach 1.8×
  |   Tile to ~96 batches, ws fits in L2
  |   Requires CUDA Graphs for 86 launches
  |   Target: ~15-16ms (1.8-1.9×)
  ↓
Level 6 (cuFFTDx)               ← stretch goal; ~8-12ms best case
```

**Current best:** vZ2Z at 19.78ms (1.48× over JAX FP64).

Each level adds a new benchmark variant. Keep all previous variants for
A/B comparison. Create separate `.cu` host wrappers for each level.

---

## ⚠️ Fatbin alignment — root cause of CUFFT_INTERNAL_ERROR (5)

`xxd -i` generates `unsigned char[]` (1-byte aligned). nvJitLink requires
≥8-byte alignment.

**Fix:** `bin2c --type longlong` → `unsigned long long[]` (8-byte aligned).

---

## 📋 Bugs found during Level 2 development

### Bug 1: Cross-field pairing leakage (rel_l2 ~ 1e-2)

**Symptom:** Z2Z packing of (iky·ϕ, ikx·f) produced ~1% error.
**Cause:** |ϕ| ≫ |f|. Machine-epsilon errors in ϕ channel leak into f
channel at catastrophic relative magnitudes.
**Fix:** Same-field pairing. Pack (ikx·ϕ, iky·ϕ) and (ikx·f, iky·f).

### Bug 2: Hermitian asymmetry at j=0 (rel_l2 ~ 3.5e-4)

**Symptom:** Same-field Z2Z still had 3.5e-4 error, unaffected by scaling.
**Cause:** Gyro-averaging (Bessel multiplication) breaks Hermitian symmetry
at j=0 (ky=0) column of phi. Measured: **14% relative asymmetry**.
Invisible to irfft2 (outputs real by construction). Fatal to Z2Z packing
(leaks parasitic imaginary into other channel).
**Proof:** dxphi imag residual = 1.292e-5 (kx≠0 at j=0).
dyphi imag residual = 9.858e-17 (ky[0]=0).
max|dyphi_error| = 1.565e-6 = exactly max|dxphi| × 1.292e-5.
**Fix:** Symmetrize j=0 inline in callback:
`src = (src + conj(src_mirror)) / 2` when `j_src == 0`.
Cost: 85 complex averages per field = negligible.
**Result:** rel_l2 drops from 3.548e-4 to **4.414e-16** (Python),
**2.58e-15** (CUDA).