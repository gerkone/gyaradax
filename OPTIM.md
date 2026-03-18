# OPTIM.md — HPC Performance Analysis & GPU Optimization Roadmap

## 1. Executive Summary

The `gyaradax` gyrokinetic Vlasov-Poisson solver is **memory-bandwidth bound** on both current (A100) and next-generation (B300) NVIDIA GPUs. The aggregate arithmetic intensity of the RK4 time-advance loop is **~2.7 FLOP/byte**, well below the FP64 roofline ridge points of both targets (A100: 4.85, B300: 5.0).

The dominant cost is the **nonlinear ExB advection** (Term III), which accounts for ~95% of FLOPs per RK4 step but achieves a moderate arithmetic intensity of ~4.4 FLOP/byte — straddling the roofline knee. The **linear RHS** and **phi solve** are deeply bandwidth-limited (AI < 0.4) and dominated by strided gathers from Python-loop stencil applications that prevent XLA fusion.

**Key findings:**

| Priority | Bottleneck | Tier | Expected Speedup |
|----------|-----------|------|------------------|
| P0 | Python-loop stencils in `_apply_parallel` | JAX / Custom Kernel | 3-5× on linear RHS |
| P0 | Per-s-slice FFT launch overhead (vmap over ns) | JAX / Custom Kernel | 2-4× on NL term |
| P1 | Phi solve broadcast + reduction | JAX Level | 1.5-2× on phi |
| P1 | RK4 accumulation passes (4 extra df reads) | JAX Level | 1.2-1.5× overall |
| P2 | Mixed-precision FFT pipeline | Algorithmic | 1.5-2× on NL bandwidth |
| P2 | IMEX time integration for kinetic electrons | Algorithmic | 5-20× via larger dt |
| P3 | Fused FFT-bracket kernel (Triton/CUDA) | Custom Kernel | 2-3× on NL term |

---

## 2. Hardware Roofline Targets

### 2.1 NVIDIA B300 (Blackwell) — Estimated from B200 Published Specs

| Metric | Value |
|--------|-------|
| HBM3e bandwidth | ~8 TB/s |
| HBM capacity | 192 GB |
| FP64 (non-tensor) | ~40 TFLOP/s |
| FP64 (tensor core) | ~90 TFLOP/s |
| FP32 (non-tensor) | ~80 TFLOP/s |
| TF32 (tensor core) | ~180 TFLOP/s |
| L2 cache | ~128 MB |
| Roofline ridge (FP64 non-tensor) | 40T / 8T = **5.0 FLOP/byte** |
| Roofline ridge (FP32 non-tensor) | 80T / 8T = **10.0 FLOP/byte** |
| Roofline ridge (FP64 tensor) | 90T / 8T = **11.25 FLOP/byte** |

**Blackwell-specific hardware features:**
- **TMA (Tensor Memory Accelerator):** Asynchronous, hardware-driven bulk data transfers between global/shared memory. Eliminates software address computation overhead for strided/tiled access patterns. Critical for stencil operations.
- **WGMMA (Warpgroup MMA):** Warpgroup-level matrix multiply-accumulate. Irrelevant for this solver (no dense GEMM), but TF32 tensor cores can accelerate batched FFT butterfly operations if cuFFT exploits them.
- **128 MB L2 cache:** At 128 MB, the L2 can hold the entire `df` array for a single species (55.8 MB) plus phi and working buffers. This is a qualitative change from A100's 40 MB L2, enabling reuse across stencil points within a single kernel launch.

### 2.2 NVIDIA A100 (Current Baseline)

| Metric | Value |
|--------|-------|
| HBM2e bandwidth | 2 TB/s |
| FP64 peak | 9.7 TFLOP/s (19.5 tensor) |
| FP32 peak | 19.5 TFLOP/s |
| L2 cache | 40 MB |
| Roofline ridge (FP64) | 9.7T / 2T = **4.85 FLOP/byte** |

### 2.3 Roofline Implications

For a kernel with arithmetic intensity AI (FLOP/byte):
- **If AI < ridge point:** Kernel is **memory-bandwidth bound**. Performance = AI × BW.
- **If AI > ridge point:** Kernel is **compute bound**. Performance = peak FLOP/s.

The solver's aggregate AI of ~2.7 means that on B300:
- Achievable throughput = 2.7 × 8 TB/s = **21.6 TFLOP/s** (54% of FP64 non-tensor peak)
- Actual throughput will be lower due to strided access patterns degrading effective bandwidth

---

## 3. Solver Architecture & Array Shapes

### 3.1 Phase-Space Discretization (Kinetic Reference Case)

```
Coordinates: (nsp, vpar, mu, s, kx, ky)
             species × parallel velocity × magnetic moment × parallel position × radial wavenumber × binormal wavenumber

Reference resolution:
  nsp  = 2     (ions + electrons)
  nvpar = 32   (parallel velocity grid)
  nmu   = 8    (magnetic moment grid)
  ns    = 16   (parallel grid points)
  nkx   = 85   (radial Fourier modes)
  nky   = 32   (binormal Fourier modes)
```

### 3.2 Array Inventory

| Array | Shape | Elements | Size (complex128) | Notes |
|-------|-------|----------|-------------------|-------|
| `df` | (2, 32, 8, 16, 85, 32) | 6,980,608 | 111.7 MB | Distribution function (hot) |
| `phi` | (16, 85, 32) | 43,520 | 696 KB | Electrostatic potential |
| `bessel` | (2, 32, 8, 16, 85, 32) | 6,980,608 | 111.7 MB | Bessel J0 (precomputed, real→complex broadcast) |
| `fmaxwl` | (2, 32, 8, 16, 85, 32) | 6,980,608 | 111.7 MB | Background Maxwellian (precomputed) |
| `s_total_upar` | (9, 2, 32, 8, 16, 85, 32) | 62,825,472 | 1005 MB | Fused parallel stencil (precomputed) |
| `s_total_t7` | (9, 2, 32, 8, 16, 85, 32) | 62,825,472 | 1005 MB | Fused Term VII stencil (precomputed) |
| `phi_weight` | (2, 1, 8, 16, 85, 32) | 436,480 | 7.0 MB | Phi solve weight (precomputed) |
| `phi_diag` | (16, 85, 32) | 43,520 | 348 KB | Phi solve diagonal (precomputed, real) |
| **Total precomputed** | | | **~2.35 GB** | Resident in HBM |

**FFT grid (dealiased, per nonlinear call):**

| Array | Shape | Elements | Precision | Size |
|-------|-------|----------|-----------|------|
| Spectral half (packed) | (nvpar, nmu, mrad, mphiw3) = (32, 8, 135, 49) | 1,693,440 | complex128 | 27.1 MB |
| Real space | (nvpar, nmu, mrad, mphi) = (32, 8, 135, 96) | 3,317,760 | float64/float32 | 26.5 / 13.3 MB |

Where `mrad = 135`, `mphi = 96`, `mphiw3 = 49` are the dealiased FFT grid sizes computed by `extended_*dim_fft_size()` for `nkx=85`, `nky=32`.

### 3.3 Time Integration Structure

```
gksolve (jax.lax.scan over n_steps)
  └─ gkstep_single (1 RK4 step)
       ├─ k1 = _rhs(df)
       ├─ k2 = _rhs(df + 0.5*dt*k1)
       ├─ k3 = _rhs(df + 0.5*dt*k2)
       ├─ k4 = _rhs(df + dt*k3)
       ├─ next_df = df + (dt/6)(k1 + 2*k2 + 2*k3 + k4)   ← 4 reads + 1 write of df
       └─ post-step phi + normalization (1 extra phi solve)

  _rhs(df):
    ├─ _compute_phi(df)            → phi (reduction over nsp, nvpar, nmu)
    ├─ _compute_linear_rhs(df, phi) → vmap over nsp → _linear_rhs_core per species
    │    ├─ _apply_parallel(df, s_total_upar)    [9-point stencil, index-map gather]
    │    ├─ _apply_vpar(df, VPAR_D1) × utrap     [5-point stencil, jnp.take]
    │    ├─ _apply_vpar(df, VPAR_D4) × abs_vp    [5-point dissipation]
    │    ├─ drift: -1j * kdotvd * df              [elementwise]
    │    ├─ hyper: hyper * df                     [elementwise]
    │    ├─ drive: 1j * scale * (...) * gyro_phi  [elementwise]
    │    └─ term_vii: _apply_parallel(gyro_phi, s_total_t7)
    └─ _compute_nonlinear_rhs(df, phi) → vmap over nsp → nonlinear_term_iii per species
         └─ vmap over ns:
              ├─ 4× irfft2(mrad, mphi)   [grad_phi_y, grad_f_x, grad_phi_x, grad_f_y]
              ├─ bracket: efun * (a*b - c*d)  [real-space multiply]
              └─ 1× rfft2(mrad, mphi)   [back to spectral]
```

### 3.4 Call Counts per RK4 Step

| Component | Calls per RK4 step | Notes |
|-----------|-------------------|-------|
| `_compute_phi` | 5 (4 RHS + 1 post-step) | Or 6 in linear mode (cond branches) |
| `_compute_linear_rhs` | 4 | Each vmaps over nsp=2 species |
| `_compute_nonlinear_rhs` | 4 | Each vmaps over nsp=2, then over ns=16 |
| `_apply_parallel` | 4 × 2 × 2 = 16 | 2 per _linear_rhs_core (df + term_vii) |
| `_apply_vpar` | 4 × 2 × 2 = 16 | 2 per _linear_rhs_core (D1 + D4) |
| FFTs (irfft2 + rfft2) | 4 × 2 × 16 × 5 = 640 | 5 per (species, s-slice) per NL call |
| RK4 accumulations | 4 | `df + coeff * dt * k_i` |

---

## 4. Stencil & Dataflow Analysis

### 4.1 `_apply_parallel` — 9-Point Parallel Stencil (solver.py:756-766)

```python
def _apply_parallel(field, coeffs):
    out = jnp.zeros_like(field)
    nky = field.shape[-1]
    ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
    for i in range(9):                         # ← Python for-loop (9 iterations)
        s_map = pre["s_shift"][i]              # (ns, nkx, nky) int32 index map
        kx_map = pre["kx_shift"][i]            # (ns, nkx, nky) int32 index map
        valid = pre["valid_shift"][i]          # (ns, nkx, nky) bool mask
        shifted = jnp.where(
            valid[None, None, :, :, :],
            field[:, :, s_map, kx_map, ky_idx], # ← gathered access
            0.0
        )
        out = out + coeffs[i] * shifted         # coeffs[i] shape: (nvpar, nmu, ns, nkx, nky)
    return out
```

**Memory access pattern:**
- `field[:, :, s_map, kx_map, ky_idx]` is a **3-index gather** along (s, kx, ky). The `s_map` and `kx_map` arrays encode the parallel boundary condition (magnetic shear connection) — neighboring s-points may map to shifted kx indices. This is **inherently non-contiguous** in the (s, kx) plane, producing scattered memory reads that cannot be coalesced by the GPU memory controller.
- `coeffs[i]` (slice of `s_total_upar`) is a full 5D array: each stencil point has a spatially-varying coefficient. This means the coefficient read is contiguous (good) but doubles the memory traffic per stencil point.

**FLOP count (per call, per species):**

| Operation | Count | FLOPs |
|-----------|-------|-------|
| Gather `field[:,:,s_map,kx_map,ky_idx]` | 9 × 3.49M | 0 (memory) |
| `valid` mask (`jnp.where`) | 9 × 3.49M | 9 × 3.49M = 31.4M |
| `coeffs[i] * shifted` | 9 × 3.49M | 9 × 3.49M × 2 = 62.9M (complex mul) |
| `out + ...` accumulation | 9 × 3.49M | 9 × 3.49M × 2 = 62.9M (complex add) |
| **Total** | | **~157M FLOPs** |

Note: Complex multiply on complex128 = 6 FP64 FLOPs. Complex add = 2 FP64 FLOPs. However, `coeffs[i]` is real-valued (complex128 with zero imaginary) from the fused stencil, so the multiply is effectively 2 FP64 FLOPs per element (real × complex = 2 muls + 0 adds for the real-valued coefficient case — but XLA won't exploit this unless the coefficient is explicitly real-typed).

**Memory traffic (per call, per species):**

| Data | Direction | Elements | Bytes | Notes |
|------|-----------|----------|-------|-------|
| `field` (gathered) | Read | 9 × 3.49M | 9 × 55.8 MB = 502 MB | Non-contiguous; effective BW ~50% |
| `coeffs` (9 slices) | Read | 9 × 3.49M | 502 MB | Contiguous slice along leading dim |
| `s_map`, `kx_map` | Read | 9 × 43.5K | 3.1 MB | Index arrays (int32) |
| `valid` mask | Read | 9 × 43.5K | 1.6 MB | Bool mask |
| `out` | Write | 3.49M | 55.8 MB | Final output |
| `out` (intermediate) | R+W | 8 × 3.49M | 8 × 111.7 MB = 894 MB | Accumulation reads + writes |
| **Total** | | | **~1.96 GB** | |

**Arithmetic intensity: 157M / 1.96 GB ≈ 0.08 FLOP/byte**

**XLA fusion analysis:** The Python `for i in range(9)` loop is unrolled at trace time into 9 separate XLA operations. Each iteration produces an intermediate `shifted` and `out` array. XLA *may* fuse the `where` + multiply + add into a single kernel per iteration, but **cannot fuse across iterations** because each depends on the previous `out`. This produces **up to 9 separate kernel launches** with 9 intermediate `out` materializations (each 55.8 MB for complex128). The intermediate accumulation buffers alone add 894 MB of unnecessary memory traffic.

**Critical issue: The 3-index gather `field[:, :, s_map, kx_map, ky_idx]` forces XLA to emit a Gather HLO rather than a simple strided load.** On GPU, this becomes a scatter/gather kernel with poor memory coalescing — adjacent threads in a warp read from non-adjacent memory locations.

### 4.2 `_apply_vpar` — 5-Point Velocity-Space Stencil (solver.py:768-776)

```python
def _apply_vpar(field, coeffs):
    nv = field.shape[0]
    out = jnp.zeros_like(field)
    for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):  # ← Python for-loop (5 iterations)
        idx = jnp.clip(jnp.arange(nv) + s, 0, nv - 1)
        valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
        shifted = jnp.take(field, idx, axis=0)      # ← gather along axis 0
        out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
    return out
```

**Memory access pattern:**
- `jnp.take(field, idx, axis=0)` with `idx = clip(arange(nv) + s, 0, nv-1)` is a **shifted read along the leading axis (vpar)**. This is equivalent to `field[idx, :, :, :, :]` — a contiguous slice for each shifted index. Because vpar is the leading dimension, the shifted access pattern is a **stride-1 offset** in the outermost dimension, which means each `take` reads a nearly-contiguous block. This is far better coalesced than `_apply_parallel`.
- The `clip` at boundaries means edge values are duplicated (Neumann-like), and the `valid` mask zeros them out. This is correct but wastes the clipped read.

**FLOP count (per call, per species):**

| Operation | FLOPs |
|-----------|-------|
| `jnp.take` gather | 0 (memory) |
| `valid` mask | 5 × 3.49M = 17.5M |
| `c * shifted` | 5 × 3.49M × 2 = 34.9M (scalar × complex) |
| `out + ...` | 5 × 3.49M × 2 = 34.9M |
| **Total** | **~87M FLOPs** |

Note: `c` is a scalar (from VPAR_D1 or VPAR_D4), so `c * shifted` is 2 FP64 FLOPs per complex element.

**Memory traffic (per call, per species):**

| Data | Direction | Bytes |
|------|-----------|-------|
| `field` (5 shifted reads) | Read | 5 × 55.8 MB = 279 MB |
| `out` (intermediate accum) | R+W | 4 × 111.7 MB = 447 MB |
| `out` (final) | Write | 55.8 MB |
| **Total** | | **~782 MB** |

**Arithmetic intensity: 87M / 782 MB ≈ 0.11 FLOP/byte**

**XLA fusion analysis:** Same Python-loop unrolling issue as `_apply_parallel`. 5 kernel launches with 4 intermediate accumulations. However, the gather pattern is much simpler (axis-0 shift), so XLA could in principle fuse this into a 1D convolution — but it does not because the explicit `jnp.take` + `jnp.where` pattern is opaque to the XLA convolution recognizer.

### 4.3 `_linear_rhs_core` — Full Linear Operator (solver.py:742-804)

This function assembles all linear terms for a single species (5D arrays). It is called via `jax.vmap` over the species dimension for kinetic cases.

**Composition:**

| Sub-operation | Function | Calls | FLOPs | Bytes R+W |
|---------------|----------|-------|-------|-----------|
| Parallel streaming (Term I + dissipation) | `_apply_parallel(df, s_total_upar)` | 1 | 157M | 1.96 GB |
| Trapping/mirror (Term IV) | `utrap * _apply_vpar(df, VPAR_D1) / dvp` | 1 | 87M + 7M = 94M | 782 MB + 112 MB |
| Velocity dissipation | `disp_vp * abs_vp * _apply_vpar(df, VPAR_D4) / dvp` | 1 | 87M + 14M = 101M | 782 MB + 168 MB |
| Drift advection (Terms II) | `-1j * kdotvd * df` | 1 | 21M | 224 MB |
| Hyper-diffusion | `hyper * df` | 1 | 14M | 168 MB |
| Gyro-averaging phi | `bessel * phi_b` | 1 | 14M | 168 MB |
| Term VII (parallel field drive) | `_apply_parallel(gyro_phi, s_total_t7)` | 1 | 157M | 1.96 GB |
| Drive term | `1j * scale * (dmax_ek - signz*kdotvd*fmax/tmp) * gyro_phi` | 1 | 42M | 504 MB |
| Final summation (7 terms) | element-wise adds | 1 | 42M | 504 MB |
| **Total per species** | | | **~635M** | **~7.3 GB** |

**Per-species arithmetic intensity: 635M / 7.3 GB ≈ 0.087 FLOP/byte**

**Per RK4 step (4 calls × 2 species): 5.1B FLOPs, 58.4 GB R+W**

**Dominant cost within _linear_rhs_core:** The two `_apply_parallel` calls account for 314M out of 635M FLOPs (49%) and 3.92 GB out of 7.3 GB (54%) of memory traffic. The stencil operations are the clear bottleneck within the linear RHS.

**vmap overhead (kinetic path):**
The kinetic path in `_compute_linear_rhs` (solver.py:991-1060) uses `jax.vmap` over species with 15 input arrays mapped on various axes. This is well-structured — vmap adds a batch dimension without data copies. However, the `in_axes=(0,0,...,0,1,1)` for `s_total_upar` and `s_total_t7` means these are mapped on axis=1 (the species axis within the 9-point stencil's leading dimension). XLA may need to transpose or insert a `gather` to extract the species slice, adding overhead.

### 4.4 `nonlinear_term_iii` / `_per_s` — FFT Poisson Bracket (solver.py:203-255)

This is the most compute-intensive kernel, implementing the pseudospectral ExB nonlinearity:

```
NL = efun × (∂φ/∂y × ∂f/∂x − ∂φ/∂x × ∂f/∂y)
```

evaluated via:
1. Spectral gradients: multiply by `1j * kx` or `1j * ky`
2. Transform to real space: `irfft2` (4 transforms)
3. Pointwise bracket in real space
4. Transform back: `rfft2` (1 transform)

**Execution structure:**
```
nonlinear_term_iii(df_5d, phi, ...)
  ├─ moveaxis: df (nv, nmu, ns, nkx, nky) → (ns, nv, nmu, nkx, nky)
  └─ jax.vmap(_per_s) over ns=16 slices
       ├─ gyro_phi = bessel_s * phi_s[None, None, :, :]        # (nv, nmu, nkx, nky)
       ├─ grad_phi_y_k = 1j * ky2d * gyro_phi                  # spectral gradient
       ├─ grad_phi_x_k = 1j * kx2d * gyro_phi                  # spectral gradient
       ├─ grad_f_x_k = 1j * kx2d * df_s                        # spectral gradient
       ├─ grad_f_y_k = 1j * ky2d * df_s                        # spectral gradient
       ├─ _to_real(spec):
       │    ├─ pack_half_spectrum: (nv,nmu,nkx,nky) → (nv,nmu,mrad,mphiw3) [scatter]
       │    ├─ .astype(complex64)                               # FP64 → FP32 cast
       │    └─ irfft2(s=(mrad,mphi))                            # (nv,nmu,135,96) float32
       ├─ nl_real = efun * (to_real(∂φ/∂y)*to_real(∂f/∂x) - to_real(∂φ/∂x)*to_real(∂f/∂y))
       │    └─ 4× irfft2, 2 multiplies, 1 subtract             # all in float32
       ├─ nl_real.astype(float64)                               # FP32 → FP64 upcast
       ├─ rfft2(nl_real, s=(mrad,mphi))                         # back to complex128
       ├─ scale by fft_prefactor * fft_scale
       └─ unpack_half_spectrum: (nv,nmu,mrad,mphiw3) → (nv,nmu,nkx,nky) [gather]
```

**FLOP count (per call, per species):**

| Operation | Per s-slice | × ns=16 | Notes |
|-----------|------------|---------|-------|
| 4× spectral gradient (1j × k × f) | 4 × 256 × 85 × 32 × 8 = 22.3M | 357M | Complex multiply |
| 4× `pack_half_spectrum` scatter | — | — | Pure memory |
| 4× FP64→FP32 cast | — | — | Pure memory |
| 4× `irfft2(135, 96)` | 4 × 256 × 2.5 × 12960 × log2(12960) ≈ 4 × 113M = 452M | 7.2B | Batched over (nv, nmu)=256 |
| Real-space bracket (2 mul + 1 sub) | 256 × 135 × 96 × 3 = 9.4M | 151M | FP32 arithmetic |
| efun multiply | 256 × 135 × 96 = 3.1M | 50M | FP32 |
| FP32→FP64 upcast | — | — | Pure memory |
| 1× `rfft2(135, 96)` | 256 × 2.5 × 12960 × log2(12960) ≈ 113M | 1.8B | FP64 precision |
| Scale by prefactor | 256 × 135 × 49 × 8 ≈ 17M | 272M | Complex multiply |
| `unpack_half_spectrum` gather | — | — | Pure memory |
| **Total per species** | | **~9.8B** | |

**Memory traffic (per call, per species):**

| Data | Direction | Per s-slice | × ns=16 |
|------|-----------|-------------|---------|
| `df_s` input | Read | 256 × 85 × 32 × 16B = 11.1 MB | 178 MB |
| `phi_s` + `bessel_s` | Read | (85×32 + 256×85×32) × 16B ≈ 11.1 MB | 178 MB |
| 4× gradient spectra | Write+Read | 4 × 11.1 MB × 2 = 89 MB | 1.42 GB |
| 4× packed spectra (complex64) | Write+Read | 4 × 256×135×49×8B × 2 = 54 MB | 864 MB |
| 4× real-space (float32) | Write | 4 × 256×135×96×4B = 50 MB | 800 MB |
| 1× bracket result (float32) | Write+Read | 256×135×96×4B × 2 = 25 MB | 400 MB |
| 1× upcast to float64 | Read+Write | 256×135×96×8B × 2 = 50 MB | 800 MB |
| 1× rfft2 output (complex128) | Write | 256×135×49×16B = 27 MB | 432 MB |
| 1× unpacked result | Write | 11.1 MB | 178 MB |
| **Total per species** | | | **~5.3 GB** |

**Arithmetic intensity: 9.8B / 5.3 GB ≈ 1.85 FLOP/byte (per species, per call)**

**Per RK4 step (4 calls × 2 species): 78.4B FLOPs, 42.4 GB R+W → aggregate AI ≈ 1.85**

**Note on vmap structure:** The nonlinear term is vmapped twice:
1. `jax.vmap(_nl_sp)` over species in `_compute_nonlinear_rhs` (solver.py:1070-1074)
2. `jax.vmap(_per_s)` over `ns` slices inside `nonlinear_term_iii` (solver.py:254)

The inner vmap over ns means each s-slice's FFTs are **independently batched**. On GPU, `jax.vmap` of `irfft2` becomes a batched cuFFT call across (ns, nvpar, nmu) — this is actually efficient if the batch size is large enough. However, the **moveaxis** at line 221 (`jnp.moveaxis(df, 2, 0)`) transposes the 5D array to bring `s` to the front, which may trigger a physical memory copy (cost: 55.8 MB).

**Mixed-precision path:**
The forward FFTs (irfft2) operate in FP32 via `.astype(complex64)`, saving ~50% bandwidth on the four gradient transforms. The inverse FFT (rfft2) operates in FP64 after upcasting. This asymmetry is intentional — the real-space bracket loses precision quadratically (product of two ~machine-epsilon quantities), but the spectral coefficients must remain FP64 for the time integrator's numerical stability.

### 4.5 `calculate_phi_kinetic` — Kinetic Phi Solve (integrals.py:209-225)

```python
def calculate_phi_kinetic(geometry, df, phi_weight=None, phi_diag=None):
    if phi_weight is None or phi_diag is None:
        phi_weight, phi_diag = precompute_phi_kinetic(geometry)
    phi_num = jnp.sum(phi_weight * df, axis=(0, 1, 2))  # sum over (nsp, nvpar, nmu)
    return -phi_num / phi_diag
```

**Memory access pattern:**
- `phi_weight` has shape `(nsp=2, 1, nmu=8, ns=16, nkx=85, nky=32)` — the `nvpar` axis is size 1 and will be **broadcast** to match `df`'s shape `(2, 32, 8, 16, 85, 32)` during the multiply. XLA handles this by either:
  1. Materializing the broadcast (copies phi_weight 32 times → 224 MB), or
  2. Using an implicit broadcast in the fused multiply-reduce kernel.

  Whether XLA fuses the broadcast-multiply-reduce depends on the reduction axes. Because the reduction is over axes (0, 1, 2) and the broadcast is on axis 1, XLA **should** be able to fuse this into a single kernel that streams phi_weight once and accumulates across nvpar in registers. But this is not guaranteed.

**FLOP count (per call):**

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| `phi_weight * df` (broadcast multiply) | 6.98M × 6 = 41.9M | Complex × complex (broadcast) |
| `jnp.sum(..., axis=(0,1,2))` | 6.98M × 2 = 14.0M | Complex addition reduction |
| `-phi_num / phi_diag` | 43.5K × 6 = 261K | Complex / real |
| **Total** | **~56M** | |

Note: The Bessel/gamma recomputation is avoided by the precompute path.

**Memory traffic (per call):**

| Data | Direction | Bytes |
|------|-----------|-------|
| `df` | Read | 111.7 MB |
| `phi_weight` | Read | 7.0 MB (if broadcast is fused) or 224 MB (if materialized) |
| `phi_diag` | Read | 348 KB |
| `phi_num` (output) | Write | 696 KB |
| **Total** | | **119 MB** (best) to **337 MB** (worst) |

**Arithmetic intensity: 56M / 119 MB ≈ 0.47 FLOP/byte (best) or 56M / 337 MB ≈ 0.17 (worst)**

**Per RK4 step (5 calls): 280M FLOPs, 0.6-1.7 GB R+W**

### 4.6 `pack_half_spectrum` / `unpack_half_spectrum` — FFT Packing (solver.py:190-200)

```python
def pack_half_spectrum(spec_kxky, jind, mrad, mphiw3):
    out_shape = spec_kxky.shape[:-2] + (mrad, mphiw3)
    out = jnp.zeros(out_shape, dtype=jnp.complex128)
    nky = spec_kxky.shape[-1]
    return out.at[..., jind, :nky].set(spec_kxky)   # scatter via jind

def unpack_half_spectrum(spec_half, jind, nky):
    return spec_half[..., jind, :nky]                 # gather via jind
```

**Memory access pattern:**
- `jind` maps `kx` indices to the FFT-compatible ordering: positive kx modes map to low indices, negative kx modes wrap to high indices. This is a **permutation scatter/gather** along the kx axis.
- `pack`: writes into a zero-initialized output at non-contiguous kx positions → **scatter write**
- `unpack`: reads from non-contiguous kx positions → **gather read**

**FLOP count:** Zero (pure memory movement).

**Memory traffic (per pack or unpack):**

| Data | Bytes |
|------|-------|
| Input spectrum read | 256 × 85 × 32 × 16B = 11.1 MB |
| Output array write | 256 × 135 × 49 × 16B = 27.1 MB (pack) or 11.1 MB (unpack) |
| Zero-init for pack | 27.1 MB |

**Per NL call per species:** 4 packs + 1 pack (rfft output) + 1 unpack = 6 pack/unpack ops per s-slice, but since the vmap batches these, effective overhead is absorbed into the FFT kernel pipeline.

### 4.7 `_fuse_stencils` — Precomputed Stencil Fusion (solver.py:407-455)

**Purpose:** Combines the upwind-selected stencil coefficients with the characteristic velocity and dissipation into a single fused coefficient array, so that the hot-path `_apply_parallel` only needs one coefficient array instead of performing the upwind selection per time step.

**Output shape:** `s_total_upar` = `(9, nsp, nvpar, nmu, ns, nkx, nky)` — 62.8M elements × 16B = 1005 MB.

**This is a one-time precomputation cost** (called once per `linear_precompute`). The trade-off is memory capacity (2 GB for s_total_upar + s_total_t7) vs. per-step compute savings.

**Analysis:** This is already an optimization — it eliminates the per-step sign-based stencil selection (`jnp.where(upar_sign > 0, ...)`) from the hot path. The resulting 2 GB memory footprint is significant (1% of B300 HBM) but acceptable given the reduction in per-step kernel complexity.

**Tier: Optimal** — The precomputed fusion is the correct approach. The memory cost is a worthwhile trade-off.

### 4.8 RK4 Accumulation (solver.py:1104-1109)

```python
k1 = _rhs(prev_df)
k2 = _rhs(prev_df + 0.5 * dt * k1)
k3 = _rhs(prev_df + 0.5 * dt * k2)
k4 = _rhs(prev_df + dt * k3)
next_df_raw = prev_df + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

**Each intermediate `prev_df + coeff * dt * k_i`:**
- Reads `prev_df` (111.7 MB) + `k_i` (111.7 MB)
- Writes intermediate df (111.7 MB)
- FLOPs: 6.98M × 4 = 27.9M (scalar × complex + complex add)

**Final accumulation `(dt/6) * (k1 + 2*k2 + 2*k3 + k4)`:**
- Reads k1, k2, k3, k4 (4 × 111.7 MB) + prev_df (111.7 MB)
- Writes next_df (111.7 MB)
- FLOPs: 6.98M × (3 adds + 4 scalar muls) × 2 = 97.7M

**Total RK4 overhead per step:**
- Memory: 3 × (2 reads + 1 write) × 111.7 MB + (5 reads + 1 write) × 111.7 MB = 1.67 GB
- FLOPs: 3 × 27.9M + 97.7M = 181M
- AI: 181M / 1.67 GB ≈ 0.11 FLOP/byte

**Note:** XLA *may* fuse `prev_df + 0.5 * dt * k2` into the beginning of the next `_rhs` call, avoiding the intermediate materialization. Whether this happens depends on whether `jax.lax.scan` and the RK4 structure allow XLA to see across the `_rhs` boundary. In practice, the `_rhs` call begins with `_compute_phi` which reads the full df → XLA is unlikely to fuse the accumulation with the phi reduction.

---

## 5. Bottleneck Identification & Roofline Placement

### 5.1 Per-Component Roofline Summary

| Component | AI (FLOP/byte) | B300 BW-limited throughput | B300 peak | Bound | % of step FLOPs |
|-----------|----------------|---------------------------|-----------|-------|----------------|
| `_apply_parallel` | 0.08 | 0.64 TFLOP/s | 40 TFLOP/s | **Mem BW** | 3.6% |
| `_apply_vpar` | 0.11 | 0.88 TFLOP/s | 40 TFLOP/s | **Mem BW** | 2.0% |
| `_linear_rhs_core` (aggregate) | 0.087 | 0.70 TFLOP/s | 40 TFLOP/s | **Mem BW** | 5.8% |
| `nonlinear_term_iii` | 1.85 | 14.8 TFLOP/s | 40 TFLOP/s | **Mem BW** | 89.5% |
| `calculate_phi_kinetic` | 0.17-0.47 | 1.4-3.8 TFLOP/s | 40 TFLOP/s | **Mem BW** | 0.3% |
| RK4 accumulations | 0.11 | 0.88 TFLOP/s | 40 TFLOP/s | **Mem BW** | 0.2% |
| **Aggregate** | **~1.67** | **13.4 TFLOP/s** | **40 TFLOP/s** | **Mem BW** | 100% |

Every component is memory-bandwidth bound. The nonlinear term (AI = 1.85) is closest to the ridge point but still firmly in the bandwidth-limited regime.

### 5.2 XLA Fusion Failures

#### 5.2.1 Python-Loop Stencils (Critical)

**Location:** `_apply_parallel` (solver.py:760-765), `_apply_vpar` (solver.py:771-775)

**Problem:** The `for i in range(9)` and `for c, s in zip(...)` Python loops are unrolled at JAX trace time. Each iteration produces independent XLA operations that **cannot be fused into a single kernel** because:
1. Each iteration reads and writes the `out` accumulator → sequential dependency
2. The gather patterns differ per iteration (different `s_map`/`kx_map`)
3. XLA's loop fusion cannot recognize these as a reduction over stencil points

**Impact:** 9 (or 5) separate kernel launches per stencil application, with 8 (or 4) intermediate materialization of the `out` buffer. For `_apply_parallel`, this wastes 894 MB of memory traffic per call on intermediate reads/writes.

**Estimated overhead:** ~50% of total linear RHS memory traffic is wasted on intermediate accumulation buffers.

#### 5.2.2 `moveaxis` Transposition in Nonlinear Term

**Location:** `nonlinear_term_iii` line 221: `jnp.moveaxis(df, 2, 0)`

**Problem:** Transposes `df` from `(nvpar, nmu, ns, nkx, nky)` to `(ns, nvpar, nmu, nkx, nky)` before the ns-vmap. This may produce a physical copy if XLA cannot represent the transpose as a view.

**Impact:** 55.8 MB per species per NL call × 8 calls/step = 446 MB of potentially unnecessary copies.

#### 5.2.3 dtype Casts in Mixed-Precision FFT Pipeline

**Location:** `_per_s` lines 235-250

**Problem:** The sequence `.astype(complex64)` → `irfft2` → multiply in FP32 → `.astype(float64)` → `rfft2` involves two precision casts that XLA must materialize as separate kernels or insert into the FFT plan. The FP32→FP64 upcast before `rfft2` is particularly expensive because it reads the entire real-space array and writes a double-sized output.

**Impact:** Additional 800 MB/species/call for the upcast, across 8 calls/step = 6.4 GB/step.

#### 5.2.4 Phi Solve Broadcast

**Location:** `calculate_phi_kinetic` line 224

**Problem:** `phi_weight * df` where phi_weight has shape `(2, 1, 8, 16, 85, 32)` and df has shape `(2, 32, 8, 16, 85, 32)`. The broadcast along axis 1 (nvpar) may be materialized as a full copy of phi_weight expanded to df's shape if XLA does not fuse the broadcast into the subsequent reduction.

### 5.3 Uncoalesced / Strided Memory Patterns

#### 5.3.1 3-Index Gather in `_apply_parallel` (Critical)

The access `field[:, :, s_map, kx_map, ky_idx]` gathers from 3 out of 5 trailing dimensions simultaneously. On GPU:
- Threads in a warp process adjacent (kx, ky) elements
- But `s_map[s, kx, ky]` and `kx_map[s, kx, ky]` may point to non-adjacent (s', kx') pairs
- This produces **random-access reads** in the (s, kx) plane → L2 cache miss rate is high

**Effective bandwidth:** ~30-50% of peak HBM bandwidth for scattered access patterns on A100/B300. This means the already-low AI of 0.08 FLOP/byte is further degraded — actual performance may be 0.04-0.05 FLOP/byte effective.

#### 5.3.2 `jind`-Based Scatter/Gather in `pack_half_spectrum`

The `jind` index permutes kx modes between the solver's spectral ordering and the FFT-compatible ordering. This is a **stride-1 permutation along one axis** — relatively benign compared to the 3-index gather, but still prevents streaming access.

### 5.4 Register Pressure from vmap

#### 5.4.1 Species vmap in `_compute_linear_rhs`

The `jax.vmap` over nsp=2 species with 15 input arrays creates a batch dimension. For nsp=2, XLA may:
1. Emit a single kernel with doubled work → register pressure doubles for all temporaries
2. Emit separate kernels per species → no register pressure issue but more launches

For nsp=2, option (1) is likely and benign. For larger nsp, this could become problematic.

#### 5.4.2 ns vmap in `nonlinear_term_iii`

The `jax.vmap(_per_s)` over ns=16 with 4 input arrays creates a batch of 16 independent FFT pipelines. XLA will batch the cuFFT calls → batch size of 256 × 16 = 4096 per FFT call, which is large enough for good GPU occupancy.

However, each s-slice maintains its own set of gradient spectra, real-space arrays, and bracket temporaries. For ns=16, the simultaneous live memory is:
- 4 gradient spectra × 16 × 256 × 85 × 32 × 8B = 2.85 GB
- 4 real-space arrays × 16 × 256 × 135 × 96 × 4B = 3.4 GB

This exceeds the B300 L2 cache (128 MB) by a factor of ~50, so there is no L2 reuse across s-slices. The vmap structure prevents **temporal reuse** — if the FFTs were executed sequentially per s, each s-slice's data would cycle through L2 with reuse for the bracket phase.

---

## 6. Optimization Roadmap

### 6.1 JAX Level — Pure JAX Refactoring

#### 6.1.1 Replace Python-Loop Stencils with Fused Operations (P0)

**Target:** `_apply_parallel` (solver.py:756-766)

**Current state:** 9 sequential kernel launches with intermediate accumulation.

**Proposed refactoring — `jnp.einsum` approach:**

The key insight is that `_apply_parallel` computes:
```
out[v, m, s, x, y] = Σ_i coeffs[i, v, m, s, x, y] × field[v, m, s_map[i,s,x,y], kx_map[i,s,x,y], y]
```

This can be restructured as:
1. Pre-gather all 9 shifted versions of `field` into a stacked array `shifted_stack[9, v, m, s, x, y]`
2. Compute `out = jnp.sum(coeffs * shifted_stack, axis=0)`

```python
def _apply_parallel_fused(field, coeffs, s_shift, kx_shift, valid_shift):
    nky = field.shape[-1]
    ky_idx = jnp.arange(nky, dtype=jnp.int32)
    # Stack all 9 shifted fields: (9, nv, nmu, ns, nkx, nky)
    shifted_stack = jnp.where(
        valid_shift[:, None, None, :, :, :],            # (9, 1, 1, ns, nkx, nky)
        field[:, :, s_shift, kx_shift, ky_idx],          # advanced indexing
        0.0,
    )
    return jnp.sum(coeffs * shifted_stack, axis=0)
```

**Why this helps:** Single kernel launch. The `jnp.sum(..., axis=0)` reduction fuses the 9-way accumulation into a single pass. XLA can emit one Gather + FusedMultiplyReduce kernel. Eliminates 8 intermediate `out` buffers (894 MB saved per call).

**Risk:** The `field[:, :, s_shift, kx_shift, ky_idx]` advanced indexing with stacked `s_shift` of shape `(9, ns, nkx, nky)` may produce a BatchGather HLO that is less efficient than 9 independent gathers if XLA cannot vectorize the batch dimension. **Profile before committing.**

**Alternative — `jax.lax.conv_general_dilated`:** Not directly applicable because the stencil is not a regular convolution — the `s_map` and `kx_map` encode magnetic shear boundary conditions that create a non-uniform connectivity pattern. A convolution would only work for the interior points.

#### 6.1.2 Replace Python-Loop Vpar Stencil (P0)

**Target:** `_apply_vpar` (solver.py:768-776)

**Proposed refactoring — 1D convolution:**

Since the vpar stencil is a regular 5-point central difference with fixed coefficients and simple boundary handling, it maps directly to a 1D convolution:

```python
def _apply_vpar_fused(field, coeffs):
    # coeffs: 5-element 1D array (from stencils.VPAR_D1 or VPAR_D4)
    # field: (nv, nmu, ns, nkx, nky) complex128
    # Reshape field to (batch, 1, nv) for conv1d, convolve, reshape back
    batch_shape = field.shape[1:]
    f_flat = field.reshape(field.shape[0], -1).T  # (batch, nv)
    f_flat = f_flat[:, None, :]                    # (batch, 1, nv)
    kernel = coeffs[::-1].reshape(1, 1, 5)         # (out_ch, in_ch, width)
    result = jax.lax.conv_general_dilated(
        f_flat, kernel, window_strides=(1,), padding=[(2, 2)],
        dimension_numbers=('NHC', 'OIH', 'NHC'),
    )
    # Zero boundary padding (Dirichlet-like)
    nv = field.shape[0]
    mask = jnp.ones(nv)
    mask = mask.at[:2].set(0.0).at[-2:].set(0.0)  # simplified; adjust for actual BC
    return (result.squeeze(1).T.reshape(field.shape)) * mask[:, None, None, None, None]
```

**Why this helps:** XLA recognizes `conv_general_dilated` and emits a single cuDNN convolution kernel. No intermediate buffers. Automatic boundary handling via padding mode.

**Caveat:** The current boundary handling uses `jnp.clip` (Neumann-like) followed by `valid` masking. The conv1d with zero-padding implements Dirichlet BCs. Verify that the boundary treatment is equivalent. For the interior points (indices 2 through nv-3), the results are identical.

**Alternative — `jnp.convolve` via `jax.lax.conv`:** Equivalent, but `conv_general_dilated` gives more control over batching.

#### 6.1.3 Eliminate Phi Solve Broadcast Materialization (P1)

**Target:** `calculate_phi_kinetic` (integrals.py:224)

**Current:** `phi_num = jnp.sum(phi_weight * df, axis=(0, 1, 2))` where phi_weight is `(2,1,8,16,85,32)` and df is `(2,32,8,16,85,32)`.

**Proposed refactoring — explicit einsum:**

```python
phi_num = jnp.einsum('sjsxy,svmsxy->sxy', phi_weight.squeeze(1), df)
# More precisely, using named axes:
phi_num = jnp.einsum('aibcde,ajbcde->bcde', phi_weight, df)
```

Wait — `phi_weight` has shape `(nsp, 1, nmu, ns, nkx, nky)` and we sum over `(nsp, nvpar, nmu)`. Since phi_weight is constant along nvpar (axis 1 is size 1), the einsum is:

```python
# Sum df over nvpar first (cheap, no broadcast needed), then multiply by weight and sum over (nsp, nmu)
df_vpar_summed = jnp.sum(df, axis=1, keepdims=True)  # (nsp, 1, nmu, ns, nkx, nky)
phi_num = jnp.sum(phi_weight * df_vpar_summed, axis=(0, 2))  # (ns, nkx, nky)
```

**Why this helps:** Summing df over nvpar (axis 1) first produces a `(2, 1, 8, 16, 85, 32)` intermediate — 7.0 MB instead of 111.7 MB. The subsequent `phi_weight * df_vpar_summed` is then a (2, 1, 8, 16, 85, 32) × (2, 1, 8, 16, 85, 32) multiply with no broadcast, followed by a sum over (nsp, nmu). Total traffic drops from 224 MB to 119 MB + 7 MB + 7 MB = 133 MB.

Actually even better: `jnp.einsum` can express the full contraction:
```python
phi_num = jnp.einsum('aimjkl,avmjkl->jkl', phi_weight, df)
```
which tells XLA to contract over (a=nsp, m=nmu) and the broadcast axis simultaneously. XLA can then emit an optimal reduce-scatter kernel.

#### 6.1.4 Fuse RK4 Stage Accumulations (P1)

**Target:** `gkstep_single` (solver.py:1104-1109)

**Current:** Each `prev_df + coeff * dt * k_i` materializes a full df-sized intermediate before passing to `_rhs`.

**Proposed — in-place accumulation with `jax.checkpoint`:**

The RK4 stages have a natural structure where the intermediate df is consumed immediately by the next `_rhs` call. Using `jax.checkpoint` (gradient checkpointing) is orthogonal, but we can restructure to minimize materializations:

```python
# Instead of materializing k1, k2, k3, k4 separately:
def _rk4_fused(df, dt, rhs_fn):
    k1 = rhs_fn(df)
    df2 = jax.lax.add(df, jax.lax.mul(0.5 * dt, k1))  # fuse scalar-mul + add
    k2 = rhs_fn(df2)
    df3 = jax.lax.add(df, jax.lax.mul(0.5 * dt, k2))
    k3 = rhs_fn(df3)
    df4 = jax.lax.add(df, jax.lax.mul(dt, k3))
    k4 = rhs_fn(df4)
    # Fused final accumulation:
    return jax.lax.add(df, jax.lax.mul(dt / 6.0,
        jax.lax.add(jax.lax.add(k1, jax.lax.mul(2.0, k2)),
                     jax.lax.add(jax.lax.mul(2.0, k3), k4))))
```

In practice, XLA should already fuse the scalar multiplies and additions. The main saving comes from ensuring that `k1` through `k4` are not all live simultaneously (they can be accumulated on-the-fly).

**More impactful alternative — use SSPRK3 (3-stage Runge-Kutta):**
This is an algorithmic change (see Section 6.3.3), but SSPRK3 replaces 4 RHS evaluations with 3, saving 25% of total compute with minimal accuracy loss for the CFL-limited regime.

### 6.2 Custom Kernel Level — Triton / CUDA

#### 6.2.1 Fused Parallel Stencil Kernel (P0)

**Motivation:** The 9-point parallel stencil is the single worst-performing kernel in the solver (AI = 0.08). The gather pattern with magnetic-shear index maps cannot be efficiently expressed in pure JAX.

**Kernel design for Triton:**

```
Kernel: fused_parallel_stencil_9pt
  Grid: (nkx_blocks, nky_blocks, nvpar*nmu_blocks)
  Block: BLOCK_S=16, BLOCK_KX=16, BLOCK_KY=32 (tunable)

  Shared memory layout:
    - coeffs_smem[9][BLOCK_S][BLOCK_KX][BLOCK_KY]  → 9×16×16×32×16B = 1.125 MB (too large)

  Revised design — stream coefficients from HBM:
    - smem_field[BLOCK_S+8][BLOCK_KX+8][BLOCK_KY]  → halo region for stencil
    - smem_coeffs[9]  → load 9 coefficient values per (s,kx,ky) point

  Algorithm:
    1. Load field block with halo into shared memory via TMA
    2. For each stencil point i=0..8:
       a. Compute source (s', kx') from s_map, kx_map (stored in constant memory)
       b. Read field value from smem (if in-block) or global (if in halo)
       c. Multiply by coeffs[i] (streamed from HBM, coalesced)
       d. Accumulate in registers
    3. Write output block to global memory
```

**Shared memory sizing for B300:**
- B300 has 228 KB shared memory per SM
- A single (s, kx, ky) output block of (16, 16, 32) needs:
  - Field halo: (16+8) × (16+8) × 32 × 16B = 295 KB → exceeds smem
  - Reduced block: (8, 8, 32) with halo (16, 16, 32) = 131 KB → fits

**TMA utilization:** B300's TMA can asynchronously fill shared memory tiles from HBM while the previous tile is being computed. Use double-buffering:
```
Pipeline:
  [TMA load tile N+1] || [compute stencil for tile N] || [write tile N-1]
```

**Expected performance:**
- Eliminates 8 intermediate buffers → ~50% memory traffic reduction
- Shared memory reuse of halo points → ~30% effective bandwidth increase
- Single kernel launch → ~10% reduction from launch overhead
- **Net: 3-5× speedup on `_apply_parallel`**

#### 6.2.2 Fused FFT-Bracket Kernel (P2)

**Motivation:** The nonlinear term's pipeline (pack → cast → irfft2 → bracket → cast → rfft2 → scale → unpack) involves multiple intermediate materializations. A fused kernel could execute the entire pipeline in a single launch.

**Design:**
This is a "callback-style" FFT kernel:
1. Load spectral data from HBM → shared memory
2. Execute irfft2 butterfly in shared memory / registers
3. Multiply bracket terms in registers (no global memory round-trip)
4. Execute rfft2 butterfly
5. Store result to HBM

**Challenge:** cuFFT operates on contiguous arrays in global memory and does not support callback fusion in JAX. A Triton implementation would require hand-coding the FFT butterfly for sizes (135, 96), which is non-trivial for non-power-of-2 sizes.

**Alternative — cuFFT with callbacks (CUDA):** NVIDIA's cuFFTDx (device FFT library) supports fused load/store callbacks. This requires a CUDA custom op, not Triton.

**Expected performance:** Eliminates 4 intermediate spectral + 4 real-space materializations per s-slice = ~100 MB per s-slice. For 16 s-slices × 2 species × 4 calls = 128 instances, saving ~12.8 GB per RK4 step. **Net: 2-3× speedup on nonlinear term.**

#### 6.2.3 TMA-Accelerated Phi Reduction (P2)

**Motivation:** The phi solve is a contraction from (2, 32, 8, 16, 85, 32) to (16, 85, 32) — a 512:1 reduction. This is a classic reduce kernel.

**Design for B300 TMA:**
```
Kernel: phi_kinetic_tma
  Grid: (nkx_blocks, nky_blocks, ns)
  Block: 256 threads

  Algorithm:
    1. TMA bulk load phi_weight[:, :, :, s, kx_block, ky_block]  → smem (small: 2×1×8×16×32×16B = 131 KB)
    2. For each nvpar chunk:
       a. TMA load df[:, nvpar_chunk, :, s, kx_block, ky_block]  → smem
       b. Multiply weight × df in registers
       c. Warp-level reduce across nvpar, nmu
    3. Block-level reduce across nsp
    4. Write phi[s, kx_block:, ky_block:] to global memory
```

**Expected performance:** Eliminates broadcast materialization entirely. Single-pass through df with register-level accumulation. **Net: 1.5-2× on phi solve.**

### 6.3 Algorithmic Level

#### 6.3.1 Full Mixed-Precision FFT Pipeline (P1)

**Current state:** The forward transforms (irfft2) use FP32 via `.astype(complex64)`, but the inverse (rfft2) is FP64. The upcast `nl_real.astype(jnp.float64)` before rfft2 is the bottleneck.

**Proposal:** Perform the entire rfft2 in FP32 and upcast only the final spectral coefficients:
```python
nl_half = jnp.fft.rfft2(nl_real, s=(mrad, mphi), axes=(-2,-1), norm="backward")
nl_half = nl_half.astype(jnp.complex128) * (fft_prefactor * fft_scale)
```

**Impact:** Saves 800 MB/species/call (the FP32→FP64 upcast of the real-space array). Over 8 calls/step → 6.4 GB saved.

**Risk:** The rfft2 in FP32 introduces ~1e-7 relative error in the spectral coefficients. For explicit time integration at typical CFL numbers, this is acceptable — the time integration error (~dt^5 for RK4) dominates. **Validate by comparing growth rates with FP64-only.**

#### 6.3.2 IMEX Time Integration for Kinetic Electrons (P2)

**Motivation:** The linear CFL condition for kinetic electrons is dominated by the electron parallel streaming: `dt_par = sgr_dist / max|v_the * ffun|`. Since `v_the / v_thi ~ sqrt(m_i/m_e) ≈ 43` for deuterium, the electron CFL is ~43× more restrictive than the ion CFL. This forces the entire solver to take tiny time steps even when the nonlinear CFL (ExB) allows much larger ones.

**Proposal:** Treat the electron parallel streaming term **implicitly** using an IMEX (Implicit-Explicit) scheme:
- Explicit: nonlinear ExB, drifts, drives (all terms except electron parallel streaming)
- Implicit: electron parallel streaming (Term I for electron species only)

The implicit solve for the parallel streaming term reduces to a **tridiagonal system** per (kx, ky, mu) point for each electron parallel velocity, since the parallel stencil is local in s with known connectivity. This is a classic Thomas algorithm — O(ns) per system, and there are nkx × nky × nmu = 21,760 independent systems.

**Impact:** Time step increases by a factor of ~43 (the electron-to-ion thermal velocity ratio). Even accounting for the implicit solve cost, the overall speedup is **5-20× for kinetic electron cases**.

**Complexity:** Moderate. Requires restructuring the RHS to separate electron parallel streaming from other terms, and implementing a batched tridiagonal solver (available in JAX via `jax.scipy.linalg.solve_banded` or a custom scan-based Thomas algorithm).

#### 6.3.3 Batched FFT Across (nvpar, nmu) for Occupancy (P1)

**Current state:** The vmap over ns produces batched FFTs with batch size = nvpar × nmu = 256 per s-slice. With the ns-vmap, the total batch size is 256 × 16 = 4096.

**Observation:** cuFFT is most efficient when the batch size is large and the transform size is moderate. The current transform size (135, 96) produces 12,960-point real FFTs, which map to ~2-3 SMs per transform on B300. With batch=4096, the 148 SMs of B300 are well-utilized.

**Potential improvement:** Restructure to eliminate the ns-vmap and instead batch across the full (nvpar, nmu, ns) = 4096 dimension simultaneously. This is already what the vmap achieves, but verifying that XLA indeed fuses the vmap into a single batched cuFFT call (rather than 16 separate calls) is important.

**Diagnostic:** Inspect the XLA HLO with `jax.make_jaxpr` or `jax.jit(f).lower(*args).compile().as_text()` to verify the FFT batching.

#### 6.3.4 SSPRK3 Time Integration (P2)

Replace the 4-stage RK4 with a 3-stage strong-stability-preserving RK3 (Shu-Osher form):
```
u^(1) = u^n + dt * L(u^n)
u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * L(u^(1)))
u^(n+1) = 1/3 u^n + 2/3 (u^(2) + dt * L(u^(2)))
```

**Impact:** 3 RHS evaluations instead of 4 → 25% reduction in total compute. The CFL number for SSPRK3 is ~1.0 vs ~2.78 for RK4, so the time step must be reduced by ~2.8×. Net: SSPRK3 requires ~2.1× more steps but each step is 25% cheaper → **overall ~1.6× slower**. This is only beneficial if combined with IMEX (where the CFL is set by the explicit nonlinear term, not the linear streaming).

**Recommendation:** Only adopt SSPRK3 if IMEX is also implemented.

### 6.4 Optimal — Already Near Roofline

#### 6.4.1 Elementwise Drift/Drive Terms

The drift term `-1j * kdotvd * df` and hyper-diffusion `hyper * df` are simple elementwise operations. XLA fuses these into the surrounding linear RHS kernel. AI is low (0.05-0.1) but these operations are bandwidth-limited by construction — there is no algorithmic way to increase the AI of a pointwise multiply.

**Status: Optimal.** These achieve near-peak bandwidth utilization when fused with adjacent operations.

#### 6.4.2 Precomputed Stencil Fusion (`_fuse_stencils`)

One-time cost. The decision to precompute `s_total_upar` and `s_total_t7` is correct — it eliminates per-step conditional logic from the hot path at the cost of 2 GB memory.

**Status: Optimal.**

#### 6.4.3 `build_jind` / FFT Size Selection

The FFT size selection (`extended_*dim_fft_size`) ensures sizes with small prime factors (≤7, preferring powers of 2) for optimal cuFFT performance. The `jind` index mapping is a necessary cost of the spectral ordering.

**Status: Optimal.**

---

## 7. Priority Matrix & Estimated Speedups

### 7.1 Priority Matrix

| ID | Optimization | Tier | Effort | Speedup (component) | Speedup (overall) | Priority |
|----|-------------|------|--------|---------------------|-------------------|----------|
| O1 | Fused `_apply_parallel` (einsum/stack) | JAX | Low | 2-3× | 1.1-1.3× | **P0** |
| O2 | Fused `_apply_vpar` (conv1d) | JAX | Low | 2-3× | 1.05-1.1× | **P0** |
| O3 | Phi solve reorder (sum-vpar-first) | JAX | Low | 1.5-2× | 1.01× | **P1** |
| O4 | RK4 accumulation fusion | JAX | Low | 1.2× | 1.02× | **P1** |
| O5 | Full FP32 rfft2 path | Algorithmic | Low | 1.3-1.5× NL | 1.2-1.4× | **P1** |
| O6 | Batched FFT verification | JAX | Low | 1.0-1.5× NL | 1.0-1.3× | **P1** |
| O7 | Fused parallel stencil (Triton) | Custom Kernel | High | 3-5× | 1.2-1.5× | **P0** |
| O8 | Fused FFT-bracket (cuFFTDx/Triton) | Custom Kernel | Very High | 2-3× NL | 1.5-2× | **P2** |
| O9 | TMA phi reduction | Custom Kernel | High | 1.5-2× | 1.01× | **P2** |
| O10 | IMEX for kinetic electrons | Algorithmic | High | 5-20× dt | 5-20× (kinetic) | **P2** |
| O11 | SSPRK3 (with IMEX only) | Algorithmic | Medium | 0.6× per step | Only with O10 | **P3** |

### 7.2 Estimated Aggregate Speedups by Implementation Phase

**Phase 1 — JAX-level quick wins (O1 + O2 + O3 + O4):**
- Eliminates intermediate buffers in stencils and phi solve
- Expected: **1.3-1.5× overall speedup**
- Effort: 1-2 days

**Phase 2 — Mixed precision + FFT verification (O5 + O6):**
- Reduces NL memory traffic by 30-50%
- Expected: **1.2-1.4× additional speedup (1.6-2.1× cumulative)**
- Effort: 1-2 days

**Phase 3 — Custom kernels (O7 + O8):**
- Fused stencil and FFT-bracket kernels
- Expected: **1.5-2.5× additional speedup (2.4-5.3× cumulative)**
- Effort: 2-4 weeks

**Phase 4 — IMEX (O10):**
- Eliminates electron CFL constraint
- Expected: **5-20× for kinetic electron cases (12-100× cumulative)**
- Effort: 2-4 weeks

### 7.3 Time-to-Solution Estimates (Kinetic Reference Case)

Assuming the reference case runs at 1 RK4 step/second on A100 (dominated by compilation and memory transfers on first step, ~0.05s/step thereafter):

| Configuration | Steps/s (A100) | Steps/s (B300, baseline) | Steps/s (B300, Phase 1-2) | Steps/s (B300, Phase 1-3) |
|---------------|----------------|--------------------------|---------------------------|---------------------------|
| Current | ~20 | ~80 (4× BW scaling) | ~130-170 | ~200-420 |
| With IMEX | ~20 (same dt) | ~80 | ~130-170 | ~200-420 at 5-20× larger dt |

**B300 baseline estimate:** The 4× bandwidth improvement (8 TB/s vs 2 TB/s) directly translates to ~4× speedup for bandwidth-bound kernels. The aggregate improvement is less because compilation overhead and kernel launch latency do not scale with bandwidth.

---

## Appendix A: Notation

| Symbol | Meaning |
|--------|---------|
| AI | Arithmetic Intensity (FLOP/byte) |
| BW | Memory Bandwidth |
| nsp | Number of species |
| nvpar | Number of parallel velocity grid points |
| nmu | Number of magnetic moment grid points |
| ns | Number of parallel position grid points |
| nkx | Number of radial Fourier modes |
| nky | Number of binormal Fourier modes |
| mrad | Dealiased radial FFT grid size |
| mphi | Dealiased binormal FFT real-space grid size |
| mphiw3 | Dealiased binormal FFT half-spectrum size |
| HBM | High Bandwidth Memory |
| TMA | Tensor Memory Accelerator (Blackwell) |
| IMEX | Implicit-Explicit time integration |
| CFL | Courant-Friedrichs-Lewy stability condition |
| SSPRK3 | Strong-Stability-Preserving Runge-Kutta 3rd order |

## Appendix B: Source File Reference

| File | Lines | Component | Section |
|------|-------|-----------|---------|
| `gyaradax/solver.py` | 756-766 | `_apply_parallel` | §4.1 |
| `gyaradax/solver.py` | 768-776 | `_apply_vpar` | §4.2 |
| `gyaradax/solver.py` | 742-804 | `_linear_rhs_core` | §4.3 |
| `gyaradax/solver.py` | 203-255 | `nonlinear_term_iii` / `_per_s` | §4.4 |
| `gyaradax/solver.py` | 190-200 | `pack_half_spectrum` / `unpack_half_spectrum` | §4.6 |
| `gyaradax/solver.py` | 407-455 | `_fuse_stencils` | §4.7 |
| `gyaradax/solver.py` | 1077-1137 | `gkstep_single` (RK4) | §4.8 |
| `gyaradax/solver.py` | 1141-1207 | `gksolve` (scan loop) | §3.3 |
| `gyaradax/solver.py` | 978-1060 | `_compute_phi`, `_compute_linear_rhs`, `_compute_nonlinear_rhs` | §3.3 |
| `gyaradax/integrals.py` | 168-206 | `precompute_phi_kinetic` | §4.5 |
| `gyaradax/integrals.py` | 209-225 | `calculate_phi_kinetic` | §4.5 |
| `gyaradax/stencils.py` | 1-77 | Stencil coefficient tables | §4.1, §4.2 |
