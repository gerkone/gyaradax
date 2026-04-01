# Fused Linear RHS CUDA Kernel — Implementation Specification

## 1. Motivation and Expected Impact

The current CUDA backend for the linear RK4 step runs at **48.2 ms** — 43% slower than the pure JAX backend (33.8 ms) — despite the individual stencil kernels being 1.5–3.3× faster than JAX's equivalents. The cause: each CUDA FFI call creates an opaque fusion barrier, forcing XLA to materialize all intermediate 5D arrays (~178 MB each) to global memory between calls.

One linear RHS evaluation currently requires:
- 1 elementwise kernel for `gyro_phi = bessel * phi_b`
- 1 FFI launch for `_apply_parallel_dual` → writes `term_par`, `term_vii`
- 1 FFI launch for `_apply_vpar_dual` → writes `out_d1`, `out_d4`
- 1 elementwise kernel for the 8-term accumulation

That is 4 global-memory round-trips of the full 5D field per RHS call, 16 over 4 RK4 stages ≈ 14 ms of pure materialization overhead.

**The fix**: a single CUDA kernel that computes the entire `_linear_rhs_core` — parallel stencil, vpar stencil, and all elementwise terms — writing one output array. All intermediates live in registers.

**Expected performance**: ~2 ms per RHS call → ~10–12 ms for the full linear RK4 step (3× over JAX, 4× over current CUDA).

---

## 2. Mathematical Specification

The kernel computes, for each grid point `(v, mu, s, kx, ky)`:

```
gyro_phi = bessel[v,mu,s,kx,ky] * phi[s,kx,ky]

term_par  = parallel_stencil(df,       s_total_upar)   // 9-point (s,kx)-stencil
term_vii  = parallel_stencil(gyro_phi, s_total_t7)     // 9-point (s,kx)-stencil

out_d1 = vpar_stencil_D1(df)   // 5-point stencil along v-dimension
out_d4 = vpar_stencil_D4(df)   // 5-point stencil along v-dimension

term_iv     = utrap * out_d1 / dvp
term_vp_diss = disp_vp * abs_dum2_vp * out_d4 / dvp

kdotvd = drift_x * kx_b + drift_y * ky_b

rhs = term_par
    + term_iv
    + term_vp_diss
    - 1j * kdotvd * df
    + hyper * df
    + 1j * drive_scale * (dmaxwel_fm_ek - signz0 * kdotvd * (fmaxwl / tmp0)) * gyro_phi
    + term_vii
```

All arithmetic is complex128 (double2) for `df`, `gyro_phi`, and the output; real64 (double) for all coefficients.

---

## 3. Data Layout and Dimensions

Field shape: `(nv, nmu, ns, nkx, nky)` — contiguous in row-major (C) order, so `nky` is the fastest dimension.

Concrete test case: `(32, 8, 16, 85, 32)` → `nv_nmu = 256`, spatial = `16 × 85 × 32 = 43,520`.

The kernel's internal indexing uses a "flattened velocity" index: `v_idx = v * nmu + mu`, ranging over `[0, nv_nmu)`.

### Thread/Block Structure

Identical to the existing parallel stencil kernel:
- **One block per `(v_idx, kx)` pair**: `num_blocks = nv_nmu * nkx`
- **One thread per `(s, ky)` element**: `threads_per_block = ns * nky`
- Thread mapping: `s = threadIdx.x / nky`, `ky = threadIdx.x % nky`

For `(ns=16, nky=32)`: 512 threads/block, 21,760 blocks.

---

## 4. Kernel Inputs

### Buffers (passed as `xla_ffi::Buffer` args)

| # | Name | Type | Shape | Description |
|---|------|------|-------|-------------|
| 1 | `df` | C128 | `(nv, nmu, ns, nkx, nky)` | Distribution function |
| 2 | `phi` | C128 | `(ns, nkx, nky)` | Electrostatic potential (3D) |
| 3 | `bessel` | F64 | `(nv, nmu, ns, nkx, nky)` | Bessel J0 coefficients |
| 4 | `s_total_upar` | F64 | `(9, nv_nmu, ns, nkx, nky)` | Fused parallel streaming + dissipation stencil weights |
| 5 | `s_total_t7` | F64 | `(9, nv_nmu, ns, nkx, nky)` | Fused parallel field-drive stencil weights |
| 6 | `packed_maps` | S32 | `(9, ns, nkx, nky, 2)` | Parallel stencil connectivity: `.x = src_s`, `.y = src_kx` (-1 = inactive) |
| 7 | `utrap` | F64 | `(nv, nmu, ns, nkx, nky)` | Trapping velocity |
| 8 | `abs_dum2_vp` | F64 | `(nv, nmu, ns, nkx, nky)` | Absolute vpar dissipation speed |
| 9 | `drift_x` | F64 | `(nv, nmu, ns, nkx, nky)` | Drift velocity x-component |
| 10 | `drift_y` | F64 | `(nv, nmu, ns, nkx, nky)` | Drift velocity y-component |
| 11 | `dmaxwel_fm_ek` | F64 | `(nv, nmu, ns, nkx, nky)` | Equilibrium drive coefficient |
| 12 | `fmaxwl` | F64 | `(nv, nmu, ns, nkx, nky)` | Maxwellian distribution |
| 13 | `hyper` | F64 | `(ns, nkx, nky)` | Hyper-diffusion coefficient (3D, broadcast over v,mu) |
| 14 | `kx_vals` | F64 | `(nkx,)` | kx grid values |
| 15 | `ky_vals` | F64 | `(nky,)` | ky grid values |

### Scalar Attributes

| Name | Type | Description |
|------|------|-------------|
| `nv` | int32 | Velocity grid points |
| `nmu` | int32 | Magnetic moment grid points |
| `ns` | int32 | Parallel grid points |
| `nkx` | int32 | Radial wavenumber grid points |
| `nky` | int32 | Toroidal wavenumber grid points |
| `dvp` | float64 | Velocity grid spacing |
| `disp_vp` | float64 | Velocity-space dissipation coefficient |
| `drive_scale` | float64 | Equilibrium drive scaling |
| `signz0` | float64 | Species charge sign |
| `tmp0` | float64 | Species temperature |

### Output Buffer

| # | Name | Type | Shape | Description |
|---|------|------|-------|-------------|
| 1 | `rhs_out` | C128 | `(nv, nmu, ns, nkx, nky)` | Complete linear RHS |

**Important note on `signz0` and `tmp0`**: In the current Python code, these are scalar (adiabatic case). For the kinetic multi-species case, the outer `jax.vmap` slices per species, so they remain scalar from the kernel's perspective. If they become per-`(v,mu,s,kx,ky)` arrays in a future refactor, they should be promoted to buffer args instead of scalar attrs.

---

## 5. Kernel Algorithm (Pseudocode)

```
// ── Block/thread identity ──
v_idx = blockIdx.x / nkx          // flattened velocity index (v*nmu + mu)
kx    = blockIdx.x % nkx
s     = threadIdx.x / NKY
ky    = threadIdx.x % NKY

v     = v_idx / nmu               // needed for vpar stencil
mu    = v_idx % nmu               // (integer division)

// ── Index helpers ──
spatial_stride = ns * nkx * nky
spatial_idx    = s * (nkx * nky) + kx * nky + ky
field_idx      = v_idx * spatial_stride + spatial_idx
                 // = (v * nmu + mu) * ns * nkx * nky + s * nkx * nky + kx * nky + ky

// ── Step 1: Load df and compute gyro_phi into shared memory ──
double2 my_df = df[field_idx]
smem_df[threadIdx.x] = my_df

// phi is 3D: index = s * nkx * nky + kx * nky + ky
double phi_val_r = phi[spatial_idx].x    // phi is complex, but bessel is real
double phi_val_i = phi[spatial_idx].y
double bes = bessel[field_idx]
double2 my_gyro_phi = make_double2(bes * phi_val_r, bes * phi_val_i)
smem_gyro[threadIdx.x] = my_gyro_phi

__syncthreads()

// ── Step 2: Parallel stencil (9-point in s,kx) ──
// Exactly as existing apply_parallel_dual_kernel
double2 acc_par  = {0, 0}   // term_par  accumulator
double2 acc_t7   = {0, 0}   // term_vii  accumulator

for i in 0..8:
    int2 map_val = packed_maps[i * spatial_stride + spatial_idx]
    src_s  = map_val.x
    src_kx = map_val.y
    if src_s >= 0:
        c_upar = s_total_upar[i * nv_nmu * spatial_stride + v_idx * spatial_stride + spatial_idx]
        c_t7   = s_total_t7  [i * nv_nmu * spatial_stride + v_idx * spatial_stride + spatial_idx]
        
        if src_kx == kx:
            val_df   = smem_df[src_s * NKY + ky]
            val_gyro = smem_gyro[src_s * NKY + ky]
        else:
            src_field_idx = v_idx * spatial_stride + src_s * (nkx * nky) + src_kx * nky + ky
            val_df   = df[src_field_idx]         // global read
            val_gyro_r = bes_at_src * phi_at_src_r  // see note below
            val_gyro_i = bes_at_src * phi_at_src_i

        acc_par.x += val_df.x * c_upar
        acc_par.y += val_df.y * c_upar
        acc_t7.x  += val_gyro.x * c_t7    // NOTE: gyro_phi at (v_idx, src_s, src_kx, ky)
        acc_t7.y  += val_gyro.y * c_t7

// ── CRITICAL NOTE on cross-kx gyro_phi reads ──
// For cross-kx neighbors, we need gyro_phi = bessel[v_idx, src_s, src_kx, ky] * phi[src_s, src_kx, ky]
// Option A: Read bessel and phi separately, multiply in register (2 global reads, 1 FMA)
// Option B: Pre-materialize gyro_phi as a full 5D array (but this defeats the purpose)
// CHOOSE Option A: 2 extra reads are cheaper than a full materialization

// ── Step 3: Vpar stencil (5-point along v-dimension) ──
// D1 coefficients: [-1/12, 8/12, 0, -8/12, 1/12]
// D4 coefficients: [1, -4, 6, -4, 1]
// Stencil reaches v ± 2 (i.e., v_idx ± nmu for same mu, or v_idx ± 2*nmu)
// Wait — the vpar stencil is along the v-axis with mu fixed.
// In the flattened (v_idx) = v*nmu + mu layout:
//   v-1 at same mu → v_idx - nmu
//   v+1 at same mu → v_idx + nmu
//   v-2 at same mu → v_idx - 2*nmu
//   v+2 at same mu → v_idx + 2*nmu

double2 df_vm2, df_vm1, df_v0, df_vp1, df_vp2
df_v0 = my_df   // already in register

// Boundary: clamp to zero for v outside [0, nv-1]
// v_idx = v * nmu + mu, so v = v_idx / nmu
// v-2 valid if v >= 2, i.e. v_idx >= 2*nmu
// v-1 valid if v >= 1, i.e. v_idx >= nmu
// v+1 valid if v <= nv-2, i.e. v_idx < (nv-1)*nmu
// v+2 valid if v <= nv-3, i.e. v_idx < (nv-2)*nmu

size_t vpar_stride = nmu * spatial_stride   // stride to move ±1 in v

df_vm2 = (v >= 2)    ? __ldg(&df[field_idx - 2*vpar_stride]) : make_double2(0,0)
df_vm1 = (v >= 1)    ? __ldg(&df[field_idx - 1*vpar_stride]) : make_double2(0,0)
df_vp1 = (v <= nv-2) ? __ldg(&df[field_idx + 1*vpar_stride]) : make_double2(0,0)
df_vp2 = (v <= nv-3) ? __ldg(&df[field_idx + 2*vpar_stride]) : make_double2(0,0)

// D1: central difference (coefficients from stencils.VPAR_D1)
// VPAR_D1 = [1/12, -8/12, 0, 8/12, -1/12]  (or similar — VERIFY from stencils.py)
double2 out_d1 = c_d1_0 * df_vm2 + c_d1_1 * df_vm1 + c_d1_2 * df_v0 
               + c_d1_3 * df_vp1 + c_d1_4 * df_vp2

// D4: 4th-order dissipation
// VPAR_D4 = [1, -4, 6, -4, 1]  (or similar — VERIFY from stencils.py)
double2 out_d4 = c_d4_0 * df_vm2 + c_d4_1 * df_vm1 + c_d4_2 * df_v0 
               + c_d4_3 * df_vp1 + c_d4_4 * df_vp2

// ── Step 4: Elementwise accumulation ──
double utrap_val    = utrap[field_idx]
double abs_vp_val   = abs_dum2_vp[field_idx]
double drift_x_val  = drift_x[field_idx]
double drift_y_val  = drift_y[field_idx]
double dmaxwel_val  = dmaxwel_fm_ek[field_idx]
double fmaxwl_val   = fmaxwl[field_idx]
double kx_val       = kx_vals[kx]    // 1D lookup
double ky_val       = ky_vals[ky]    // 1D lookup

// hyper is 3D (ns, nkx, nky) — index by spatial_idx
double hyper_val    = hyper[spatial_idx]

double kdotvd       = drift_x_val * kx_val + drift_y_val * ky_val
double inv_dvp      = 1.0 / dvp
double inv_tmp      = 1.0 / max(tmp0, 1e-15)

// term_iv = utrap * out_d1 / dvp
double2 term_iv = make_double2(utrap_val * out_d1.x * inv_dvp,
                               utrap_val * out_d1.y * inv_dvp)

// term_vp_diss = disp_vp * abs_vp * out_d4 / dvp
double vp_diss_coeff = disp_vp * abs_vp_val * inv_dvp
double2 term_vp_diss = make_double2(vp_diss_coeff * out_d4.x,
                                     vp_diss_coeff * out_d4.y)

// -1j * kdotvd * df  →  (kdotvd * df.y, -kdotvd * df.x)
double2 drift_term = make_double2(kdotvd * my_df.y, -kdotvd * my_df.x)

// hyper * df
double2 hyper_term = make_double2(hyper_val * my_df.x, hyper_val * my_df.y)

// Drive: 1j * drive_scale * (dmaxwel - signz0 * kdotvd * fmaxwl / tmp0) * gyro_phi
double drive_coeff = drive_scale * (dmaxwel_val - signz0 * kdotvd * fmaxwl_val * inv_tmp)
// 1j * drive_coeff * gyro_phi = (-drive_coeff * gyro_phi.y, drive_coeff * gyro_phi.x)
double2 drive_term = make_double2(-drive_coeff * my_gyro_phi.y,
                                    drive_coeff * my_gyro_phi.x)

// Final accumulation
double2 result
result.x = acc_par.x + acc_t7.x + term_iv.x + term_vp_diss.x + drift_term.x + hyper_term.x + drive_term.x
result.y = acc_par.y + acc_t7.y + term_iv.y + term_vp_diss.y + drift_term.y + hyper_term.y + drive_term.y

rhs_out[field_idx] = result
```

---

## 6. Critical Implementation Details

### 6.1. Cross-kx `gyro_phi` Reads

The parallel stencil for `term_vii` needs `gyro_phi[v_idx, src_s, src_kx, ky]` at scattered locations. Since `gyro_phi = bessel * phi`, and we don't have `gyro_phi` materialized for cross-kx neighbors, we must compute it on-the-fly:

```cuda
// For cross-kx neighbor in term_vii stencil:
size_t src_field_idx = v_idx * spatial_stride + src_s * (nkx * nky) + src_kx * nky + ky;
size_t src_spatial_idx = src_s * (nkx * nky) + src_kx * nky + ky;

double bes_src = __ldg(&bessel[src_field_idx]);
double2 phi_src = __ldg(&phi[src_spatial_idx]);   // phi is 3D
double2 gyro_src = make_double2(bes_src * phi_src.x, bes_src * phi_src.y);
```

This adds 2 global reads per cross-kx neighbor for the `term_vii` path (bessel + phi) compared to 1 read in the pre-materialized approach. However:
- `phi` is only `(ns, nkx, nky)` = 0.7 MB — it stays in L2 cache
- `bessel` at scattered `(src_s, src_kx)` has the same access pattern as `df` — no additional cache pressure
- This avoids materializing a 178 MB `gyro_phi` array

### 6.2. Vpar Stencil Memory Access Pattern

The vpar stencil reads `df` at `v ± 1` and `v ± 2` (with mu, s, kx, ky fixed). In memory:
- `df[v±1, mu, s, kx, ky]` is at offset `±nmu * ns * nkx * nky * sizeof(double2)` from the current element
- For `nmu=8, ns=16, nkx=85, nky=32`: stride = `8 * 43520 * 16B = 5.6 MB`

These are large strides, but:
- All 512 threads in a block read from the same `v±delta` offset (they differ only in `s, ky`)
- Adjacent ky-threads read adjacent memory → **coalesced within each v-neighbor read**
- 4 extra coalesced global reads per thread (vm2, vm1, vp1, vp2) is a modest cost

### 6.3. Shared Memory Usage

```
smem_df:   NS * NKY * sizeof(double2)  =  512 * 16 = 8,192 bytes
smem_gyro: NS * NKY * sizeof(double2)  =  512 * 16 = 8,192 bytes
Total:     16,384 bytes = 16 KB
```

This is identical to the current dual parallel kernel. No additional shared memory is needed for the vpar or elementwise portions.

### 6.4. Vpar Stencil Coefficients

The vpar D1 and D4 stencil coefficients are **constant** (same for all grid points). Pass them as scalar attributes to the kernel, not as buffer arrays. There are 10 scalar doubles total (5 for D1, 5 for D4).

**CRITICAL**: Verify the exact coefficient values from `stencils.py`. The expected values are:
```python
VPAR_D1 = [1/12, -8/12, 0, 8/12, -1/12]    # 4th-order central difference
VPAR_D4 = [1, -4, 6, -4, 1]                  # 4th-order dissipation
```
But the code may use a different convention (e.g., the sign or ordering). Check `stencils.VPAR_D1` and `stencils.VPAR_D4` and confirm the index mapping: does index 0 correspond to `v-2` or `v+2`?

### 6.5. Boundary Conditions for Vpar Stencil

At the edges of the velocity grid (`v = 0, 1` or `v = nv-2, nv-1`), the stencil reaches outside the domain. The existing JAX implementation uses zero-padding (out-of-bounds values are 0). Replicate this in CUDA:

```cuda
const int v = v_idx / nmu;
df_vm2 = (v >= 2)      ? __ldg(&df[field_idx - 2*vpar_stride]) : make_double2(0.0, 0.0);
df_vm1 = (v >= 1)      ? __ldg(&df[field_idx - 1*vpar_stride]) : make_double2(0.0, 0.0);
df_vp1 = (v <= nv - 2) ? __ldg(&df[field_idx + 1*vpar_stride]) : make_double2(0.0, 0.0);
df_vp2 = (v <= nv - 3) ? __ldg(&df[field_idx + 2*vpar_stride]) : make_double2(0.0, 0.0);
```

Where `vpar_stride = (size_t)nmu * spatial_stride` (stride of one `v`-step in the flattened array).

### 6.6. Coefficient Indexing for `s_total_upar` and `s_total_t7`

These have shape `(9, nv_nmu, ns, nkx, nky)`. The existing kernel uses:

```cuda
const size_t c_i_stride = (size_t)nv_nmu * spatial_stride;     // stride between stencil indices
const size_t c_idx_base = (size_t)v_idx * spatial_stride + spatial_idx;  // within one stencil index

coeffs[i * c_i_stride + c_idx_base]
```

**IMPORTANT**: The existing `apply_parallel_kernel` has a subtlety in the dynamic kernel path where `c_idx_base` uses `v_idx / nmu` instead of `v_idx` (for coefficient broadcasting across mu). The templated kernel uses `v_idx` directly. Verify which convention applies in the fused kernel. Looking at the existing code:

- Templated kernel: `c_idx_base = v_idx * spatial_stride + spatial_idx` and `c_i_stride = nv_total * spatial_stride` where `nv_total = nv_nmu`. This indexes into coeffs shape `(9, nv_nmu, ...)`.
- Dynamic kernel: `c_idx_base = (v_idx / nmu) * spatial_stride + spatial_idx` and `c_i_stride = nv_raw * spatial_stride` where `nv_raw = nv_nmu / nmu`. This indexes into coeffs shape `(9, nv, ...)`.

The Python `CUDAOps._apply_parallel` always broadcasts coefficients to shape `(9, nv_nmu, ns, nkx, nky)` before calling FFI. So for the fused kernel, use the templated convention: `v_idx` directly, `nv_total = nv_nmu`.

---

## 7. CUDA Kernel Signature

```cuda
template <int NS, int NKY>
__global__ __launch_bounds__(NS * NKY)
void linear_rhs_fused_kernel(
    // Fields (complex128 = double2)
    const double2* __restrict__ df,          // (nv_nmu, ns, nkx, nky)
    const double2* __restrict__ phi,         // (ns, nkx, nky)
    // Parallel stencil data
    const double*  __restrict__ bessel,      // (nv_nmu, ns, nkx, nky)
    const double*  __restrict__ s_total_upar,// (9, nv_nmu, ns, nkx, nky)
    const double*  __restrict__ s_total_t7,  // (9, nv_nmu, ns, nkx, nky)
    const int2*    __restrict__ packed_maps,  // (9, ns, nkx, nky)
    // 5D coefficient arrays (all real64)
    const double*  __restrict__ utrap,       // (nv_nmu, ns, nkx, nky)
    const double*  __restrict__ abs_dum2_vp, // (nv_nmu, ns, nkx, nky)
    const double*  __restrict__ drift_x,     // (nv_nmu, ns, nkx, nky)
    const double*  __restrict__ drift_y,     // (nv_nmu, ns, nkx, nky)
    const double*  __restrict__ dmaxwel_fm_ek,//(nv_nmu, ns, nkx, nky)
    const double*  __restrict__ fmaxwl,      // (nv_nmu, ns, nkx, nky)
    // Low-dimensional arrays
    const double*  __restrict__ hyper,       // (ns, nkx, nky) — 3D only
    const double*  __restrict__ kx_vals,     // (nkx,)
    const double*  __restrict__ ky_vals,     // (nky,)
    // Output
    double2*       __restrict__ rhs_out,     // (nv_nmu, ns, nkx, nky)
    // Scalar params
    int nv, int nmu, int nkx, int nky_param, int nv_nmu,
    // Vpar stencil coefficients (constant)
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    // Physics scalars
    double dvp, double disp_vp, double drive_scale, double signz0, double tmp0
);
```

Launch configuration:
```cuda
num_blocks = nv_nmu * nkx;
threads_per_block = NS * NKY;
shared_mem = 2 * NS * NKY * sizeof(double2);  // smem_df + smem_gyro

linear_rhs_fused_kernel<NS, NKY><<<num_blocks, threads_per_block, shared_mem, stream>>>(...);
```

---

## 8. FFI Binding

```cuda
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    linear_rhs_fused_ffi, LinearRhsFusedImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // bessel
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // s_total_upar
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // s_total_t7
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()   // packed_maps
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // utrap
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // abs_dum2_vp
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // drift_x
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // drift_y
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // dmaxwel_fm_ek
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // fmaxwl
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // hyper
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // kx_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // ky_vals
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // rhs_out
        .Attr<int32_t>("nv")
        .Attr<int32_t>("nmu")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("nv_nmu")
        .Attr<double>("c_d1_0").Attr<double>("c_d1_1").Attr<double>("c_d1_2")
        .Attr<double>("c_d1_3").Attr<double>("c_d1_4")
        .Attr<double>("c_d4_0").Attr<double>("c_d4_1").Attr<double>("c_d4_2")
        .Attr<double>("c_d4_3").Attr<double>("c_d4_4")
        .Attr<double>("dvp")
        .Attr<double>("disp_vp")
        .Attr<double>("drive_scale")
        .Attr<double>("signz0")
        .Attr<double>("tmp0")
);
```

---

## 9. Python-Side Integration

### 9.1. New Method on `CUDAOps`

Add to `gyaradax/backends/cuda_ops.py`:

```python
def _linear_rhs_fused(
    self,
    df: jnp.ndarray,
    phi: jnp.ndarray,          # 3D: (ns, nkx, nky)
    pre: GKPre,
    params_dvp: float,
    params_disp_vp: float,
    params_drive_scale: float,
) -> jnp.ndarray:
    """Fused linear RHS: parallel stencil + vpar stencil + elementwise, single kernel."""
    nv, nmu, ns, nkx, nky = df.shape
    nv_nmu = nv * nmu

    # Prepare coefficient arrays — broadcast to (9, nv_nmu, ns, nkx, nky) for stencil coeffs
    s_total_upar = self._prepare_stencil_coeffs(pre["s_total_upar"], nv, nmu, ns, nkx, nky)
    s_total_t7   = self._prepare_stencil_coeffs(pre["s_total_t7"], nv, nmu, ns, nkx, nky)
    
    # Packed maps (reuse existing logic)
    valid_jax = jnp.array(pre["valid_shift"])
    s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
    kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
    packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1)

    # 5D arrays — ensure contiguous and correct shape
    bessel = jnp.broadcast_to(pre["bessel"], df.shape)
    utrap = jnp.broadcast_to(pre["utrap"], df.shape)
    abs_vp = jnp.broadcast_to(pre["abs_dum2_vp"], df.shape)
    drift_x = jnp.broadcast_to(pre["drift_x"], df.shape)
    drift_y = jnp.broadcast_to(pre["drift_y"], df.shape)
    dmaxwel = jnp.broadcast_to(pre["dmaxwel_fm_ek"], df.shape)
    fmaxwl = jnp.broadcast_to(pre["fmaxwl"], df.shape)
    
    # 3D hyper array
    hyper = pre["hyper"].reshape(ns, nkx, nky)  # squeeze leading dims if needed
    # ... may need to handle broadcasting from (1,1,ns,nkx,nky) to (ns,nkx,nky)
    
    # 1D grids
    kx_vals = pre["kx_b"].reshape(-1)[:nkx]  # extract 1D from broadcast shape
    ky_vals = pre["ky_b"].reshape(-1)[:nky]
    
    # Vpar stencil coefficients — read from stencils module
    from gyaradax import stencils
    d1 = stencils.VPAR_D1  # verify: should be length-5 array
    d4 = stencils.VPAR_D4
    
    return ffi.ffi_call(
        "linear_rhs_fused_ffi",
        [jax.ShapeDtypeStruct(df.shape, df.dtype)]
    )(
        df, phi,
        bessel, s_total_upar, s_total_t7, packed_maps,
        utrap, abs_vp, drift_x, drift_y, dmaxwel, fmaxwl,
        hyper, kx_vals, ky_vals,
        nv=np.int32(nv), nmu=np.int32(nmu), nkx=np.int32(nkx),
        nky=np.int32(nky), nv_nmu=np.int32(nv_nmu),
        c_d1_0=float(d1[0]), c_d1_1=float(d1[1]), c_d1_2=float(d1[2]),
        c_d1_3=float(d1[3]), c_d1_4=float(d1[4]),
        c_d4_0=float(d4[0]), c_d4_1=float(d4[1]), c_d4_2=float(d4[2]),
        c_d4_3=float(d4[3]), c_d4_4=float(d4[4]),
        dvp=float(params_dvp),
        disp_vp=float(params_disp_vp),
        drive_scale=float(params_drive_scale),
        signz0=float(pre["signz0"]),
        tmp0=float(pre["tmp0"]),
    )[0]
```

### 9.2. Integration into the Solver

In `solver.py`, modify `_linear_rhs_core` to dispatch to the fused kernel when the backend supports it:

```python
def _linear_rhs_core(df, phi_b, pre, params_dvp, params_disp_vp, params_drive_scale, ops):
    # If ops supports fused RHS, use it
    if hasattr(ops, '_linear_rhs_fused'):
        phi_3d = phi_b.reshape(phi_b.shape[-3], phi_b.shape[-2], phi_b.shape[-1])
        return ops._linear_rhs_fused(df, phi_3d, pre, params_dvp, params_disp_vp, params_drive_scale)
    
    # Fallback: existing decomposed implementation
    gyro_phi = pre["bessel"] * phi_b
    term_par, term_vii = ops._apply_parallel_dual(df, gyro_phi, pre["s_total_upar"], pre["s_total_t7"])
    # ... rest unchanged
```

---

## 10. Compilation

Add to the `Makefile` / `CMakeLists.txt` alongside existing kernels:

```makefile
# Add to existing CUDA compilation
KERNELS += linear_rhs_fused.cu

# Same flags as existing kernels
NVCC_FLAGS = -O3 --use_fast_math -arch=sm_80 -Xcompiler -fPIC
```

Register the new FFI symbol in `cuda_ops.py`'s `_register_ffi()`:

```python
targets = {
    # ... existing targets ...
    "linear_rhs_fused_ffi": _lib.linear_rhs_fused_ffi,
}
```

---

## 11. Testing Strategy

### 11.1. Correctness Test

Compare against the JAX reference (`JAXOps._linear_rhs_core` equivalent):

```python
# Reference: decomposed JAX path
rhs_ref = _linear_rhs_core(df, phi_b, pre, dvp, disp_vp, drive_scale, ops_jax)

# Test: fused CUDA kernel
rhs_fused = ops_cuda._linear_rhs_fused(df, phi_3d, pre, dvp, disp_vp, drive_scale)

err = float(jnp.linalg.norm(rhs_ref - rhs_fused) / jnp.linalg.norm(rhs_ref))
assert err < 1e-13, f"Fused RHS error: {err:.2e}"
```

### 11.2. Component Isolation Tests

Test each sub-computation in isolation to localize any errors:

1. **Parallel stencil only**: Verify `term_par` and `term_vii` match the existing `_apply_parallel_dual` output. To do this, create a temporary debug version of the kernel that writes intermediate accumulators.

2. **Vpar stencil only**: Compare against `ops_jax._apply_vpar_dual(df, VPAR_D1, VPAR_D4)`.

3. **Cross-kx gyro_phi**: Verify that on-the-fly `bessel * phi` at scattered locations matches pre-materialized `gyro_phi` at those locations.

### 11.3. Benchmark

Adapt the existing `bench_rk4_step.py` to add a `--backend cuda_fused` option, or compare within the existing benchmark:

```python
# V0: JAX reference
# V1: CUDA with decomposed kernels  
# V2: CUDA with fused RHS kernel
```

Expected results:
- V2 linear RK4: ~10–12 ms (vs V0 = 33.8 ms, V1 = 48.2 ms)
- Numerical error: < 1e-13 relative L2 norm

---

## 12. Implementation Checklist

Ordered by dependency:

- [ ] **Step 1**: Verify vpar stencil coefficients. Print `stencils.VPAR_D1` and `stencils.VPAR_D4`, confirm index ordering (which index is v-2, v-1, etc.), and confirm boundary treatment (zero-padding vs. something else).

- [ ] **Step 2**: Verify coefficient array shapes. Print shapes of `pre["s_total_upar"]`, `pre["s_total_t7"]`, `pre["bessel"]`, `pre["utrap"]`, `pre["abs_dum2_vp"]`, `pre["drift_x"]`, `pre["drift_y"]`, `pre["dmaxwel_fm_ek"]`, `pre["fmaxwl"]`, `pre["hyper"]`, `pre["signz0"]`, `pre["tmp0"]`. Confirm which are 5D, which are broadcastable, and which are scalar.

- [ ] **Step 3**: Write the CUDA kernel `linear_rhs_fused.cu` following the template in §5 and §7. Start with the dynamic (non-templated) version for correctness. Use the existing `apply_parallel_dual_kernel` as a starting point — copy it and extend with the vpar stencil and elementwise accumulation.

- [ ] **Step 4**: Write the FFI dispatch function `LinearRhsFusedImpl` with the `DISPATCH_CASE` macro for `(16,32)`, `(32,32)`, `(16,64)`, plus the dynamic fallback.

- [ ] **Step 5**: Add FFI registration and the Python `_linear_rhs_fused` method to `CUDAOps`.

- [ ] **Step 6**: Write a standalone correctness test comparing the fused kernel output against `_linear_rhs_core` with JAX ops.

- [ ] **Step 7**: Integrate into `_linear_rhs_core` with the `hasattr` dispatch.

- [ ] **Step 8**: Run the full RK4 benchmark and verify accuracy + speedup.

- [ ] **Step 9** (optional): Add templated specializations for common `(NS, NKY)` pairs. Profile to see if the template specialization matters (it likely won't for this kernel since the vpar and elementwise portions dominate register usage, not loop unrolling).

---

## 13. Risks and Mitigations

**Register pressure**: The kernel uses many registers (2× double2 smem loads, 5× double2 vpar reads, ~10 double coefficient reads, 4 double2 accumulators). At 255 registers/thread max on A100, this should be fine, but high register usage may reduce occupancy. Mitigation: check `--ptxas-options=-v` for register count; if >128, consider `__launch_bounds__` with explicit maxreg.

**Coefficient broadcasting**: The Python side must ensure all 5D arrays are actually materialized to `(nv, nmu, ns, nkx, nky)` before the FFI call. `jnp.broadcast_to` returns a view, not a copy — XLA/FFI may or may not handle stride-0 dimensions correctly. Use `.copy()` if needed. This adds a one-time cost per RK4 step but is negligible if the arrays are already the right shape (which they are for `pre["bessel"]`, `pre["utrap"]`, etc.).

**Kinetic multi-species**: The current design handles the adiabatic (5D) case. For kinetic electrons, `jax.vmap` over species slices the 6D arrays into 5D slices before calling `_linear_rhs_core`. The fused kernel receives 5D arrays in both cases — no changes needed. However, verify that `signz0` and `tmp0` are indeed scalar per-species (they are: see `_compute_species_coeffs`).

**`phi_b` reshape**: In `_linear_rhs_core`, phi is passed as `phi_b` with shape `(1, 1, ns, nkx, nky)`. The fused kernel expects 3D `(ns, nkx, nky)`. The Python dispatch should squeeze the leading dimensions.