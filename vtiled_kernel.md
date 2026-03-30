# V-Tiled Fused Linear RHS Kernel — Implementation Specification

## 1. Core Idea

The current kernel assigns one block per `(v_idx, kx)` pair, meaning 256 blocks (for nv_nmu=256) independently scatter-gather the same stencil neighbors at the same `(src_s, src_kx, ky)` addresses. Each block reads one `double2` from each scattered location.

The v-tiled kernel assigns one block per `(v_tile, kx)` pair, where each block handles V_TILE consecutive v_idx values. The stencil map lookup happens once per thread per stencil neighbor, then the data reads are done V_TILE times at sequential memory addresses. This converts scattered independent reads into sequential streaming reads that the memory controller can prefetch.

This matches XLA's strategy: its `fused_add` uses `gather` with `slice_sizes={32,8,1,1,1}`, fetching entire (nv,nmu) vectors per spatial scatter point.

---

## 2. Block/Thread Structure

```
Old: num_blocks = nv_nmu * nkx = 256 * 85 = 21,760
     threads    = ns * nky = 16 * 32 = 512
     each thread: 1 output element

New: num_blocks = (nv_nmu / V_TILE) * nkx = 32 * 85 = 2,720  (for V_TILE=8)
     threads    = ns * nky = 16 * 32 = 512
     each thread: V_TILE output elements (loop over v within tile)
```

Thread identity:
```cuda
const int tile_idx = blockIdx.x / nkx;     // which v-tile (0..nv_nmu/V_TILE - 1)
const int kx       = blockIdx.x % nkx;     // which kx
const int s        = threadIdx.x / NKY;
const int ky       = threadIdx.x % NKY;
const int v_base   = tile_idx * V_TILE;    // first v_idx in this tile
```

V_TILE should divide nv_nmu evenly. For nv=32, nmu=8, nv_nmu=256:
- V_TILE=8: 32 tiles, 2720 blocks — good balance
- V_TILE=4: 64 tiles, 5440 blocks — safer for register pressure
- V_TILE=16: 16 tiles, 1360 blocks — more aggressive, may hit smem limits

Start with V_TILE=8. Make it a template parameter so we can tune.

---

## 3. Shared Memory Layout

Each block must cache `df` and `gyro_phi` for all V_TILE velocity slices at the current kx, so that same-kx stencil lookups hit smem.

```
smem_df:   V_TILE * NS * NKY * sizeof(double2) = 8 * 16 * 32 * 16 = 65,536 bytes = 64 KB
smem_gyro: V_TILE * NS * NKY * sizeof(double2) = 64 KB
Total:     128 KB
```

128 KB exceeds the default 48 KB smem limit but is within A100's 164 KB maximum. Use dynamic shared memory with `cudaFuncSetAttribute`:

```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
```

If 128 KB is too tight for occupancy, we have two options:
- **Option A**: Drop smem_gyro — compute gyro_phi on-the-fly from smem_df + bessel + phi. This halves smem to 64 KB. The cost: for same-kx stencil neighbors in term_vii, we do `bessel_lookup * phi_lookup` per access instead of a single smem read. But bessel is now small (fits in L2) and phi is tiny (0.7 MB, fits in L2), so this is cheap.
- **Option B**: Reduce V_TILE to 4, halving smem to 64 KB total (32 KB per array).

**Recommendation: Start with Option A (no smem_gyro, 64 KB total).** This avoids the 128 KB smem requirement while the on-the-fly gyro_phi computation from L2-resident bessel+phi is nearly free.

### Revised Shared Memory (Option A):
```
smem_df only: V_TILE * NS * NKY * sizeof(double2) = 8 * 512 * 16 = 65,536 bytes = 64 KB
```

Set max dynamic smem to 65536.

### Indexing into smem_df:
```cuda
// smem_df is laid out as [V_TILE][NS * NKY]
// For v_offset in 0..V_TILE-1, thread (s, ky):
//   smem_df[v_offset * (NS * NKY) + s * NKY + ky]

// For stencil lookup at (v_offset, src_s, ky) where src_kx == kx:
//   smem_df[v_offset * (NS * NKY) + src_s * NKY + ky]
```

---

## 4. Kernel Algorithm

```cuda
template <int NS, int NKY, int V_TILE>
__global__ void linear_rhs_vtiled_kernel(
    const double2* __restrict__ df,
    const double2* __restrict__ phi,          // (ns, nkx, nky)
    const double*  __restrict__ bessel,       // MINIMAL: (nmu, ns, nkx, nky)
    const double*  __restrict__ s_total_upar, // MINIMAL: (9, nv, ns, nkx, nky)
    const double*  __restrict__ s_total_t7,   // MINIMAL: (9, nv, ns, nkx, nky)
    const int2*    __restrict__ packed_maps,   // (9, ns, nkx, nky)
    const double*  __restrict__ utrap,        // MINIMAL: (nmu, ns)
    const double*  __restrict__ abs_dum2_vp,  // MINIMAL: (nmu, ns)
    const double*  __restrict__ drift_x,      // MINIMAL: (nv, nmu, ns)
    const double*  __restrict__ drift_y,      // MINIMAL: (nv, nmu, ns)
    const double*  __restrict__ dmaxwel_fm_ek,// MINIMAL: (nv, nmu, ns, nky)
    const double*  __restrict__ fmaxwl,       // MINIMAL: (nv, nmu, ns)
    const double*  __restrict__ hyper,        // (ns, nkx, nky)
    const double*  __restrict__ kx_vals,      // (nkx,)
    const double*  __restrict__ ky_vals,      // (nky,)
    double2*       __restrict__ rhs_out,      // (nv_nmu, ns, nkx, nky)
    int nv, int nmu, int nkx, int nky_param, int nv_nmu,
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    double dvp, double disp_vp, double drive_scale, double signz0, double tmp0
) {
    extern __shared__ double2 smem_df[];  // [V_TILE * NS * NKY]

    const int tile_idx  = blockIdx.x / nkx;
    const int kx        = blockIdx.x % nkx;
    const int s         = threadIdx.x / NKY;
    const int ky        = threadIdx.x % NKY;
    const int local_tid = threadIdx.x;
    const int v_base    = tile_idx * V_TILE;

    const size_t spatial_stride = (size_t)NS * nkx * NKY;
    const size_t spatial_idx    = (size_t)s * (nkx * NKY) + (size_t)kx * NKY + ky;
    const int    smem_block     = NS * NKY;

    // ── Phase 0: Load df into shared memory for all V_TILE slices ──
    // Each thread loads V_TILE values (one per velocity slice)
    #pragma unroll
    for (int vv = 0; vv < V_TILE; vv++) {
        const int v_idx = v_base + vv;
        const size_t field_idx = (size_t)v_idx * spatial_stride + spatial_idx;
        smem_df[vv * smem_block + local_tid] = __ldg(&df[field_idx]);
    }
    __syncthreads();

    // ── Load per-thread constants (same for all v in tile) ──
    const double2 phi_val = __ldg(&phi[spatial_idx]);
    const double  kx_val  = __ldg(&kx_vals[kx]);
    const double  ky_val  = __ldg(&ky_vals[ky]);
    const double  hyp_val = __ldg(&hyper[spatial_idx]);
    const double  inv_dvp = 1.0 / dvp;
    const double  inv_tmp = 1.0 / fmax(tmp0, 1e-15);

    // ── Phase 1: Parallel stencil — compute map once, apply V_TILE times ──

    // Preload all 9 stencil map entries for this (s, kx, ky)
    int    src_s_arr[9], src_kx_arr[9];
    bool   valid_arr[9];
    size_t src_spatial_arr[9];  // precomputed spatial index for cross-kx

    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const int2 map_val = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        src_s_arr[i]  = map_val.x;
        src_kx_arr[i] = map_val.y;
        valid_arr[i]  = (map_val.x >= 0);
        if (valid_arr[i] && src_kx_arr[i] != kx) {
            src_spatial_arr[i] = (size_t)src_s_arr[i] * (nkx * NKY)
                               + (size_t)src_kx_arr[i] * NKY + ky;
        }
    }

    // ── Process each v_idx in the tile ──
    #pragma unroll
    for (int vv = 0; vv < V_TILE; vv++) {
        const int v_idx = v_base + vv;
        const int v     = v_idx / nmu;
        const int mu    = v_idx % nmu;
        const size_t field_idx = (size_t)v_idx * spatial_stride + spatial_idx;

        double2 my_df = smem_df[vv * smem_block + local_tid];

        // Compute gyro_phi for this (v_idx, s, kx, ky) on-the-fly
        // bessel is (nmu, ns, nkx, nky)
        const size_t bes_spatial = (size_t)mu * (NS * nkx * NKY) + spatial_idx;
        const double bes = __ldg(&bessel[bes_spatial]);
        double2 my_gyro_phi = make_double2(bes * phi_val.x, bes * phi_val.y);

        // ── Parallel stencil accumulation ──
        // s_total_upar/t7: (9, nv, ns, nkx, nky) — index by v (not v_idx)
        const size_t reduced_spatial = (size_t)NS * nkx * NKY;
        const size_t c_reduced_stride = (size_t)nv * reduced_spatial;
        const size_t c_v_base = (size_t)v * reduced_spatial + spatial_idx;

        double acc_par_r = 0.0, acc_par_i = 0.0;
        double acc_t7_r  = 0.0, acc_t7_i  = 0.0;

        #pragma unroll
        for (int i = 0; i < 9; i++) {
            if (valid_arr[i]) {
                const double c_upar = __ldg(&s_total_upar[(size_t)i * c_reduced_stride + c_v_base]);
                const double c_t7   = __ldg(&s_total_t7  [(size_t)i * c_reduced_stride + c_v_base]);

                double2 v_df;
                double2 v_gyro;

                if (src_kx_arr[i] == kx) {
                    // Same-kx: read df from shared memory
                    v_df = smem_df[vv * smem_block + src_s_arr[i] * NKY + ky];
                    // Compute gyro_phi on-the-fly from bessel + phi
                    const size_t src_bes = (size_t)mu * (NS * nkx * NKY)
                                         + (size_t)src_s_arr[i] * (nkx * NKY)
                                         + (size_t)kx * NKY + ky;
                    const double bes_src = __ldg(&bessel[src_bes]);
                    const size_t src_phi_idx = (size_t)src_s_arr[i] * (nkx * NKY)
                                             + (size_t)kx * NKY + ky;
                    const double2 phi_src = __ldg(&phi[src_phi_idx]);
                    v_gyro = make_double2(bes_src * phi_src.x, bes_src * phi_src.y);
                } else {
                    // Cross-kx: read df from global memory
                    // Key insight: for consecutive vv iterations, these reads are
                    // at addresses separated by spatial_stride — sequential in memory!
                    const size_t src_field = (size_t)v_idx * spatial_stride + src_spatial_arr[i];
                    v_df = __ldg(&df[src_field]);

                    // gyro_phi on-the-fly — bessel from L2, phi from L2
                    const size_t src_bes = (size_t)mu * (NS * nkx * NKY) + src_spatial_arr[i];
                    const double bes_src = __ldg(&bessel[src_bes]);
                    const double2 phi_src = __ldg(&phi[src_spatial_arr[i]]);
                    v_gyro = make_double2(bes_src * phi_src.x, bes_src * phi_src.y);
                }

                acc_par_r += v_df.x * c_upar;
                acc_par_i += v_df.y * c_upar;
                acc_t7_r  += v_gyro.x * c_t7;
                acc_t7_i  += v_gyro.y * c_t7;
            }
        }

        // ── Vpar stencil ──
        const size_t vpar_stride = (size_t)nmu * spatial_stride;

        // Boundary checks based on v (not v_idx)
        double2 df_vm2 = (v >= 2)      ? __ldg(&df[field_idx - 2*vpar_stride]) : make_double2(0,0);
        double2 df_vm1 = (v >= 1)      ? __ldg(&df[field_idx - 1*vpar_stride]) : make_double2(0,0);
        double2 df_vp1 = (v <= nv - 2) ? __ldg(&df[field_idx + 1*vpar_stride]) : make_double2(0,0);
        double2 df_vp2 = (v <= nv - 3) ? __ldg(&df[field_idx + 2*vpar_stride]) : make_double2(0,0);

        double2 out_d1 = make_double2(
            c_d1_0*df_vm2.x + c_d1_1*df_vm1.x + c_d1_2*my_df.x + c_d1_3*df_vp1.x + c_d1_4*df_vp2.x,
            c_d1_0*df_vm2.y + c_d1_1*df_vm1.y + c_d1_2*my_df.y + c_d1_3*df_vp1.y + c_d1_4*df_vp2.y
        );
        double2 out_d4 = make_double2(
            c_d4_0*df_vm2.x + c_d4_1*df_vm1.x + c_d4_2*my_df.x + c_d4_3*df_vp1.x + c_d4_4*df_vp2.x,
            c_d4_0*df_vm2.y + c_d4_1*df_vm1.y + c_d4_2*my_df.y + c_d4_3*df_vp1.y + c_d4_4*df_vp2.y
        );

        // ── Elementwise terms — minimal-shape indexing ──
        // utrap, abs_dum2_vp: (nmu, ns)
        const size_t utrap_idx = (size_t)mu * NS + s;
        const double utrap_val   = __ldg(&utrap[utrap_idx]);
        const double abs_vp_val  = __ldg(&abs_dum2_vp[utrap_idx]);

        // drift_x, drift_y, fmaxwl: (nv, nmu, ns)
        const size_t drift_idx = (size_t)v * nmu * NS + (size_t)mu * NS + s;
        const double drift_x_val = __ldg(&drift_x[drift_idx]);
        const double drift_y_val = __ldg(&drift_y[drift_idx]);
        const double fmaxwl_val  = __ldg(&fmaxwl[drift_idx]);

        // dmaxwel_fm_ek: (nv, nmu, ns, nky)
        const size_t dmax_idx = (size_t)v * nmu * NS * NKY + (size_t)mu * NS * NKY + (size_t)s * NKY + ky;
        const double dmaxwel_val = __ldg(&dmaxwel_fm_ek[dmax_idx]);

        double kdotvd = drift_x_val * kx_val + drift_y_val * ky_val;

        // term_iv
        double2 term_iv = make_double2(utrap_val * out_d1.x * inv_dvp,
                                       utrap_val * out_d1.y * inv_dvp);

        // term_vp_diss
        double vp_coeff = disp_vp * abs_vp_val * inv_dvp;
        double2 term_vp_diss = make_double2(vp_coeff * out_d4.x, vp_coeff * out_d4.y);

        // -1j * kdotvd * df
        double2 drift_term = make_double2(kdotvd * my_df.y, -kdotvd * my_df.x);

        // hyper * df
        double2 hyper_term = make_double2(hyp_val * my_df.x, hyp_val * my_df.y);

        // drive = 1j * drive_scale * (dmaxwel - signz0 * kdotvd * fmaxwl / tmp0) * gyro_phi
        double drive_c = drive_scale * (dmaxwel_val - signz0 * kdotvd * fmaxwl_val * inv_tmp);
        double2 drive_term = make_double2(-drive_c * my_gyro_phi.y, drive_c * my_gyro_phi.x);

        // Final sum
        double2 res;
        res.x = acc_par_r + term_iv.x + term_vp_diss.x + drift_term.x + hyper_term.x + drive_term.x + acc_t7_r;
        res.y = acc_par_i + term_iv.y + term_vp_diss.y + drift_term.y + hyper_term.y + drive_term.y + acc_t7_i;

        rhs_out[field_idx] = res;
    }
}
```

---

## 5. Why This Is Faster

### 5.1. Stencil Map Amortization
The 9 packed_maps lookups and address computations happen once per thread, then are reused V_TILE=8 times. Old kernel: 9 map reads per output element. New: 9 map reads per 8 output elements = 1.125 per element.

### 5.2. Sequential Cross-kx Reads
For cross-kx stencil neighbor at `(src_s, src_kx)`, the reads across the v-tile are:
```
df[v_base+0, src_s, src_kx, ky]  → address A
df[v_base+1, src_s, src_kx, ky]  → address A + spatial_stride * sizeof(double2)
df[v_base+2, src_s, src_kx, ky]  → address A + 2 * spatial_stride * sizeof(double2)
...
```
These are strided but predictable — the hardware prefetcher can help, and the memory controller can coalesce them better than 8 independent requests from separate blocks.

### 5.3. L2 Cache Reuse for Coefficients
With V_TILE=8 and nmu=8, each tile covers one complete mu sweep. The coefficient arrays indexed by `v` (s_total_upar/t7) are read for the same `v` value across `nmu=8` consecutive v_idx values, so 8 of the V_TILE iterations hit the same coefficient cache line.

### 5.4. Reduced Block Count
Old: 21,760 blocks. New: 2,720 blocks. Fewer blocks means less scheduling overhead and less contention for shared resources (L2 partitions, memory controllers).

### 5.5. Bessel + Phi L2 Residency  
Bessel `(nmu, ns, nkx, nky)` = 2.8 MB and phi `(ns, nkx, nky)` = 0.7 MB both fit in L2. The on-the-fly gyro_phi computation `bessel[mu,...] * phi[...]` for cross-kx neighbors becomes an L2 read instead of a DRAM read. This was impossible before because bessel was `(nv_nmu, ns, nkx, nky)` = 89 MB.

---

## 6. Shared Memory Configuration

```cuda
// In the dispatch function:
cudaFuncSetAttribute(
    linear_rhs_vtiled_kernel<NS, NKY, V_TILE>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    V_TILE * NS * NKY * sizeof(double2)  // 65536 for V_TILE=8, NS=16, NKY=32
);

linear_rhs_vtiled_kernel<NS, NKY, V_TILE>
    <<<num_blocks, NS * NKY, V_TILE * NS * NKY * sizeof(double2), stream>>>(...);
```

---

## 7. Vpar Stencil Optimization Within the V-Tile

The vpar stencil reads `df[v±1, mu, s, kx, ky]` and `df[v±2, mu, s, kx, ky]`. When the neighbor is **within the same V_TILE**, we can read from smem instead of global memory.

For v_idx within a tile [v_base, v_base + V_TILE), the vpar stencil at v_idx needs:
- `v_idx - nmu` and `v_idx + nmu` (±1 in v, stride nmu in v_idx)
- `v_idx - 2*nmu` and `v_idx + 2*nmu` (±2 in v)

With V_TILE=8 and nmu=8: `v_idx ± nmu` means a jump of 8 v_idx positions — exactly the tile size. So vpar neighbors are **never within the same tile** for V_TILE=nmu=8.

This means vpar stencil reads always go to global memory regardless of V_TILE. This is fine — the vpar reads are coalesced (all threads read the same v-offset, different (s,ky)) and there are only 4 per output element.

**Alternative V_TILE choices**: If V_TILE=16 (2*nmu), then v_idx ± nmu falls within the tile for the middle 8 entries, saving half the vpar global reads. But this doubles smem to 128 KB and may hurt occupancy. Not recommended as a first step.

---

## 8. Register Pressure Considerations

Each thread now processes V_TILE=8 output elements in a loop. Within the loop body, the live registers are:
- `my_df`: 2 doubles (1 double2)
- `my_gyro_phi`: 2 doubles
- `out_d1`, `out_d4`: 4 doubles
- `df_vm2/vm1/vp1/vp2`: 8 doubles
- Stencil accumulators (`acc_par_r/i`, `acc_t7_r/i`): 4 doubles
- Various scalar coefficients: ~8 doubles
- Final result: 2 doubles
Total: ~30 doubles = 240 bytes = 30 registers

Plus the map arrays stored across iterations:
- `src_s_arr[9]`, `src_kx_arr[9]`, `valid_arr[9]`, `src_spatial_arr[9]`: ~36 registers

Total: ~66 registers. Well within A100's 255 limit. The compiler may spill some to local memory but this should be manageable.

Check with `--ptxas-options=-v` after compilation.

---

## 9. Dispatch Macro

```cuda
#define DISPATCH_VTILE_CASE(NS_VAL, NKY_VAL, VT)                                        \
    case (((NS_VAL) << 16) | (NKY_VAL)):                                                 \
    {                                                                                     \
        const size_t smem_bytes = (VT) * (NS_VAL) * (NKY_VAL) * sizeof(double2);        \
        cudaFuncSetAttribute(                                                             \
            linear_rhs_vtiled_kernel<NS_VAL, NKY_VAL, VT>,                               \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);                     \
        linear_rhs_vtiled_kernel<NS_VAL, NKY_VAL, VT>                                    \
            <<<num_blocks, (NS_VAL) * (NKY_VAL), smem_bytes, stream>>>(                  \
                /* all args */);                                                          \
    }                                                                                     \
    break;
```

In `LinearRhsVtiledImpl`:
```cuda
const int V_TILE = 8;  // or make this an attr for tuning
const int num_blocks = (nv_nmu / V_TILE) * nkx;
// assert nv_nmu % V_TILE == 0

switch ((ns << 16) | nky) {
    DISPATCH_VTILE_CASE(16, 32, 8)
    DISPATCH_VTILE_CASE(32, 32, 8)
    DISPATCH_VTILE_CASE(16, 64, 8)
    default: { /* dynamic fallback */ }
}
```

---

## 10. Python-Side Changes

The `_linear_rhs_fused` method in `CUDAOps` needs minimal changes:

1. Pass V_TILE as an attribute (or hardcode 8)
2. Ensure `nv_nmu % V_TILE == 0` (add assertion)
3. All coefficient arrays are already in minimal shape from Phase 6

```python
def _linear_rhs_fused(self, df, phi, pre, params_dvp, params_disp_vp, params_drive_scale):
    nv, nmu, ns, nkx, nky = df.shape
    nv_nmu = nv * nmu
    V_TILE = 8
    assert nv_nmu % V_TILE == 0, f"nv_nmu={nv_nmu} must be divisible by V_TILE={V_TILE}"
    
    # ... same coefficient preparation as Phase 6 (minimal shapes) ...
    
    return ffi.ffi_call(
        "linear_rhs_vtiled_ffi",
        [jax.ShapeDtypeStruct(df.shape, df.dtype)]
    )(
        df, phi, bessel, s_total_upar, s_total_t7, packed_maps,
        utrap, abs_vp, drift_x, drift_y, dmaxwel, fmaxwl,
        hyper, kx_vals, ky_vals,
        nv=np.int32(nv), nmu=np.int32(nmu), nkx=np.int32(nkx),
        nky=np.int32(nky), nv_nmu=np.int32(nv_nmu),
        # ... same scalar attrs as before ...
    )[0]
```

---

## 11. Testing Strategy

### 11.1. Correctness
Compare against the non-v-tiled fused kernel (Phase 5/6) which is already verified:
```python
rhs_ref = ops_cuda_old._linear_rhs_fused(df, phi, pre, dvp, disp_vp, drive_scale)
rhs_new = ops_cuda_vtiled._linear_rhs_fused(df, phi, pre, dvp, disp_vp, drive_scale)
err = jnp.linalg.norm(rhs_ref - rhs_new) / jnp.linalg.norm(rhs_ref)
assert err < 1e-15  # should be exact since same FP operations
```

### 11.2. Performance
Run `bench_rk4_step.py` and compare:
- V0: JAX reference (33.8 ms)
- V1: Phase 6 fused kernel (29.5 ms)
- V2: V-tiled kernel (target: ~22 ms)

### 11.3. Occupancy Check
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    --kernel-name vtiled \
    python bench_rk4_step.py --backend cuda
```

---

## 12. Implementation Checklist

- [ ] **Step 1**: Create `linear_rhs_vtiled.cu` with the v-tiled kernel. Start from the Phase 6 kernel code, restructure blocking, add the v-loop.

- [ ] **Step 2**: Handle the smem configuration — `cudaFuncSetAttribute` for 64 KB dynamic shared memory. Verify this doesn't fail on your MIG partition (MIG might have reduced smem limits — check with `cudaDeviceGetAttribute`).

- [ ] **Step 3**: Register the new FFI symbol `linear_rhs_vtiled_ffi`.

- [ ] **Step 4**: Correctness test against Phase 6 kernel output.

- [ ] **Step 5**: Benchmark full RK4 step.

- [ ] **Step 6**: Tune V_TILE. Try 4, 8, 16 and measure. If V_TILE=8 is limited by smem (only 1 block/SM due to 64 KB), try V_TILE=4 (32 KB smem, potentially 2 blocks/SM, better occupancy).

- [ ] **Step 7** (optional): If V_TILE=8 wins but occupancy is low, try the RK4-stage-fused variant from the earlier discussion — write `df + weight*rhs` directly instead of `rhs`, eliminating one round-trip per stage.

---

## 13. Fallback: V_TILE=nmu Variant

If V_TILE=8 (=nmu) works well, note that all 8 iterations within a tile share the same `v` value but differ in `mu`. This means:
- `s_total_upar[i, v, ...]` is the same for all 8 iterations → read once, reuse 8 times
- `drift_x[v, mu, s]`, `drift_y[v, mu, s]` differ only in mu → 8 different reads but from contiguous memory (mu is second-fastest dimension)

This is the optimal V_TILE for coefficient reuse. Larger V_TILE (e.g., 16) crosses a `v` boundary, requiring new coefficient reads.