# Task: CUDA Kernel for `_apply_vpar` (5-Point Stencil)

## Objective

Replace the JAX `_apply_vpar` function with a fused CUDA kernel via XLA FFI.
Target: **0.3–0.5 ms** (vs 2.0 ms JAX = **4–6× speedup**).

## Current JAX Implementation

```python
def _apply_vpar(field, coeffs):
    # field: [nv=32, nmu=8, nspec=16, nkx=85, nky=32] complex128
    # coeffs: 5 real scalars for offsets (-2, -1, 0, 1, 2)
    nv = field.shape[0]
    out = jnp.zeros_like(field)
    for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):
        idx = jnp.clip(jnp.arange(nv) + s, 0, nv - 1)
        valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
        shifted = jnp.take(field, idx, axis=0)
        out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
    return out
```

## Baseline Performance

```
JAX FP64:   1.997 ms, 178.5 GB/s (11% of 1587 GB/s peak), AI=0.594
JAX FP64:   1.981 ms, 180.0 GB/s (11% of peak)
```

Called twice per RHS (D1 streaming + D4 dissipation) = **~4.0 ms total**.

## Why JAX Is Slow

The `for c, s in zip(...)` loop produces 5 iterations, each materializing:
1. `jnp.take(field, idx, axis=0)` → 170 MB gather (full copy)
2. `jnp.where(valid, shifted, 0)` → 170 MB conditional copy
3. `out + c * (...)` → 170 MB read-modify-write

That's ~510 MB per iteration × 5 = **2.55 GB** of HBM traffic.
Minimum is 340 MB (one read + one write of the full array).
**JAX does ~7.5× the minimum traffic.**

## CUDA Kernel Design

### Core Idea

A single kernel where each thread owns one position in the inner dimensions
`(mu, spec, kx, ky)` and loops over all `nv=32` positions along the stencil
axis. For each output `v`, it reads 5 neighbors (with boundary checks) and
writes one result. Every element is read exactly once and written exactly once.

### Memory Access Pattern

The data is laid out as `[nv, inner_size]` where `inner_size = 348,160`.
Adjacent threads access adjacent elements within a v-slice → **perfectly
coalesced** reads and writes. The stride between v-slices is 5.3 MB.

The kernel performs 32 coalesced sweeps over the inner dimension (one per v),
each sweep reading 5.3 MB. This is a bandwidth-optimal access pattern.

### Register Strategy: Sliding Window

For a 5-point stencil with offsets (-2, -1, 0, 1, 2), only 5 values
are needed simultaneously. Use a sliding window of 5 registers:

```
v=0: need v[-2]=0, v[-1]=0, v[0], v[1], v[2]    → window holds [0,0,f0,f1,f2]
v=1: need v[-1]=0, v[0],    v[1], v[2], v[3]    → shift, load f3
v=2: need v[0],    v[1],    v[2], v[3], v[4]    → shift, load f4
...
```

5 `double2` values = 10 doubles = 20 registers (32-bit GPU registers).
Plus 5 coefficient registers + loop overhead ≈ **35 registers total**.
This gives excellent occupancy (~60-75% on A100).

### Kernel Pseudocode

```cpp
__global__ void apply_vpar_stencil(
    const double2* __restrict__ field,   // [nv, inner_size] as complex128
    double2*       __restrict__ output,  // [nv, inner_size]
    int nv, int inner_size,
    double c0, double c1, double c2, double c3, double c4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inner_size) return;

    // Coefficients in registers
    double coeffs[5] = {c0, c1, c2, c3, c4};

    // Preload first 3 values for window (v=0 needs v[0], v[1], v[2])
    // Window: w[0..4] corresponds to stencil positions -2..+2 relative to current v
    double2 w[5];
    w[0] = {0.0, 0.0};  // v=-2: out of bounds
    w[1] = {0.0, 0.0};  // v=-1: out of bounds
    w[2] = field[0 * inner_size + idx];                        // v=0
    w[3] = (nv > 1) ? field[1 * inner_size + idx] : double2{0,0};  // v=1
    w[4] = (nv > 2) ? field[2 * inner_size + idx] : double2{0,0};  // v=2

    for (int v = 0; v < nv; v++) {
        // Compute stencil sum
        double2 result = {0.0, 0.0};
        #pragma unroll
        for (int s = 0; s < 5; s++) {
            result.x += coeffs[s] * w[s].x;
            result.y += coeffs[s] * w[s].y;
        }
        output[v * inner_size + idx] = result;

        // Slide window: shift left by 1
        w[0] = w[1];
        w[1] = w[2];
        w[2] = w[3];
        w[3] = w[4];

        // Load next value into w[4]
        int next_v = v + 3;  // v+3 because w[4] is 2 ahead of current output v
        w[4] = (next_v < nv) ? field[next_v * inner_size + idx] : double2{0,0};
    }
}
```

### Launch Configuration

```
inner_size = 348,160
block_size = 256 (or 128 — benchmark both)
grid_size  = ceil(348,160 / 256) = 1,360 blocks
```

### HBM Traffic

```
Reads:   nv × inner_size × 16 bytes = 32 × 348,160 × 16 = 178.3 MB
Writes:  nv × inner_size × 16 bytes = 32 × 348,160 × 16 = 178.3 MB
Total:   356.5 MB (theoretical minimum — cannot go lower)
```

At 1587 GB/s A100 peak: **0.22 ms** theoretical floor.
Realistic (75% bandwidth efficiency): **0.30 ms**.

### Boundary Handling

The JAX code uses `jnp.clip` to clamp indices but then zeros out
contributions where the original (unclipped) index is out of bounds.
This means: **zero-pad boundaries**, not clamp. The sliding window
naturally handles this — out-of-bounds positions enter the window as
`{0.0, 0.0}`.

## XLA FFI Integration

### Registration

Follow the same pattern as the LTO bracket FFI:
- C++ wrapper function registered via `XLA_FFI_DEFINE_HANDLER`
- Python binding via `jax.extend.ffi.ffi_lowering`
- Takes `field` as input buffer, `coeffs` as 5 scalar attributes
- Returns `output` buffer of same shape

### Python Side

```python
def _apply_vpar_cuda(field, coeffs):
    """Drop-in replacement for _apply_vpar using CUDA kernel."""
    nv = field.shape[0]
    inner_size = 1
    for d in field.shape[1:]:
        inner_size *= d
    return jax.extend.ffi.ffi_call(
        "apply_vpar_stencil",
        jax.ShapeDtypeStruct(field.shape, field.dtype),
        field,
        c0=float(coeffs[0].real), c1=float(coeffs[1].real),
        c2=float(coeffs[2].real), c3=float(coeffs[3].real),
        c4=float(coeffs[4].real),
        nv=nv, inner_size=inner_size,
    )
```

### Why Not cuBLAS/cuDNN?

This is a 1D stencil on the outermost axis of a 5D array. No standard
library handles this natively. A custom kernel is the right approach —
it's straightforward, the access pattern is simple, and the theoretical
speedup (4–6×) is large enough to justify the effort.

## Optimization Variants to Benchmark

### V0: Baseline sliding window (as described above)

Single kernel, one thread per inner-dimension element, sliding window of 5
registers. This should achieve ~0.3–0.5 ms.

### V1: Two-call fusion (D1 + D4)

Both `_apply_vpar` calls operate on the same input `field` with different
coefficients. A fused kernel reads the field once and writes two outputs:

```cpp
// field is read ONCE, two stencils computed simultaneously
double2 result_d1 = {0.0, 0.0};
double2 result_d4 = {0.0, 0.0};
for (int s = 0; s < 5; s++) {
    result_d1.x += c_d1[s] * w[s].x;
    result_d1.y += c_d1[s] * w[s].y;
    result_d4.x += c_d4[s] * w[s].x;
    result_d4.y += c_d4[s] * w[s].y;
}
output_d1[...] = result_d1;
output_d4[...] = result_d4;
```

Traffic: 178 MB read + 2 × 178 MB write = **535 MB** (vs 712 MB for two
separate calls). At 1587 GB/s: **0.34 ms** for both D1 and D4 combined.
This replaces 4.0 ms of JAX time with 0.34 ms — **11.7× total speedup**.

### V2: Coefficient generalization

Instead of hardcoding 5 coefficients as scalar attributes, pass them as a
small device array. This allows arbitrary stencil widths (3-point, 7-point)
without recompilation. Negligible performance impact since coefficients
are tiny (40 bytes) and cached in L1 after first access.

## Implementation Steps

- [ ] **Step 0:** Create benchmark script `bench_apply_vpar_cuda.py`.
    - Load same test data as `bench_apply_vpar.py`.
    - Compare CUDA output against JAX reference (rel_l2 < 1e-14).
    - Time CUDA kernel vs JAX baseline.

- [x] **Step 1:** Implement V0 kernel `apply_vpar_stencil.cu`.
    - Sliding window, single stencil.
    - XLA FFI wrapper.
    - Add to CMakeLists.txt.
    - Target: 0.3–0.5 ms, rel_l2 < 1e-14.

- [x] **Step 2:** Implement V1 kernel `apply_vpar_dual_stencil.cu`.
    - Fused D1+D4: one read, two outputs.
    - Separate FFI target.
    - Target: 0.3–0.4 ms for both combined.

- [x] **Step 3:** Integrate into solver.
    - Replace `_apply_vpar` calls with CUDA FFI variants.
    - Validate full solver output unchanged (rel_l2 < 1e-14).

- [x] **Step 4:** Benchmark block sizes (64, 128, 256, 512) and
      measure actual bandwidth utilization with `ncu`.

## Expected Impact on Solver

```
Current:  2 × _apply_vpar = 2 × 2.0 ms = 4.0 ms per RHS
V0:       2 × 0.35 ms = 0.7 ms per RHS → saves 3.3 ms
V1:       1 × 0.35 ms = 0.35 ms per RHS → saves 3.65 ms
```

Combined with LTO bracket (saved ~12 ms):
- Total RHS savings: ~15.7 ms (bracket) + ~3.3 ms (vpar) = **~19 ms**

## Files to Create

```
cuda_augmentations/
├── apply_vpar_stencil.cu          # V0 kernel + FFI wrapper
├── apply_vpar_dual_stencil.cu     # V1 fused kernel + FFI wrapper
├── CMakeLists.txt                 # Updated with new targets
└── jax_ffi_benchmark.py           # Updated with vpar variants

solver_components_benchmarks/
└── bench_apply_vpar_cuda.py       # Standalone benchmark
```

## Boundary Handling Summary

For output at position `v` with stencil offset `s`:
- Source index: `v + s` where `s ∈ {-2, -1, 0, 1, 2}`
- If `v + s < 0` or `v + s >= nv`: contribute **0.0** (not clamped edge value)
- This matches the JAX `valid` mask behavior

Concretely for `nv=32`:
```
v=0:  uses [  0,  0, f[0], f[1], f[2] ]   (first two zeroed)
v=1:  uses [  0, f[0], f[1], f[2], f[3] ] (first one zeroed)
v=2+: uses [ f[v-2], f[v-1], f[v], f[v+1], f[v+2] ]  (all valid)
...
v=30: uses [ f[28], f[29], f[30], f[31], 0 ]  (last one zeroed)
v=31: uses [ f[29], f[30], f[31],   0,   0 ]  (last two zeroed)
```