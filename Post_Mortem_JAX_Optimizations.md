# Post-Mortem: JAX-Level Optimization of the gyaradax Gyrokinetic Solver

**Date**: 2026-03-18
**Solver**: `gyaradax` — Vlasov-Poisson gyrokinetic code in JAX
**Grid**: `(nvpar=32, nmu=8, ns=16, nkx=85, nky=32)`, adiabatic electrons, complex128
**Hardware**: NVIDIA A100 80GB (HBM2e, 2 TB/s, 9.7 TFLOP/s FP64)
**Config**: `configs/iteration_13.yaml`

---

## 0. Abstract

Five pure-JAX optimizations (O1-O4, O6) were proposed in OPTIM.md targeting a predicted aggregate 1.3-1.5x speedup. All five were implemented, benchmarked, and failed: two caused regressions (-2.6% and -4.4%), one caused a catastrophic -16% regression in its best variant, and two were neutral. The aggregate delivered speedup is **1.00x** — zero improvement. This section maps each prediction to its measured outcome and quantifies the gap.

---

## 1. Discrepancy Mapping

### 1.1 Prediction vs. Measurement Table

Sources: OPTIM.md Section 7.1 (predictions), OPTIM_measurements.md (measurements).

| ID | Target | Predicted Component | Predicted Overall | Best Measured | Worst Measured | Error Factor | Category |
|----|--------|--------------------:|------------------:|--------------:|---------------:|-------------:|----------|
| O1 | `_apply_parallel` | 2-3x | 1.1-1.3x | 1.00x (v4 isolated: 1.03x) | 0.29x (v1/v2/v3 component), 0.84x (v4 full solver) | **6.9-10.3x** on component | Direction-wrong |
| O2 | `_apply_vpar` | 2-3x | 1.05-1.1x | 1.00x (baseline kept) | 0.36x (v3 lax.scan component) | **5.6-8.3x** on component | Direction-wrong |
| O3 | `calculate_phi_kinetic` | 1.5-2x | 1.01x | 1.00x (6.10 vs 6.11 steps/s) | 1.00x | **1.5-2x** on component | Magnitude-wrong |
| O4 | RK4 accumulation | 1.2x | 1.02x | 1.00x (6.12 vs 6.11 steps/s) | 1.00x | **1.2x** on component | Magnitude-wrong |
| O6 | Nonlinear FFT batching | 1.0-1.5x | 1.0-1.3x | — | 0.96x (5.84 vs 6.11 steps/s) | **1.0-1.6x** | Direction-wrong |

### 1.2 Aggregate Phase 1 Outcome

| Metric | Predicted (OPTIM.md §7.2) | Actual |
|--------|---------------------------:|-------:|
| Overall speedup | 1.3-1.5x | **1.00x** |
| Effort estimate | 1-2 days | ~3 days (accurate on time, zero on value) |
| Optimizations that improved perf | 4 of 5 (O6 uncertain) | **0 of 5** |
| Optimizations that degraded perf | 0 | **3 of 5** (O1, O2 variants, O6) |

### 1.3 Worst Offenders — Ranked by Impact Severity

Severity metric: `(component weight in step time) x (magnitude of prediction miss)`.

Component time weights (Device 6 adiabatic, isolated single-call benchmarks):
- Linear RHS: 66.61 ms/call (~57% of single-RHS cost)
- Nonlinear RHS: 50.12 ms/call (~43%)
- Phi solve: 0.63 ms/call (~0.5%)

Within linear RHS, `_apply_parallel` accounts for ~49% of FLOPs and ~54% of memory traffic (OPTIM.md §4.3). `_apply_vpar` (x2 calls) accounts for ~31% of traffic.

| Rank | ID | Severity Score | Why |
|-----:|:---|:--------------:|:----|
| 1 | **O1** | Critical | Targeted the dominant bottleneck (parallel stencil: ~28% of step cost). Predicted 2-3x, achieved 0.29x. The single most valuable optimization target was provably unoptimizable at the JAX level. |
| 2 | **O6** | High | Targeted 43% of step cost (nonlinear FFT). Predicted up to 1.5x component speedup. Caused 4.4% regression by inflating peak memory. |
| 3 | **O2** | Moderate | Targeted ~15% of step cost (vpar stencils). All 3 alternatives were slower. Best baseline was already optimal. |
| 4 | **O3** | Low | Targeted 0.5% of step cost. Even a perfect 2x component speedup = 0.25% overall. Correctly predicted as low-impact in OPTIM.md (1.01x overall), but component prediction (1.5-2x) was still wrong. |
| 5 | **O4** | Low | Targeted ~1% of step cost. XLA already fused the original form optimally. |

### 1.4 Microbenchmark Detail: O1 Variants

All variants tested against the Python-loop baseline on isolated `_apply_parallel` (Device 1, 20 trials):

| Variant | Approach | ms/call | vs Baseline | Status |
|---------|----------|--------:|:-----------:|:------:|
| v0 | Python `for i in range(9)` (baseline) | 5.565 +/- 0.012 | 1.00x | **Optimal** |
| v1 | Batch-gather + moveaxis | 19.091 +/- 0.036 | 0.29x | Regression |
| v2 | `jax.vmap` over 9 stencils | 19.098 +/- 0.028 | 0.29x | Regression |
| v3 | Batch-gather + einsum | 19.088 +/- 0.037 | 0.29x | Regression |
| v4 | `jax.lax.scan` accumulation | 5.393 +/- 0.020 | 1.03x | Neutral (but -16% in full solver) |

v1, v2, v3 produce **identical HLO** — all route through the same BatchGather + large intermediate. The 3.4x regression is remarkably consistent across all three JAX-level expressions.

### 1.5 Microbenchmark Detail: O2 Variants

All variants tested on isolated `_apply_vpar` (Device 1, 20 trials, COEFFS_D1):

| Variant | Approach | ms/call | vs Baseline | Status |
|---------|----------|--------:|:-----------:|:------:|
| v0 | `jnp.take` + clip + valid mask (baseline) | 1.755 +/- 0.020 | 1.00x | **Optimal** |
| v1 | `jnp.pad` + slice | 3.364 +/- 0.085 | 0.52x | Regression |
| v2 | `conv_general_dilated` | 2.138 +/- 0.031 | 0.82x | Regression + numerically wrong |
| v3 | `jax.lax.scan` | 4.890 +/- 0.038 | 0.36x | Regression |

### 1.6 Retained Changes (Neutral, Kept for Code Quality)

Two optimizations were kept despite being performance-neutral:

- **O3** (einsum phi solve): `jnp.einsum('avmjkl,avmjkl->jkl', phi_weight, df)` replaces `jnp.sum(phi_weight * df, axis=(0,1,2))`. Identical HLO. Semantically clearer.
- **O4** (expanded RK4): `prev_df + dt6*k1 + dt3*k2 + dt3*k3 + dt6*k4` replaces `prev_df + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)`. Identical HLO. Marginally cleaner intent.

### 1.7 Summary

Every optimization that was predicted to deliver measurable speedup either failed or regressed. The two that were retained (O3, O4) are performance-neutral code-quality changes. The theoretical analysis in OPTIM.md was **systematically overconfident** about the gap between XLA's actual code generation and the proposed alternatives.

**The core analytical error**: OPTIM.md assumed that visible Python-level inefficiencies (for-loops, intermediate buffers, broadcast patterns) translate to XLA-level inefficiencies. They do not. XLA's trace-time unrolling, elementwise fusion, and GPU scheduling already handle these patterns near-optimally for this workload.

---

## 2. Root-Cause Analysis

This section traces the exact compiler or hardware mechanism that caused each optimization to fail. The analysis proceeds from the most impactful failure (O1) to the least.

### 2.1 O1: `_apply_parallel` — BatchGather vs. Independent Gathers

**Code reference**: `solver.py:758-768`

```python
for i in range(9):
    s_map = pre["s_shift"][i]        # (ns, nkx, nky) int32
    kx_map = pre["kx_shift"][i]      # (ns, nkx, nky) int32
    valid = pre["valid_shift"][i]    # (ns, nkx, nky) bool
    shifted = jnp.where(valid[None, None], field[:, :, s_map, kx_map, ky_idx], 0.0)
    out = out + coeffs[i] * shifted
```

#### 2.1.1 What the Python for-loop actually compiles to

The Python `for i in range(9)` is **not a loop in the compiled program**. At JAX trace time, the loop body is executed 9 times with concrete Python integer `i`. Each iteration produces independent HLO operations:

```
Iteration 0: Gather_0 → Where_0 → Multiply_0 → Add_0
Iteration 1: Gather_1 → Where_1 → Multiply_1 → Add_1
...
Iteration 8: Gather_8 → Where_8 → Multiply_8 → Add_8
```

XLA sees all 36 ops (9 × 4) simultaneously in a flat computation graph. This is critical: the XLA scheduler can **overlap** memory-bound gather operations from different iterations across different SMs. While iteration `i+1` depends on `out` from iteration `i` (sequential accumulation), the `Gather_{i+1}` can begin executing while `Add_i` is still writing — the gather reads `field`, not `out`. XLA's dataflow analysis recognizes this and pipelines the operations.

The XLA elementwise fusion pass then merges `Where_i + Multiply_i + Add_i` into a single fused kernel per iteration — 3 elementwise ops become 1 kernel. So the compiled code is 9 gather kernels interleaved with 9 fused elementwise kernels, with the gathers pipelined ahead of the elementwise ops.

#### 2.1.2 Why the batch-gather alternatives are 3.4x slower

All three batch variants (v1/v2/v3) express the same semantic: "gather all 9 stencil points first, then reduce." In HLO, this becomes:

```
BatchGather → (9, nv, nmu, ns, nkx, nky) intermediate → Multiply + Reduce
```

The `BatchGather` produces a single output tensor of shape `(nv, nmu, 9, ns, nkx, nky)` — that's **9 × 55.8 MB = 502 MB** of complex128 data written to HBM in a single kernel.

**Mechanism 1 — Intermediate materialization cost**: The 502 MB intermediate must be fully written to HBM before the multiply-reduce can begin (XLA cannot fuse a Gather with a subsequent Reduce because the gather output indexing is data-dependent). At A100's 2 TB/s: 502 MB write = 0.25 ms. The baseline's intermediate `out` buffers total 8 × 111.7 MB = 894 MB of R+W traffic, but these are pipelined across 9 kernel launches. The batch version writes 502 MB in a single non-overlappable burst, then reads it all back for the reduce — total 1004 MB sequential traffic just for the intermediate.

**Mechanism 2 — L2 cache thrashing**: A100 has 40 MB L2 cache. The baseline processes one stencil point at a time: each fused kernel reads ~56 MB of gathered field + ~56 MB of coefficients + ~56 MB of `out` = ~168 MB working set. Only 24% fits in L2, but the sequential nature means recently-used `out` data from the previous iteration's write may still be in L2 for the current iteration's read.

The batched version writes 502 MB to HBM, then reads it back — the L2 has been completely evicted by the write, guaranteeing 100% cache misses on the read-back. This alone costs 502 MB / (0.5 × 2 TB/s) ≈ 0.5 ms (assuming 50% effective bandwidth for the sequential read).

**Mechanism 3 — moveaxis physical copy** (v1 only, but v2/v3 have equivalent costs): `jnp.moveaxis(gathered, 2, 0)` transposes `(nv, nmu, 9, ns, nkx, nky)` to `(9, nv, nmu, ns, nkx, nky)`. In XLA's row-major layout, axis 2 (the stencil dimension) has stride `ns × nkx × nky × 16B = 16 × 85 × 32 × 16 = 696 KB`. Moving it to axis 0 requires a full physical copy: 502 MB read + 502 MB write = 1004 MB. At 2 TB/s, that's another 0.5 ms — and v2 (vmap) and v3 (einsum) produce equivalent transposes internally.

**Measured impact**: Baseline 5.565 ms vs. batch variants ~19.09 ms = +13.5 ms overhead. This is consistent with: 502 MB intermediate write (0.25 ms) + 502 MB readback (0.25 ms) + 1004 MB transpose (0.5 ms) + degraded gather efficiency from batching = ~1 ms of pure overhead, plus the loss of pipelined gather-fuse scheduling that accounts for the remaining ~12.5 ms gap.

The 12.5 ms gap deserves further explanation. The `BatchGather` HLO emits a single GPU kernel where each thread computes all 9 gathers for its output element. This means each thread issues 9 non-contiguous loads from `field`, each at a different `(s', kx')` address. With 9 outstanding loads per thread, the register file must hold 9 × 16B = 144B of gathered data per thread, plus the index computation state. On A100, each SM has 65,536 32-bit registers (256 KB). With 2048 threads per SM, that's 32 registers per thread — exactly enough for 8 float64 values. The 9 concurrent gathers exceed the register budget, causing **register spilling to local memory** (which is actually HBM-backed). This converts the gather from a single-round HBM access to a multi-round spill+reload pattern, effectively doubling the memory traffic.

#### 2.1.3 Why `lax.scan` works in isolation but fails in the full solver

The v4 variant replaces the Python for-loop with `jax.lax.scan`:

```python
out, _ = jax.lax.scan(body, jnp.zeros_like(field), (s_shift, kx_shift, valid_shift, coeffs))
```

**In isolation** (benchmarked via `jax.jit(make_v4(...))`): XLA's loop unroller recognizes that the scan has a static trip count of 9 with no data-dependent exit conditions. It unrolls the 9 iterations into flat HLO — essentially recovering the same op sequence as the Python for-loop. The 1.03x speedup comes from slightly different scheduling of the unrolled ops (the scan's structured output avoids one redundant buffer allocation for the initial `jnp.zeros_like`).

**In the full solver** (`gksolve` at `solver.py:1192-1202`): The outer `jax.lax.scan` over `n_steps` wraps the entire `gkstep_single` → `_rhs` → `_linear_rhs_core` → `_apply_parallel` call chain. When XLA compiles this, the computation graph contains:

```
Outer While (n_steps iterations):
  └─ Body:
       └─ ... → _apply_parallel →
            Inner While (9 iterations):    ← lax.scan
              └─ carry: out (55.8 MB complex128)
```

XLA's loop unroll heuristic has a **cost threshold**: it will only unroll an inner loop if the total HLO instruction count of the unrolled body stays below a limit (typically ~1000 ops). Inside `gksolve`'s body, the `_rhs` function expands to thousands of HLO ops (all of _linear_rhs_core + nonlinear_term_iii + phi solve). The inner 9-iteration scan, even though small, is part of this massive body. XLA's optimizer conservatively refuses to unroll inner scans inside large loop bodies because unrolling increases the body size by 9x, potentially blowing compilation time and memory.

The result: the inner scan compiles to a real `While` HLO loop with:
- The `out` carry stored to HBM at each loop back-edge (55.8 MB write per iteration, 9 iterations = 502 MB extra writes)
- The gather for iteration `i+1` cannot begin until `out` from iteration `i` is committed to HBM (no pipelining)
- Total extra HBM traffic vs. the Python for-loop baseline: 8 × 2 × 55.8 MB = 894 MB (same amount as the "wasted" intermediate traffic in the baseline, but now it's **serialized** instead of pipelined)

**Measured**: 5.12 steps/s vs 6.11 steps/s baseline = -16.2%. Per step, that's 195 ms vs 164 ms = +31 ms. With 4 RHS calls/step, 2 `_apply_parallel` calls per RHS (term_par + term_vii), that's 16 `_apply_parallel` calls/step. Extra HBM traffic per call: ~894 MB serialized → at ~352 GB/s effective (measured baseline rate): 894 MB / 352 GB/s = 2.5 ms. Over 16 calls: 16 × 2.5 ms = 40 ms, which overpredicts the observed 31 ms (consistent with partial pipelining of some iterations by the GPU scheduler even in While-loop mode).

#### 2.1.4 Is the baseline already at the hardware limit?

**Measured**: 5.565 ms/call for `_apply_parallel` in isolation.

**Irreducible memory traffic** (minimum bytes that must be read/written regardless of code structure):
- `field` read: each of the 9 stencil points gathers from `field`. Due to the scattered `(s_map, kx_map)` pattern, there is no reuse across stencil points — worst case, each stencil point reads the entire `field`. 9 × 55.8 MB = 502 MB.
- `coeffs` read: 9 × 55.8 MB = 502 MB (contiguous slices, near-full bandwidth).
- `out` write: 55.8 MB (single output).
- Intermediate `out` R+W: 8 × 111.7 MB = 894 MB (baseline's accumulation buffers — currently irreducible without shared-memory fusion).
- **Total irreducible**: 502 + 502 + 55.8 = 1060 MB (ideal, with fused accumulation) or 1954 MB (with intermediate buffers).

**Bandwidth analysis**:
- Measured throughput: 1954 MB / 5.565 ms = **351 GB/s** effective.
- A100 peak: 2048 GB/s.
- **Efficiency: 17.1%** of peak bandwidth.

This 17% efficiency is entirely explained by the 3-index scattered gather pattern `field[:, :, s_map, kx_map, ky_idx]`:
- `ky_idx` is contiguous (identity mapping) → the innermost 32 elements (512 bytes = 4 cache lines of 128B) are contiguous per `(s', kx')` pair.
- `s_map` and `kx_map` scatter across the `(ns=16, nkx=85)` plane → each `(s', kx')` lookup jumps by `nky × 16B = 512 bytes` within a row, or `nkx × nky × 16B = 43,520 bytes` across rows.
- A warp of 32 threads processes 32 adjacent `ky` values at a fixed `(nv, nmu, s, kx)` → the 32 threads read from the **same** `(s', kx')` location (good coalescing for the ky dimension). But across different `(s, kx)` tiles processed by different thread blocks, the `s_map[s, kx, :]` and `kx_map[s, kx, :]` values vary — the L2 sees effectively random addresses at the 43.5 KB granularity of an s-row.
- With 16 × 85 = 1,360 distinct (s, kx) pairs and 9 stencil points, the gather issues up to 12,240 independent 512-byte requests. At 128-byte cache line granularity, that's up to 48,960 cache line requests. Assuming ~50% L2 hit rate (overlapping stencil lookups for adjacent s-values), the effective HBM request count is ~24,480 cache lines = 3.1 MB per gather. But the total data needed is 55.8 MB per gather — so the cache lines are spread across the full 55.8 MB working set.

The key insight: **17% effective bandwidth is the hardware-imposed ceiling for this scatter pattern**. No JAX-level restructuring can make a 3-index gather contiguous. The only path to higher effective bandwidth is loading `field` into shared memory once per tile and reusing it across all 9 stencil points — that requires a custom kernel (O7).

### 2.2 O1: The Analytical Error in OPTIM.md

OPTIM.md §4.1 correctly identified the 894 MB of intermediate accumulation traffic and §5.2.1 correctly diagnosed the Python loop as preventing "fusion across iterations." But the prescription was wrong because:

1. **"Fusion across iterations" is not the bottleneck.** The dominant cost is the 502 MB of scattered gathers from `field`, not the 894 MB of contiguous intermediate R+W. Eliminating intermediate buffers saves the cheaper traffic while introducing more expensive alternatives (BatchGather, transposes).

2. **OPTIM.md assumed XLA cannot pipeline across iterations.** It can. The gathers for iteration `i+1` read from `field` (unchanged), not from `out`. XLA schedules `Gather_{i+1}` concurrently with `FusedAdd_i` on different SMs. The "sequential dependency" on `out` only serializes the fused kernels, not the gathers.

3. **The roofline analysis (AI = 0.08) was correct but drew the wrong conclusion.** An AI of 0.08 means the kernel is purely memory-bound. The proposal assumed that reducing memory traffic (via fewer intermediate buffers) would proportionally improve performance. But the batched alternatives **increased** total traffic (502 MB intermediate + 1004 MB transpose) while also degrading the **quality** of memory access (scattered BatchGather vs. pipelined independent gathers).

### 2.3 O2: `_apply_vpar` — Baseline Access Pattern is Already Optimal

**Code reference**: `solver.py:770-778`

```python
for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):
    idx = jnp.clip(jnp.arange(nv) + s, 0, nv - 1)
    shifted = jnp.take(field, idx, axis=0)
    out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
```

#### 2.3.1 Why the baseline is fundamentally different from `_apply_parallel`

`jnp.take(field, idx, axis=0)` with `idx = [0, 0, 0, 1, 2, ..., 29]` (for shift s=-2) is a **nearly contiguous** access pattern:
- `field` has shape `(nv=32, nmu=8, ns=16, nkx=85, nky=32)`. The leading axis (vpar) has stride `nmu × ns × nkx × nky × 16B = 8 × 16 × 85 × 32 × 16 = 5.57 MB`.
- `jnp.take` along axis 0 with a monotonically increasing index reads `field[0], field[0], field[0], field[1], field[2], ...` — that's 30 unique rows plus 2 duplicated rows. Each "row" is 5.57 MB and is contiguous in memory.
- XLA compiles this to a Gather HLO with unit-stride access along the gathered dimension. The memory controller can prefetch effectively because the access pattern is predictable.

Compare with `_apply_parallel`'s `field[:, :, s_map, kx_map, ky_idx]`:
- Gathers along 3 dimensions simultaneously with data-dependent indices
- Each `(s, kx)` lookup jumps to an unpredictable memory location
- No prefetch opportunity

The vpar stencil's per-call measured time (1.755 ms) is 3.2x faster than the parallel stencil (5.565 ms), despite operating on the same array size. This ratio is consistent with the difference between near-contiguous (vpar) and scattered (parallel) gather patterns.

#### 2.3.2 Why pad+slice is slower (v1: 0.52x)

`jnp.pad(field, ((2, 2), (0, 0), (0, 0), (0, 0), (0, 0)))` allocates a `(36, 8, 16, 85, 32)` buffer — 62.8 MB.

Cost breakdown:
1. **Pad allocation + zero-fill**: XLA's `Pad` HLO first allocates the output (62.8 MB), zeros it, then copies `field` into the interior. Cost: 62.8 MB write (zeros) + 55.8 MB read (field) + 55.8 MB write (into padded) = 174 MB traffic.
2. **5 slices**: Each `padded[s:nv+s]` reads 55.8 MB. With 5 slices: 279 MB reads.
3. **5 scalar multiplies + accumulation**: Same as baseline: 87M FLOPs.
4. **Total traffic**: 174 + 279 + ~780 MB (accumulation) = ~1233 MB.

Baseline traffic: 5 × 55.8 MB (takes) + 4 × 111.7 MB (accum R+W) + 55.8 MB (output) = 782 MB.

The pad+slice version has **58% more memory traffic** than the baseline, primarily from the pad operation itself. Measured slowdown: 1.92x (3.364 ms / 1.755 ms), which overstates the 58% traffic increase because the pad also disrupts XLA's fusion: the `Pad` HLO is not fusible with the subsequent slices (it produces a new, larger buffer), so XLA inserts a materialization boundary.

#### 2.3.3 Why conv_general_dilated fails (v2: 0.82x, numerically wrong)

The convolution approach requires reshaping the complex128 field:

```python
f_flat = field.reshape(nv, batch).T.reshape(batch, nv, 1)
```

This `.T` transposes a `(nv=32, batch=348160)` matrix — swapping a stride-348160 axis with a stride-1 axis. For complex128 (16 bytes per element), this is a full 55.8 MB physical transpose at ~50% peak bandwidth. The subsequent `conv_general_dilated` runs two 1D convolutions (real + imag separately), each requiring the transposed layout. After convolution, another `.T` is needed to restore the original layout. Total: 2 transposes × 55.8 MB × 2 (R+W) = 223 MB of pure reshaping overhead — 28% of the baseline's total traffic, spent doing zero useful computation.

The numerical incorrectness comes from boundary handling: `conv_general_dilated` with `padding=[(2, 2)]` applies zero-padding (Dirichlet BCs), but the solver's `jnp.clip` implements value-clamping (Neumann-like BCs). The first and last 2 vpar points produce different values.

#### 2.3.4 Why `lax.scan` is catastrophic (v3: 0.36x)

Same nested-scan mechanism as O1 §2.1.3, but worse because the stencil has only 5 points (less work per iteration to amortize the While-loop overhead). Each of the 5 iterations forces a 55.8 MB `out` carry round-trip to HBM. With only 87M FLOPs total, the overhead is proportionally larger than for `_apply_parallel` (157M FLOPs).

Even in isolation (no outer scan), the v3 achieves only 0.36x — this is because the benchmark uses `jax.jit(v3)`, which wraps the scan in a JIT boundary. XLA's unroller has less context than when the scan is inside a larger computation, and the small trip count (5) with a 55.8 MB carry exceeds the cost-to-unroll threshold.

### 2.4 O6: `vmap` as an Implicit Memory Manager

**Code reference**: `solver.py:223-256`

```python
df_by_s = jnp.moveaxis(df, 2, 0)            # (ns, nv, nmu, nkx, nky)
bessel_by_s = jnp.moveaxis(bessel, 2, 0)    # same
nl = jnp.moveaxis(jax.vmap(_per_s)(df_by_s, phi, bessel_by_s, dum_s), 0, 2)
```

#### 2.4.1 How XLA handles vmap on GPU

When `jax.vmap(_per_s)` is applied over `ns=16` slices, JAX's batching rules transform each operation inside `_per_s` by adding a batch dimension. For `jnp.fft.irfft2`, this produces a batched cuFFT call with total batch = `ns × nvpar × nmu = 16 × 32 × 8 = 4096`. Crucially, XLA does **not** sequentially execute 16 independent copies of `_per_s`. It fuses them into a single batched computation.

However, XLA's **memory planning** pass treats the vmap differently from an explicit batch: it knows that the batch dimension was introduced by vmap and can potentially **reuse intermediate buffers across batch indices**. Whether XLA actually does this depends on the specific HLO ops — for operations where the vmap batch dimension maps cleanly to a cuFFT batch parameter (like irfft2), XLA emits a single kernel with the full batch. For operations where it doesn't (like the `pack_half_spectrum` scatter), XLA may process batch indices sequentially with buffer reuse.

#### 2.4.2 Why removing vmap causes a regression

The O6 optimization eliminated the vmap and passed the full `(nv, nmu, ns, nkx, nky)` array directly, treating `(ns)` as part of the inner dimensions to be batched over in the FFT.

The `_per_s` function creates these intermediate arrays per call:
- 4 gradient spectra: `grad_phi_y_k`, `grad_phi_x_k`, `grad_f_x_k`, `grad_f_y_k` — each `(nv, nmu, nkx, nky)` complex128 = 14.0 MB
- 4 packed spectra: each `(nv, nmu, mrad, mphiw3)` complex64 = 6.5 MB
- 4 real-space arrays: each `(nv, nmu, mrad, mphi)` float32 = 3.2 MB
- 1 bracket result: 3.2 MB float32
- 1 upcast result: 25.1 MB float64
- 1 rfft2 output: 27.1 MB complex128
- 1 unpacked result: 14.0 MB complex128

**With vmap** (per s-slice, XLA may reuse across slices): peak live intermediates ~90 MB
**Without vmap** (all 16 slices simultaneously): peak live intermediates ~90 × 16 = **1.44 GB**

The 1.44 GB of simultaneously live intermediates completely evicts the L2 cache (40 MB) and competes for HBM bandwidth with the FFT data itself. XLA's buffer assignment pass cannot reuse the gradient spectrum buffer from s-slice 0 for s-slice 1 because they are all computed simultaneously in the non-vmap version.

**The moveaxis cost is negligible**: `jnp.moveaxis(df, 2, 0)` on a 55.8 MB array costs ~55.8 MB read + write = 111.6 MB at 2 TB/s = 0.056 ms. Three moveaxis calls = 0.17 ms. The measured regression is 171.18 - 163.76 = 7.42 ms, which is 44x larger than the moveaxis savings.

#### 2.4.3 The analytical error

OPTIM.md §5.4.2 (lines 548-556) actually correctly analyzed the memory pressure issue: "the simultaneous live memory is 4 gradient spectra × 16 × 256 × 85 × 32 × 16B = 2.85 GB... exceeds B300 L2 by 50x." But §6.3.3 proposed eliminating the vmap anyway, suggesting that "verifying that XLA indeed fuses the vmap into a single batched cuFFT call is important." The verification showed that XLA **already batches the cuFFT correctly through vmap** — the optimization had no upside, only the downside of inflated memory pressure.

### 2.5 O3/O4: Neutral Because XLA Already Optimizes Both Forms Identically

#### 2.5.1 O3: Phi solve einsum

**Original**: `jnp.sum(phi_weight * df, axis=(0, 1, 2))`
**Changed**: `jnp.einsum('avmjkl,avmjkl->jkl', phi_weight, df)`

Both expressions lower to the same HLO:
```
Reduce(Multiply(phi_weight, df), init=0.0, dimensions={0, 1, 2})
```

XLA's `DotDecomposer` and `AlgebraicSimplifier` passes canonicalize `einsum` contractions of matching-shaped operands (same shape on all axes, contraction over a subset) into `Multiply + Reduce`. The `sum(a * b, axes)` pattern is already in this canonical form. Identical HLO → identical compiled code → identical performance.

The component cost (0.63 ms) is 0.4% of the step. Even a hypothetical 10x component speedup would yield 0.36% overall improvement — below the measurement noise floor (6.46 ± 0.87 steps/s has 13.5% relative uncertainty on Device 6).

#### 2.5.2 O4: RK4 expanded accumulation

**Original**: `prev_df + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)`
**Changed**: `prev_df + dt6*k1 + dt3*k2 + dt3*k3 + dt6*k4`

XLA's elementwise fusion pass (`InstructionFusion`) groups all scalar-multiply and add operations on the same-shaped operands into a single kernel. Both forms read 5 arrays (`prev_df`, `k1`, `k2`, `k3`, `k4`) and write 1 output. XLA produces a single fused kernel in both cases:

```
// Fused kernel (pseudocode):
for each element i:
    output[i] = prev_df[i] + dt6*k1[i] + dt3*k2[i] + dt3*k3[i] + dt6*k4[i]
```

The intermediate `(k1 + 2*k2 + 2*k3 + k4)` in the original form is never materialized — XLA's fusion eliminates it. This is a textbook case of XLA doing exactly what it was designed for.

The slight rel_l2 difference (1.08e-15 vs 1.15e-16) comes from non-associativity of IEEE 754 addition. The original form computes `((k1 + 2*k2) + 2*k3) + k4) * (dt/6)`, while the expanded form computes `(dt6*k1) + (dt3*k2) + (dt3*k3) + (dt6*k4)` with different intermediate rounding. Both are within 5 ULPs of the exact result — numerically irrelevant for a CFL-limited explicit time integrator.

### 2.6 Root-Cause Summary

| ID | Failure Mechanism | Was it Predictable a Priori? |
|----|-------------------|------------------------------|
| O1 (batch) | BatchGather intermediate 502 MB + transpose 1 GB + register spilling from 9 concurrent loads; destroys pipeline overlap that Python-loop gives for free | Partially. The intermediate cost was calculable. The register spilling and pipeline effects require GPU architecture knowledge. |
| O1 (scan) | Nested `lax.scan` inside `gksolve` `lax.scan` → XLA emits real `While` loop → carry stored to HBM each iteration, serializing gathers | Yes. Known XLA limitation. Should have been tested in the full solver context from the start. |
| O2 (all) | Baseline `jnp.take` on axis-0 is already near-contiguous; alternatives add allocation, transpose, or loop overhead | Yes. The access pattern analysis should have recognized axis-0 shifts as fundamentally different from 3-index gathers. |
| O6 | vmap provides implicit buffer reuse across batch indices; removing it inflates peak live memory 16x | Partially. OPTIM.md §5.4.2 identified the memory issue but §6.3.3 dismissed it. |
| O3 | `einsum` and `sum(a*b, axes)` lower to identical HLO | Yes. This is documented XLA behavior. |
| O4 | XLA elementwise fusion eliminates all temporaries in both forms | Yes. This is XLA's core competency. |
