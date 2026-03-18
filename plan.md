# Plan: Post-Mortem Analysis of Failed JAX/XLA Optimizations

## Context

OPTIM.md predicted a 1.3-1.5x aggregate speedup from 5 pure-JAX optimizations (O1-O4, O6) targeting the `gyaradax` gyrokinetic solver. OPTIM_measurements.md shows every optimization was either neutral or a regression. The worst offender (O1: fused `_apply_parallel`) was predicted at 2-3x component speedup but measured at 0.29x (3.4x slower) for batch variants, and -16% in full solver for `lax.scan`. The user wants a rigorous root-cause analysis determining whether further JAX-level optimization is physically possible or if XLA has reached the hardware roofline.

## Deliverable

A single report `Post_Mortem_JAX_Optimizations.md` written in 4 incremental sections, each requiring user approval before proceeding to the next.

---

## Step 1: Discrepancy Mapping

**Goal**: Quantitative table mapping each OPTIM.md prediction to its measured outcome.

**Data sources** (read-only):
- OPTIM.md Section 7.1 (Priority Matrix, predicted speedups)
- OPTIM_measurements.md (all empirical results)

**Output**: Table with columns: Optimization ID | Target Component | Predicted Component Speedup | Predicted Overall Speedup | Actual Result | Error Factor | Prediction Category (direction-wrong / magnitude-wrong / correct-neutral)

**Ranking** worst offenders by: (component weight in step time) x (magnitude of prediction error)

**Key narrative**: Aggregate Phase 1 prediction was 1.3-1.5x. Actual: 1.00x. Zero value delivered from JAX-level refactoring.

---

## Step 2: Root-Cause Analysis

For each failed optimization, trace the exact compiler/hardware mechanism.

### O1: `_apply_parallel` — 3 distinct failure modes
1. **BatchGather vs 9 independent Gathers (v1/v2/v3 = 0.29x)**: Batched index `(9, ns, nkx, nky)` creates 502 MB intermediate. 9 independent gathers pipeline across SMs with partial L2 reuse; BatchGather forces all 9 stencil reads per thread block, exploding working set past 40 MB L2. The `moveaxis` on the 502 MB result is a physical copy (stencil dim not contiguous in row-major).
2. **Nested lax.scan penalty (v4 = -16% in solver)**: Inner `lax.scan(range=9)` inside outer `jax.lax.scan(n_steps)` in `gksolve` → XLA emits real `While` HLO with HBM-materialized carry. Python `for` unrolls at trace time, exposing all 9 ops for concurrent scheduling.
3. **Baseline already near hardware limit**: Measured 5.565 ms/call. Irreducible traffic ~1.06 GB at ~50% effective BW for scattered reads → theoretical minimum ~1.06 ms. Baseline at ~19% of peak scattered BW — consistent with 3-index gather on A100.

### O2: `_apply_vpar` — baseline access pattern is already optimal
- `jnp.take(field, idx, axis=0)` with sequential idx is near-contiguous (leading axis shift). XLA compiles this to strided loads.
- Pad+slice (0.52x): allocates padded buffer + full copy + 5 non-fusible slices.
- conv_general_dilated (0.82x): requires complex→real decomposition reshape (physical transpose) and has wrong boundary conditions.
- lax.scan (0.36x): same nested-scan penalty.

### O6: vmap as implicit memory manager
- `jax.vmap(_per_s)` over ns=16 reuses HBM buffers across slices (sequential execution, same buffer recycled). Removing vmap materializes 16 slices' intermediates simultaneously: 4 gradient spectra × 16 × 256 × 85 × 32 × 16B = 2.85 GB live. JAX vmap already produces optimal batched cuFFT calls — removing vmap only inflated peak memory.

### O3/O4: Neutral because negligible + XLA already optimal
- O3: phi solve = 0.63 ms (0.4% of step). `einsum` and `sum(weight*df)` generate identical HLO.
- O4: XLA algebraic simplifier fuses elementwise ops regardless of Python-level grouping. Both forms produce same kernel.

**Additional investigation during execution**:
- Read `scripts/bench_apply_parallel.py` and `scripts/bench_apply_vpar.py` for exact variant implementations
- Read `solver_components_benchmarks/` if present for roofline data
- Compute irreducible bandwidth per step and compare to measured time

---

## Step 3: Missing Element Investigation

Systematic search for overlooked JAX-level opportunities:

1. **Data layout reordering**: Current `(nsp, nvpar, nmu, ns, nkx, nky)`. The scatter in `_apply_parallel` is over (s, kx) — these are dimensions 3,4 (stride 2720, 32). Moving them outermost would make inner dimensions (nvpar, nmu, nky) contiguous per gather, improving warp coalescing. But this degrades `_apply_vpar` (needs nvpar leading) and FFT (needs kx, ky trailing). Layout is a zero-sum game for this operator mix.

2. **Pallas kernels**: `jax.experimental.pallas` as intermediate between pure JAX and CUDA. Better target for `_apply_vpar` (regular stride-1 pattern) than `_apply_parallel` (inherently scattered). Could fuse the 5-point stencil into a single tiled kernel with register accumulation.

3. **Hidden materializations**: Check if `jnp.where(valid[None,None,:,:,:], gathered, 0.0)` forces broadcast materialization of the bool mask. Check if `bessel * phi_b` (broadcast on nvpar, nmu) fuses with downstream `_apply_parallel(gyro_phi, s_total_t7)`.

4. **`jax.custom_call` / XLA custom kernels**: Direct HLO injection without full CUDA. Limited documentation but theoretically allows inserting optimized kernels into XLA graphs.

5. **What won't help**: `custom_vjp` (solver is forward-only, no autodiff), `ensure_compile_time_eval` (indices already static), `named_scope` (annotation only), single-GPU sharding constraints.

---

## Step 4: Final Verdict + Pivot Strategy

### Saturation argument (3 pillars):
1. **Roofline**: Every component is memory-bandwidth bound (AI 0.08–1.85, all below A100 ridge point 4.85).
2. **XLA optimality**: Python loop unrolling IS the optimal trace strategy. All alternatives produce same-or-worse HLO.
3. **Access pattern ceiling**: 3-index scatter in `_apply_parallel` hits ~20% of peak BW — hardware limit for non-contiguous reads. No JAX restructuring converts scattered reads to contiguous.

### Irreducible bandwidth estimate:
- Per step minimum: ~33 GB (4 RHS × ~2.4 GB linear + 4 × ~5.3 GB NL + ~2 GB overhead)
- At 2 TB/s: theoretical minimum 16.5 ms
- Measured: 154.74 ms → 9.4x overhead
- Sources: intermediate materializations, scatter BW penalty, FFT intermediates
- Custom kernels could cut 2-3x; eliminating more requires algorithm change

### Pivot recommendations (prioritized):
1. **O7: Custom Triton kernel for `_apply_parallel`** — shared-memory halo, single write. 2-4x on 40.6% component → 1.2-1.5x overall.
2. **O10: IMEX for kinetic electrons** — eliminate electron CFL constraint. 5-20x on kinetic cases.
3. **O5: FP32 rfft2** — low effort, saves ~6.4 GB/step cast traffic. Validate growth rates.
4. **Multi-GPU species sharding** — embarrassingly parallel except phi solve.

---

## Files to Read/Reference

| File | Purpose |
|------|---------|
| `gyaradax/solver.py:758-768` | `_apply_parallel` baseline |
| `gyaradax/solver.py:770-778` | `_apply_vpar` baseline |
| `gyaradax/solver.py:205-257` | `nonlinear_term_iii` + vmap structure |
| `gyaradax/solver.py:1079-1141` | `gkstep_single` RK4 |
| `gyaradax/solver.py:1144-1211` | `gksolve` outer scan |
| `gyaradax/solver.py:409-457` | `_fuse_stencils` precomputation |
| `gyaradax/integrals.py:209-225` | `calculate_phi_kinetic` |
| `scripts/bench_apply_parallel.py` | O1 variant implementations |
| `scripts/bench_apply_vpar.py` | O2 variant implementations |
| `OPTIM.md` Section 7.1 | Predicted speedups |
| `OPTIM_measurements.md` | All empirical results |

## Verification

- Cross-check every claim against measured data in OPTIM_measurements.md
- Ensure no prediction is stated without its empirical counterpart
- Verify irreducible bandwidth calculation against array sizes in OPTIM.md Section 3.2
- Final report must be self-contained (reader needs no other documents)
