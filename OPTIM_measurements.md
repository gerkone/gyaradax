# OPTIM_measurements.md — Empirical Results

All runs: `configs/iteration_13.yaml`, adiabatic mode, grid `(32, 8, 16, 85, 32)`.

Benchmark command:
```bash
PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_volkmann \
  /system/apps/userenv/galletti/mhd/bin/python scripts/solver_benchmark.py \
  configs/iteration_13.yaml --device <N> [--reference baseline_adiabatic.npz]
```

---

## Baselines

### Device 1 — adiabatic

| steps/s | ms/step |
|---------|---------|
| 6.11 ± 0.00 | 163.76 |

### Device 6 — adiabatic

| Component | ms |
|-----------|----|
| phi solve | 0.63 |
| linear rhs | 66.61 |
| nonlinear rhs | 50.12 |
| **steps/s** | **6.46 ± 0.87** |
| ms/step | 154.74 |

### Device 6 — kinetic

| Component | ms |
|-----------|----|
| phi solve | 0.71 |
| linear rhs | 57.94 |
| nonlinear rhs | 54.69 |
| **steps/s** | **6.88 ± 0.01** |
| ms/step | 145.36 |

---

## O1 — Fused `_apply_parallel` (`solver.py:758`)

**Goal:** Replace 9-iteration Python for-loop with a single fused gather+reduce,
eliminating 8 intermediate `out` materializations (~894 MB wasted traffic/call).

### Attempt 1 — batch-gather + moveaxis (full solver, device 1)

```python
gathered = field[:, :, pre["s_shift"], pre["kx_shift"], ky_idx]  # (nv,nmu,9,ns,nkx,nky)
shifted_stack = jnp.moveaxis(gathered, 2, 0)                      # (9,nv,nmu,ns,nkx,nky)
shifted_stack = jnp.where(pre["valid_shift"][:, None, None], shifted_stack, 0.0)
return jnp.sum(coeffs * shifted_stack, axis=0)
```

| Version | steps/s | ms/step | rel_l2 |
|---------|---------|---------|--------|
| Baseline | 6.11 | 163.76 | — |
| batch-gather + moveaxis | 5.95 | 168.13 | 1.15e-16 ✓ |

**−2.6%. Reverted.** `moveaxis` on 502 MB tensor forces a physical copy. XLA BatchGather
HLO for the stacked `(9, ns, nkx, nky)` index is less efficient than 9 independent 56 MB gathers.

### Attempt 2 — microbenchmark of all JAX variants (`scripts/bench_apply_parallel.py`, device 1)

Isolated `_apply_parallel` call, 20 trials:

| Variant | ms/call | speedup |
|---------|---------|---------|
| v0 Python loop (baseline) | 5.565 ± 0.012 | 1.00× |
| v1 batch-gather + moveaxis | 19.091 ± 0.036 | 0.29× |
| v2 jax.vmap over 9 stencils | 19.098 ± 0.028 | 0.29× |
| v3 batch-gather + einsum | 19.088 ± 0.037 | 0.29× |
| v4 lax.scan accumulation | 5.393 ± 0.020 | 1.03× |

v1/v2/v3 are all the same HLO underneath — the 502 MB batched intermediate tensor is the bottleneck regardless of how it's expressed in JAX.

v4 looks promising in isolation (+3%). Full solver result:

| Variant | steps/s | vs baseline |
|---------|---------|-------------|
| Python loop | 6.11 | 1.00× |
| lax.scan | 5.12 | **0.84×** |

**−16% in full solver. Reverted.** The `lax.scan` body is nested inside `gksolve`'s outer
`jax.lax.scan`. XLA cannot unroll the inner 9-iteration scan, emitting a real `While` loop
with a global-memory round-trip on `out` per iteration. The Python `for` loop unrolls at
trace time — XLA sees all 9 ops simultaneously and can pipeline/fuse across them.

Note: `lax.scan` also required fixing the carry initializer from `jnp.zeros_like(field)`
to `jnp.zeros(jnp.broadcast_shapes(field.shape, coeffs.shape[1:]), dtype=...)` because
`gyro_phi` has shape `(1, nmu, ...)` while `s_total_t7` coeffs are `(nv, nmu, ...)` —
the Python loop silently widens `out` via broadcast, but scan requires a static carry type.

### Conclusion

**O1 is not achievable at the JAX level.** The Python loop is the XLA-optimal form:
unrolling exposes all 9 gathers to the XLA scheduler simultaneously. A real speedup
requires **O7** (custom Triton/CUDA kernel) with shared-memory halo storage across
all 9 stencil points, eliminating global-memory round-trips entirely.

---

## O3 — Phi solve einsum (`integrals.py:224`)

**Change:** `jnp.sum(phi_weight * df, axis=(0, 1, 2))` → `jnp.einsum('avmjkl,avmjkl->jkl', phi_weight, df)`

**Goal:** Let XLA fuse the multiply-reduce, avoiding materialising the full `(nsp, nvpar, nmu, ns, nkx, nky)` intermediate (~112 MB).

| Version | steps/s | ms/step | rel_l2 |
|---------|---------|---------|--------|
| Baseline | 6.11 | 163.76 | — |
| O3 einsum | 6.10 | 163.89 | 1.15e-16 ✓ |

**Neutral — kept.** Phi solve is ~0.6 ms out of 164 ms/step (~0.4% of step time), so any
speedup there is invisible at the overall steps/s level. Change is retained since it is
semantically cleaner and avoids the explicit broadcast with no downside.

---

## O4 — RK4 accumulation restructure (`solver.py:1111`)

**Change:** `prev_df + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)` →
`prev_df + dt6*k1 + dt3*k2 + dt3*k3 + dt6*k4`

**Goal:** Let XLA read k1..k4 and prev_df as direct additive terms in one fused
kernel rather than first building the weighted sum as an intermediate.

| Version | steps/s | ms/step | rel_l2 |
|---------|---------|---------|--------|
| Baseline | 6.11 | 163.76 | — |
| O4 expanded accumulation | 6.12 | 163.37 | 1.08e-15 ✓ |

**Neutral — kept.** RK4 accumulation is ~1.7 GB R+W out of ~100 GB/step; XLA already
fused the elementwise ops in the original form. The slightly higher rel_l2 (vs 1.15e-16)
is expected from reordered FP ops — still <5 ULPs of float64. Cleaner structure retained.

---

## O6 — Batched FFT restructure (`solver.py:nonlinear_term_iii`)

**Change:** Eliminated `vmap(_per_s)` over ns=16 + 3 `moveaxis` calls. Replaced with direct
batch over full `(nv,nmu,ns,nkx,nky)` arrays, letting `irfft2` batch over all 4096 transforms
in one call rather than 16 vmap iterations of 256.

| Version | steps/s | ms/step | rel_l2 |
|---------|---------|---------|--------|
| Baseline | 6.11 | 163.76 | — |
| O6 batched FFT | 5.84 | 171.18 | 1.08e-15 ✓ |

**−4.4%. Reverted.** JAX's vmap already batches cuFFT calls correctly through its batching
rules — the 3 moveaxis ops were cheap. The real cost: eliminating vmap makes all 4 gradient
spectra and real-space arrays simultaneously 16× larger `(nv,nmu,ns,...)` vs vmap's per-slice
`(nv,nmu,...)`, dramatically increasing peak HBM pressure. The vmap lets XLA process one
s-slice at a time, reusing intermediate buffers across slices.

---

## O2 — Fused `_apply_vpar` (`solver.py:770`)

**Goal:** Replace 5-iteration Python for-loop with a fused convolution or pad+slice,
eliminating 4 intermediate `out` accumulation buffers.

### Microbenchmark (`scripts/bench_apply_vpar.py`, isolated call, device 1)

| Variant | ms/call | vs baseline |
|---------|---------|-------------|
| v0 baseline (take + clip + valid) | 1.755 ± 0.020 | 1.00× |
| v1 pad + slice (no Gather HLO) | 3.364 ± 0.085 | **0.52×** |
| v2 conv_general_dilated | 2.138 ± 0.031 (wrong) | 0.82× |
| v3 lax.scan | 4.890 ± 0.038 | **0.36×** |

**Not implemented — baseline is optimal.** Same conclusion as O1: the Python
`for` loop unrolls at trace time, giving XLA all 5 ops simultaneously for fusion.
- v1 (pad+slice): allocating the `(nv+4,...)` padded buffer adds overhead; XLA
  was already treating `jnp.take` with a sequential idx as near-strided access.
- v2 (conv): numerically incorrect (complex/real dim confusion) and slower.
- v3 (lax.scan): 2.8× slower — same nested-scan problem as O1 revisit.
