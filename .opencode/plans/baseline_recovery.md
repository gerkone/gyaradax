# Baseline Recovery Plan

**Date:** 2026-04-04  
**Status:** CRITICAL — Baselines incompatible with current code

---

## Executive Summary

Baselines in `solver_components_benchmarks/baselines/` (generated 2026-03-30, commit `4bd9b71`) are **numerically incompatible** with current code due to bug fixes committed after baseline generation.

**Key finding:** `linear_rhs.npz` and `phi_solve.npz` contain **buggy output** from code that incorrectly hardcoded zonal mode at `ky=0` instead of detecting actual zonal mode at `iyzero = argmin(|krho|)`. Fixed in commit `7ff332d` (2026-04-01).

**Measured deviation:** `rel_l2 ≈ 3.97e-03` for `linear_rhs` (fails 1e-10 threshold by 7 orders of magnitude).

---

## Timeline

| Date | Commit | Description |
|------|--------|-------------|
| 2026-03-30 | `4bd9b71` | **Baseline generation** — baselines created with buggy phi code |
| 2026-04-01 | `7ff332d` | **Bug fix** — zonal mode detection corrected |
| 2026-04-04 | `ae871dc` | Current HEAD — extensive refactoring |

---

## Files Changed Since Baseline

| File | Pre-baseline | Current | Delta |
|------|-------------|---------|-------|
| `gyaradax/geometry.py` | 510 lines | 510 lines | Heavily modified |
| `gyaradax/integrals.py` | 381 lines | 381 lines | Heavily modified |
| `gyaradax/solver.py` | 1196 lines | 1196 lines | Heavily modified |
| `gyaradax/backends/_jax.py` | ~400 | 510 | +110 (backend refactor) |
| `gyaradax/backends/_cuda.py` | ~200 | 276 | +76 (backend refactor) |
| `gyaradax/backends/ops.py` | ~60 | 85 | +25 (backend refactor) |

---

## Numerical Differences

### 1. Phi Solve Bug (CRITICAL)

**Buggy code (pre-7ff332d):**
```python
matz = matz.at[..., 1:].set(0.0)  # Hardcoded ky=0
y_mask = y_mask.at[..., 0].set(1.0)
```

**Fixed code (post-7ff332d):**
```python
iyzero = jnp.argmin(jnp.abs(krho_flat))
has_zonal = jnp.where(jnp.abs(krho_flat[iyzero]) < 1e-10, 1.0, 0.0)
ky_is_zonal = jnp.arange(matz.shape[-1]) == iyzero
matz = matz * ky_is_zonal * has_zonal
y_mask = y_mask.at[..., iyzero].set(has_zonal)
```

**Impact:** For grids where `iyzero != 0`, phi solve produces wrong results at all non-zonal modes.

### 2. Geometry Refactoring

- Converted NumPy → JAX for differentiability
- Added s-alpha geometry model
- Vectorized branch correction in `_dzetadeps()`
- Numerically equivalent but not bit-identical

### 3. Solver Restructuring

- Fused stencil computation
- Precomputed phi weights
- Different memory layout
- Mathematically equivalent, different operation ordering

---

## Baseline Status

| File | Status | Reason |
|------|--------|--------|
| `linear_rhs.npz` | INVALID | Buggy phi + old precomputation |
| `phi_solve.npz` | INVALID | Buggy phi (hardcoded ky=0) |
| `nonlinear.npz` | Likely valid | Term III unaffected by phi bug |
| `apply_parallel.npz` | Likely valid | Stencil application |
| `apply_vpar.npz` | Likely valid | Velocity stencil |
| `pack_spectrum.npz` | Likely valid | FFT packing |
| `rk4_step.npz` | INVALID | Uses phi |

---

## Recovery Plan: Option A (RECOMMENDED)

**Goal:** Restore code to commit `4bd9b71^` (pre-baseline), keep ONLY backend refactoring.

### Step 1: Backup Current Work
```bash
git checkout -b backup/refactoring-work
```

### Step 2: Restore Pre-Baseline Files
```bash
git checkout 4bd9b71^ -- gyaradax/geometry.py
git checkout 4bd9b71^ -- gyaradax/integrals.py
git checkout 4bd9b71^ -- gyaradax/solver.py
```

### Step 3: Keep Backend Refactoring
Do NOT revert:
- `gyaradax/backends/_jax.py`
- `gyaradax/backends/_cuda.py`
- `gyaradax/backends/ops.py`

### Step 4: Regenerate Baselines
```bash
cd solver_components_benchmarks
python generate_baselines.py
```

### Step 5: Verify All Benchmarks
```bash
python run_all.py
# Expected: All rel_l2 < 1e-10
```

---

## Recovery Plan: Option B (Alternative)

**Goal:** Keep all current changes, regenerate baselines.

```bash
# Document current state (this file)
# Regenerate baselines
python solver_components_benchmarks/generate_baselines.py

# Commit with clear message
git add solver_components_benchmarks/baselines/
git commit -m "Regenerate baselines after phi bug fix and refactoring"
```

**Risk:** Loses ability to compare against original GKW validation trajectory.

---

## Verification Commands

After restoration:

```bash
# Test linear_rhs
python solver_components_benchmarks/bench_linear_rhs.py

# Test phi_solve
python solver_components_benchmarks/bench_phi_solve.py

# Test nonlinear
python solver_components_benchmarks/bench_nonlinear.py
```

**Expected:** All tests pass with `rel_l2 < 1e-10`.

---

## Commit Map

```
Pre-baseline (correct phi, old structure)
  ↓
4bd9b71 — baseline generation (buggy phi captured)
  ↓
7ff332d — phi bug fix (too late, baselines already wrong)
  ↓
ae871dc — current HEAD (refactored + fixed)
```

**Critical:** Phi bug fix came AFTER baseline generation.

---

## Files to Restore (Exact Commands)

```bash
# Save current work
git stash push -m "refactoring work" -- \
  gyaradax/geometry.py \
  gyaradax/integrals.py \
  gyaradax/solver.py

# Restore pre-baseline versions
git checkout 4bd9b71^ -- \
  gyaradax/geometry.py \
  gyaradax/integrals.py \
  gyaradax/solver.py

# Backend refactoring stays (already correct)
# git checkout HEAD -- gyaradax/backends/
```
