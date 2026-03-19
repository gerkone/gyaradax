# solver_components_benchmarks — Implementation Plan

## Goal

A directory of standalone, reproducible benchmarks for every solver component
analysed in OPTIM.md, each covering:
- **Performance:** ms/call, achieved memory bandwidth (GB/s vs A100 peak 2 TB/s)
- **Numerical accuracy:** rel_l2 vs saved baseline output
- **Roofline context:** theoretical AI, achieved AI, % of BW-limited roofline

Components (from OPTIM.md §4):
| ID | Component | Source location |
|----|-----------|----------------|
| C1 | `_apply_parallel` | `solver.py:758` (closure inside `_linear_rhs_core`) |
| C2 | `_apply_vpar` | `solver.py:770` (closure inside `_linear_rhs_core`) |
| C3 | `_linear_rhs_core` (via `_compute_linear_rhs`) | `solver.py` |
| C4 | `nonlinear_term_iii` | `solver.py:205` |
| C5 | `calculate_phi_kinetic` (via `_compute_phi`) | `integrals.py:209` |
| C6 | `pack_half_spectrum` / `unpack_half_spectrum` | `solver.py:192,201` |

---

## Directory layout

```
solver_components_benchmarks/
  common.py                  # load_setup(), BenchTimer, roofline_report()
  generate_baselines.py      # run once → writes baselines/*.npz
  baselines/                 # saved (inputs, expected_output) per component
  bench_apply_parallel.py    # C1
  bench_apply_vpar.py        # C2
  bench_linear_rhs.py        # C3
  bench_nonlinear.py         # C4
  bench_phi_solve.py         # C5
  bench_pack_spectrum.py     # C6
  run_all.py                 # runs all, prints summary table
```

---

## Tasks

### Phase 0 — Setup & common infrastructure

- [ ] Create `solver_components_benchmarks/` directory and `baselines/` subdir
- [ ] Write `common.py`:
  - `load_setup(config_path, device)` — loads real df + geom + params + pre from
    iteration_13.yaml + K01 checkpoint (same logic as solver_benchmark.py lines 73-134)
    Returns `(df, phi, geom, params, pre)` with df shape `(nv,nmu,ns,nkx,nky)` (5D adiabatic)
  - `BenchTimer(fn_jit, n_warmup=3, n_trials=10)` — warms up then times, returns
    `(mean_ms, std_ms)` using `block_until_ready`
  - `roofline_report(label, mean_ms, flops, bytes_rw)` — prints ms, GB/s achieved,
    theoretical BW-limited roofline (AI × 2000 GB/s for A100), % utilisation
  - `check_accuracy(out, baseline_path, key)` — loads npz, computes rel_l2, prints pass/fail

### Phase 1 — Baseline generator

- [ ] Write `generate_baselines.py`:
  - Calls `load_setup("configs/iteration_13.yaml", device=1)`
  - For each component, runs it once with real data, saves `{inputs, output}` to
    `baselines/<component>.npz`
  - Components that are closures (C1, C2) are extracted by running a thin wrapper
    that calls `_linear_rhs_core` and captures intermediates via explicit re-implementation
    of the same closure logic (matching exactly what's in solver.py)
  - Run this script once; the baselines are committed to the repo

### Phase 2 — Individual benchmark files

Each file follows this structure:
1. Parse `--device` before any JAX import, set `CUDA_VISIBLE_DEVICES`
2. Enable x64, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
3. Call `load_setup()` from common.py
4. Build the JIT-compiled function under test
5. Run `BenchTimer` → report timing
6. Run `check_accuracy` vs baseline
7. Run `roofline_report` with OPTIM.md FLOP/byte numbers

#### bench_apply_parallel.py (C1)
- Extract inputs from `pre`: `s_shift`, `kx_shift`, `valid_shift`, `s_total_upar`
- Use real `df[0]` (first-species 5D slice) as `field`
- JIT: `_apply_parallel` re-implemented inline (same logic as solver.py:758-768)
- Baseline key: `apply_parallel_output`
- OPTIM.md figures: 157M FLOPs/call, ~1.96 GB R+W → AI = 0.08 FLOP/byte

#### bench_apply_vpar.py (C2)
- Inputs: `df[0]` (5D field), `stencils.VPAR_D1`, `stencils.VPAR_D4`
- JIT: `_apply_vpar` re-implemented inline (same logic as solver.py:770-778)
- Benchmark both VPAR_D1 and VPAR_D4 coefficients
- Baseline keys: `apply_vpar_d1_output`, `apply_vpar_d4_output`
- OPTIM.md figures: 87M FLOPs, ~782 MB R+W → AI = 0.11 FLOP/byte

#### bench_linear_rhs.py (C3)
- Import `_compute_linear_rhs` directly from `gyaradax.solver`
- Inputs: `df`, `phi`, `geom`, `params`, `pre`
- Baseline key: `linear_rhs_output`
- OPTIM.md figures: ~635M FLOPs/species, ~7.3 GB R+W → AI = 0.087 FLOP/byte

#### bench_nonlinear.py (C4)
- Import `nonlinear_term_iii` directly from `gyaradax.solver`
- Inputs: `df[0]`, `phi`, `geom`, `pre`  (5D, single species)
- Baseline key: `nonlinear_output`
- OPTIM.md figures: ~9.8B FLOPs/species, ~5.3 GB R+W → AI = 1.85 FLOP/byte

#### bench_phi_solve.py (C5)
- Import `_compute_phi` from `gyaradax.solver` (wraps `calculate_phi_kinetic`)
- Inputs: `df`, `geom`, `params`, `pre`
- Baseline key: `phi_output`
- OPTIM.md figures: ~56M FLOPs, ~119 MB R+W → AI = 0.47 FLOP/byte

#### bench_pack_spectrum.py (C6)
- Import `pack_half_spectrum`, `unpack_half_spectrum` from `gyaradax.solver`
- Inputs: spectral array `(nv,nmu,nkx,nky)` extracted from real phi/df, `pre["nl_jind"]`
- Benchmark both pack and unpack
- Baseline keys: `pack_output`, `unpack_output`
- OPTIM.md: pure memory movement, 0 FLOPs → just report GB/s

### Phase 3 — run_all.py

- [ ] Write `run_all.py`:
  - Imports each bench module's `run(device)` function
  - Runs all sequentially on the specified device
  - Prints a summary table:
    ```
    Component           ms/call   GB/s    AI       % roofline  rel_l2
    _apply_parallel     X.XX      X.X     0.08     XX%         X.Xe-16
    _apply_vpar         X.XX      X.X     0.11     XX%         X.Xe-16
    _linear_rhs_core    X.XX      X.X     0.087    XX%         X.Xe-16
    nonlinear_term_iii  X.XX      X.X     1.85     XX%         X.Xe-16
    phi_solve           X.XX      X.X     0.47     XX%         X.Xe-16
    pack_spectrum       X.XX      X.X     0.00     XX%         X.Xe-16
    ```

---

## Implementation order

1. `common.py`                        ✅
2. `generate_baselines.py` + run it   ✅
3. `bench_apply_parallel.py`          ✅
4. `bench_apply_vpar.py`              ✅
5. `bench_phi_solve.py`               ✅
6. `bench_linear_rhs.py`              ✅
7. `bench_nonlinear.py`               ✅
8. `bench_pack_spectrum.py`           ✅
9. `run_all.py` + smoke-test          ✅

## Results (device 1, iteration_13.yaml, adiabatic)

| Component | ms/call | Achieved GB/s | % A100 BW | rel_l2 |
|-----------|---------|---------------|-----------|--------|
| `_apply_parallel` | 6.18 | 317 | 16% | 0 |
| `_apply_vpar` | 1.77 | 442 | 22% | 0 |
| `_compute_linear_rhs` | **4.20** | **1737** | **87%** | 0 |
| `nonlinear_term_iii` (mp) | 26.9 | 197 | **10%** | 0 |
| `_compute_phi` | 0.36 | 333–943 | 17–47% | 0 |
| `pack_half_spectrum` | 1.92 | 34 | **2%** | 0 |
| `unpack_half_spectrum` | 2.00 | 19 | **1%** | 0 |

Key findings:
- `_compute_linear_rhs` at 87% BW explains why O1/O2 couldn't help — XLA already fuses the whole operator near roofline
- `nonlinear_term_iii` at 10% BW and `pack/unpack` at 1–2% are the real targets
- `pack_half_spectrum` scatter write is 34 GB/s vs 2 TB/s peak — 59× below roofline; this feeds every FFT

---

## Notes

- All benchmarks default to `--device 1` (matching OPTIM_measurements.md baselines)
- Baselines are generated from the current solver state (post O3+O4 edits)
- C1 and C2 closures must replicate the exact logic from solver.py — any drift
  invalidates the comparison; add an assertion that the closure matches _compute_linear_rhs
  output on the same input
- For C4 (nonlinear), benchmark both `mixed_precision=True` and `mixed_precision=False`
  since O5 (FP32 rfft2) will target this path
