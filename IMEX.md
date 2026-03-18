# IMEX Time Integration Analysis

This document designs the implicit-explicit (IMEX) time integration scheme for
gyrokinetics-JAX, motivated by `OPTIM.md §6.3.2` (original proposal) and
`Post_Mortem_JAX_Optimizations.md §4.3` (priority ranking after pure-JAX
optimization was exhausted).

The central questions:
1. How to build the implicit solver for kinetic-electron parallel streaming.
2. **What does IMEX mean for the adiabatic case?**
3. What are the data layout implications.

---

## 1. Problem Statement: The CFL Bottleneck

The explicit RK4 time integrator in `solver.py:1079-1113` is subject to CFL
constraints from three sources, computed in `estimate_timestep` (line 342):

| CFL source | Formula | Adiabatic (iter_13) | Kinetic (991_double_rlt) |
|:-----------|:--------|:-------------------:|:------------------------:|
| Parallel streaming | `dt_par = 0.5 × sgr_dist / max\|upar\|` | ~0.01 | ~0.00017 (electron-limited) |
| Trapping | `dt_trap = 0.5 × dvp / max\|utrap\|` | ~0.09 | ~0.001 (electron-limited) |
| Nonlinear ExB | `dt_nl = 0.95 × 2 / max\|∇φ\|` | ~0.01–∞ (amplitude-dependent) | ~0.01–∞ |

The characteristic speeds are (`solver.py:569-570`):

```
upar  = -ffun × vthrat × vp        # parallel streaming
utrap =  vthrat × mu × bn × gfun   # magnetic trapping
```

For kinetic electrons, `vthrat_e = 60.634` (`configs/kinetic_991_double_rlt.yaml:24`),
making electron streaming **60.6× faster** than ions. This forces `dt_par_e` to
be ~60× smaller than `dt_par_i`, dominating the combined CFL.

**Numerical verification** (adiabatic, `configs/iteration_13.yaml`):
```
upar_max = ffun_max × vthrat × vp_max = ~1.0 × 1.0 × 3.0 = 3.0
dt_par   = 0.5 × 0.0625 / 3.0 = 0.0104  ← matches configured dt = 0.01
```

**Numerical verification** (kinetic electrons):
```
upar_e_max = ffun_max × 60.634 × 3.0 ≈ 181.9
dt_par_e   = 0.5 × 0.0625 / 181.9 ≈ 0.000172
```

The kinetic config sets `dt = 0.004` with `adaptive_dt: true` — the adaptive CFL
reduces the actual step to ~0.00017, a ~23× reduction from the configured maximum.

---

## 2. What IMEX Means for Each Case

### 2.1 Kinetic Electrons — The Big Win

Remove electron parallel streaming from the CFL. The next constraint is:

| Remaining CFL | Estimate | Ratio to current dt |
|:-------------|:---------|:-------------------:|
| Electron trapping | ~0.001 | ~6× |
| Ion parallel streaming | ~0.01 | ~58× |
| Nonlinear ExB (saturated) | ~0.01 | ~58× |

**Key subtlety**: electron trapping is also scaled by `vthrat_e = 60.634`,
so `dt_trap_e ≈ 0.5 × 0.1875 / (60.634 × mu_max × bn_max × gfun_max)`.
With Gauss-Laguerre nodes for `nmu = 8`, `mu_max ≈ 22.9`, and for circular
geometry `gfun_max ≈ eps/q ≈ 0.04`:

```
utrap_e_max ≈ 60.634 × 22.9 × 1.2 × 0.04 ≈ 66.7
dt_trap_e   ≈ 0.5 × 0.1875 / 66.7 ≈ 0.0014
```

So with **parallel streaming only** treated implicitly:
- dt increases from ~0.00017 to ~0.0014 → **~8× larger dt**
- Speedup after implicit solve overhead: **~7×**

With **both parallel streaming and trapping** treated implicitly (§8):
- dt increases to ~0.01 (ion CFL / NL CFL) → **~25–60× larger dt**
- Speedup: **~20–50×**

The difference between 7× and 50× makes the case for treating trapping
implicitly alongside streaming. See §8.

### 2.2 Adiabatic Ions — The Critical Question

For the adiabatic case (`vthrat = 1.0`, `dt = 0.01`), the ion parallel
streaming CFL is **approximately binding** — the configured dt already
matches `dt_par_i ≈ 0.01`.

Remove ion streaming from the CFL:

| Phase | Next binding CFL | dt estimate | Speedup |
|:------|:-----------------|:------------|:-------:|
| Linear growth | Trapping (~0.09) | ~0.05–0.08 | **5–8×** |
| Nonlinear saturation | NL ExB (~0.01–0.05) | ~0.01–0.03 | **1–3×** |
| Late turbulence | NL ExB + trapping | ~0.01–0.02 | **1–2×** |

**Assessment**: IMEX provides a **significant speedup during the linear phase**
(where NL CFL is inactive and trapping is soft) but only a **marginal
improvement in saturated turbulence** where the nonlinear CFL becomes binding.

For simulations dominated by the linear growth phase (e.g., convergence studies,
growth rate scans), the 5–8× improvement is valuable. For long nonlinear
simulations, the payoff is modest and must be weighed against implementation
complexity. The implicit solve adds a batched banded system to every step.

**Bottom line**: IMEX is transformative for kinetic electrons, beneficial for
the linear phase of adiabatic simulations, and marginal for saturated adiabatic
turbulence.

---

## 3. The Implicit Linear System

### 3.1 Operator Structure

The parallel streaming term is computed by `_apply_parallel` (`solver.py:758-768`):

```python
def _apply_parallel(field, coeffs):
    out = jnp.zeros_like(field)
    for i in range(9):
        s_map = pre["s_shift"][i]
        kx_map = pre["kx_shift"][i]
        shifted = field[:, :, s_map, kx_map, ky_idx]
        out = out + coeffs[i] * shifted
    return out
```

The fused coefficients `s_total_upar` are built in `_fuse_stencils`
(`solver.py:409-457`):

```python
s_total_upar = (
    upar * s_d1_upar + disp_par * abs_par * s_d4_upar
) / sgr_dist
```

where `s_d1_upar` and `s_d4_upar` are the upwinded D1/D4 stencil arrays
(selected by sign of `upar`), normalized by 12.0 in `_parallel_coefficients`
(`solver.py:369-371`).

For the implicit system, we solve:

```
[I - dt × A_par] f^{n+1} = f^n + dt × R_explicit(f^n)
```

where `A_par` is the linear operator encoded by `s_total_upar`.

### 3.2 Stencil Bandwidth

The stencils (`stencils.py`) store 9 entries per s-point (shifts -4 to +4),
but **only shifts -2 to +2 have non-zero coefficients**:

| Stencil | Interior row | Non-zero shifts | Source |
|:--------|:-------------|:----------------|:-------|
| D1_IPW_POS | `[0,0, 1,-8, 0, 8,-1, 0,0]` | -2,-1,+1,+2 | `stencils.py:28` |
| D1_IPW_NEG | `[0,0, 1,-8, 0, 8,-1, 0,0]` | -2,-1,+1,+2 | `stencils.py:42` |
| D4_IPW_POS | `[0,0,-1, 4,-6, 4,-1, 0,0]` | -2,-1, 0,+1,+2 | `stencils.py:54` |
| D4_IPW_NEG | `[0,0,-1, 4,-6, 4,-1, 0,0]` | -2,-1, 0,+1,+2 | `stencils.py:65` |

Boundary stencils (rows 0, 1, 3, 4 in each table) use **even narrower**
subsets, all within ±2. The implicit matrix `[I - dt × A]` is therefore
**pentadiagonal** (bandwidth (2, 2)).

### 3.3 Boundary Conditions and kx-Chains

The parallel shift maps are built in `geometry.py:312-362`
(`_build_parallel_shift_maps`). Two cases:

**Zonal modes (ky = 0)**: periodic in s, independent per kx. Each kx gives
a **periodic pentadiagonal** system of size `ns = 16`. Solvable via:
- Sherman-Morrison-Woodbury (rank-4 correction to pentadiagonal LU), or
- Dense solve (ns = 16 is small enough: O(16³) = 4096 FLOPs)

**Non-zonal modes (ky ≠ 0)**: magnetic shear connects kx modes at
s-boundaries via `ixplus`/`ixminus`. With `ikxspace = 5` and `nkx = 85`,
each non-zonal ky has 5 independent kx-chains of 17 members each.

**Key insight**: linearizing the indices as `global = chain_pos × ns + s`
**preserves the pentadiagonal bandwidth**. When s overflows past `ns - 1`,
the shift map connects to `s = 0` of the next kx in the chain — which is
`global + 1`. Similarly, underflow at `s = 0` connects to `s = ns - 1` of the
previous kx, which is `global - 1`. The kx-boundary coupling fills exactly the
same diagonals as the interior stencil.

At chain endpoints (`ixplus = -1` or `ixminus = -1`), the shift is invalid
(`valid = False`), giving **open boundary conditions**. The resulting system is
a standard (non-periodic) pentadiagonal matrix of size `17 × 16 = 272`.

### 3.4 System Sizes

| Category | Systems per ky | System size | BCs | Total systems |
|:---------|:--------------:|:-----------:|:---:|:-------------:|
| Zonal (ky = 0) | 85 (one per kx) | 16 | periodic | 85 |
| Non-zonal (ky ≠ 0) | 5 chains × 31 ky | 272 | open | 155 |

Each system is independent per `(nvpar, nmu)` point, giving a batch dimension
of `nvpar × nmu = 32 × 8 = 256`.

**Total independent solves per step**:
- Zonal: `85 × 256 = 21,760` systems of size 16
- Non-zonal: `155 × 256 = 39,680` systems of size 272
- **Grand total: 61,440 solves**

For IMEX-SSP2 (2 implicit solves per step): **122,880 solves**.

For kinetic electrons (2 species, each treated independently): multiply by
`nsp = 2`.

---

## 4. Data Layout Considerations

**Current layout**: `(nvpar, nmu, ns, nkx, nky)` — s is axis 2 with stride
`nkx × nky × 16 bytes = 85 × 32 × 16 = 43,520 bytes` (non-contiguous).

The banded solver needs s contiguous. Options:

1. **Transpose to `(nvpar × nmu, nkx, nky, ns)`** with s trailing.
   Cost: read + write of the full array.

2. **Flatten non-s dims, vmap**: reshape to `(batch, ns)` where
   `batch = nvpar × nmu × nkx × nky`, then `jax.vmap(solve_banded)`.

3. **For kx-chains**: reshape to `(batch, chain_size)` with linearized
   `(chain_pos × ns + s)` contiguous.

**Array sizes** (adiabatic, complex128):
```
32 × 8 × 16 × 85 × 32 × 16 bytes = 178 MB
```

**Transpose cost**: `2 × 178 MB` (read + write) at 2 TB/s HBM bandwidth
= **0.18 ms** — negligible vs. the ~155 ms/step baseline.

**Coefficient matrix assembly**: `s_total_upar` already stores per-point
coefficients with shape `(9, nvpar, nmu, ns, nkx, nky)`. Extracting the 5
non-zero diagonals (indices 2–6) and assembling the banded format
`(l+u+1, n) = (5, ns)` is a gather/reshape from existing arrays.

---

## 5. IMEX Scheme Design

### Operator Splitting

```
df/dt = L_imp(f) + L_exp(f)
```

| Component | Terms | Reason |
|:----------|:------|:-------|
| `L_imp` | `_apply_parallel(f, s_total_upar)` | Stiff parallel streaming |
| `L_exp` | All remaining: vpar stencils, drifts, hyper, drive, term_vii, nonlinear | Non-stiff or different structure |

### Recommended: IMEX-SSP2(2,2,2) (Pareschi & Russo 2005)

Second-order, strong-stability-preserving, 2 implicit solves per step:

```
# Stage 1
k1_exp = L_exp(f^n)
Solve [I - dt·γ·A] w1 = L_imp(f^n) + dt·γ·A·f^n        → 1 banded solve
f* = f^n + dt·(k1_exp + w1)

# Stage 2
k2_exp = L_exp(f*)
Solve [I - dt·γ·A] w2 = L_imp(f*) + dt·γ·A·f*           → 1 banded solve
f^{n+1} = f^n + dt/2·(k1_exp + w1 + k2_exp + w2)
```

where `γ = 1 - 1/√2 ≈ 0.2929` is the DIRK parameter.

**Cost per step**: 2 explicit RHS evaluations + 2 implicit solves. Comparable
to 2 explicit RK stages (vs. 4 for current RK4), but allows ~8–60× larger dt.

### Alternative: First-Order Semi-Implicit (simpler, for prototyping)

```
[I - dt·A] f^{n+1} = f^n + dt·L_exp(f^n)
```

1 implicit solve, 1 explicit RHS. Cheapest per step, but only O(dt) accurate.
Useful for initial validation before committing to the 2nd-order scheme.

---

## 6. Implementation Sketch

### New function: `_implicit_parallel_solve`

```python
def _implicit_parallel_solve(
    rhs: jnp.ndarray,       # (nvpar, nmu, ns, nkx, nky) — RHS of implicit system
    dt: jnp.ndarray,        # scalar timestep
    pre: GKPre,             # precomputed coefficients
    gamma: float = 1.0,     # DIRK parameter (1.0 for semi-implicit, 0.2929 for SSP2)
) -> jnp.ndarray:
    """Solve [I - dt*gamma*A_par] x = rhs via batched pentadiagonal solve."""

    # 1. Extract 5 non-zero diagonals from s_total_upar (indices 2..6)
    #    Shape: (5, nvpar, nmu, ns, nkx, nky)
    diags = pre["s_total_upar"][2:7]

    # 2. Assemble banded matrix [I - dt*gamma*A] in (5, ns) format per batch point
    #    For zonal (ky=0): handle periodic BCs separately (dense solve for ns=16)
    #    For non-zonal: linearize kx-chains to (5, chain_size=272)

    # 3. Transpose rhs to make s (or chain_s) the last contiguous axis
    #    rhs_t shape: (batch, ns) or (batch, chain_size)

    # 4. vmap solve_banded over batch dimensions
    #    x_t = jax.vmap(jax.scipy.linalg.solve_banded, ...)((2,2), ab, rhs_t)

    # 5. Transpose back to original layout
    return x
```

### Modified `gkstep_single_imex`

```python
def gkstep_single_imex(prev_df, geometry, params, state, pre, dt_override=None):
    dt = dt_override if dt_override is not None else jnp.array(params.dt, ...)

    def _rhs_explicit(df):
        phi = _compute_phi(df, geometry, params, pre)
        rhs = _compute_linear_rhs_explicit(df, phi, geometry, params, pre)
        if params.non_linear:
            rhs += _compute_nonlinear_rhs(df, phi, geometry, params, pre)
        return rhs

    gamma = 1.0 - 1.0 / jnp.sqrt(2.0)  # SSP2 DIRK parameter

    # Stage 1
    k1_exp = _rhs_explicit(prev_df)
    k1_imp = _apply_parallel(prev_df, pre["s_total_upar"])
    rhs1 = k1_imp + dt * gamma * _apply_parallel(prev_df, pre["s_total_upar"])  # A·f^n term
    w1 = _implicit_parallel_solve(rhs1, dt, pre, gamma)
    f_star = prev_df + dt * (k1_exp + w1)

    # Stage 2
    k2_exp = _rhs_explicit(f_star)
    rhs2 = _apply_parallel(f_star, pre["s_total_upar"]) + dt * gamma * ...
    w2 = _implicit_parallel_solve(rhs2, dt, pre, gamma)
    next_df = prev_df + (dt / 2.0) * (k1_exp + w1 + k2_exp + w2)

    # ... normalization and state tracking as in gkstep_single ...
    return next_df, (phi, (z, z, z)), next_state
```

### Modified CFL: `estimate_timestep_imex`

Remove `dt_par` from the linear CFL computation (`solver.py:320-339`):

```python
def estimate_linear_timestep_imex(pre, safety_factor=0.5):
    max_utrap = jnp.max(jnp.abs(pre["utrap"]))
    dvp = pre["dvp"]
    dt_trap = jnp.where(max_utrap > 1e-30, safety_factor * dvp / max_utrap, 1e10)
    return dt_trap  # parallel streaming no longer constrains dt
```

---

## 7. Expected Performance

### Implicit Solve Cost

**FLOPs per pentadiagonal solve**: O(12 × n) real operations (LU + forward/back
substitution for bandwidth (2,2)). Complex128 ≈ 4× real.

| System type | Size n | FLOPs/solve | Count (×2 stages) | Total FLOPs |
|:------------|:------:|:----------:|:------------------:|:-----------:|
| Zonal | 16 | ~770 | 43,520 | 33.5 M |
| Non-zonal chains | 272 | ~13,100 | 79,360 | 1,039 M |
| **Total** | | | **122,880** | **~1.1 GFLOP** |

At 9.7 TFLOP/s (A100 FP64): **0.11 ms** — negligible vs. 155 ms/step.

Memory traffic (transpose + solve + transpose back): `4 × 178 MB ≈ 712 MB`
at 2 TB/s: **0.36 ms**.

**Total implicit overhead: ~0.5 ms/step (< 0.3% of step time).**

### Time-to-Solution Speedup

Assuming parallel streaming only implicit (trapping still explicit):

| Case | Current dt | IMEX dt | dt ratio | Step cost | **Net speedup** |
|:-----|:----------:|:-------:|:--------:|:---------:|:---------------:|
| Kinetic electrons | ~0.00017 | ~0.0014 | 8× | +0.3% | **~8×** |
| Adiabatic (linear) | 0.01 | ~0.05 | 5× | +0.3% | **~5×** |
| Adiabatic (saturated) | 0.01 | ~0.01–0.03 | 1–3× | +0.3% | **1–3×** |

With implicit trapping extension (§8):

| Case | IMEX dt | dt ratio | **Net speedup** |
|:-----|:-------:|:--------:|:---------------:|
| Kinetic electrons | ~0.004–0.01 | 25–60× | **~20–50×** |
| Adiabatic (linear) | ~0.05–0.08 | 5–8× | **~5–8×** |
| Adiabatic (saturated) | ~0.01–0.03 | 1–3× | **1–3×** |

---

## 8. Extension: Implicit Trapping

The trapping term (`_apply_vpar` in `solver.py:770-778`) uses the same
pentadiagonal structure but operates along the **vpar axis** instead of s:

```
VPAR_D1 = [1, -8, 0, 8, -1] / 12    # stencils.py:74
VPAR_D4 = [-1, 4, -6, 4, -1] / 12   # stencils.py:77
```

Treating trapping implicitly requires a **second pentadiagonal solve** in vpar
for each `(mu, s, kx, ky)` point:

- System size: `nvpar = 32`
- Batch: `nmu × ns × nkx × nky = 8 × 16 × 85 × 32 = 348,160` per species
- Boundary: open (vpar has absorbing boundaries)
- Cost: ~348,160 × 2 species × 2 stages × 32 × 48 FLOPs ≈ 2.1 GFLOP → 0.22 ms

Since the parallel streaming (s-direction) and trapping (vpar-direction) operators
act on **different axes**, they commute to leading order. Within the implicit
part, we can use **dimensional splitting**:

```
[I - dt·γ·A_par] [I - dt·γ·A_trap] f^{n+1} ≈ [I - dt·γ·(A_par + A_trap)] f^{n+1}
```

This avoids solving a coupled (s, vpar) system and instead solves two
independent pentadiagonal systems sequentially — one sweep in s, one in vpar.
The splitting error is O(dt² · γ² · [A_par, A_trap]) where the commutator is
small (the operators share no common axes).

**Total implicit overhead with trapping**: ~0.5 + 0.7 = **~1.2 ms/step** —
still < 1% of step time.

This extension is what unlocks the full 20–50× speedup for kinetic electrons.

---

## 9. Risks and Validation

1. **Accuracy**: IMEX-SSP2 is 2nd order, vs. RK4's 4th order. For CFL-limited
   runs the temporal error is dominated by the CFL timestep, not the order.
   Validate by comparing growth rates (linear) and heat flux spectra (nonlinear)
   at matched physical time.

2. **Stability**: the implicit part must be A-stable (DIRK with γ > 0 is).
   The explicit part must satisfy CFL for the remaining terms. Monitor for
   instabilities at the explicit/implicit interface.

3. **Operator splitting error**: for the trapping extension, the
   dimensional splitting introduces an O(dt²) error from the commutator
   `[A_par, A_trap]`. This is small because the operators act on orthogonal
   axes, but should be monitored in long nonlinear runs.

4. **JAX compatibility**: `jax.scipy.linalg.solve_banded` wraps LAPACK's
   `zgbsv` for complex128. Verify it supports batched execution via `vmap`
   and compiles correctly under `jit`.

---

## Appendix A: Files to Modify (Implementation)

| File | Change |
|:-----|:-------|
| `gyaradax/solver.py:1079–1141` | New `gkstep_single_imex` alongside existing RK4 |
| `gyaradax/solver.py:1144–1211` | `gksolve` to support IMEX mode |
| `gyaradax/solver.py:320–339` | `estimate_linear_timestep_imex` without parallel CFL |
| `gyaradax/solver.py` (new) | `_implicit_parallel_solve()` function |
| `gyaradax/params.py` | `imex: bool` parameter |

## Appendix B: Files Referenced (Read-Only)

| File | Purpose |
|:-----|:--------|
| `solver.py:260–356` | CFL estimation functions |
| `solver.py:459–595` | `_compute_species_coeffs` — upar, utrap construction |
| `solver.py:409–457` | `_fuse_stencils` — s_total_upar structure |
| `solver.py:359–406` | `_precompute_shared` — stencil normalization (`/12.0`) |
| `solver.py:758–768` | `_apply_parallel` — explicit stencil being replaced |
| `solver.py:770–778` | `_apply_vpar` — trapping operator (§8 extension) |
| `geometry.py:312–362` | `_build_parallel_shift_maps` — boundary conditions |
| `stencils.py` | D1/D4 stencil coefficients, bandwidth verification |
| `configs/iteration_13.yaml` | Adiabatic reference: dt=0.01, vthrat=1.0 |
| `configs/kinetic_991_double_rlt.yaml` | Kinetic reference: dt=0.004, vthrat=[1.0, 60.634] |
| `OPTIM.md §6.3.2` | Original IMEX proposal |
| `Post_Mortem_JAX_Optimizations.md §4.3` | IMEX as Priority 2 pivot |
