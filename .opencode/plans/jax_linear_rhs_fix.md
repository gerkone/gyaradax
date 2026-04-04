# Fix: Implement `_linear_rhs_core` in JAX Backend

## Problem
The JAX backend's `linear_rhs()` method calls `self._linear_rhs_core()` which doesn't exist. During refactoring, the implementation was removed from `solver.py` but never added to `_jax.py`.

## Solution
Add `_linear_rhs_core` method to `JAXOps` class in `gyaradax/backends/_jax.py`.

## Implementation

**File:** `gyaradax/backends/_jax.py`

**Location:** Insert between `nonlinear_term_iii()` and `linear_rhs()` methods (after line 141)

**Code to add:**
```python
def _linear_rhs_core(
    self,
    df: jnp.ndarray,
    phi: jnp.ndarray,
    params,
    pre,
) -> jnp.ndarray:
    """Fused linear RHS for single species (5D df).

    Implements Terms I, II, IV, V, VII, VIII + dissipation.
    Matches GKW linear_terms.f90 and GKW's calc_linear_terms.
    """
    gyro_phi = pre["bessel"] * phi[None, None, :, :, :]

    term_par, term_vii = self._apply_parallel_dual(
        df, gyro_phi, pre["s_total_upar"], pre["s_total_t7"]
    )

    out_d1, out_d4 = self._apply_vpar_dual(df, stencils.VPAR_D1, stencils.VPAR_D4)
    term_iv = pre["utrap"] * out_d1 / params.dvp
    term_vp_diss = params.disp_vp * pre["abs_dum2_vp"] * out_d4 / params.dvp

    kdotvd = pre["drift_x"] * pre["kx_b"] + pre["drift_y"] * pre["ky_b"]

    return (
        term_par
        + term_iv
        + term_vp_diss
        - 1j * kdotvd * df
        + pre["hyper"] * df
        + 1j * params.drive_scale * (
            pre["dmaxwel_fm_ek"]
            - pre["signz0"] * kdotvd * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1e-15))
        )
        * gyro_phi
        + term_vii
    )
```

**Required import:** Add `from gyaradax import stencils` at the top of the file (if not already present).

## Architecture

```
JAXOps.linear_rhs(df, phi, geometry, params, pre)
  ├─ 5D case → _linear_rhs_core(df, phi, params, pre)
  │   ├─ _apply_parallel_dual() → term_par (I,II), term_vii (VII)
  │   ├─ _apply_vpar_dual() → term_iv, term_vp_diss
  │   └─ drift terms (V, VIII) + hyper + kdotvd
  └─ 6D case → vmap over species with split pre arrays
      ├─ Per-species: bessel, fmaxwl, drift_*, upar, utrap, etc. (axis 0)
      ├─ Stencil coeffs: s_total_upar, s_total_t7 (axis 1)
      └─ Shared: kx_b, ky_b, hyper, shift maps
```

## Testing
- Run `tests/unit/test_nonlinear.py::test_kinetic_iteration_parity` to verify 6D kinetic electrons work
- Run full test suite to ensure no regressions
- Verify linear RHS matches CUDA backend output for 5D case

## Related Files
- `gyaradax/backends/_jax.py` - needs _linear_rhs_core added
- `gyaradax/backends/_cuda.py` - reference implementation via _linear_rhs_fused
- `gyaradax/stencils.py` - provides VPAR_D1, VPAR_D4 stencils
