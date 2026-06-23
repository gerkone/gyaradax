"""Eigenvalue solver for the gyaradax linear operator.

Mirrors GKW's `eiv_integration.F90` `mat_vec_product_rhs` / `mat_vec_product_exp`:
the linear RHS (including the self-consistent field solve and, for EM runs,
the mixed-variable g -> f transform) is wrapped as a matrix-free matvec and
handed to an Arnoldi eigensolver to find the top-k eigenvalues.  This exposes
subdominant linear modes that an initial-value solver cannot reach since IVP
only converges to the dominant root.

The matvec is the EXACT operator integrated by the IVP (`gkstep_single._rhs`
with ``non_linear=False``): fields are solved from the evolved variable g,
then ``g_to_f`` is applied before ``ops.linear_rhs`` (collisions included via
``pre['coll_stencil']`` when enabled).

Conventions (matches GKW `trafo_eiv_to_gf`, mat_vec_routine=2):

  lambda = gamma + i*omega    with d(g)/dt = L g, g ~ exp(lambda t)

  Re(lambda) = growth rate gamma  == IVP ``state.last_growth_rate`` (per ky)
  Im(lambda) = real frequency omega, same sign as GKW's frequencies.dat.

Two matvec modes are supported:

  - mode='rhs': matvec is L(g) = ops.linear_rhs(g_to_f(g), fields(g)).
    Eigenvalues are returned directly (lambda).  ARPACK selector defaults to
    'LR' (largest real part).  Convergence can be slow because the dominant
    physical modes are not the largest-magnitude eigenvalues of L.

  - mode='exp' (recommended): matvec is ``n_steps_per_matvec`` RK4 steps of
    the linear operator; eigenvalues of the step operator are
    mu = exp(lambda * n_steps * dt) (to RK4 accuracy) and the dominant
    physical mode IS the largest-|mu| eigenvalue, so Arnoldi converges fast.
    ARPACK selector defaults to 'LM'.  ``dt`` must be RK4-stable for the
    linear operator (use the IVP's working dt).  Raw eigenvalues are
    recovered as log(mu)/(n_steps*dt), which is branch-ambiguous when
    |Im(lambda)|*n_steps*dt > pi; with the default ``refine=True`` each
    converged eigenvector v is re-evaluated through the 'rhs' matvec via the
    Rayleigh quotient lambda = <v, L v>/<v, v>, which removes both the log
    branch ambiguity and the RK4 discretization error.

Per-ky spectra: the linear operator block-diagonalizes over ky (and over
connected-kx chains within each ky).  A global Arnoldi mixes all blocks and
returns globally-dominant eigenvalues, whereas the IVP reports per-ky growth
rates.  Pass ``ky_select=<iky>`` to restrict the solve to a single ky block
(the start vector and every matvec output are masked to that ky column);
the dominant eigenvalue then matches the IVP's ``state.last_growth_rate[iky]``.

Example
-------
    geometry = compute_geometry_from_input("input.dat")
    params = gkparams_from_input_and_geometry("input.dat", geometry)
    eigvals, eigvecs = eigensolve_linear(
        geometry, params, k=4, mode="exp", n_steps_per_matvec=50, ky_select=1
    )
    gamma, omega = eigvals[0].real, eigvals[0].imag
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

from gyaradax.backends import create_ops
from gyaradax.solver import _compute_fields, linear_precompute, g_to_f
from gyaradax.simulate import gk_init


def _df_shape(geometry: Dict[str, jnp.ndarray], n_species: int, kinetic: bool) -> Tuple[int, ...]:
    # match init_f shape: 5D for adiabatic, 6D for kinetic
    nv = int(geometry["intvp"].shape[0])
    nmu = int(geometry["intmu"].shape[0])
    ns = int(geometry["ints"].shape[0])
    nkx = int(geometry["kxrh"].shape[0])
    nky = int(geometry["krho"].shape[0])
    if kinetic:
        return (n_species, nv, nmu, ns, nkx, nky)
    return (nv, nmu, ns, nkx, nky)


def _resolve_n_species(params: Any, n_species: int) -> int:
    """For kinetic runs, derive the species count from params.mas."""
    if bool(params.adiabatic_electrons):
        return 1
    mas = np.atleast_1d(np.asarray(params.mas))
    return max(int(n_species), int(mas.shape[0]))


def _check_linear(params: Any) -> None:
    if bool(params.non_linear):
        raise ValueError(
            "eigensolve requires a linear operator: set params.non_linear=False"
        )


def _ky_mask(df_shape: Tuple[int, ...], ky_select: Optional[int]) -> Optional[jnp.ndarray]:
    """Boolean (1.0/0.0) mask over the trailing ky axis, broadcast to df_shape."""
    if ky_select is None:
        return None
    nky = df_shape[-1]
    iky = int(ky_select)
    if not (0 <= iky < nky):
        raise ValueError(f"ky_select={ky_select} out of range [0, {nky})")
    mask = jnp.zeros((nky,), dtype=jnp.float64).at[iky].set(1.0)
    return mask.reshape((1,) * (len(df_shape) - 1) + (nky,))


def _build_rhs_matvec(geometry, params, pre, ops, ky_mask=None):
    """Pure linear-operator matvec: L(dg) = linear_rhs(g_to_f(dg), fields(dg)).

    Identical to the operator integrated by ``gkstep_single`` when
    ``params.non_linear`` is False (solver.py `_rhs`).  ``ky_mask`` restricts
    input and output to one ky block (the operator is exactly block-diagonal
    over ky in linear runs; the mask only guards against roundoff leakage).
    """

    @jax.jit
    def matvec(df):
        if ky_mask is not None:
            df = df * ky_mask
        phi, apar, bpar = _compute_fields(df, geometry, params, pre)
        df_for_rhs = g_to_f(df, apar, params, pre) if apar is not None else df
        out = ops.linear_rhs(df_for_rhs, phi, geometry, params, pre, apar=apar, bpar=bpar)
        if ky_mask is not None:
            out = out * ky_mask
        return out

    return matvec


def _build_exp_matvec(geometry, params, pre, ops, dt, n_steps=1, ky_mask=None):
    """N-step linear RK4 matvec: M(df) = (one_step)^n_steps · df.

    Each one_step is M_1 = df + dt/6 (k1+2k2+2k3+k4); eigenvalues of M are
    exp(lambda * n_steps * dt) (to RK4 accuracy). Larger n_steps amplifies
    the magnitude gap between unstable and stable modes, dramatically
    improving Arnoldi convergence. Matches GKW's advance_large_step_explicit
    pattern in mat_vec_product_exp.
    """
    rhs = _build_rhs_matvec(geometry, params, pre, ops, ky_mask=ky_mask)

    def one_step(df):
        k1 = rhs(df)
        k2 = rhs(df + 0.5 * dt * k1)
        k3 = rhs(df + 0.5 * dt * k2)
        k4 = rhs(df + dt * k3)
        return df + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @jax.jit
    def matvec(df):
        if ky_mask is not None:
            df = df * ky_mask
        if n_steps == 1:
            out = one_step(df)
        else:
            out = jax.lax.fori_loop(0, n_steps, lambda _, x: one_step(x), df)
        if ky_mask is not None:
            out = out * ky_mask
        return out

    return matvec


def _rayleigh_refine(rhs_matvec_flat: Callable, eigvecs_flat: np.ndarray) -> np.ndarray:
    """lambda_i = <v_i, L v_i> / <v_i, v_i> via the direct 'rhs' matvec.

    For an exact eigenvector the Rayleigh quotient is exact regardless of
    operator normality; for ARPACK-converged eigenvectors of the exp-step
    operator (which shares eigenvectors with L) this recovers lambda without
    the log-branch ambiguity of log(mu)/dt.
    """
    out = np.empty(eigvecs_flat.shape[0], dtype=np.complex128)
    for i in range(eigvecs_flat.shape[0]):
        v = jnp.asarray(eigvecs_flat[i], dtype=jnp.complex128)
        lv = rhs_matvec_flat(v)
        out[i] = complex(jnp.vdot(v, lv) / jnp.vdot(v, v))
    return out


def eigensolve_linear(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    *,
    pre=None,
    n_species: int = 1,
    k: int = 10,
    which: Optional[str] = None,
    tol: float = 1e-8,
    mode: str = "exp",
    backend: str = "jax",
    seed: int = 42,
    v0: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
    ncv: Optional[int] = None,
    dt: Optional[float] = None,
    n_steps_per_matvec: int = 1,
    ky_select: Optional[int] = None,
    refine: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvalues of the gyaradax linear operator (matrix-free ARPACK).

    Uses scipy.sparse.linalg.eigs with a LinearOperator wrapping a JIT-compiled
    JAX matvec. The operator is non-Hermitian and complex-valued (complex128).

    Args:
        geometry: gyaradax geometry dict.
        params: GKParams (non_linear must be False).
        pre: optional precomputed coefficients (else recomputed).
        n_species: ignored for kinetic runs (derived from params.mas);
            1 for adiabatic.
        k: number of eigenpairs to return.
        which: ARPACK selector. Defaults to 'LM' for mode='exp' (largest
            |mu| == largest growth rate) and 'LR' for mode='rhs'.
        tol: ARPACK relative tolerance.
        mode: 'exp' (eigenvalues of the RK4 step operator, recommended) or
            'rhs' (direct eigenvalues of L).
        backend: solver backend ('jax' recommended; 'cuda' is not differentiable).
        seed: RNG seed for v0 if not provided.
        v0: optional initial Arnoldi vector (flat complex128).
        maxiter, ncv: ARPACK knobs.
        dt: time step for mode='exp' (defaults to params.dt; must be RK4-stable).
        n_steps_per_matvec: RK4 steps per matvec for mode='exp'.
        ky_select: restrict the solve to one ky block (see module docstring).
        refine: mode='exp' only — re-evaluate each eigenvalue with a Rayleigh
            quotient through the 'rhs' matvec (fixes log-branch ambiguity).

    Returns:
        eigenvalues: shape (k,), complex (lambda = gamma + i*omega); sorted by
            descending Re(lambda).
        eigenvectors: shape (k, *df_shape), complex; eigvecs[i] matches eigvals[i].
    """
    if mode not in ("rhs", "exp"):
        raise ValueError(f"mode must be 'rhs' or 'exp', got {mode!r}")
    _check_linear(params)

    kinetic = not bool(params.adiabatic_electrons)
    n_species = _resolve_n_species(params, n_species)
    df_shape = _df_shape(geometry, n_species=n_species, kinetic=kinetic)
    n = int(np.prod(df_shape))
    ky_mask = _ky_mask(df_shape, ky_select)

    if pre is None:
        pre = linear_precompute(geometry, params)

    ops = create_ops(
        pre,
        backend=backend,
        use_z2z=getattr(params, "use_z2z", False),
        mixed_precision=getattr(params, "mixed_precision", False),
    )

    rhs_matvec = _build_rhs_matvec(geometry, params, pre, ops, ky_mask=ky_mask)
    if mode == "rhs":
        jmatvec = rhs_matvec
        which = which or "LR"
    else:
        dt_val = float(dt) if dt is not None else float(params.dt)
        jmatvec = _build_exp_matvec(
            geometry,
            params,
            pre,
            ops,
            jnp.asarray(dt_val, dtype=jnp.float64),
            n_steps=n_steps_per_matvec,
            ky_mask=ky_mask,
        )
        which = which or "LM"

    # numpy <-> jax bridge for scipy LinearOperator
    def matvec_np(x: np.ndarray) -> np.ndarray:
        df = jnp.asarray(x, dtype=jnp.complex128).reshape(df_shape)
        out = jmatvec(df)
        out.block_until_ready()
        return np.asarray(out).reshape(-1)

    op = spla.LinearOperator((n, n), matvec=matvec_np, dtype=np.complex128)

    if v0 is None:
        rng = np.random.default_rng(seed)
        v0 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
    else:
        v0 = np.asarray(v0, dtype=np.complex128).reshape(-1)
    if ky_mask is not None:
        v0 = v0 * np.asarray(jnp.broadcast_to(ky_mask, df_shape)).reshape(-1)
    v0 = v0 / np.linalg.norm(v0)

    eigvals, eigvecs = spla.eigs(
        op, k=k, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv
    )

    # mode='exp': eigenvalues of the time-step operator -> growth rates
    if mode == "exp":
        dt_eff = dt_val * n_steps_per_matvec
        if refine:
            eigvals = _rayleigh_refine(
                lambda v: rhs_matvec(v.reshape(df_shape)).reshape(-1),
                np.ascontiguousarray(eigvecs.T),
            )
        else:
            eigvals = np.log(eigvals) / dt_eff

    # sort by descending real part
    order = np.argsort(-eigvals.real)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # reshape eigenvectors to df-shape, return as (k, *df_shape)
    eigvecs_reshaped = np.stack(
        [eigvecs[:, i].reshape(df_shape) for i in range(eigvals.shape[0])], axis=0
    )
    return eigvals, eigvecs_reshaped


def _make_arnoldi(matvec_flat: Callable, ncv: int):
    """Build a JIT'd Arnoldi iteration with `matvec_flat` closed over.

    Returns a function `arnoldi(v0_flat) -> (V, H)` where V is (ncv+1, n)
    orthonormal and H is (ncv+1, ncv) upper Hessenberg. The leading-k
    eigenvalues of L are approximated by the eigenvalues of H[:ncv, :ncv].
    Two-pass modified Gram-Schmidt for numerical stability.  Breakdown
    (norm of the new direction ~ 0, i.e. an exact invariant subspace) is
    guarded with a safe division; the corresponding subdiagonal entry of H
    is ~0 so the Ritz values of the converged block are unaffected.
    """

    @jax.jit
    def arnoldi(v0_flat: jnp.ndarray):
        n = v0_flat.shape[0]
        dtype = v0_flat.dtype

        V = jnp.zeros((ncv + 1, n), dtype=dtype)
        H = jnp.zeros((ncv + 1, ncv), dtype=dtype)
        V = V.at[0].set(v0_flat / jnp.linalg.norm(v0_flat))

        def step(j, state):
            V, H = state
            w = matvec_flat(V[j])
            mask = jnp.arange(ncv + 1) <= j
            # two-pass Gram-Schmidt
            coeffs1 = V.conj() @ w
            coeffs1 = jnp.where(mask, coeffs1, jnp.zeros_like(coeffs1))
            w = w - coeffs1 @ V
            coeffs2 = V.conj() @ w
            coeffs2 = jnp.where(mask, coeffs2, jnp.zeros_like(coeffs2))
            w = w - coeffs2 @ V
            coeffs = coeffs1 + coeffs2

            norm_w = jnp.linalg.norm(w)
            col = jnp.where(mask, coeffs, jnp.zeros_like(coeffs))
            col = col.at[j + 1].set(norm_w)
            H = H.at[:, j].set(col)
            # breakdown guard: norm_w ~ 0 means an invariant subspace was hit
            safe_norm = jnp.where(norm_w > 1e-300, norm_w, 1.0)
            V = V.at[j + 1].set(w / safe_norm)
            return V, H

        V, H = jax.lax.fori_loop(0, ncv, step, (V, H))
        return V, H

    return arnoldi


def eigensolve_linear_jax(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    *,
    pre=None,
    n_species: int = 1,
    k: int = 4,
    ncv: int = 20,
    mode: str = "exp",
    backend: str = "jax",
    seed: int = 42,
    v0: Optional[jnp.ndarray] = None,
    dt: Optional[float] = None,
    n_steps_per_matvec: int = 1,
    ky_select: Optional[int] = None,
    refine: bool = True,
    return_eigvecs: bool = True,
):
    """Pure-JAX Arnoldi eigensolve (single fixed-size Krylov subspace).

    No scipy LinearOperator host roundtrips per matvec; the entire Arnoldi
    loop is one jitted computation.  No restarts: accuracy is controlled by
    ``ncv`` and (for mode='exp') by ``n_steps_per_matvec``.

    `n_steps_per_matvec` (only for mode='exp'): each Arnoldi matvec advances
    the df by this many RK4 steps. Larger values widen the magnitude gap
    between unstable/stable modes — Arnoldi converges in fewer iterations.
    For typical ITG (gamma ~ 0.3, dt ~ 0.01), use ~50-100.

    ``ky_select``/``refine``: see :func:`eigensolve_linear`.

    Returns:
        eigvals: (k,) complex (lambda = gamma + i*omega), sorted by
            descending Re(lambda).
        eigvecs: (k, *df_shape) complex if return_eigvecs else None.
    """
    if mode not in ("rhs", "exp"):
        raise ValueError(f"mode must be 'rhs' or 'exp', got {mode!r}")
    _check_linear(params)

    kinetic = not bool(params.adiabatic_electrons)
    n_species = _resolve_n_species(params, n_species)
    df_shape = _df_shape(geometry, n_species=n_species, kinetic=kinetic)
    n = int(np.prod(df_shape))
    ky_mask = _ky_mask(df_shape, ky_select)

    if pre is None:
        pre = linear_precompute(geometry, params)
    ops = create_ops(
        pre,
        backend=backend,
        use_z2z=getattr(params, "use_z2z", False),
        mixed_precision=getattr(params, "mixed_precision", False),
    )

    rhs_mv = _build_rhs_matvec(geometry, params, pre, ops, ky_mask=ky_mask)
    if mode == "rhs":
        _mv = rhs_mv
        dt_eff = None
    else:
        dt_val = float(dt) if dt is not None else float(params.dt)
        _mv = _build_exp_matvec(
            geometry,
            params,
            pre,
            ops,
            jnp.asarray(dt_val, dtype=jnp.float64),
            n_steps=n_steps_per_matvec,
            ky_mask=ky_mask,
        )
        dt_eff = dt_val * n_steps_per_matvec

    # flat-vector matvec (df_shape is closed over)
    def matvec_flat(v):
        return _mv(v.reshape(df_shape)).reshape(-1)

    if v0 is None:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        v0_flat = (jax.random.normal(k1, (n,), dtype=jnp.float64)
                   + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64))
    else:
        v0_flat = jnp.asarray(v0, dtype=jnp.complex128).reshape(-1)
    if ky_mask is not None:
        v0_flat = (v0_flat.reshape(df_shape) * ky_mask).reshape(-1)
    v0_flat = v0_flat / jnp.linalg.norm(v0_flat)

    arnoldi = _make_arnoldi(matvec_flat, ncv)
    V, H = arnoldi(v0_flat)

    # eigendecompose the small Hessenberg on host
    H_host = np.asarray(H[:ncv, :ncv])
    ritz_vals, ritz_vecs = np.linalg.eig(H_host)

    # mode='exp': convert Ritz of M = exp-step operator back to lambda
    if mode == "exp":
        with np.errstate(divide="ignore", invalid="ignore"):
            eigvals = np.log(ritz_vals) / dt_eff
        eigvals = np.where(np.isfinite(eigvals), eigvals, -np.inf + 0j)
    else:
        eigvals = ritz_vals
    order = np.argsort(-eigvals.real)

    eigvals = eigvals[order][:k]

    # reconstruct full-space eigenvectors: y_i = V[:ncv].T @ ritz_vecs[:, i]
    V_host = np.asarray(V[:ncv])
    rv = ritz_vecs[:, order][:, :k]
    eigvecs_full_flat = (V_host.T @ rv).T  # (k, n)

    if mode == "exp" and refine:
        eigvals = _rayleigh_refine(
            lambda v: rhs_mv(v.reshape(df_shape)).reshape(-1), eigvecs_full_flat
        )
        order2 = np.argsort(-eigvals.real)
        eigvals = eigvals[order2]
        eigvecs_full_flat = eigvecs_full_flat[order2]

    if not return_eigvecs:
        return eigvals, None
    return eigvals, eigvecs_full_flat.reshape((eigvals.shape[0],) + df_shape)


def random_initial_df(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    *,
    n_species: int = 1,
    seed: int = 0,
    amp: float = 1e-3,
) -> jnp.ndarray:
    """Random complex df shaped like init_f's output, for IVP-baseline runs.

    Uses gk_init solely for the shape; the returned array is a normalized
    random complex perturbation of that shape so the IVP regression is
    started from a generic state with non-zero projection on the dominant
    eigenmode.
    """
    df, _, _ = gk_init(geometry, params, n_species=n_species)
    rng = np.random.default_rng(seed)
    shape = df.shape
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x = x / np.linalg.norm(x)
    return jnp.asarray(x * amp, dtype=jnp.complex128)
