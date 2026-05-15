"""Eigenvalue solver for the gyaradax linear operator.

Mirrors GKW's `eiv_integration.F90` `mat_vec_product_rhs`: the linear RHS
(including the self-consistent field solve) is wrapped as a matrix-free
matvec and handed to scipy ARPACK to find the top-k eigenvalues. This
exposes subdominant linear modes that an initial-value solver cannot reach
since IVP only converges to the dominant root.

Two modes are supported:

  - mode='rhs': matvec is L(df) = ops.linear_rhs(df, fields(df)).
    Eigenvalues are returned directly (lambda).

  - mode='exp': matvec is one RK4 step of the linear-only operator,
    eigenvalues come out as exp(lambda * dt). Better conditioned for
    very dominant modes; result is converted back via log(mu)/dt.
"""

from __future__ import annotations

from functools import partial
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


def _build_rhs_matvec(geometry, params, pre, ops):
    """Pure linear-operator matvec: L(df) = linear_rhs(g_to_f(df), fields(df))."""

    @jax.jit
    def matvec(df):
        phi, apar, bpar = _compute_fields(df, geometry, params, pre)
        df_for_rhs = g_to_f(df, apar, params, pre) if apar is not None else df
        return ops.linear_rhs(df_for_rhs, phi, geometry, params, pre, apar=apar, bpar=bpar)

    return matvec


def _build_exp_matvec(geometry, params, pre, ops, dt, n_steps=1):
    """N-step linear RK4 matvec: M(df) = (one_step)^n_steps · df.

    Each one_step is M_1 = df + dt/6 (k1+2k2+2k3+k4); eigenvalues of M are
    exp(lambda * n_steps * dt) (to RK4 accuracy). Larger n_steps amplifies
    the magnitude gap between unstable and stable modes, dramatically
    improving Arnoldi convergence. Matches GKW's advance_large_step_explicit
    pattern in mat_vec_product_exp.
    """
    rhs = _build_rhs_matvec(geometry, params, pre, ops)

    def one_step(df):
        k1 = rhs(df)
        k2 = rhs(df + 0.5 * dt * k1)
        k3 = rhs(df + 0.5 * dt * k2)
        k4 = rhs(df + dt * k3)
        return df + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @jax.jit
    def matvec(df):
        if n_steps == 1:
            return one_step(df)
        return jax.lax.fori_loop(0, n_steps, lambda _, x: one_step(x), df)

    return matvec


def eigensolve_linear(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    *,
    pre=None,
    n_species: int = 1,
    k: int = 10,
    which: str = "LR",
    tol: float = 1e-8,
    mode: str = "rhs",
    backend: str = "jax",
    seed: int = 42,
    v0: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
    ncv: Optional[int] = None,
    dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k eigenvalues of the gyaradax linear operator (matrix-free Arnoldi).

    Uses scipy.sparse.linalg.eigs with a LinearOperator wrapping a JIT-compiled
    JAX matvec. The operator is non-Hermitian and complex-valued (complex128).

    Args:
        geometry: gyaradax geometry dict.
        params: GKParams (non_linear should be False).
        pre: optional precomputed coefficients (else recomputed).
        n_species: 1 for adiabatic; matches kinetic species count otherwise.
        k: number of eigenpairs to return.
        which: ARPACK selector ('LR' = largest real part, dominant + subdominants).
        tol: ARPACK relative tolerance.
        mode: 'rhs' (direct eigenvalues of L) or 'exp' (eigenvalues of one RK4 step).
        backend: solver backend ('jax' recommended; 'cuda' is not differentiable).
        seed: RNG seed for v0 if not provided.
        v0: optional initial Arnoldi vector (flat complex128).
        maxiter, ncv: ARPACK knobs.
        dt: time step for mode='exp' (defaults to params.dt).

    Returns:
        eigenvalues: shape (k,), complex; sorted by descending Re(lambda).
        eigenvectors: shape (k, *df_shape), complex; eigvecs[i] matches eigvals[i].
    """
    if mode not in ("rhs", "exp"):
        raise ValueError(f"mode must be 'rhs' or 'exp', got {mode!r}")

    kinetic = not bool(params.adiabatic_electrons)
    df_shape = _df_shape(geometry, n_species=n_species, kinetic=kinetic)
    n = int(np.prod(df_shape))

    if pre is None:
        pre = linear_precompute(geometry, params)

    ops = create_ops(
        pre,
        backend=backend,
        use_z2z=getattr(params, "use_z2z", False),
        mixed_precision=getattr(params, "mixed_precision", False),
    )

    if mode == "rhs":
        jmatvec = _build_rhs_matvec(geometry, params, pre, ops)
    else:
        dt_val = float(dt) if dt is not None else float(params.dt)
        jmatvec = _build_exp_matvec(geometry, params, pre, ops, jnp.asarray(dt_val, dtype=jnp.float64))

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
        v0 /= np.linalg.norm(v0)

    eigvals, eigvecs = spla.eigs(
        op, k=k, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv
    )

    # mode='exp': eigenvalues of the time-step operator -> growth rates
    if mode == "exp":
        eigvals = np.log(eigvals) / dt_val

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
    Two-pass modified Gram-Schmidt for numerical stability.
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
            V = V.at[j + 1].set(w / norm_w)
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
    return_eigvecs: bool = True,
):
    """Pure-JAX Arnoldi eigensolve. No scipy, no host roundtrips.

    `n_steps_per_matvec` (only for mode='exp'): each Arnoldi matvec advances
    the df by this many RK4 steps. Larger values widen the magnitude gap
    between unstable/stable modes — Arnoldi converges in fewer iterations.
    For typical ITG (gamma ~ 0.3, dt ~ 0.01), use ~50-100.

    Returns:
        eigvals: (k,) complex, sorted by descending Re(lambda).
        eigvecs: (k, *df_shape) complex if return_eigvecs else None.
    """
    if mode not in ("rhs", "exp"):
        raise ValueError(f"mode must be 'rhs' or 'exp', got {mode!r}")

    kinetic = not bool(params.adiabatic_electrons)
    df_shape = _df_shape(geometry, n_species=n_species, kinetic=kinetic)
    n = int(np.prod(df_shape))

    if pre is None:
        pre = linear_precompute(geometry, params)
    ops = create_ops(
        pre,
        backend=backend,
        use_z2z=getattr(params, "use_z2z", False),
        mixed_precision=getattr(params, "mixed_precision", False),
    )

    if mode == "rhs":
        _mv = _build_rhs_matvec(geometry, params, pre, ops)
        dt_eff = None
    else:
        dt_val = float(dt) if dt is not None else float(params.dt)
        _mv = _build_exp_matvec(
            geometry, params, pre, ops,
            jnp.asarray(dt_val, dtype=jnp.float64),
            n_steps=n_steps_per_matvec,
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
    v0_flat = v0_flat / jnp.linalg.norm(v0_flat)

    arnoldi = _make_arnoldi(matvec_flat, ncv)
    V, H = arnoldi(v0_flat)

    # eigendecompose the small Hessenberg on host
    H_host = np.asarray(H[:ncv, :ncv])
    ritz_vals, ritz_vecs = np.linalg.eig(H_host)

    # mode='exp': convert Ritz of M = exp-step operator back to lambda
    if mode == "exp":
        eigvals = np.log(ritz_vals) / dt_eff
        order = np.argsort(-eigvals.real)
    else:
        eigvals = ritz_vals
        order = np.argsort(-eigvals.real)

    eigvals = eigvals[order][:k]
    if not return_eigvecs:
        return eigvals, None

    # reconstruct full-space eigenvectors: y_i = V[:ncv].T @ ritz_vecs[:, i]
    V_host = np.asarray(V[:ncv])
    rv = ritz_vecs[:, order][:, :k]
    eigvecs_full = (V_host.T @ rv).T.reshape((k,) + df_shape)
    return eigvals, eigvecs_full


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
