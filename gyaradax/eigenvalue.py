"""Eigenvalue solver for the gyaradax linear operator.

Mirrors GKW's `eiv_integration.F90`: the linear RHS -- field solve, g -> f
transform and collisions included -- is wrapped as a matrix-free matvec and
handed to Arnoldi. Unlike an initial-value run, which only ever converges to
the dominant root, this reaches the subdominant and damped modes too. The
matvec is the exact operator the IVP integrates (`gkstep_single._rhs` with
``non_linear=False``).

Eigenvalues follow GKW's convention, lambda = gamma + i*omega for
d(g)/dt = L g: Re is the growth rate (matching the IVP's per-ky
``last_growth_rate``) and Im the real frequency, signed as in frequencies.dat.

:func:`eigensolve` selects the driver with ``solver=``: 'arpack' is scipy's
implicitly restarted Arnoldi on a jitted matvec (the reference -- slower but
battle-tested), 'jax' a thick-restarted Krylov-Schur Arnoldi kept on device,
differentiable apart from the small dense eigendecomposition.

``mode='exp'`` (recommended) uses ``n_steps_per_matvec`` RK4 steps, whose
eigenvalues mu = exp(lambda*n_steps*dt) make the dominant physical mode the
largest-|mu| one, so Arnoldi converges quickly; ``dt`` must be RK4-stable.
What separates the modes is the time window n_steps*dt, not the step count,
so scale `n_steps_per_matvec` with 1/dt rather than fixing it: kinetic
electrons push dt to the electron Alfven CFL (~3e-4 against ~1e-2 adiabatic),
where a count tuned on an adiabatic run spans too little time and Arnoldi
stops converging -- residual 2e-1 at t = 0.017, 7e-14 at t = 2.5.
``mode='rhs'`` applies L directly and converges slowly, since the physical
modes are not the largest-magnitude eigenvalues of L. Recovering lambda from
log(mu) is branch-ambiguous once |Im(lambda)|*n_steps*dt > pi, so by default
``refine=True`` re-evaluates each eigenvector through the 'rhs' matvec as a
Rayleigh quotient, removing both that ambiguity and the RK4 error.

The operator block-diagonalizes over ky, so a global solve returns
globally-dominant eigenvalues while the IVP reports per-ky rates. Passing
``ky_select=<iky>`` restricts the problem to that block -- geometry and
coefficients are sliced, not masked, so the solve is nky times smaller -- and
the dominant eigenvalue then matches ``last_growth_rate[iky]``.

Example
-------
    eigvals, eigvecs = eigensolve(
        geometry, params, k=4, mode="exp", n_steps_per_matvec=50, ky_select=1
    )
    gamma, omega = eigvals[0].real, eigvals[0].imag
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla

from gyaradax.backends import create_ops
from gyaradax.solver import linear_precompute
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


# every ky-dependent geometry array carries ky as its trailing axis
_KY_GEOM_KEYS = (
    "ixminus", "ixplus", "krho", "kx_shift", "mode_label", "parseval",
    "pos_par_grid_class", "s_shift", "valid_shift",
)


def _slice_ky_geometry(geometry: Dict[str, jnp.ndarray], iky: int) -> Dict[str, jnp.ndarray]:
    """Geometry restricted to a single ky.

    The linear operator is exactly block-diagonal over ky, so one block is an
    independent problem of size n/nky. Restricting beats masking the full grid:
    the work drops by nky and the masked-out directions no longer sit in the
    Krylov space as an artificial kernel.
    """
    nky = int(np.shape(geometry["krho"])[0])
    if not (0 <= iky < nky):
        raise ValueError(f"ky_select={iky} out of range [0, {nky})")
    out = dict(geometry)
    for key in _KY_GEOM_KEYS:
        v = geometry[key]
        if int(np.shape(v)[-1]) != nky:
            raise ValueError(
                f"geometry[{key!r}] has shape {np.shape(v)}, expected a trailing "
                f"ky axis of size {nky}"
            )
        out[key] = v[..., iky:iky + 1]
    return out


def _embed_ky(vecs: np.ndarray, full_shape: Tuple[int, ...], iky: int) -> np.ndarray:
    """Scatter single-ky eigenvectors back into the full df shape."""
    out = np.zeros((vecs.shape[0],) + tuple(full_shape), dtype=vecs.dtype)
    out[..., iky:iky + 1] = vecs
    return out


def _build_rhs_matvec(geometry, params, pre, ops):
    """Pure linear-operator matvec: L(dg) = linear_rhs(g_to_f(dg), fields(dg)).

    Identical to the operator integrated by ``gkstep_single`` when
    ``params.non_linear`` is False (solver.py `_rhs`).
    """

    @jax.jit
    def matvec(df):
        phi, apar, bpar = ops.compute_fields(df, geometry, params, pre)
        return ops.linear_rhs_from_g(df, phi, geometry, params, pre, apar=apar, bpar=bpar)

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


def _residual(rhs_matvec: Callable, lam: complex, vec: np.ndarray) -> float:
    """||L v - lambda v|| / (|lambda| ||v||) against the direct operator."""
    v = jnp.asarray(vec, dtype=jnp.complex128)
    return float(
        jnp.linalg.norm(rhs_matvec(v) - lam * v) / (abs(lam) * jnp.linalg.norm(v))
    )


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
    return_residuals: bool = False,
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
        tol: ARPACK relative tolerance. Together with `ncv` and
            `n_steps_per_matvec` this is the accuracy/cost dial: at n~3e5 the
            same eigenvalue costs 165 s at tol=1e-10 with the default ncv and
            75 s at tol=1e-7 with ncv=60.
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
        return_residuals: also return ||L v - lambda v|| / (|lambda| ||v||)
            per eigenpair, measured against the direct 'rhs' operator.

    Returns:
        eigenvalues: shape (k,), complex (lambda = gamma + i*omega); sorted by
            descending Re(lambda).
        eigenvectors: shape (k, *df_shape), complex; eigvecs[i] matches eigvals[i].
        residuals: shape (k,), float, only when `return_residuals`.
    """
    if mode not in ("rhs", "exp"):
        raise ValueError(f"mode must be 'rhs' or 'exp', got {mode!r}")
    _check_linear(params)

    kinetic = not bool(params.adiabatic_electrons)
    n_species = _resolve_n_species(params, n_species)
    full_shape = _df_shape(geometry, n_species=n_species, kinetic=kinetic)
    iky = None if ky_select is None else int(ky_select)
    if iky is not None:
        geometry = _slice_ky_geometry(geometry, iky)
        pre = linear_precompute(geometry, params)
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

    rhs_matvec = _build_rhs_matvec(geometry, params, pre, ops)
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
    res = None
    if return_residuals:
        res = np.array([
            _residual(rhs_matvec, eigvals[i], eigvecs_reshaped[i])
            for i in range(eigvals.shape[0])
        ])
    if iky is not None:
        eigvecs_reshaped = _embed_ky(eigvecs_reshaped, full_shape, iky)
    if not return_residuals:
        return eigvals, eigvecs_reshaped
    return eigvals, eigvecs_reshaped, res






def eigensolve(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    *,
    solver: str = "jax",
    **kwargs,
):
    """Top-k eigenpairs of the linear operator, via the JAX Arnoldi or ARPACK.

    `solver='jax'` uses the thick-restarted on-device Arnoldi,
    `solver='arpack'` the scipy reference. Remaining keywords are forwarded to
    the chosen driver; see those functions for the knobs each accepts.
    """
    if solver == "jax":
        return eigensolve_linear_jax(geometry, params, **kwargs)
    if solver == "arpack":
        return eigensolve_linear(geometry, params, **kwargs)
    raise ValueError(f"solver must be 'jax' or 'arpack', got {solver!r}")


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


def eigensolve_ky_spectrum(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    *,
    pre=None,
    kys=None,
    k: int = 2,
    solver: str = "jax",
    **kwargs,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Top-k eigenpairs per ky block, TGLF style.

    The linear operator is exactly block-diagonal over ky, so each block is an
    independent problem of size n/nky; solving them separately is both cheaper
    and the form a saturation rule wants (a few unstable roots per ky). Blocks
    are solved in sequence here; they are independent and can be fanned out.

    Returns {iky: (eigenvalues, residuals)}, each sorted by descending growth
    rate.
    """
    if pre is None:
        pre = linear_precompute(geometry, params)
    nky = int(np.asarray(geometry["krho"]).shape[0])
    if kys is None:
        kys = range(nky)
    out = {}
    for iky in kys:
        vals, _vecs, res = eigensolve(
            geometry, params, solver=solver, pre=pre, k=k, ky_select=int(iky),
            return_residuals=True, **kwargs
        )
        out[int(iky)] = (vals, res)
    return out

def _arnoldi_factory(matvec_flat: Callable, m: int, dtype, start: int = 0):
    """Jitted Arnoldi filling columns `start`..m-1, two-pass Gram-Schmidt.

    With ``start > 0`` the first `start` basis vectors and the leading block
    of H are taken as given, which is what a thick restart needs.
    """

    @jax.jit
    def arnoldi(V, H):
        def step(j, state):
            V, H = state
            w = matvec_flat(V[j]).astype(dtype)
            keep = jnp.arange(m + 1) <= j
            c1 = jnp.where(keep, V.conj() @ w, 0)
            w = w - c1 @ V
            c2 = jnp.where(keep, V.conj() @ w, 0)
            w = w - c2 @ V
            nw = jnp.linalg.norm(w)
            col = jnp.where(keep, c1 + c2, 0).at[j + 1].set(nw)
            safe = jnp.where(nw > 1e-30, nw, 1.0)
            return V.at[j + 1].set(w / safe), H.at[:, j].set(col)

        return jax.lax.fori_loop(start, m, step, (V, H))

    return arnoldi


def _thick_restart(V, H, m, k, wanted):
    """Collapse an m-step factorization onto the k wanted Ritz directions.

    Returns the restarted (V, H) plus the Arnoldi residual bound. V holds an
    orthonormal basis of the k wanted Ritz directions followed by the residual
    vector, so the next cycle continues from column k instead of starting over.
    """
    Hs = np.asarray(H[:m, :m])
    h_last = complex(np.asarray(H[m, m - 1]))
    theta, Y = np.linalg.eig(Hs)
    sel = np.argsort(wanted(theta))[:k]
    Q, _ = np.linalg.qr(Y[:, sel])
    T = Q.conj().T @ Hs @ Q
    resid = (np.abs(h_last) * np.abs(Y[m - 1, sel])
             / np.maximum(np.abs(theta[sel]), 1e-300))

    Vk = jnp.asarray(Q, dtype=V.dtype).T @ V[:m]
    V_new = jnp.zeros_like(V).at[:k].set(Vk).at[k].set(V[m])
    H_new = jnp.zeros_like(H)
    H_new = H_new.at[:k, :k].set(jnp.asarray(T, dtype=H.dtype))
    H_new = H_new.at[k, :k].set(jnp.asarray(h_last * Q[m - 1, :], dtype=H.dtype))
    return V_new, H_new, float(np.max(resid))


def eigensolve_linear_jax(
    geometry: Dict[str, jnp.ndarray],
    params: Any,
    *,
    pre=None,
    n_species: int = 1,
    k: int = 4,
    ncv: int = 40,
    mode: str = "exp",
    backend: str = "jax",
    seed: int = 42,
    v0: Optional[jnp.ndarray] = None,
    dt: Optional[float] = None,
    n_steps_per_matvec: int = 1,
    ky_select: Optional[int] = None,
    refine: bool = True,
    tol: float = 1e-6,
    max_restarts: int = 40,
    return_eigvecs: bool = True,
    return_residuals: bool = False,
):
    """Thick-restarted Arnoldi in JAX, matching :func:`eigensolve_linear`.

    The Krylov window stays at `ncv`; each cycle keeps all `k` wanted Ritz
    directions and extends from column `k`, so accuracy no longer requires an
    unaffordable `ncv`. Convergence uses the Arnoldi residual bound
    |h_{m+1,m} y_m| / |theta|, which costs no extra matvec.

    `tol` is the accuracy/cost dial: a restart cycle costs `ncv - k` matvecs.
    """
    if mode not in ("rhs", "exp"):
        raise ValueError(f"mode must be 'rhs' or 'exp', got {mode!r}")
    _check_linear(params)

    kinetic = not bool(params.adiabatic_electrons)
    n_species = _resolve_n_species(params, n_species)
    full_shape = _df_shape(geometry, n_species=n_species, kinetic=kinetic)
    iky = None if ky_select is None else int(ky_select)
    if iky is not None:
        geometry = _slice_ky_geometry(geometry, iky)
        pre = linear_precompute(geometry, params)
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

    rhs_mv = _build_rhs_matvec(geometry, params, pre, ops)
    if mode == "rhs":
        _mv = rhs_mv
        dt_eff = None
    else:
        dt_val = float(dt) if dt is not None else float(params.dt)
        _mv = _build_exp_matvec(
            geometry, params, pre, ops, jnp.asarray(dt_val, dtype=jnp.float64),
            n_steps=n_steps_per_matvec,
        )
        dt_eff = dt_val * n_steps_per_matvec

    def matvec_flat(v):
        return _mv(v.reshape(df_shape)).reshape(-1)

    if v0 is None:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        v = (jax.random.normal(k1, (n,), dtype=jnp.float64)
             + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64))
    else:
        v = jnp.asarray(v0, dtype=jnp.complex128).reshape(-1)
    v = v / jnp.linalg.norm(v)

    first = _arnoldi_factory(matvec_flat, ncv, jnp.complex128, start=0)
    cont = _arnoldi_factory(matvec_flat, ncv, jnp.complex128, start=k)
    # step operator wants the largest |mu|, L the largest real part
    wanted = (lambda w: -np.abs(w)) if mode == "exp" else (lambda w: -w.real)

    V = jnp.zeros((ncv + 1, n), dtype=jnp.complex128).at[0].set(v)
    H = jnp.zeros((ncv + 1, ncv), dtype=jnp.complex128)
    V, H = first(V, H)
    converged = False
    for _ in range(max_restarts):
        V, H, resid = _thick_restart(V, H, ncv, k, wanted)
        converged = resid < tol
        if converged:
            break
        V, H = cont(V, H)
    else:
        # the loop ended on an extension, collapse it so H[:k, :k] is the block
        V, H, resid = _thick_restart(V, H, ncv, k, wanted)
    if not converged:
        warnings.warn(
            f"eigensolve_linear_jax: residual bound {resid:.2e} > tol {tol:g} "
            f"after {max_restarts} restarts; raise max_restarts or ncv.",
            RuntimeWarning,
            stacklevel=2,
        )
    # V[:k] spans the wanted subspace, the Ritz pairs live in the k x k block
    Tk = np.asarray(H[:k, :k])
    theta, Z = np.linalg.eig(Tk)
    eigvecs_flat = np.asarray(jnp.asarray(Z, dtype=jnp.complex128).T @ V[:k])
    if mode == "exp":
        if refine:
            eigvals = _rayleigh_refine(matvec_rhs_flat(rhs_mv, df_shape), eigvecs_flat)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                eigvals = np.log(theta) / dt_eff
    else:
        eigvals = theta

    order = np.argsort(-eigvals.real)
    eigvals = eigvals[order]
    eigvecs_flat = eigvecs_flat[order]

    res = None
    if return_residuals:
        res = np.array([
            _residual(rhs_mv, eigvals[i], eigvecs_flat[i].reshape(df_shape))
            for i in range(eigvals.shape[0])
        ])
    out_vecs = eigvecs_flat.reshape((eigvals.shape[0],) + df_shape) if return_eigvecs else None
    if out_vecs is not None and iky is not None:
        out_vecs = _embed_ky(out_vecs, full_shape, iky)
    if not return_residuals:
        return eigvals, out_vecs
    return eigvals, out_vecs, res


def matvec_rhs_flat(rhs_mv: Callable, df_shape) -> Callable:
    """Flat-vector view of the direct operator, for the Rayleigh refinement."""
    return lambda v: rhs_mv(v.reshape(df_shape)).reshape(-1)
