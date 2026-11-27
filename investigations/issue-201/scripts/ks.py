"""Krylov-Schur on the gyaradax RK4 step operator of one ky block, converged on the growing Ritz values only."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from gyaradax.backends import create_ops
from gyaradax.operators.precompute import linear_precompute
from gyaradax.solver.eigenvalue import (_arnoldi_factory, _build_exp_matvec, _build_rhs_matvec, _df_shape,
                                        _residual, _slice_ky_geometry, _thick_restart)


def block(geom, params, iky):
    g = _slice_ky_geometry(geom, iky)
    pre = linear_precompute(g, params)
    shape = _df_shape(g, n_species=len(np.atleast_1d(params.mas)), kinetic=not params.adiabatic_electrons)
    ops = create_ops(pre, backend="jax", use_z2z=False, mixed_precision=False)
    return g, pre, ops, _build_rhs_matvec(g, params, pre, ops), shape


def spectral_radius(rhs, shape, n_it=400, seed=1):
    """Largest |lambda| of L by power iteration (max of the last norm ratios)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    v = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)

    @jax.jit
    def run(v):
        def step(v, _):
            w = rhs(v)
            nw = jnp.linalg.norm(w)
            return w / nw, nw / jnp.linalg.norm(v)

        return jax.lax.scan(step, v / jnp.linalg.norm(v), None, length=n_it)

    _, ratios = run(v)
    return float(jnp.max(ratios[-50:]))


def build_filter(rhs, dt, nsteps, omega0, sigma_frac=0.125):
    """Gaussian-windowed frequency filter sum_n w(t_n) exp(-i omega0 t_n) x(t_n) along an RK4 trajectory of length T."""
    T = dt * nsteps
    t = jnp.arange(nsteps + 1) * dt
    w = jnp.exp(-0.5 * ((t - 0.5 * T) / (sigma_frac * T)) ** 2) * jnp.exp(-1j * omega0 * t) * dt

    def one_step(x):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @jax.jit
    def mv(x):
        def body(i, c):
            y, acc = c
            return one_step(y), acc + w[i] * y

        y, acc = jax.lax.fori_loop(0, nsteps, body, (x, jnp.zeros_like(x)))
        return acc + w[nsteps] * y

    return mv


def solve(g, params, pre, ops, rhs, shape, dt, nsteps, k=10, keep=14, ncv=40, tol=1e-6, gmin=0.0, maxr=40,
          seed=42, log=print, mv=None):
    """Leading eigenpairs of the step operator; stops when every top-k Ritz value with growth > gmin has bound < tol.

    With ``mv`` (a filter operator) the Ritz values are not exp(lambda T): every top-k one must converge.
    """
    n = int(np.prod(shape))
    T = dt * nsteps
    filt = mv is not None
    if mv is None:
        mv = _build_exp_matvec(g, params, pre, ops, jnp.asarray(dt, dtype=jnp.float64), n_steps=nsteps)

    def mv_flat(v):
        return mv(v.reshape(shape)).reshape(-1)

    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    v = jax.random.normal(k1, (n,), dtype=jnp.float64) + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    V = jnp.zeros((ncv + 1, n), dtype=jnp.complex128).at[0].set(v / jnp.linalg.norm(v))
    H = jnp.zeros((ncv + 1, ncv), dtype=jnp.complex128)
    first = _arnoldi_factory(mv_flat, ncv, jnp.complex128, start=0)
    cont = _arnoldi_factory(mv_flat, ncv, jnp.complex128, start=keep)
    t0 = time.time()
    V, H = first(V, H)
    nmv = ncv
    hist = []
    for it in range(maxr + 1):
        Hs = np.asarray(H[:ncv, :ncv])
        if not np.all(np.isfinite(Hs)):
            raise np.linalg.LinAlgError("non-finite Hessenberg matrix")
        theta, Y = np.linalg.eig(Hs)
        order = np.argsort(-np.abs(theta))
        resid = np.abs(complex(np.asarray(H[ncv, ncv - 1]))) * np.abs(Y[ncv - 1]) / np.maximum(np.abs(theta), 1e-300)
        lam = np.log(theta.astype(complex)) / T
        top = order[:k]
        need = list(top) if filt else [i for i in top if lam[i].real > gmin]
        conv = len(need) > 0 and bool(np.all(resid[need] < tol))
        hist.append(dict(it=it, nmv=nmv, wall=time.time() - t0,
                         ritz=[[float(lam[i].real), float(lam[i].imag), float(resid[i])] for i in top]))
        log(f"restart {it} matvecs {nmv} wall {time.time() - t0:.0f}s | " + " ".join(
            f"{lam[i].real:+.4f}{lam[i].imag:+.3f}i({resid[i]:.0e})" for i in top[:10]))
        if conv or it == maxr:
            break
        V, H, _ = _thick_restart(V, H, ncv, keep, lambda w: -np.abs(w))
        V, H = cont(V, H)
        nmv += ncv - keep
    vecs = np.array(jnp.asarray(Y[:, top].T, dtype=jnp.complex128) @ V[:ncv])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vals = np.empty(len(top), dtype=complex)
    res = np.empty(len(top))
    for i in range(len(top)):
        x = jnp.asarray(vecs[i].reshape(shape))
        lx = rhs(x)
        vals[i] = complex(jnp.vdot(x, lx) / jnp.vdot(x, x))
        res[i] = _residual(rhs, vals[i], vecs[i].reshape(shape))
    ritz_bound = resid[top]
    o = np.argsort(-vals.real)
    return vals[o], vecs[o].reshape((len(top),) + tuple(shape)), res[o], ritz_bound[o], conv, nmv, hist
