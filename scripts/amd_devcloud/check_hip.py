"""Compare the HIP operators against the JAX operators on the benchmark grids."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import gyaradax as gyx
from gyaradax.backends import create_ops
from gyaradax.operators.fields import _compute_fields


def err(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-300))


def timed(f, n=8):
    jax.block_until_ready(f())
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


for cfg in sys.argv[1:]:
    df, geometry, params, state, pre = gyx.gk_from_config(cfg)
    rng = np.random.default_rng(0)
    df = df + 1e-3 * jnp.asarray(rng.normal(size=df.shape) + 1j * rng.normal(size=df.shape)) * jnp.max(jnp.abs(df))
    phi, apar, bpar = _compute_fields(df, geometry, params, pre)
    npts = df.size
    print(f"== {cfg} shape {df.shape}", flush=True)
    jx = create_ops(pre, backend="jax", use_z2z=True, mixed_precision=False)
    ref_nl = jx.nonlinear_term_iii(df, phi, geometry)
    ref_lin = jx.linear_rhs(df, phi, geometry, params, pre)
    for mp in (False, True):
        ops = create_ops(pre, backend="hip", mixed_precision=mp)
        nl = jax.jit(lambda d, p: ops.nonlinear_term_iii(d, p, geometry))
        lin = jax.jit(lambda d, p: ops.linear_rhs(d, p, geometry, params, pre))
        fld = jax.jit(lambda d: ops.compute_fields(d, geometry, params, pre)[0])
        out_nl, out_lin = nl(df, phi), lin(df, phi)
        print(f" mp={mp} bracket err {err(out_nl, ref_nl):.2e}  linear err {err(out_lin, ref_lin):.2e}  fields err {err(fld(df), phi):.2e}")
        t_nl = timed(lambda: nl(df, phi))
        t_lin = timed(lambda: lin(df, phi))
        t_fld = timed(lambda: fld(df))
        print(f"   bracket {t_nl*1e3:8.2f} ms ({t_nl/npts*1e12:7.1f} ps/pt)  linear {t_lin*1e3:8.2f} ms ({t_lin/npts*1e12:7.1f} ps/pt)  fields {t_fld*1e3:7.2f} ms", flush=True)
