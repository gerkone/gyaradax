"""Ozaki Stage 1: how many INT8 slices does OUR parallel stencil need for
FP64-level accuracy, on random AND on physically evolved data?

out(p) = sum_{i=2..6} c_i(p) * f_i(p) is a length-5 dot product per point
(real coefficients, complex data -> two real dot products). Ozaki-I splits each
operand into k INT8 slices with a per-point power-of-two scale, forms the
slice-pair products exactly (INT8*INT8 -> INT32), and sums the triangular set
i+j <= k+1. The error that matters is relative to |out|, not to the inputs:
derivative stencils on smooth data cancel heavily.
"""
import sys
import numpy as np
sys.path.insert(0, "/system/user/galletti/git/gyaradax-perf")
import jax
import jax.numpy as jnp
from tests.perf.cases import SMALL, build, case_specs
from gyaradax.solver import default_state, gksolve

spec = next(s for s in case_specs(SMALL) if s["name"] == "kinetic")
c = build(spec); geom, p, pre = c["geom"], c["params"], c["pre"]
sh = c["shape"]

def taps(field):
    """The five neighbour fields and coefficients of the parallel stencil."""
    tab = np.asarray(pre["s_upar_tab"]); cls = np.asarray(pre["par_stencil_class"])
    s_sh = np.asarray(pre["s_shift"]); kx_sh = np.asarray(pre["kx_shift"])
    val = np.asarray(pre["valid_shift"])
    nsp, nv, nmu, ns, nkx, nky = field.shape
    ky_i = np.arange(nky).reshape(1, 1, -1); s_i = np.arange(ns)[:, None, None]
    F, C = [], []
    for i in range(2, 7):
        sh_f = field[:, :, :, s_sh[i], kx_sh[i], ky_i]
        F.append(np.where(val[i][None, None, None], sh_f, 0.0))
        coef = tab[..., s_i, cls, i]                      # (sp, v, mu|1, s, kx, ky)
        C.append(np.broadcast_to(coef, field.shape))
    return np.stack(C, -1), np.stack(F, -1)               # (..., 5)

def split_int8(x, k):
    """Ozaki-I: x ~= s * sum_{i<k} q_i 2^{-7(i+1)}, q_i in [-127,127], per point."""
    m = np.max(np.abs(x), axis=-1, keepdims=True)
    e = np.where(m > 0, np.ceil(np.log2(np.where(m > 0, m, 1.0))), 0.0)
    s = np.exp2(e)
    r = x / s
    qs = []
    for i in range(k):
        q = np.trunc(r * 2.0 ** (7 * (i + 1)))
        q = np.clip(q, -127, 127)
        qs.append(q)
        r = r - q * 2.0 ** (-7 * (i + 1))
    return s, qs

def ozaki_dot(c, f, k):
    """Triangular Ozaki-I dot product over the last axis; every slice product exact."""
    sa, qa = split_int8(c, k); sb, qb = split_int8(f, k)
    acc = np.zeros(c.shape[:-1])
    for i in range(k):
        for j in range(k - i):                            # i + j <= k - 1
            prod = np.sum(qa[i].astype(np.int64) * qb[j].astype(np.int64), axis=-1)
            acc = acc + prod * 2.0 ** (-7 * (i + j + 2))
    return acc * sa[..., 0] * sb[..., 0]

def report(tag, field):
    C, F = taps(field)
    for part, Fp in (("re", F.real), ("im", F.imag)):
        ref = np.sum(C * Fp, axis=-1)
        scale = np.max(np.abs(ref))
        # cancellation: how much smaller is the result than the terms that make it
        terms = np.sum(np.abs(C * Fp), axis=-1)
        canc = np.median(terms[np.abs(ref) > 1e-300] / np.abs(ref[np.abs(ref) > 1e-300]))
        row = []
        for k in (3, 4, 5, 6, 7, 8, 9):
            got = ozaki_dot(C, Fp, k)
            err = np.abs(got - ref)
            row.append(f"k={k}:{np.max(err)/scale:.1e}")
        print(f"  {tag:8s} {part}  median cancellation x{canc:6.1f}   " + "  ".join(row))

rng = np.random.default_rng(0)
f_rand = (rng.standard_normal(sh) + 1j*rng.standard_normal(sh)) * 1e-3
print("  max |ozaki - fp64| / max|fp64 result|, by INT8 slice count k\n")
report("random", f_rand)

sim = default_state(nky=sh[-1])
f_evo = np.asarray(gksolve(jnp.asarray(f_rand), geom, p, sim, n_steps=300, pre=pre)[0])
report("evolved", f_evo)
print("\n  FP64 roundoff reference ~1e-16; a usable k must reach ~1e-15 on EVOLVED data.")
