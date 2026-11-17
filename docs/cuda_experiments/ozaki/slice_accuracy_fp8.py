"""Ozaki stage 1b: FP8 e4m3 slices (what B300 actually runs fast), same operator
and evolved data as stage 1. Each slice is exactly representable in e4m3 under
its own power-of-two scale, so slice-pair products are exact in an FP32
accumulator; the reconstruction runs in FP64."""
import sys
import numpy as np
import ml_dtypes
sys.path.insert(0, "/system/user/galletti/git/gyaradax-perf")
import jax, jax.numpy as jnp
from tests.perf.cases import SMALL, build, case_specs
from gyaradax.solver import default_state, gksolve
exec(open("/tmp/ozaki/stage1.py").read().split("def split_int8")[0].split("spec = next")[0])
spec = next(s for s in case_specs(SMALL) if s["name"] == "kinetic")
c = build(spec); geom, p, pre = c["geom"], c["params"], c["pre"]; sh = c["shape"]
src = open("/tmp/ozaki/stage1.py").read()
exec(src[src.index("def taps"):src.index("def split_int8")])

E4M3 = ml_dtypes.float8_e4m3fn
MAXN = 448.0
def split_fp8(x, k):
    """x ~= sum_i q_i * 2^{e_i}, q_i in e4m3, e_i chosen per point per slice so the
    residual sits at the top of the e4m3 range (no underflow as it shrinks)."""
    out = []
    r = x.copy()
    for _ in range(k):
        m = np.max(np.abs(r), axis=-1, keepdims=True)
        e = np.where(m > 0, np.ceil(np.log2(np.where(m > 0, m, 1.0) / MAXN)) , 0.0)
        sc = np.exp2(e)
        q = (r / sc).astype(E4M3).astype(np.float64)       # round to e4m3
        out.append((q, sc))
        r = r - q * sc
    return out

def ozaki_fp8_dot(cc, ff, k):
    A = split_fp8(cc, k); B = split_fp8(ff, k)
    acc = np.zeros(cc.shape[:-1])
    for i in range(k):
        for j in range(k - i):
            qa, sa = A[i]; qb, sb = B[j]
            # e4m3*e4m3 has a <=8-bit significand: exact in FP32, and a length-5
            # sum stays exact there too; the scaled partials accumulate in FP64
            prod = np.sum((qa * qb).astype(np.float32), axis=-1).astype(np.float64)
            acc = acc + prod * sa[..., 0] * sb[..., 0]
    return acc

rng = np.random.default_rng(0)
f_rand = (rng.standard_normal(sh) + 1j*rng.standard_normal(sh)) * 1e-3
sim = default_state(nky=sh[-1])
f_evo = np.asarray(gksolve(jnp.asarray(f_rand), geom, p, sim, n_steps=300, pre=pre)[0])
C, F = taps(f_evo)
print("  FP8 e4m3 Ozaki-I on EVOLVED data: max|err|/max|result| by slice count k")
print("  (triangular cost = k(k+1)/2 FP8 slice-products per FP64 product)\n")
for k in (4, 6, 8, 10, 12, 14, 16):
    errs = []
    for Fp in (F.real, F.imag):
        ref = np.sum(C * Fp, axis=-1)
        got = ozaki_fp8_dot(C, Fp, k)
        errs.append(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    print(f"    k={k:2d}  err {max(errs):.1e}   products {k*(k+1)//2:4d}")
