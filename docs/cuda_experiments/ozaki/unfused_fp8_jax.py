"""End-to-end FP8 Ozaki-I for the stencil-as-GEMM on GPU: split, every slice
GEMM, reconstruct -- all on device -- timed against native FP64, at the
production shape. This is the real cost of the ~1e-9 variant, overhead included.

C = A @ F with A:(B,M,K) and F:(B,K,N). A is scaled per row and F per column
(over the contraction axis K), each split into k e4m3 slices; slice GEMMs run in
FP8 with FP32 accumulation exactly as the tensor cores do; reconstruction in FP64.
"""
import statistics, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

NSP, NV, NMU, NS, NKX, NKY = 2, 32, 12, 18, 191, 32
B, N, M = NSP * NV, NMU * NKX * NKY, 32           # M=K=32 padded from ns=18
MAXN = 448.0
rng = np.random.default_rng(0)
A = np.zeros((B, M, M)); F = np.zeros((B, M, N))
A[:, :NS, :NS] = rng.standard_normal((B, NS, NS)); F[:, :NS, :] = rng.standard_normal((B, NS, N))
A, F = jnp.asarray(A), jnp.asarray(F)

def split(x, axis, k):
    """k e4m3 slices of x, each with its own power-of-two scale along `axis`."""
    out, r = [], x
    for _ in range(k):
        m = jnp.max(jnp.abs(r), axis=axis, keepdims=True)
        e = jnp.ceil(jnp.log2(jnp.where(m > 0, m, 1.0) / MAXN))
        sc = jnp.exp2(e)
        q = (r / sc).astype(jnp.float8_e4m3fn)
        out.append((q, sc))
        r = r - q.astype(jnp.float64) * sc
    return out

def ozaki(A, F, k):
    As, Fs = split(A, -1, k), split(F, -2, k)       # A over K (rows), F over K (cols)
    C = jnp.zeros((A.shape[0], A.shape[1], F.shape[2]))
    for i in range(k):
        for j in range(k - i):
            qa, sa = As[i]; qf, sf = Fs[j]
            p = jnp.einsum("bij,bjn->bin", qa, qf, preferred_element_type=jnp.float32)
            C = C + p.astype(jnp.float64) * sa * sf
    return C

def tm(fn, *a, n=10):
    jax.block_until_ready(fn(*a)); r = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n): o = fn(*a)
        jax.block_until_ready(o); r.append((time.perf_counter()-t0)/n*1e3)
    return statistics.median(r)

with jax.default_matmul_precision("highest"):
    ref_fn = jax.jit(lambda a, f: jnp.einsum("bij,bjn->bin", a, f))
    ref = ref_fn(A, F); t64 = tm(ref_fn, A, F)
scale = float(jnp.max(jnp.abs(ref)))
print(f"  device {jax.devices()[0].device_kind}")
print(f"  native FP64 : {t64:8.3f} ms\n")
print(f"  {'k':>3} {'products':>9} {'time ms':>9} {'vs FP64':>9} {'max rel err':>12}")
for k in [int(a) for a in sys.argv[1:]] or [4, 6, 8]:
    f = jax.jit(lambda a, ff, k=k: ozaki(a, ff, k))
    got = f(A, F); t = tm(f, A, F)
    err = float(jnp.max(jnp.abs(got - ref))) / scale
    print(f"  {k:3d} {k*(k+1)//2:9d} {t:9.3f} {t64/t:8.2f}x {err:12.1e}")
