"""Ozaki-II (CRT) reference for a 5-tap stencil dot product, against exact integers.

S = sum_t c_t f_t. Scale c and f to integers with B bits, compute the exact
integer dot product modulo r primes p_i < 2^14 (products < 2^28, a 5-term sum
stays inside int32), reconstruct by Garner's mixed radix, unscale. Python ints
give the exact reference, so this isolates the scheme's own error from any
implementation choice. Tile-wide f scaling (what a fused kernel can afford)
versus per-output scaling shows what sharing deconstruction costs."""
import numpy as np

def primes_below(n, k):
    out, x = [], n - 1
    while len(out) < k:
        if x > 1 and all(x % d for d in range(2, int(x**0.5) + 1)): out.append(x)
        x -= 1
    return out

P = primes_below(1 << 14, 12)

def to_int(x, bits, scale_exp):
    """round(x * 2^(bits-1-scale_exp)) as exact Python ints."""
    return [int(round(v * 2.0 ** (bits - 1 - scale_exp))) for v in x]

def garner(res, ps):
    """Mixed-radix digits -> exact signed integer."""
    a, M, tot = [], 1, 0
    for i, (r, p) in enumerate(zip(res, ps)):
        v = r
        for j in range(i):
            v = (v - a[j]) * pow(ps[j], -1, p) % p
        a.append(v)
    for i, v in enumerate(a):
        tot += v * M
        M *= ps[i]
    if tot > M // 2: tot -= M            # back to signed
    return tot

rng = np.random.default_rng(0)
NOUT = 2000
c = rng.standard_normal((NOUT, 5)) * np.array([0.08, 0.6, 0.0, 0.6, 0.08])[None] * 300
c[:, 2] = rng.standard_normal(NOUT) * 50                      # centre tap
f = rng.standard_normal((NOUT, 5)) * 1e-3 * np.exp(rng.standard_normal((NOUT, 1)))
ref = np.array([float(sum(np.longdouble(a)*np.longdouble(b) for a, b in zip(ci, fi)))
                for ci, fi in zip(c, f)])

print(f"  primes: {P[:4]}... (all < 2^14)\n")
print(f"  {'bits B':>6} {'r':>3} {'scaling':>11} {'max rel err':>12} {'med rel err':>12}")
for B, r in ((31, 5), (40, 6), (53, 8), (53, 9)):
    ps = P[:r]
    for mode in ("per-output", "tile-wide"):
        ec = np.ceil(np.log2(np.abs(c).max(axis=1)))                # per output
        ef = (np.ceil(np.log2(np.abs(f).max(axis=1))) if mode == "per-output"
              else np.full(NOUT, np.ceil(np.log2(np.abs(f).max()))))
        errs = []
        for o in range(NOUT):
            ci = to_int(c[o], B, ec[o]); fi = to_int(f[o], B, ef[o])
            res = [sum(a * b for a, b in zip(ci, fi)) % p for p in ps]   # exact mod
            S = garner(res, ps)
            got = S * 2.0 ** (ec[o] - B + 1) * 2.0 ** (ef[o] - B + 1)
            errs.append(abs(got - ref[o]) / max(abs(ref[o]), 1e-300))
        bound = 2 * 5 * 2 ** (2 * B)
        ok = "" if np.prod([float(p) for p in ps]) > bound else "  (product < 2|S|: WRAPS)"
        print(f"  {B:6d} {r:3d} {mode:>11} {max(errs):12.1e} {np.median(errs):12.1e}{ok}")
