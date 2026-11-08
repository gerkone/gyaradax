"""J0 at small argument: jax bessel_jn, the series fix and scipy, and the size of bessel_jn's backward recurrence.
usage: python docs/_static/figs/numerics/make_bessel.py docs/_static/figs/numerics/bessel_small_x.png
"""
import sys

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
from jax.scipy.special import bessel_jn

from gyaradax.operators.poisson import j0

x = np.logspace(-7, -3, 4001)
old = np.asarray(bessel_jn(jnp.asarray(x), v=0)[0])
new = np.asarray(j0(jnp.asarray(x)))
nan = ~np.isfinite(old)
wrong = np.isfinite(old) & (np.abs(old - ss.j0(x)) > 1e-12)


def largest_term(z, n_iter=50):
    # bessel_jn's recurrence in extended precision: the largest of f_k and the normalisation sum
    z = np.longdouble(z)
    f0, f1, bs, big = np.longdouble(0), np.longdouble(1e-16), np.longdouble(0), np.longdouble(0)
    for k in range(n_iter, -1, -1):
        f = 2 * (k + 1) * f1 / z - f0
        if k % 2 == 0:
            bs += 2 * f
        f0, f1 = f1, f
        big = max(big, abs(f), abs(bs))
    return float(np.log10(big))


xr = np.logspace(-7, -3, 400)
lg = np.array([largest_term(v) for v in xr])
edge = x[nan | wrong].max()

ink, muted, c_old, c_new = "#1f2328", "#6e7781", "#d1582a", "#2a6fd0"
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
a = ax[0]
a.axvspan(x[nan].min(), x[nan].max(), color="0.92", lw=0)
a.text(np.sqrt(x[nan].min() * x[nan].max()), 1e-3, "bessel_jn\nreturns NaN", ha="center", va="center", color=muted, fontsize=11)
a.plot(x, 1 - ss.j0(x), color=muted, lw=1.2, ls=(0, (4, 3)), label="exact (scipy)", zorder=3)
a.plot(x, np.where(nan, np.nan, 1 - old), color=c_old, lw=2.2, label="jax bessel_jn (before)")
a.plot(x, 1 - new, color=c_new, lw=1.4, label="series below x = 2e-5 (after)", zorder=4)
a.annotate("finite but wrong: J0 = 0", xy=(x[wrong].mean(), 1.0), xytext=(6e-5, 3e-2),
           arrowprops=dict(arrowstyle="-", color=muted, lw=0.8), color=ink, fontsize=10)
a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-16, 30)
a.set_xlabel("x = k_perp rho sqrt(2 mu / B)"); a.set_ylabel("1 - J0(x)")
a.set_title("J0 at small argument", loc="left", color=ink)
a.legend(loc="lower right", frameon=False, fontsize=10)
b = ax[1]
b.plot(xr, lg, color=c_old, lw=2, label="largest value in the recurrence")
b.axhline(np.log10(np.finfo(np.float64).max), color=muted, lw=1, ls=(0, (4, 3)))
b.text(1.3e-7, np.log10(np.finfo(np.float64).max) - 14, "fp64 maximum (1.8e308)", color=muted, fontsize=10)
b.axvline(edge, color="0.8", lw=1)
b.set_xscale("log"); b.set_xlabel("x"); b.set_ylabel("log10 of the largest value")
b.set_title("why: the backward recurrence from order 50 overflows", loc="left", color=ink)
b.legend(loc="upper right", frameon=False, fontsize=10)
for a in ax:
    a.grid(True, color="0.92"); a.tick_params(colors=ink)
    for s in a.spines.values():
        s.set_color("0.75")
fig.savefig(sys.argv[1], dpi=120)
print(sys.argv[1], "edge", edge)
