# Applications

Besides time stepping, `gyx` gives three tools built on the same linear operator:

- {doc}`eigenmodes`: the dominant and subdominant linear modes, per $k_y$, with an Arnoldi solver.
- {doc}`quasilinear`: turbulent fluxes from one linear solve and a saturation rule.
- {doc}`differentiable`: gradients of anything computed from a run, by JAX automatic differentiation, for sensitivities and inverse problems.

The examples on these pages share one small linear case: an ITG mode in circular geometry close to the Cyclone Base Case, on a $(n_{v_\parallel}, n_\mu, n_s, n_{k_x}, n_{k_y})$ = (24, 8, 16, 9, 8) grid. It runs in seconds on a GPU and in about a minute on a CPU.

```python
import dataclasses

import jax
import jax.numpy as jnp
import gyaradax as gyx
from gyaradax import GKParams, compute_geometry, gk_init, gk_run
from gyaradax.operators.precompute import linear_precompute

geometry = compute_geometry(
    q=1.4, shat=0.78, eps=0.19,
    nvpar=24, nmu=8, ns=16, nkx=9, nky=8, krhomax=1.4, kxmax=2 * 9 * 0.78,
)
scalars = ("shat", "q", "eps", "kthnorm", "Rref", "d2X", "signB", "dvp", "sgr_dist", "kxmax", "kymax")
base = GKParams(
    rlt=6.9, rln=2.22, dt=0.01, naverage=40, non_linear=False,
    disp_par=0.1, disp_vp=0.1, disp_x=0.1, disp_y=0.1,
    **{k: float(geometry[k]) for k in scalars},
)
```

`import gyaradax` switches JAX to FP64. All three tools run on the JAX backend, which is the default of `GKParams`.

```{toctree}
:hidden:

eigenmodes
quasilinear
differentiable
```
