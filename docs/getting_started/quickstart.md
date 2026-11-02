# Quickstart

## From the command line

Every run is described by a YAML config; command-line flags only override it.
`configs/iteration_13.yaml` is a nonlinear adiabatic ITG case on the standard
grid ($n_{v_\parallel}, n_\mu, n_s, n_{k_x}, n_{k_y}$) = (32, 8, 16, 85, 32):

```bash
gyaradax run configs/iteration_13.yaml --device 0
```

The run writes `outputs_adiabatic_iteration_13/`: the effective config, the
geometry, the diagnostic histories (`fluxes.npz`, `kyspec.npz`, `kxspec.npz`,
`growth.npz`, `dt_history.npz`) and a restart snapshot `step_NNNNNN.npz`. Pass
the directory instead of the config to continue it:

```bash
gyaradax run outputs_adiabatic_iteration_13 --n-steps 2000
```

The heat flux of the run:

```python
import numpy as np
import matplotlib.pyplot as plt

h = np.load("outputs_adiabatic_iteration_13/fluxes.npz")
plt.plot(h["time"], h["fluxes"][:, 1])   # columns: particle, heat, momentum flux
plt.xlabel("t [R/v_th]"); plt.ylabel("Q_i")
```

Kinetic electrons, electromagnetic fields and collisions are switched on in the
config, not with flags: `configs/kinetic.yaml`, `configs/nl_em_apar.yaml` and
`configs/kinetic_collisions.yaml` are examples. {doc}`running` covers the
remaining commands and options.

## From Python

The same run, step by step:

```python
import gyaradax as gyx

df, geometry, params, state, pre = gyx.gk_from_config("configs/iteration_13.yaml")
df, phi, fluxes, state = gyx.gk_run(df, geometry, params, state, 400, pre=pre)
pflux, eflux, vflux = fluxes
```

{func}`~gyaradax.solver.driver.gk_run` does no I/O.
{func}`~gyaradax.solver.driver.gksimulate` adds the run directory, the
diagnostics and snapshots:

```python
df, phi, fluxes, state = gyx.gksimulate(
    df, geometry, params, state, 4000, pre=pre,
    output_dir="outputs", checkpoint_interval=400,
)
```

Parameters can also be built directly. The geometry comes from equilibrium
parameters ({func}`~gyaradax.geometry.geom.compute_geometry`) and the run
parameters are a {class}`~gyaradax.core.params.GKParams`, which needs the
geometry's scalars (grid spacings, largest wavenumbers) copied in; the YAML
path does this for you:

```python
from gyaradax import GKParams, compute_geometry, gk_init, gk_run

geometry = compute_geometry(
    q=1.4, shat=0.78, eps=0.19, geom_type="s-alpha",
    nvpar=32, nmu=8, ns=16, nkx=1, nky=8, krhomax=0.8,
)
scalars = ("shat", "q", "eps", "kthnorm", "Rref", "d2X", "signB", "dvp", "sgr_dist", "kxmax", "kymax")
params = GKParams(
    rlt=6.9, rln=2.2, dt=0.01, naverage=100, non_linear=False,
    **{k: float(geometry[k]) for k in scalars},
)
df, geometry, state = gk_init(geometry, params)
df, phi, fluxes, state = gk_run(df, geometry, params, state, 2000)
gamma = state.last_growth_rate          # per-ky growth rate
```

Pass `backend="cuda"` to `GKParams` for the CUDA kernels. Linear runs also have
an eigenvalue solver, {func}`~gyaradax.solver.eigenvalue.eigensolve`, which finds
subdominant and damped modes as well.

## From a GKW run

A GKW run directory converts to a config, or can be continued directly from its
last K-file dump:

```bash
gyaradax convert /path/to/gkw_run configs/my_case.yaml
```

```python
df, geometry, params, state, pre = gyx.gk_from_gkw_dir("/path/to/gkw_run")
```

## Differentiating through a run

The JAX backend is differentiable end to end. A gradient of a loss on the
potential with respect to the temperature gradient:

```python
import dataclasses
import jax
import jax.numpy as jnp
from gyaradax.operators.precompute import linear_precompute

def loss(rlt):
    p = dataclasses.replace(params, rlt=rlt)
    pre = linear_precompute(geometry, p)
    _, phi, _, _ = gk_run(df, geometry, p, state, 400, pre=pre)
    return jnp.mean(jnp.abs(phi - phi_target) ** 2)

dloss = jax.grad(loss)(6.9)
```

{doc}`../verification/index` shows the inverse problem and the sensitivity study
built this way. The CUDA kernels are not differentiable; use `backend="jax"`.
