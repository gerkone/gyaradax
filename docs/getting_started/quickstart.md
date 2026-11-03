# Quickstart

## From the command line

Every run is described by a YAML config. Command-line flags only override it. `configs/adiabatic/nonlinear/itg_high_shear.yaml` is a nonlinear adiabatic ITG case on the standard grid ($n_{v_\parallel}, n_\mu, n_s, n_{k_x}, n_{k_y}$) = (32, 8, 16, 85, 32):

```bash
gyaradax run configs/adiabatic/nonlinear/itg_high_shear.yaml --device 0 --backend cuda --n-steps 400
```

`--device 0` sets `CUDA_VISIBLE_DEVICES=0`. On a shared machine pick a free GPU, or leave the flag out and set `CUDA_VISIBLE_DEVICES` yourself. `XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX from taking 75 % of the GPU memory at start. The backend defaults to `jax`. `--backend cuda` uses the CUDA kernels once they are built (2.6x faster on an A100 here), and `backend: auto` in the config picks them when they are available.

The config holds the whole GKW run it reproduces (32 000 steps, t = 320); `--n-steps 400` stops at t = 4, still in the linear phase. The run writes `outputs_adiabatic_itg_high_shear/`: the effective config (`config.yaml`), the geometry, the diagnostic histories (`fluxes.npz`, `fluxes_em.npz`, `kyspec.npz`, `kxspec.npz`, `growth.npz`, `dt_history.npz`), `run_info.jsonl` and a restart snapshot `step_NNNNNN.npz`. Only the latest snapshot is kept (about 180 MB on this grid). Pass the directory instead of the config to continue it. On a resume `--n-steps` adds steps:

```bash
gyaradax run outputs_adiabatic_itg_high_shear --n-steps 12000 --backend cuda   # to t = 124, saturated
```

The heat flux of the run:

```python
import numpy as np
import matplotlib.pyplot as plt

h = np.load("outputs_adiabatic_itg_high_shear/fluxes.npz")
plt.plot(h["time"], h["fluxes"][:, 1])   # columns: particle, heat, momentum flux
plt.xlabel("t [R/v_th]")
plt.ylabel("Q_i")
plt.savefig("heat_flux.png")
```

Kinetic electrons, electromagnetic fields and collisions are switched on in the config, not with flags: `configs/kinetic/nonlinear/benchmark.yaml`, `configs/em/nl_em_apar.yaml` and `configs/kinetic/nonlinear/itg_mid_flux.yaml` are examples. {doc}`running` covers the remaining commands and options.

## From Python

The same run, step by step:

```python
import gyaradax as gyx

df, geometry, params, state, pre = gyx.gk_from_config("configs/adiabatic/nonlinear/itg_high_shear.yaml")
df, phi, fluxes, state = gyx.gk_run(df, geometry, params, state, 400, pre=pre)
pflux, eflux, vflux = fluxes
```

{func}`~gyaradax.solver.driver.gk_run` does no I/O.
{func}`~gyaradax.solver.driver.gksimulate` adds the run directory, the diagnostics and snapshots:

```python
df, phi, fluxes, state = gyx.gksimulate(
    df, geometry, params, state, 4000, pre=pre,
    output_dir="outputs", checkpoint_interval=400,
)
```

Parameters can also be built directly. The geometry comes from equilibrium parameters ({func}`~gyaradax.geometry.geom.compute_geometry`) and the run parameters are a {class}`~gyaradax.core.params.GKParams`, which needs the geometry's scalars (grid spacings, largest wavenumbers) copied in. The YAML path does this for you:

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

Pass `backend="cuda"` to `GKParams` for the CUDA kernels. Eigenmodes, quasilinear fluxes and gradients through a run are in {doc}`../applications/index`.

## From a GKW run

A GKW run directory converts to a config, or can be continued directly from its last K-file dump:

```bash
gyaradax convert /path/to/gkw_run configs/my_case.yaml
```

```python
df, geometry, params, state, pre = gyx.gk_from_gkw_dir("/path/to/gkw_run")
```
