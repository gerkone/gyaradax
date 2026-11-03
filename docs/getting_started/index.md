# Getting started

`gyaradax` runs from a YAML config, either with the `gyaradax` command or from Python. Install it with the CUDA 13 extra on a GPU machine:

```bash
pip install -e ".[cuda13]"
```

Run a nonlinear adiabatic ITG case on the standard grid and continue it:

```bash
gyaradax run configs/examples/iteration_13.yaml --device 0 --backend cuda
gyaradax run outputs_adiabatic_iteration_13 --n-steps 2000
```

The same from Python:

```python
import gyaradax as gyx

df, geometry, params, state, pre = gyx.gk_from_config("configs/examples/iteration_13.yaml")
df, phi, fluxes, state = gyx.gk_run(df, geometry, params, state, 400, pre=pre)
```

The pages below cover the installation (including the CUDA kernels), the quickstart, running and resuming simulations with the command reference, the config sections and the example runs and notebooks.

```{toctree}
:maxdepth: 1

installation
quickstart
running
configuration
examples
```
