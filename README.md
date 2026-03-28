# `gyaradax`: Gyrokinetics in JAX

<p align="center">
  <img src="figs/gyaradax_small.png" width="500" alt="gyaradax Logo">
</p>

`gyaradax` is a JAX code for local flux-tube gyrokinetic simulations. It is based on [GKW](https://bitbucket.org/gkw/gkw). At the current stage, it provides a differentiable solver for the electrostatic, collisionless Vlasov-Poisson system.

This was made possible with significant usage of agentic workflows. [PROMPT.md](docs/PROMPT.md) contains the prompt used to obtain the initial working version of `gyaradax`

See [agent notes](docs/NOTES.md) for a detailed walkthrough of GKW and this reimplementation.

## Installation
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

This installs `gyaradax` in editable mode with JAX (CUDA 12), numpy, and dev tools (pytest, ruff, black).

## Structure

- **`solver.py`**: Linear and nonlinear Terms (I-VIII), RK4 integrator.
- **`simulate.py`**: Interface for trajectory generation.
- **`integrals.py`**: Field solvers and flux integrals.
- **`geometry.py`**: Parsers for GKW geometry files and metric tensor coefficients.
- **`params.py`**: Configuration pytrees.
- **`stencils.py`**: Finite difference stencil definitions.
- **`diag.py`**: Diagnostics (growth rate, frequency, spectral).
- **`plot_utils.py`**: Visualization.

## Running Simulations

### Basic usage

The `scripts/run.py` script provides a convenient way to execute simulations, supporting single or multiple configuration files, batch execution, and specifying runtime options like the device and number of blocks.

```bash
# Run a single configuration
python -m scripts.run configs/iteration_13.yaml --device 0
```

When multiple YAML configuration files are provided, and they share the same grid resolution and static parameters, `scripts/run.py` can automatically batch them using `jax.vmap` for parallel execution on a single device.

```bash
# Run two configurations in parallel on device 0
python -m scripts.run configs/adiabatic_a.yaml configs/adiabatic_b.yaml --device 0
```

### Usage
#### Run a simulation
```python
from gyaradax.simulate import gk_from_config, gksimulate

# load yaml and run with IO/checkpointing
df, geometry, params, state, pre = gk_from_config("configs/my_sim.yaml")
df, phi, fluxes, state = gksimulate(
  df, geometry, params, state, 1200, pre=pre,
  output_dir="outputs", checkpoint_interval=120
)
```

#### Resume from GKW checkpoints
`gyaradax` can resume from GKW binary `K` files.
```python
import jax.numpy as jnp

from gyaradax.utils import load_gkw_k_dump, load_geometry, read_gkw_dump_time
from gyaradax.params import gkparams_from_input_and_geometry
from gyaradax.solver import GKState, mode_amplitude
from gyaradax.simulate import gksimulate, _compute_phi_for_init

# resume from GKW dump K01
geometry = load_geometry("/path/to/gkw/run/")
params = gkparams_from_input_and_geometry("/path/to/gkw/run/input.dat", geometry)

res = tuple(len(geometry[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
df_k = load_gkw_k_dump("/path/to/gkw/run/K01", res, n_species=1)

t_start = read_gkw_dump_time("/path/to/gkw/run/K01.dat")
nky = len(geometry["krho"])

phi0 = _compute_phi_for_init(df, geometry, params)
amp0 = mode_amplitude(phi0, geometry, params.norm_eps)

state_k = GKState(
    time=jnp.array(t_start, dtype=jnp.float64),
    step=jnp.array(0, dtype=jnp.int32),
    accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
    window_start_amp=amp0,
    last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
)

df, phi, fluxes, state = gksimulate(df_k, geometry, params, state_k, 120)
```

#### Configuration from GKW
If you have an existing GKW run, you can extract its parameters and geometry into yaml:
```bash
python -m scripts.gkw_to_yaml /path/to/gkw_run configs/my_sim.yaml
```

## State of the project and TODOs

**Verification**:
- [x] Empirical validation against reference GKW trajectories.
- [x] Anaytical validation on RH and Cyclone Base Case.
- [ ] GKW tests and benchmarks (see [the gkw paper](docs/gkw.pdf) and Chapter 11 in the manual).
- [ ] Solver-in-the-Loop and PINNs as an ML showcase.

**Physics and solver extensions**:
- [x] Linear solver.
- [x] Adiabatic electrons corrections and cases (ion only, single species).
- [x] Kinetic electrons (multi-species).
- [ ] Electromagnetic effects.
- [ ] Collisoinality.

**Optimization**:
- [x] JAX-based improvements.
- [ ] Fully spectral solver.
- [ ] Implicit/explicit integration (IMEX).


## Citing
```
```