# Gyaradax: JAX-accelerated Gyrokinetics

Gyaradax is a high-performance, JAX-idiomatic scientific library for local flux-tube gyrokinetic simulations. It is designed for strict numerical precision (fp64) and functional composition.

## Key Features
- **JAX-Native:** Fully compatible with `jit`, `vmap`, and `grad`.
- **High Precision:** Enforced `fp64` (double precision) across all calculations.
- **Efficient I/O:** Snapshot checkpointing using compressed `.npz` files.
- **Flexible Initialization:** Supports starting from analytical profiles or resuming from GKW reference data.
- **Automated Workflows:** High-level `simulate` function for running full trajectories from YAML configurations.

## Installation
```bash
pip install -r requirements.txt
```

## Running a Simulation

The easiest way to run a simulation is using a YAML configuration file:

```python
from gyaradax import simulate

df, final_state = simulate(
    "configs/iteration_13.yaml",
    output_dir="outputs",
    checkpoint_interval=40,
    n_steps=400,
    verbose=True
)
```

### Resuming Simulations
Gyaradax supports resuming from both internal checkpoints and GKW reference files:

```python
# Resume from an .npz checkpoint
simulate("configs/iteration_13.yaml", resume_from="outputs/step_000040.npz")

# Resume from a GKW K-file (e.g., dump '100')
simulate("configs/iteration_13.yaml", resume_k_file="/path/to/gkw/run/100")
```

## Configuration

Configurations are managed via YAML files. You can generate a configuration from a GKW directory using the provided script:

```bash
python scripts/gkw_to_yaml.py /path/to/gkw_run config.yaml
```

### Configuration Schema
- `solver`: Time-step (`dt`), total steps (`n_steps`), and dissipation controls.
- `physics`: Species-level gradients (`rlt`, `rln`) and masses.
- `geometry`: Local magnetic field and metric tensor coefficients.
- `grid`: Resolution for all 5 dimensions.

## Diagnostics & Plotting

The `gyaradax.plot_utils` module provides publication-quality visualization tools:
- `plot_flux_trace`: Temporal evolution of heat and particle fluxes.
- `plot_spectra`: Spectral density analysis in $k_x$ and $k_y$.
- `plot_mode_growth`: Detailed growth rate analysis for specific modes.

Refer to `notebooks/linear_v1_inspection.ipynb` for interactive examples.
