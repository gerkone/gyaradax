# Gyaradax: Differentiable JAX-Accelerated Gyrokinetics

<p align="center">
  <img src="gyaradax.png" width="500" alt="Gyaradax Logo">
</p>

Gyaradax is a high-performance, JAX-idiomatic scientific library for local flux-tube gyrokinetic simulations. It provides a differentiable simulation core for the electrostatic, adiabatic-electron Vlasov-Poisson system, designed for high-precision (fp64) research and integration with modern optimization workflows.

## Key Features
- **JAX-Native:** Built for `jit` compilation, `vmap` parallelism, and `grad` automatic differentiation.
- **High Numerical Fidelity:** 4th-order finite difference stencils and pseudospectral nonlinear terms, validated against GKW.
- **Strict Precision:** Enforced `fp64` (double precision) across all calculations.
- **Flexible Workflows:** Support for YAML configurations, checkpointing, and direct GKW reference loading.
- **Modern Hardware:** Native acceleration on GPUs and TPUs via the JAX XLA backend.

## Installation
```bash
pip install -r requirements.txt
```

## Library Structure

The core logic resides in the `gyaradax/` package:

- **`solver.py`**: The core RK4 time-stepper and RHS implementation (Terms I-VIII).
- **`simulate.py`**: High-level orchestration for running trajectories and managing I/O.
- **`integrals.py`**: JAX-native field solvers (Poisson) and flux integrations.
- **`geometry.py`**: Loaders for GKW geometry files and metric tensor coefficients.
- **`params.py`**: Pytree-registered configuration and state containers.
- **`stencils.py`**: High-order finite difference operator definitions.
- **`diag.py`**: Growth rate, frequency, and spectral diagnostics.
- **`plot_utils.py`**: Visualization tools for fluxes and mode evolution.

## Common Workflows

### 1. Generate Configuration from GKW
If you have an existing GKW run, you can extract its parameters and geometry into a Gyaradax YAML:
```bash
python scripts/gkw_to_yaml.py /path/to/gkw_run configs/my_sim.yaml
```

### 2. Run a Simulation
Run a simulation using the high-level `simulate` entry point:
```python
from gyaradax import simulate

df, final_state = simulate(
    "configs/my_sim.yaml",
    output_dir="outputs",
    n_steps=400,
    checkpoint_interval=40
)
```

### 3. Resume from Checkpoints
Gyaradax supports resuming from internal `.npz` snapshots or GKW binary `K` files:
```python
# Resume from internal checkpoint
simulate("configs/my_sim.yaml", resume_from="outputs/step_000040.npz")

# Resume from GKW dump 100
simulate("configs/my_sim.yaml", resume_k_file="/path/to/gkw/run/100")
```

## Utility Scripts
The `scripts/` directory contains tools for validation and management:
- **`gkw_to_yaml.py`**: Converts GKW runs to Gyaradax configurations.
- **`validate_physics.py`**: Compares Gyaradax snapshots against GKW reference data.
- **`validate_time_averaged.py`**: Validates long-term flux averages.
- **`verify_yaml_params.py`**: Sanity checks for YAML configurations.

## Testing & Validation
Run the unit and integration test suite:
```bash
pytest tests/
```
Gyaradax maintains strict numerical parity with GKW (relative error $< 10^{-5}$). You can run the physics validation script to verify this on your local machine:
```bash
python scripts/validate_physics.py --config configs/iteration_13.yaml --ref_dir gkw_ref/data/iteration_13
```

## Development
Gyaradax follows functional programming principles. The core solver is a pure function, making it compatible with all JAX transformations.

- **Adding Physics:** Implement new RHS terms in `solver.py` and register them in the `GKParams` container.
- **New Diagnostics:** Add diagnostic logic to `diag.py` and ensure they are JIT-compatible.
