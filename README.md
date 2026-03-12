# Gyaradax: JAX-Native Gyrokinetics Solver

Gyaradax is a high-performance, differentiable, and JAX-native port of the GKW gyrokinetics code. It is designed for researchers who require a modular, easily extensible, and GPU-accelerated environment for plasma turbulence simulations.

## Key Features

- **JAX-Native:** Entirely written in JAX for automatic differentiation, JIT compilation, and seamless GPU/TPU acceleration.
- **GKW Parity:** Rigorously verified against GKW reference trajectories (e.g., `iteration_13`) for both linear and nonlinear regimes.
- **Adiabatic Electron Model:** Optimized for electrostatic adiabatic-electron physics with support for nonlinear E x B advection (Term III).
- **Stateful API:** Purely functional core interface (`gksolve`) compatible with JAX transformations like `scan`, `vmap`, and `grad`.
- **High-Precision:** Built from the ground up for `float64` precision to meet scientific computing standards.

## Project Structure

- `gyaradax/`: Core library package.
  - `solver.py`: Core time integration and RHS assembly (RK4, stencils).
  - `integrals.py`: JAX implementation of phase-space flux integrals and potentials.
  - `geometry.py`: GKW input/geometry parsing and connectivity mapping.
  - `utils.py`: Data loading and I/O utilities for GKW dumps.
- `tests/unit/`: Comprehensive test suite verifying physics, shapes, and parity.
- `docs/`: Detailed technical documentation.

## Getting Started

### Installation

Ensure you have a modern JAX environment set up (CUDA recommended for large grids).

```bash
# Example environment setup
pip install jax[cuda] einops numpy
export PYTHONPATH=$PYTHONPATH:.
```

### Basic Usage

```python
import jax
from gyaradax import gksolve, load_geometry, default_state, GKParams

# Load geometry from a GKW run directory
geom = load_geometry("path/to/gkw_run")

# Initialize parameters and state
params = GKParams(dt=0.01, naverage=40, non_linear=True)
state = default_state()

# Your distribution function (vpar, mu, s, kx, ky)
df = jax.numpy.zeros((32, 8, 16, 85, 32), dtype=jax.numpy.complex128)

# Single step update
next_df, (phi, fluxes) = gksolve(df, geom, params, state)
```

## Documentation

- [Physics & Numerical Implementation](docs/PHYSICS.md): Details on the implemented gyrokinetic equations and finite-difference schemes.
- [Parity Notes](GKW.md): Historical notes on the porting process and GKW verification snapshots.

## Testing

Run the full suite using pytest:

```bash
pytest tests/unit
```
