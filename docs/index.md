---
hide-toc: true
---

# gyaradax

```{image} figs/gyaradax_small.png
:class: hero-logo
:alt: gyaradax logo
```


```{rst-class} hero-tagline
```

Local flux-tube gyrokinetics in JAX, with CUDA kernels, multi-GPU sharding and
automatic differentiation.

gyaradax solves the δf gyrokinetic equation in a flux tube: electrostatic or
electromagnetic ($A_\parallel$, $B_\parallel$), adiabatic or kinetic electrons,
linearised Fokker-Planck collisions, linear and nonlinear. It follows
[GKW](https://bitbucket.org/gkw/gkw) ([Peeters et al. 2009](https://doi.org/10.1016/j.cpc.2009.07.001))
closely, with the same normalisations, grids and input files, so a GKW run can
be converted and continued in gyaradax.

It is meant for people who want a gyrokinetic code that fits in a Python
workflow: generating turbulence datasets, coupling to transport solvers,
fitting reduced models, or differentiating through a simulation. The core is
plain JAX (`jit`, `vmap`, `grad`); an optional CUDA backend makes nonlinear runs
several times faster.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install, run a first simulation from the CLI or Python, configure runs.
:::

:::{grid-item-card} Performance
:link: performance/index
:link-type: doc

Backends, precision and kernel switches, grid sizes, multi-GPU, and how the
code was optimised.
:::

:::{grid-item-card} Numerics
:link: numerics/index
:link-type: doc

The equations as discretised: grids, stencils, time stepping, field solves,
and where gyaradax differs from GKW.
:::

:::{grid-item-card} Verification
:link: verification/index
:link-type: doc

Analytical benchmarks, statistical agreement with GKW, and differentiable
experiments.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Generated from the docstrings: drivers, solver, operators, backends,
geometry, diagnostics.
:::

:::{grid-item-card} Paper
:link: https://arxiv.org/abs/2604.06085

*gyaradax: Local Gyrokinetics JAX Code*, Galletti, Volkmann and Brandstetter (2026).
:::
::::

## At a glance

```bash
pip install -e ".[cuda13]"                          # JAX with CUDA 13
gyaradax run configs/iteration_13.yaml --device 0   # adiabatic ITG turbulence
```

```python
import gyaradax as gyx

df, geometry, params, state, pre = gyx.gk_from_config("configs/iteration_13.yaml")
df, phi, fluxes, state = gyx.gk_run(df, geometry, params, state, 400, pre=pre)
```

```{image} figs/torus.gif
:alt: nonlinear ITG turbulence on a torus
:width: 100%
```

## Citing

```bibtex
@misc{galletti2026gyaradax,
  title         = {gyaradax: Local Gyrokinetics JAX Code},
  author        = {Gianluca Galletti and Eric Volkmann and Johannes Brandstetter},
  year          = {2026},
  eprint        = {2604.06085},
  archivePrefix = {arXiv},
  primaryClass  = {physics.plasm-ph},
  url           = {https://arxiv.org/abs/2604.06085},
}
```

```{toctree}
:hidden:
:caption: User guide

getting_started/index
performance/index
numerics/index
verification/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
references
```
