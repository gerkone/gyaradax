---
hide-toc: true
---

# gyaradax

```{rst-class} hero-tagline
```

Your local gyrokinetics JAX code.

`gyaradax` (short `gyx`) solves the δf gyrokinetic equation in a flux tube, electrostatic or
electromagnetic, with adiabatic or kinetic electrons and collisions. It follows
[GKW](https://bitbucket.org/gkw/gkw/src/develop/) ([Peeters et al. 2009](https://doi.org/10.1016/j.cpc.2009.07.001))
closely: same normalisations, grids and input files, so a GKW run can be
converted and continued in `gyx`.

It is meant for people who want a gyrokinetic code that fits in a Python
workflow: generating turbulence datasets, coupling to transport solvers,
fitting reduced models, or differentiating through a simulation. An optional
CUDA backend makes nonlinear runs several times faster.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install, run a first simulation from the CLI or Python, configure runs.
:::

:::{grid-item-card} Performance
:link: performance/index
:link-type: doc

Backends, precision and kernel switches, FFT sizes, memory and multi-GPU.
:::

:::{grid-item-card} Numerics
:link: numerics/index
:link-type: doc

The equations as discretised: grids, stencils, time stepping, field solves,
and where `gyx` differs from GKW.
:::

:::{grid-item-card} Development and validation
:link: validation/development
:link-type: doc

How the code was built with agents, and how it is checked against GKW and
analytical results.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Generated from the docstrings: drivers, solver, operators, backends,
geometry, diagnostics.
:::

:::{grid-item-card} Paper
:link: https://arxiv.org/abs/2604.06085

*`gyaradax`: Local Gyrokinetics JAX Code*, Galletti, Volkmann and Brandstetter (2026).
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
```

```{toctree}
:hidden:
:caption: Development and validation

validation/development
validation/verification
```

```{toctree}
:hidden:
:caption: Reference

api/index
references
```
