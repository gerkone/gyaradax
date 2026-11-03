# Differentiable runs

The JAX backend is differentiable end to end: the field solve, the RK4 steps
and the nonlinear term are plain JAX, so `jax.grad`, `jax.jacrev` and
`jax.jacobian` work through a whole run. The gradient of a scalar costs a small
multiple of one run, whatever the number of parameters. Finite differences need
two runs per parameter and a step size that balances truncation against
round-off.

## What can be differentiated

- **Inputs.** Anything that enters through
  {func}`~gyaradax.operators.precompute.linear_precompute`: gradients
  (`rlt`, `rln`), species parameters and dissipation. Call
  `linear_precompute` inside the function you differentiate, so the
  coefficients are traced. The geometry scalars copied into `GKParams` (`q`,
  `shat`, ...) are not read by the solver, so their gradients are zero.
- **Outputs.** Anything computed from the returned `df`, `phi`, fluxes or state,
  for example `state.last_growth_rate`.
- **Backend.** JAX only. The CUDA kernels are custom calls without derivatives,
  and the eigenvalue solver restarts on the host.
- **Memory.** Reverse mode keeps every step for the backward pass, so memory
  grows with the number of steps and the grid. Forward mode does not (see
  below). The examples use the small case of {doc}`index`.

## Sensitivity of the growth rate

The growth rate spectrum as a function of $R/L_T$, with `geometry` and `base`
from {doc}`index`:

```python
df0, geom, state0 = gk_init(geometry, base)  # the start does not depend on the gradients

def growth_rate(rlt, rln=2.22, n_steps=400):
    p = dataclasses.replace(base, rlt=rlt, rln=rln)
    _, _, _, state = gk_run(df0, geom, p, state0, n_steps, pre=linear_precompute(geom, p))
    return state.last_growth_rate           # (nky,)

gamma = growth_rate(6.9)
dgamma = jax.jacrev(growth_rate)(6.9)       # d gamma(ky) / d(R/L_T), (nky,)
```

`jax.jacrev` runs one backward pass per output, here one per $k_y$. Several
inputs at once:

```python
d_rlt, d_rln = jax.jacrev(growth_rate, argnums=(0, 1))(6.9, 2.22)
```

`state.last_growth_rate` is the growth rate over the last `naverage` steps, so
after 400 steps ($t = 4$) it describes the start of the run, not yet the
eigenmode. By $t \approx 40$ the four most unstable $k_y$ of this case have
converged and agree with {doc}`eigenmodes` to about $10^{-4}$. The marginal one,
with $\gamma \approx 0.01$, is still off by $10^{-2}$. The damped $k_y$ never
settle: their rate keeps changing as the damped modes decay against each other.

Scanning $R/L_T$ from 3 to 10 shows where each $k_y$ responds most to the drive
(`notebooks/sensitivity_analysis.ipynb`, 400 steps):

```{figure} ../_static/figs/paper/sensitivity_gamma.png
:width: 70%
:class: paper-fig

Growth rate $\gamma(k_y, R/L_T)$.
```

```{figure} ../_static/figs/paper/sensitivity_dgamma.png
:width: 70%
:class: paper-fig

Its derivative $\partial\gamma/\partial(R/L_T)$ by AD.
```

## Long runs: forward mode

Reverse mode stores every step, which limits it to short runs. Forward mode
carries the derivative along with the run instead, so its memory does not grow
with the number of steps. Its cost grows with the number of inputs, not
outputs, which suits one or two parameters and many outputs, such as the
whole spectrum. The converged growth rates of the example, at $t = 40$:

```python
converged = lambda rlt: growth_rate(rlt, n_steps=4000)
dgamma_converged = jax.jit(jax.jacfwd(converged))(6.9)
```

On an A100 this takes about 6 s with compilation, against 4.5 s for the run alone,
and it agrees with a central difference to the printed four digits. `jax.jvp`
gives the same derivative together with the value.

## Inverse problem

Given a target potential $\phi^*$, find the $R/L_T$ that produces it. The loss is
the mean squared mismatch of the potential after 400 steps, and its gradient
flows back through all of them. The block reuses `df0`, `geom` and `state0` from
the first block on this page. The optimiser is [Optax](https://github.com/google-deepmind/optax)'s
Adam (`pip install optax`):

```python
import optax

def potential(rlt):
    p = dataclasses.replace(base, rlt=rlt)
    _, phi, _, _ = gk_run(df0, geom, p, state0, 400, pre=linear_precompute(geom, p))
    return phi

phi_target = potential(6.9)

def loss(rlt):
    return jnp.mean(jnp.abs(potential(rlt) - phi_target) ** 2)

value_and_grad = jax.jit(jax.value_and_grad(loss))
opt = optax.adam(0.1)
rlt = jnp.asarray(10.0)
opt_state = opt.init(rlt)
for _ in range(100):
    value, grad = value_and_grad(rlt)
    updates, opt_state = opt.update(grad, opt_state)
    rlt = optax.apply_updates(rlt, updates)
```

From $R/L_T = 10$ it reaches 6.904, against the target of 6.9. The loss has a
single minimum over $R/L_T \in [2, 12]$, so the start does not matter much. The
figures come from `notebooks/inverse_problem.ipynb`, whose slightly different
setup ends at 6.908:

```{figure} ../_static/figs/paper/inverse_spectrum.png
:width: 60%
:class: paper-fig

Initial, target and recovered $k_y$ spectra (top) and $R/L_T$ over the
iterations (bottom).
```

```{figure} ../_static/figs/paper/inv_landscape.png
:width: 50%
:class: paper-fig

Loss and its AD gradient over $R/L_T$, with the optimiser's path.
```

## Checking a gradient

Compare AD with a central difference before trusting a new loss:

```python
f = lambda rlt: jnp.sum(growth_rate(rlt) ** 2)
ad = jax.grad(f)(6.9)
fd = (f(6.9 + 1e-4) - f(6.9 - 1e-4)) / 2e-4
```

The test suite checks that single linear and nonlinear steps differentiate on
every JAX backend configuration (`tests/unit/test_gradients.py`).
