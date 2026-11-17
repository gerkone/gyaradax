# Linear eigenmodes

A linear gyrokinetic problem is $\partial_t g = L g$, with $L$ the linear operator: streaming, drifts, drives, dissipation, collisions and, through the fields, the coupling of all velocities. Its eigenmodes $L v = \lambda v$ are the linear modes of the plasma, with $\lambda = \gamma + i\omega$: the growth rate and the frequency, signed as in GKW.

An initial-value run only ever shows the fastest-growing mode of each $k_y$, because it outgrows all others. The eigenvalue solver,
{func}`~gyaradax.solver.eigenvalue.eigensolve`, returns several modes per $k_y$: the dominant one, subdominant unstable ones (a TEM below an ITG, say) and damped ones. It is the counterpart of GKW's `eiv_integration`.

## How it works

- **Matrix free.** $L$ has as many rows as the distribution has points per $k_y$ (about $3\cdot10^4$ on the small grid, $3\cdot10^5$ to $3\cdot10^6$ on the production grids), so it is never stored. The solver only needs products $L v$, and these are exactly the right-hand side of an initial-value step, with the field solve and collisions. The eigenvalues are those of the operator the time stepper integrates.
- **One $k_y$ at a time.** Linear modes do not couple different $k_y$, so $L$ is block diagonal. `ky_select` cuts the geometry and the coefficients to one block, which makes the problem $n_{k_y}$ times smaller.
- **Arnoldi.** The solver builds a Krylov basis $v, Av, A^2v, \dots$ and takes the eigenvalues of $A$ projected onto it. These converge first for the eigenvalues of largest magnitude.
- **Time stepping as a filter (`mode="exp"`).** The eigenvalues of $L$ with the largest magnitude are fast, strongly damped modes, not the physical ones. So the solver uses $A = $ `n_steps_per_matvec` RK4 steps instead of $L$. Its eigenvalues are $\mu = e^{\lambda\, n\, \Delta t}$, and the fastest-growing mode now has the largest $|\mu|$, so Arnoldi finds it first.
- **Refinement.** $\lambda = \log\mu / (n\,\Delta t)$ carries the RK4 error, and its imaginary part is ambiguous once $|\omega|\, n\,\Delta t > \pi$. With `refine=True` (the default) each eigenvalue is recomputed as the Rayleigh quotient $v^\dagger L v / v^\dagger v$ of its eigenvector, with $L$ itself.
- **Residual.** $\lVert L v - \lambda v\rVert / (|\lambda|\,\lVert v\rVert)$, again with $L$ itself, says how well each mode has converged. Values near round-off mean a converged mode.

## One $k_y$

With `geometry` and `base` from {doc}`index`:

```python
vals, vecs = gyx.eigensolve(
    geometry, base, k=3, mode="exp", n_steps_per_matvec=250, ky_select=3
)
gamma, omega = vals[0].real, vals[0].imag
```

`vals` holds the `k` eigenvalues sorted by growth rate, and `vecs[i]` is the distribution of mode `i`, with the shape of `df`. The leading eigenvalue matches the initial-value growth rate of the same $k_y$ once that has converged. Here $\gamma \approx 0.256$, which the initial-value run reaches at $t \approx 40$. The other two modes of this $k_y$ grow more slowly, at $\gamma \approx 0.152$ and $0.057$.

What separates the modes is the time window $n\,\Delta t$ of one product, not the number of steps:

- a window of a few time units works. Here $250 \times 0.01 = 2.5$ gives residuals below $10^{-7}$ for every unstable $k_y$ of the spectrum below. A window of 0.5 leaves some at $10^{-4}$ to $10^{-1}$ after the default 40 restarts.
- with kinetic electrons the stable `dt` is about 30 times smaller than with adiabatic ones, so raise `n_steps_per_matvec` by the same factor.
- `dt` (default `params.dt`) must be stable for RK4.

`mode="rhs"` applies $L$ directly. It needs no time step but converges slowly.

## The whole spectrum

{func}`~gyaradax.solver.eigenvalue.eigensolve_ky_spectrum` loops over the $k_y$ blocks and returns the eigenvalues and residuals of each:

```python
spectrum = gyx.eigensolve_ky_spectrum(
    geometry, base, k=2, mode="exp", n_steps_per_matvec=250
)
for iky, (vals, residuals) in spectrum.items():
    print(iky, vals[0], residuals[0])
```

The $k_y = 0$ block holds the zonal modes, whose eigenvalues are close to zero, so its relative residual is large and says nothing about convergence.

## Solvers and accuracy

- `solver="jax"` (default) is a thick-restarted Arnoldi (Krylov-Schur) whose products run on the GPU. Each restart keeps the `k` wanted vectors and extends the basis again up to `ncv`. `solver="arpack"` is SciPy's ARPACK on the same operator, slower, and kept as the reference.
- `tol`, the basis size `ncv` and `n_steps_per_matvec` trade accuracy for cost.
- The solver needs a linear `GKParams` (`non_linear=False`). Its restarts run on the host, so it is not differentiable. Use an initial-value run for gradients
  ({doc}`differentiable`).

{doc}`../validation/verification` compares the eigenvalues with the initial-value runs for adiabatic, kinetic and electromagnetic cases.
