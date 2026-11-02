# How it was built

gyaradax was written with coding agents under human supervision, first to port
GKW's Fortran and later to optimise it. What made this work is that numerical
code can be checked: every change had to pass a measurable test before it was
kept.

## The port

The agents started from a prepared workspace: GKW I/O and parsing utilities, a
JAX field solver and flux integrals from earlier work, GKW reference
trajectories with periodic dumps of the full distribution function, and unit
tests comparing against them. The prompt ({ref}`below <prompt>`) asked first for
reading the core Fortran files and manual chapters and taking notes, then for
alternating implementation and tests: linear terms first, the nonlinear term
after. Models that could not reach the tests' tolerances were dropped; the
paper lists which ones did the work.

## The optimisation loop

Performance work ran as a closed loop:

1. **Proposals.** Reasoning models proposed optimisations at three levels (JAX,
   CUDA, algorithmic), reviewed against each other and by a person before any
   code was written.
2. **Implementation** in a separate worktree per proposal, one change at a time.
3. **Per-kernel checks.** Each kernel is compared with the JAX reference on the
   production grids (relative L2 ~1e-16 for the FP64 linear RHS, ~1e-15 for the
   FP64 bracket, ~1e-6 for the mixed-precision bracket), and register and stack
   use are compared across builds.
4. **Trajectory checks.** Restarts from saturated snapshots of long nonlinear
   runs are stepped with the old and the new code and compared bitwise in the
   distribution function. A change that is not bitwise has to reproduce the
   long-run averaged fluxes of the old code, for adiabatic and kinetic electrons.
5. **Benchmarks.** `gyaradax bench` steady-state throughput on an exclusive GPU,
   interleaving old and new code, on the production grids.
6. **Test suite** on GPU, CUDA and JAX backends.

A change ships only when 3-6 pass; options that are faster on some GPUs and
slower on others (`fused_bracket`, `linear_arith: ds`) stay off by default.

Two examples of what the loop catches: the packed "two-for-one" inverse FFT of
the bracket failed the per-kernel check until a Hermitian symmetrisation at
$k_y = 0$ was added (the gyro-averaged potential is not exactly Hermitian
there), and on some grids cuFFT silently skips an FP32 load callback, which a
plan-time probe now detects and works around.

(prompt)=
## The initial prompt

```{include} ../PROMPT.md
:heading-offset: 2
```
