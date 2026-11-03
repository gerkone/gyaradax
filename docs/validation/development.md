# How it was built

`gyaradax` was written by coding agents, guided and reviewed by the authors. The agents ported GKW's Fortran to JAX, extended it and later wrote the CUDA kernels. The method relies on one property of numerical code: correctness can be measured. Every piece of work had a test to pass before it was kept.

## Porting GKW

The authors first prepared a workspace for the agents.

- **Utilities.** GKW input parsing and file readers, plus a JAX field solver and flux integrals from earlier work. The agents could then focus on the solver.
- **Reference runs.** GKW trajectories with periodic dumps of the full distribution function, not only of fluxes. A mismatch could then be found in the state itself, term by term, instead of in an averaged observable.
- **Tests.** Unit tests that load a GKW dump, step `gyx` and compare the distribution with GKW's next dump.

The [initial prompt](https://github.com/gerkone/gyaradax/blob/master/docs/PROMPT.md) asked the agent to read the core Fortran files and the relevant manual chapters, take notes, and then alternate between implementing and testing. The work was split: the linear terms first, each checked on its own, then the nonlinear term. Some models never reached the tolerances of the tests over several attempts and were dropped. GPT-5.3 Codex did the core translation and Claude 4.6 Opus the kinetic electrons, as reported in the paper. The later extensions (electromagnetic fields, collisions, Miller geometry, multi-GPU sharding) followed the same pattern: a GKW reference case and a test first, then the code.

## Checking against GKW and theory

Agreement is checked at four levels, from exact to statistical.

1. **Same state, few steps.** Starting from the same GKW dump, both codes advance the same distribution. After 80 linear steps the adiabatic distribution agrees with GKW's to 9e-6 (relative L2). With kinetic electrons it agrees to 8e-7 for ions and 2e-6 for electrons after 300 steps. The collision increment of one step matches GKW to 3e-8. Formulas that barely change saturated fluxes, such as the $B_\parallel$ terms, are checked this way at fixed fields, where a wrong factor cannot hide.
2. **Linear growth rates.** Growth rates and frequencies come from long linear runs and are compared with GKW runs of the same input. The Cyclone Base Case matches GKW within 1 % over $k_\theta\rho$ and $R/L_T$. Electromagnetic growth rates match within 0.1 % over a $\beta$ scan through the KBM transition.
3. **Analytical results.** The Rosenbluth-Hinton residual of the zonal flow converges to the Xiao-Catto prediction within 0.1 %, and follows it over $\varepsilon$. The CBC growth rates also match the published benchmark value. Both cases run as unit tests.
4. **Nonlinear statistics.** Turbulent runs are chaotic, so two codes (or two runs of one code on a GPU) diverge after the linear phase. They are compared through time-averaged fluxes and spectra over 46 adiabatic cases, three runs per code, and through kinetic and electromagnetic cases.

The figures and numbers are on {doc}`verification`.

## The optimisation loop

The CUDA kernels came from a closed loop of agents and tests.

1. **Baselines.** Each solver component (field solve, linear RHS, bracket, RK4 step) was benchmarked for time, memory bandwidth and its L2 error against the validated JAX code.
2. **Proposals.** Reasoning models proposed optimisations at three levels: JAX, CUDA and algorithmic. The proposals were reviewed by a second model and by a person before any code was written.
3. **Implementation.** Faster models implemented one proposal at a time, in a separate worktree, against an XLA FFI scaffold and build system set up by the authors.
4. **Per-kernel checks.** Each kernel is compared with the JAX reference on the production grids. The FP64 linear RHS agrees to about 1e-16, the FP64 bracket to about 1e-15 and the mixed-precision bracket to about 1e-6. Register and stack use are compared across builds.
5. **Trajectory checks.** Saturated nonlinear runs are restarted from a snapshot and stepped with the old and the new code. Their distribution functions must be bitwise identical. A change that cannot be bitwise, such as a new FFT length, must reproduce the long-run averaged fluxes of the old code, for adiabatic and kinetic electrons.
6. **Benchmarks and tests.** `gyaradax bench` measures steady-state throughput on an exclusive GPU, alternating old and new code. The test suite runs on GPU for both backends.

A change is kept only when steps 4 to 6 pass. Options that are faster on some GPUs and slower on others (`fused_bracket`, `linear_arith: ds`) stay off by default.

The loop also finds bugs. The packed inverse FFT of the bracket failed the per-kernel check until a Hermitian symmetrisation at $k_y = 0$ was added, because the gyro-averaged potential is not exactly Hermitian there. On some grids cuFFT silently skips an FP32 load callback. A probe at plan time now detects this and switches to an explicit kernel.

## What made it work

- **A measurable target for every task.** Each task came with a test against GKW or against the previous code. Without one, agents produce code that looks right and is not. With one, they iterate until it passes.
- **Full-state references.** GKW dumps of the whole distribution function let a test point at the term that is wrong, not only at a flux that is off.
- **Small steps.** The port went term by term, linear before nonlinear. Kernels were optimised one proposal at a time, each in its own worktree.
- **Written context.** The agents kept notes on the GKW conventions they found, and the repository has an agent guide with the file map, the call graph and the common pitfalls. Every session starts from them instead of from scratch.
- **Frozen references.** Optimisations are compared bitwise with a frozen copy of the previous code. Any difference is either explained or rejected.
- **Strict rules for what counts as done.** A change that is not bitwise needs long runs with matching averaged fluxes. Benchmarks run on an exclusive GPU, alternating old and new code.
- **People where it matters.** The authors chose the physics scope, reviewed the proposals before implementation and judged results the tests could not, such as GKW conventions that differ from the manual or a trend in a long run.
