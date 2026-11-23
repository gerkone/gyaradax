---
name: gyaradax-cli
description: Configure, run, resume and inspect gyaradax simulations with the `gyaradax` CLI (YAML configs, adiabatic or kinetic, linear or nonlinear, electromagnetic, collisions, multi-GPU, batched runs, GKW conversion, diagnostics and plots). Use whenever a simulation has to be set up, run, continued or looked at.
---

# gyaradax CLI

References: `docs/getting_started/configuration.md` (every config key), `docs/getting_started/running.md` (commands, run directory, resume, batching, multi-GPU), `docs/getting_started/examples.md` (ready configs, plotting), `docs/performance/index.md` (kernel flags).

## Configure

Start from the closest config in `configs/{adiabatic,kinetic}/{nonlinear,linear}` or `configs/em/` and change only what the case needs; `gyaradax convert GKW_DIR out.yaml` makes one from a GKW run. Physics, kinetic electrons, EM fields, collisions and sharding live in the config, not in flags.

| Want | Set |
|---|---|
| kinetic electrons | `grid.adiabatic_electrons: false`; per-species lists in `physics`, ions first (`mas [1.0, 2.72e-4]`, `signz [1, -1]`, `tmp [1, Te/Ti]`, one shared `rln`) |
| linear / nonlinear | `solver.non_linear`; nonlinear runs use `adaptive_dt: true`, `finit: noise` |
| electromagnetic | `solver.nlapar` / `nlbpar` and `physics.beta > 0` |
| collisions | a `collisions:` section (`rref`, `nref`, `tref` in m, 1e19 m^-3, keV) |
| shaped geometry | `geometry.geometry_model: miller` and the Miller keys |
| several GPUs | `sharding.n_gpus: N` |

Grids: production kinetic (nvpar, nmu, ns, nkx, nky) = (32, 12, 18, 192, 32), adiabatic (32, 8, 16, 85, 32); kinetic nonlinear runs need nkx 192, and `ikxspace` = floor(6.53 |shat|) for them. Label any other grid as a test grid. Never set `drive_scale: 0`.

## Run

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false
gyaradax run CONFIG --dry-run                                  # build and print the summary
gyaradax run CONFIG --device N --backend cuda --fused-linear [--n-steps S | --t-end T] [--output-dir DIR]
gyaradax run RUN_DIR [--n-steps S]                             # resume bitwise from the last snapshot
gyaradax run a.yaml b.yaml --output-dir scan                   # batched, same grid and static parameters
gyaradax info                                                  # devices, backends, built kernels
```

- Pick a GPU nobody else uses. Add `--linear-arith ds` only on weak-FP64 GPUs (B300).
- Start long runs detached (`nohup setsid ... &`); SIGTERM stops after the running block with a snapshot.
- A directory holding another run is refused without `--overwrite`; `--from-scratch` ignores snapshots and K-files.
- Many independent runs or several machines: the `datagen` skill.

## Inspect

A run directory holds `config.yaml`, `geometry.pkl` (`krho`, `kxrh`), `fluxes.npz` (`time`, `fluxes`: (n, 3) adiabatic or (n, species, 3) kinetic; particle, heat, momentum in gyroBohm units), `kyspec.npz`, `kxspec.npz`, `growth.npz`, `step_*.npz` (`gyaradax.io.rundir.load_checkpoint`) and `DIVERGED` if df went non-finite.

- Plots and run-against-reference comparisons: the `gyaradax-plot` skill.
- Saturated flux: mean over the last 30 % of the run, with its error, and check it no longer drifts. Linear growth rate: the converged late value of `growth` per ky.
- Two runs of a chaotic system are compared by statistics (averaged fluxes, spectra, growth rates), not trajectories.
