# Configs

Run any of them with `gyaradax run <config>`; [docs/getting_started/configuration.md](../docs/getting_started/configuration.md) lists the keys.

| folder | content |
|---|---|
| `examples/` | short runs for trying the code: `iteration_13.yaml` (adiabatic ITG, the case of the docs and benchmarks), `adiabatic_a.yaml` and `adiabatic_b.yaml` (two adiabatic ITG cases for batched runs), `linear.yaml`, `kinetic.yaml`, `kinetic_collisions.yaml` |
| `em/` | electromagnetic runs: `nl_em_apar.yaml` (small $A_\parallel$ test grid), `nl_em_waltz_b005.yaml` and `nl_em_waltz_b01.yaml` (Waltz case at $\beta$ = 0.005 and 0.01) |
| `reference/` | every case of the reference dataset [gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests), with the folder layout of the dataset |

`reference/nonlinear/`, `reference/linear/` and `reference/kinetic/` hold the 11 GKW runs of the test suite, with `n_steps` set to the whole GKW run (to $t$ = 320, or 170 for kinetic electrons) and GKW's initial condition. `reference/long_horizon/` holds the long runs, one `_nonlinear` and one `_linear` config per case. Their JSON diagnostics in the dataset are what a run of these configs should reproduce on average: [`notebooks/validation.ipynb`](../notebooks/validation.ipynb) compares them, and [Production cases](../docs/applications/production.md) explains the settings and the physics of each case.
