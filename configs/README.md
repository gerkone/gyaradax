# Configs

Run any of them with `gyaradax run <config>`; [docs/getting_started/configuration.md](../docs/getting_started/configuration.md) lists the keys.

| folder | content |
|---|---|
| `examples/` | short runs for trying the code: `iteration_13.yaml` (adiabatic ITG, the case of the docs and benchmarks), `adiabatic_a.yaml` and `adiabatic_b.yaml` (two adiabatic ITG cases for batched runs), `linear.yaml`, `kinetic.yaml`, `kinetic_collisions.yaml` |
| `em/` | electromagnetic runs: `nl_em_apar.yaml` (small $A_\parallel$ test grid), `nl_em_waltz_b005.yaml` and `nl_em_waltz_b01.yaml` (Waltz case at $\beta$ = 0.005 and 0.01) |
| `reference/adiabatic/`, `reference/kinetic/` | the runs of the reference dataset [gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests), split into `nonlinear/` and `linear/` |

The reference configs reproduce whole runs, GKW ones from GKW's initial condition, with `n_steps` set to the original length. The first line of each names the dataset diagnostics it should reproduce on average, for example `# reference: long_horizon/kinetic_tem_low_flux/linear.json`. [`notebooks/validation.ipynb`](../notebooks/validation.ipynb) makes the comparison, and [Production cases](../docs/applications/production.md) explains the settings and the physics of each case.
