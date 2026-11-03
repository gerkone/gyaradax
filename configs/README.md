# Configs

Run any of them with `gyaradax run <config>`; [docs/getting_started/configuration.md](../docs/getting_started/configuration.md) lists the keys.

| folder | content |
|---|---|
| `adiabatic/nonlinear/`, `adiabatic/linear/` | adiabatic-electron ITG runs |
| `kinetic/nonlinear/`, `kinetic/linear/` | kinetic-electron runs |
| `em/` | electromagnetic runs: `nl_em_apar.yaml` (small $A_\parallel$ test grid), `nl_em_waltz_b005.yaml` and `nl_em_waltz_b01.yaml` (Waltz case at $\beta$ = 0.005 and 0.01) |

Most configs in `adiabatic/` and `kinetic/` reproduce a run of the reference dataset [gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests), GKW ones from GKW's initial condition, with `n_steps` set to the original length. Their first line names the dataset diagnostics they should reproduce on average, for example `# reference: long_horizon/kinetic_tem_low_flux/linear.json`. [`notebooks/validation.ipynb`](../notebooks/validation.ipynb) makes the comparison, and [Production cases](../docs/applications/production.md) explains the settings and the physics of each case.

The benchmarks of the [performance page](../docs/performance/index.md) use `adiabatic/nonlinear/itg_high_shear.yaml` and `kinetic/nonlinear/benchmark.yaml`. The second is not in the dataset.
