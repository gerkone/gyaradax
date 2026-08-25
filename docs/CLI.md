# The `gyaradax` CLI

`gyaradax` is the single entry point for running simulations. Everything that
defines a run lives in the YAML config; command-line flags only ever *override*
it. There is no `--kinetic` switch and no separate electromagnetic or
multi-GPU script — the config decides.

```bash
gyaradax run   CONFIG [CONFIG ...]   # run a simulation
gyaradax bench CONFIG                # measure solver throughput
gyaradax info                        # devices, backends, versions
```

## What is auto-detected

| Behaviour | Config key | Detected as |
|---|---|---|
| Adiabatic vs kinetic electrons | `grid.adiabatic_electrons` | run mode, species count, output dir name |
| Electromagnetic `A_parallel` | `solver.nlapar` | mixed-variable `g` evolution, Ampere solve |
| Electromagnetic `B_parallel` | `solver.nlbpar` | coupled Poisson–Bpar solve |
| Velocity-space sharding | `sharding.n_gpus_{sp,vp,mu}` | device mesh, multi-GPU XLA flags, sharded precompute |
| Kernel backend | `solver.backend` | `jax` or `cuda` |
| Total steps | `solver.n_steps` | run length |
| Checkpoint cadence | `solver.dump_interval` x `solver.naverage` | steps between checkpoints |

The CUDA backend has no electromagnetic `linear_rhs`. Asking for
`--backend cuda` on a config with `nlapar`/`nlbpar` is a hard error;
`--backend auto` silently falls back to JAX.

## Quick start

```bash
# adiabatic, single GPU
gyaradax run configs/adiabatic_a.yaml --device 0

# same, on the fused CUDA kernels (~1.8x faster)
gyaradax run configs/adiabatic_a.yaml --device 0 --backend cuda

# electromagnetic, kinetic electrons — no extra flags needed
gyaradax run configs/nl_em_apar.yaml --device 0

# check the setup without stepping
gyaradax run configs/nl_em_apar.yaml --dry-run
```

## Multi-GPU

Add a `sharding` block to the config and drop `--device`:

```yaml
sharding:
  n_gpus_sp: 1    # species axis (kinetic multi-species only)
  n_gpus_vp: 4    # vparallel axis
  n_gpus_mu: 1    # mu axis
```

```bash
gyaradax run configs/my_big_case.yaml          # mesh comes from the config
gyaradax run configs/my_case.yaml --n-gpus-vp 4  # or override on the fly
```

The product `n_gpus_sp * n_gpus_vp * n_gpus_mu` must equal the number of
visible GPUs, and each sharded axis must divide evenly. Combining `--device`
with a mesh larger than one is refused rather than silently ignored.

When a mesh is active the CLI uses `sharding.precompute_sharded()`, which
builds the coefficient pytree already distributed — full-size arrays never
materialise on a single device. This is what makes grids larger than one GPU
possible; the plain `linear_precompute()` path would OOM first.

**Sharding buys capacity, not speed.** On grids that already fit in one GH200
it is a net loss (all-reduces over the velocity axes dominate). Reach for it
when a grid does not fit, not to go faster.

## Common options

| Flag | Meaning |
|---|---|
| `--backend {auto,jax,cuda}` | kernel backend (default: config, else `jax`) |
| `--mp` / `--dp` | mixed precision (FP32 nonlinear FFTs) or full FP64 |
| `--device N` | pin to one GPU (single-device runs only) |
| `--device-list 0,1` | restrict which GPUs are visible |
| `--n-gpus-{sp,vp,mu} N` | override the config's mesh |
| `--n-steps` / `--n-blocks` / `--block-size` | run length and checkpoint cadence |
| `--from-scratch` | ignore K-files, cold start |
| `--resume-from K03` | resume from a specific K-dump |
| `--output-dir DIR` | override the output directory |
| `--dry-run` | build everything, print the summary, do not step |

Passing several configs with the same grid runs them batched under one `vmap`:

```bash
gyaradax run configs/case_a.yaml configs/case_b.yaml
```

## Benchmarking

```bash
gyaradax bench configs/adiabatic_a.yaml --device 0 --backend cuda --steps 100 --blocks 5
```

Always cold-starts, does no I/O, and reports the **steady-state** median
throughput — the first block carries lazy-initialisation overhead and would
otherwise skew the mean badly.

## Legacy entry point

`python scripts/run.py CONFIG ...` still works and forwards to `gyaradax run`.
The retired `--kinetic` flag is accepted and ignored with a note.
