# Running simulations

`gyaradax` is the single entry point. Everything that defines a run is in the
YAML config ({doc}`configuration`); command-line flags only override it.

```bash
gyaradax run     CONFIG [CONFIG ...]   # run one or several configs
gyaradax run     RUN_DIR               # resume / extend a previous run
gyaradax bench   CONFIG                # measure solver throughput
gyaradax convert GKW_DIR OUT.yaml      # YAML config from a GKW run directory
gyaradax info                          # devices, backends, versions
```

`python -m gyaradax ...` is the same command. `--dry-run` builds everything and
prints the run summary without stepping.

## What the config decides

| Behaviour | Config key |
|---|---|
| adiabatic or kinetic electrons | `grid.adiabatic_electrons` |
| electromagnetic $A_\parallel$ / $B_\parallel$ | `solver.nlapar` / `solver.nlbpar` |
| collisions | a `collisions:` section |
| multi-GPU layout | `sharding.n_gpus` or `sharding.n_gpus_{sp,vp,mu}` |
| backend | `solver.backend` (`jax`, `cuda`, `auto`) |
| run length | `solver.n_steps` |
| checkpoint cadence | `solver.dump_interval` x `solver.naverage` steps |

## Run directory

Every run writes into its output directory (`--output-dir`, default
`outputs_<adiabatic|kinetic>_<run.name>`):

| File | Content |
|---|---|
| `config.yaml` | the input config with the command-line choices folded in |
| `geometry.pkl` | the geometry dict |
| `fluxes.npz`, `fluxes_em.npz`, `kxspec.npz`, `kyspec.npz`, `growth.npz`, `dt_history.npz` | diagnostic histories, one entry per block (`dt_history`: one per step) |
| `step_NNNNNN.npz` | restart snapshot (df, phi, state), refreshed every `--snapshot-every` blocks and after the last |
| `run_info.jsonl` | one line per invocation: command, host, git revision, start and target step |
| `DIVERGED` | written when df turns non-finite; the run stops at that block |

Diagnostics are appended every block; the snapshot, the large file, is written
every `--snapshot-every K` blocks (default 1) and after the last block.
`--save-dumps` keeps every snapshot instead of only the latest.

## Resuming

```bash
gyaradax run outputs_kinetic_my_case                  # continue up to solver.n_steps
gyaradax run outputs_kinetic_my_case --n-steps 2000   # or run 2000 more steps
gyaradax run my_case.yaml --resume-from outputs_kinetic_my_case/step_004000.npz
gyaradax run my_case.yaml --resume-from K03           # from a GKW K-file in run.data_dir
```

A resumed run continues the solver state bitwise as if it had not stopped,
appends to the histories and rewrites `config.yaml` with the new total. Flags
still override the stored config, e.g. `--n-gpus 8` on the next leg. A run never
mixes its outputs with another run's: starting a new trajectory in a directory
that holds run outputs, or resuming an older snapshot while later ones exist,
stops with an error unless `--overwrite` is given. `--from-scratch` cold-starts
and ignores K-files and snapshots.

## Batched runs

Configs that share the grid and the static parameters run batched under one
`jax.vmap`; each writes its own run directory exactly as a single run would:

```bash
gyaradax run configs/case_a.yaml configs/case_b.yaml
gyaradax run configs/case_a.yaml configs/case_b.yaml --output-dir scan   # scan/<run.name>
gyaradax run scan/case_a scan/case_b --n-steps 2000                      # resume both
```

Members may differ in array-valued settings (gradients, `beta`, ...). Configs
that cannot share one solve (different grids, geometries, start steps, cadence,
sharded runs, or different static parameters such as `dt`, the dissipation
coefficients or the species masses) run one after another. A member that
diverges stops writing while the others continue.

## Multi-GPU

```bash
gyaradax run configs/my_big_case.yaml --n-gpus 8             # first 8 visible GPUs
gyaradax run configs/my_big_case.yaml --device-list 0,1,2,3  # these four
```

or in the config:

```yaml
sharding:
  n_gpus: 8         # layout chosen from the grid
  # or one entry per axis:
  # n_gpus_sp: 2    # species (kinetic runs)
  # n_gpus_vp: 1    # vpar
  # n_gpus_mu: 4    # mu
```

The automatic layout fills species first, then mu, then vpar (8 GPUs on a
two-species grid with `nmu = 16` give `sp = 2, mu = 4`). Before JAX starts the
CLI checks that every axis divides its grid dimension and that enough GPUs are
visible, and suggests a valid layout otherwise. Per-axis flags
`--n-gpus-{sp,vp,mu}` override the config. {doc}`../performance/index` discusses
when sharding pays off.

## Debug output

Off by default, written into the run directory:

```bash
gyaradax run my_case.yaml --telemetry   # telemetry.jsonl: per-block wall time, dt, memory
gyaradax run my_case.yaml --profile     # profile/ trace of one block + profile_summary.json
gyaradax run my_case.yaml --debug       # both
```

`telemetry.jsonl` has a `start` record (host, git revision, devices, backend,
bracket version, precision, mesh, grid, relevant environment), the compile time,
one record per block and an `end` record. `--profile` traces the first compiled
block with the JAX profiler (open it in Perfetto or TensorBoard) and writes the
GPU time per kernel to `profile_summary.json`.

## Benchmarking

```bash
gyaradax bench configs/adiabatic_a.yaml --device 0 --backend cuda --steps 100 --blocks 5
```

`bench` always cold-starts, does no I/O and reports the steady-state median
steps per second (the first block carries the compilation), plus the peak device
memory. `--profile` adds the per-kernel breakdown.

## Options

| Flag | Meaning |
|---|---|
| `--backend {auto,jax,cuda}` | kernel backend |
| `--mp` / `--dp` | mixed precision (FP32 nonlinear FFTs, the default) or full FP64 |
| `--fused-linear` | CUDA: fused electrostatic RK stage with collisions |
| `--linear-arith {fp64,ds}` | CUDA: double-single linear RHS for GPUs with weak FP64 |
| `--fused-bracket` | CUDA: mixed-precision bracket packing inside the cuFFT callbacks |
| `--device N` | one GPU (single-device runs only) |
| `--device-list 0,1` | GPUs to use; several shard automatically |
| `--n-gpus N`, `--n-gpus-{sp,vp,mu} N` | mesh size and layout |
| `--mem-fraction F` | fraction of GPU memory XLA may use (e.g. 0.95 near the limit) |
| `--n-steps N` | run N more steps |
| `--n-blocks`, `--block-size` | run length in blocks, checkpoint cadence |
| `--snapshot-every K` | restart snapshot every K blocks |
| `--save-dumps` / `--no-save-dumps` | keep every snapshot / only the latest |
| `--from-scratch`, `--resume-from`, `--overwrite` | start, resume and replace (above) |
| `--telemetry`, `--profile`, `--debug` | debug output |
| `--output-dir DIR` | output directory |
| `--dry-run` | set up and print the summary only |

`gyaradax run --help` lists every flag.
