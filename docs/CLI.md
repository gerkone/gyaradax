# The `gyaradax` CLI

`gyaradax` is the single entry point for running simulations. Everything that
defines a run lives in the YAML config; command-line flags only ever *override*
it. There is no `--kinetic` switch and no separate electromagnetic or
multi-GPU script — the config decides.

```bash
gyaradax run     CONFIG [CONFIG ...]   # run a simulation
gyaradax run     RUN_DIR               # resume / extend a previous run
gyaradax bench   CONFIG                # measure solver throughput
gyaradax convert GKW_DIR OUT.yaml      # YAML config from a GKW run directory
gyaradax info                          # devices, backends, versions
```

`python -m gyaradax ...` is the same command, for environments where the
console script is not on `PATH` (`pip install -e .` installs it).

## What is auto-detected

| Behaviour | Config key | Detected as |
|---|---|---|
| Adiabatic vs kinetic electrons | `grid.adiabatic_electrons` | run mode, species count, output dir name |
| Electromagnetic `A_parallel` | `solver.nlapar` | mixed-variable `g` evolution, Ampere solve |
| Electromagnetic `B_parallel` | `solver.nlbpar` | coupled Poisson–Bpar solve |
| Velocity-space sharding | `sharding.n_gpus` or `sharding.n_gpus_{sp,vp,mu}` | device mesh, multi-GPU XLA flags, sharded precompute |
| Kernel backend | `solver.backend` | `jax` or `cuda` |
| Total steps | `solver.n_steps` | run length |
| Checkpoint cadence | `solver.dump_interval` x `solver.naverage` | steps between checkpoints |

Both backends run electromagnetic (`nlapar`/`nlbpar`) configs;
`--backend auto` picks CUDA whenever the kernel library is built.

## Quick start

```bash
# adiabatic, single GPU
gyaradax run configs/adiabatic_a.yaml --device 0

# same, on the fused CUDA kernels (~3x faster per step on an H100)
gyaradax run configs/adiabatic_a.yaml --device 0 --backend cuda

# electromagnetic, kinetic electrons — no extra flags needed; CUDA is ~3x faster here too
gyaradax run configs/nl_em_apar.yaml --device 0 --backend cuda

# shard a large run over 4 GPUs, layout chosen from the grid
gyaradax run configs/my_big_case.yaml --n-gpus 4 --backend cuda

# check the setup without stepping
gyaradax run configs/nl_em_apar.yaml --dry-run
```

## Run directory and resuming

Every run writes into its output directory (`--output-dir`, default
`outputs_<mode>_<name>`):

| File | Content |
|---|---|
| `config.yaml` | the input config with the command-line choices folded in (backend, precision, mesh, total steps, checkpoint cadence) |
| `geometry.pkl` | the geometry dict |
| `fluxes.npz`, `fluxes_em.npz`, `kxspec.npz`, `kyspec.npz`, `growth.npz`, `dt_history.npz` | diagnostics history, one entry per block |
| `step_NNNNNN.npz` | restart snapshot (df, phi, state), refreshed every block; `--save-dumps` keeps one per block |
| `run_info.jsonl` | one line per invocation: command, host, git revision, start and target step |
| `telemetry.jsonl`, `profile/`, `profile_summary.json` | debug output, see below |

Pass the directory instead of a config to resume it in place:

```bash
gyaradax run outputs_kinetic_my_case                  # continue up to solver.n_steps
gyaradax run outputs_kinetic_my_case --n-steps 2000   # or run 2000 more steps
gyaradax run my_case.yaml --resume-from outputs_kinetic_my_case/step_004000.npz
```

A resumed run continues the solver state bitwise as if it had not stopped,
appends to the diagnostics history, and rewrites `config.yaml` with the new
total. Flags
still override the stored config (e.g. `--n-gpus 8` on the next leg).
`--resume-from K03` keeps resuming from GKW K-files in the config's `data_dir`.

## Debugging: telemetry and profiling

Both are off by default and write into the run directory:

```bash
gyaradax run my_case.yaml --telemetry   # telemetry.jsonl
gyaradax run my_case.yaml --profile     # profile/ + profile_summary.json
gyaradax run my_case.yaml --debug       # both
```

or in the config: `debug: {telemetry: true, profile: true}`.

- `telemetry.jsonl` has a `start` record (host, git revision, JAX version,
  devices, backend and bracket, precision, mesh, grid, relevant `XLA_*` /
  `CUDA_*` / `GYARADAX_*` environment), the compile time, one `block` record
  per checkpoint block (wall time, ms/step, dt min/mean/max, peak and current
  memory per device) and an `end` record.
- `--profile` traces the first block after compilation with the JAX profiler
  (open `profile/` in Perfetto or TensorBoard) and writes the GPU time per
  kernel to `profile_summary.json`; that block's telemetry record is marked
  `profiled` because tracing slows it down.
- `gyaradax bench CONFIG --profile` prints the same kernel breakdown for a
  timed block.

## Multi-GPU

Give the number of GPUs and let the CLI choose the layout:

```bash
gyaradax run configs/my_big_case.yaml --n-gpus 8             # the first 8 visible GPUs
gyaradax run configs/my_big_case.yaml --device-list 0,1,2,3  # these four GPUs
```

or put it in the config:

```yaml
sharding:
  n_gpus: 8         # layout chosen from the grid
  # or explicitly, one entry per axis:
  # n_gpus_sp: 2    # species axis (kinetic multi-species only)
  # n_gpus_vp: 1    # vparallel axis
  # n_gpus_mu: 4    # mu axis
```

The automatic layout fills the species axis first, then mu, then vparallel:
8 GPUs on a two-species grid with `nmu = 16` give `sp=2, mu=4`. Species and
mu shards need nothing from their neighbours; vparallel shards exchange two
vpar planes with each neighbour per RK stage (a fraction of a percent of the
step) and are the reserve when more GPUs are wanted than `nsp * nmu`. All
three axes measure about the same speed.

Precedence: `--n-gpus-{sp,vp,mu}` (each overrides the config's value for that
axis) > `--n-gpus` > the config's per-axis entries > `sharding.n_gpus` >
a `--device-list` with several GPUs. Before JAX starts, the CLI checks that
each axis divides its grid dimension (vparallel keeps at least two points per
GPU) and that enough GPUs are visible, and suggests a valid layout otherwise.
The run summary prints the mesh and the df block each GPU holds. Combining
`--device` with a mesh larger than one is refused rather than silently ignored.

When a mesh is active the CLI uses `sharding.precompute_sharded()`, which
builds the coefficient pytree already distributed — full-size arrays never
materialise on a single device — and the solver runs the CUDA kernels and the
nonlinear bracket on each GPU's local block. Multi-GPU runs need NCCL
(`nvidia-nccl-cu13`, part of the `cuda13` extra); `gyaradax info` reports it.

Sharding scales close to linearly on large grids (two B300:
593 -> 292 ms per step on 2 x 64 x 16 x 32 x 85 x 64, CUDA). On small grids the
per-step field all-reduce and launch overheads weigh more; shard when a grid
is large or does not fit one GPU.

## Common options

| Flag | Meaning |
|---|---|
| `--backend {auto,jax,cuda}` | kernel backend (default: config, else `jax`) |
| `--mp` / `--dp` | mixed precision (FP32 nonlinear FFTs) or full FP64 |
| `--device N` | pin to one GPU (single-device runs only) |
| `--device-list 0,1` | GPUs to use; several with no mesh given shard over them automatically |
| `--n-gpus N` | shard over N GPUs, layout chosen from the grid |
| `--n-gpus-{sp,vp,mu} N` | set the mesh per axis (overrides the config) |
| `--mem-fraction F` | fraction of GPU memory XLA may use (e.g. 0.95 for grids near the limit) |
| `--n-steps` / `--n-blocks` / `--block-size` | run length and checkpoint cadence |
| `--from-scratch` | ignore K-files and snapshots, cold start |
| `--resume-from K03` / `--resume-from DIR/step_000400.npz` | resume from a GKW K-dump or a gyaradax snapshot |
| `--save-dumps` | keep a df snapshot every block (default: only the latest) |
| `--telemetry` / `--profile` / `--debug` | debug output into the run directory (off by default) |
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
otherwise skew the mean badly. With a mesh (`--n-gpus 4`) it reports the peak
memory of the busiest GPU.

## Performance notes

- `--backend cuda` is 1.6-1.7x faster than the previous CUDA kernels on
  electrostatic runs and 2.8-4.4x faster than JAX on electromagnetic ones
  (H100); JAX stays the reference for gradients and bitwise comparisons.
- The CUDA nonlinear bracket uses cuFFTDx (v6) when the library was built with
  `nvidia-mathdx`; `GYARADAX_BRACKET=v5` forces the cuFFT-only pipeline.
- Grids close to one GPU's memory can fail on allocator fragmentation even when
  they fit: try `--mem-fraction 0.95`, or shard.
- `gyaradax info` shows the devices, whether the CUDA library and the v6
  bracket are available, and whether NCCL is installed.
- Details and measurements: `docs/NOTES.md` §13.

## Legacy entry point

`python scripts/run.py CONFIG ...` still works and forwards to `gyaradax run`.
The retired `--kinetic` flag is accepted and ignored with a note.
