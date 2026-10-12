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
| `config.yaml` | the input config with the command-line choices folded in (backend, precision, mesh, total steps, checkpoint and snapshot cadence, `--save-dumps`) |
| `geometry.pkl` | the geometry dict |
| `fluxes.npz`, `fluxes_em.npz`, `kxspec.npz`, `kyspec.npz`, `growth.npz`, `dt_history.npz` | diagnostics history, one entry per block |
| `step_NNNNNN.npz` | restart snapshot (df, phi, state), refreshed every `--snapshot-every` blocks and after the last; `--save-dumps` keeps them all |
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
total. Resuming a directory continues from its latest snapshot and removes it
once a newer one is written, unless the run archives snapshots (`--save-dumps`,
which the run directory remembers until `--no-save-dumps`); a snapshot named
with `--resume-from` is never removed. Flags still override the stored config
(e.g. `--n-gpus 8` on the next leg).

A run never mixes its outputs with another run's: starting a new trajectory
(cold start, K-file, a snapshot from elsewhere) in a directory that already
holds run outputs, or resuming an older snapshot while later ones exist, stops
with an error; `--overwrite` replaces them instead.

A run shorter than the checkpoint interval (`--n-steps 50` with a block size of
1000) is a single block. The CLI asks whether to keep that short interval for
later legs, which would write a snapshot every few steps; it keeps the original
interval when nobody answers (non-interactive runs).

Diagnostics are appended every block; the restart snapshot, the large file, is
written every `--snapshot-every K` blocks (config `run.snapshot_every`, default
1) and always after the last block. A run that stops in between resumes from
the last snapshot and rewrites the diagnostics after it.
`--resume-from K03` keeps resuming from GKW K-files in the config's `data_dir`.

## Preprocessed dataset output

`--dataset DIR` also writes the run as a training-ready trajectory `DIR/<name>_ifft_realpotens/`,
in the layout the neugk loaders read (`preprocessed_kvikio`), so no intermediate dumps or separate
conversion are needed. The shared kinetic dataset is
`/restricteddata/ukaea/gyrokinetics/preprocessed/kinetic`:

```bash
gyaradax run configs/kinetic_603.yaml --dataset /restricteddata/ukaea/gyrokinetics/preprocessed/kinetic \
    --dataset-name iteration_603_dp4
```

- The trajectory belongs to the run directory that created it (`.gyaradax_run`). An existing
  trajectory of another run, or one not written by a run (including symlinks to the stored
  dataset), is refused: a rerun needs a new `--dataset-name`.

- One frame per checkpoint block: `data/timestep_NNNNN.zstd16.bin` holds df in real space,
  (re/im, species, vpar, mu, s, x, y); adiabatic runs have no species axis.
  `data/poten_NNNNN.zstd16.bin` holds phi in real space, (x, s, y).
- `--dataset-bits zstd16` (default) is lossless zstd over bf16, the format of most of the kinetic
  dataset, read by neugk with `dataset.prefer_dtype: zstd16`; `bf16` stores the raw values, read
  with any `prefer_dtype`.
- `metadata.pkl` and `metadata_light.pkl` carry the run's parameters, the geometry, and per frame
  the time, solver step, fluxes per species, ky/kx spectra and growth rate. The trajectory also
  gets a copy of the run's `config.yaml`. Keys and transforms are those of neugk's
  `preprocess_gyaradax_kinetic`. As in the dump streams that dataset was built from, df is
  rounded to bf16 in spectral space before the transform: on stored trajectories the frames
  match bitwise up to FFT round-off (phi exactly; df in all but ~0.04% of values).
- The df and phi normalisation statistics are computed from the stored frames when the run ends.
  `--dataset-stats-from T` limits them to frames from time T on. Once the transient is known,
  recompute them with:

  ```bash
  gyaradax dataset-stats /restricteddata/ukaea/gyrokinetics/preprocessed/kinetic/iteration_603_dp4_ifft_realpotens --from-time 40
  ```

- The settings are stored in the run's `config.yaml` (`dataset:` section): resuming the run
  directory keeps appending frames to the same trajectory. Frames past the snapshot a run
  resumes from are dropped first.
- Electromagnetic fields (A_par, B_par) are not written yet.

## Debugging: telemetry and profiling

Both are off by default and write into the run directory:

```bash
gyaradax run my_case.yaml --telemetry   # telemetry.jsonl
gyaradax run my_case.yaml --profile     # profile/ + profile_summary.json
gyaradax run my_case.yaml --debug       # both
```

or in the config: `debug: {telemetry: true, profile: true}`. The flags apply to
one invocation and are not stored in the run's `config.yaml`.

- `telemetry.jsonl` has a `start` record (host, git revision, JAX version,
  devices, backend and bracket, precision, mesh, grid, relevant `XLA_*` /
  `CUDA_*` / `GYARADAX_*` environment), the compile time, one `block` record
  per checkpoint block (wall time, ms/step, dt min/mean/max, memory in use and
  its peak since the process started, per device) and an `end` record.
- `--profile` traces the first block after compilation with the JAX profiler
  into `profile/step_NNNNNN/` (the block's first step; open it in Perfetto or
  TensorBoard) and writes that block's GPU time per kernel to
  `profile_summary.json`; that block's telemetry record is marked `profiled`
  because tracing slows it down. A batched run traces once, into the first
  member's directory, and copies the summary to the others.
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
8 GPUs on a two-species grid with `nmu = 16` give `sp=2, mu=4`. Species shards
need nothing from their neighbours; vparallel shards exchange two vpar planes
with each neighbour per RK stage, and with collisions on, vparallel and mu
shards also exchange one plane for the collision stencil (each a fraction of
the step). vparallel is the reserve when more GPUs are wanted than
`nsp * nmu`. All three axes measure about the same speed, with or without
collisions.

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
| `--n-steps N` | run N more steps (default: up to the config's `solver.n_steps`) |
| `--n-blocks` / `--block-size` | run length in blocks, checkpoint cadence |
| `--snapshot-every K` | restart snapshot every K blocks (default 1) |
| `--from-scratch` | cold start: ignore K-files and the run directory's snapshots |
| `--resume-from K03` / `--resume-from DIR/step_000400.npz` | resume from a GKW K-dump or a gyaradax snapshot |
| `--save-dumps` / `--no-save-dumps` | keep every snapshot (default: only the latest); stored in the run directory |
| `--overwrite` | replace another run's outputs (or snapshots past the resumed step) in the output directory |
| `--telemetry` / `--profile` / `--debug` | debug output into the run directory (off by default) |
| `--output-dir DIR` | override the output directory |
| `--dry-run` | build everything, print the summary, do not step |

Passing several configs with the same grid runs them batched under one `vmap`:

```bash
gyaradax run configs/case_a.yaml configs/case_b.yaml
gyaradax run configs/case_a.yaml configs/case_b.yaml --output-dir scan   # scan/<run.name>
gyaradax run scan/case_a scan/case_b --n-steps 2000                      # resume both
```

Each member writes its own run directory exactly as a single run would (config,
diagnostics, snapshots, `run_info.jsonl`, telemetry, `DIVERGED`) and gets the
results of its single run; a member that diverges stops writing while the others go on.
With `--output-dir` the members go to `<dir>/<run.name>`. Members may differ in
array-valued settings (e.g. gradients, `beta`); configs that cannot share one
solve run one after another: different grids, geometries, start steps, step
counts, block or snapshot cadence, sharded runs, or different static
parameters, which include `dt`, the dissipation coefficients and the species
masses, temperatures, densities and thermal velocities (see
`GKParams._STATIC_FIELDS`).

## Benchmarking

```bash
gyaradax bench configs/adiabatic_a.yaml --device 0 --backend cuda --steps 100 --blocks 5
```

Always cold-starts, does no I/O, and reports the **steady-state** median
throughput — the first block carries lazy-initialisation overhead and would
otherwise skew the mean badly. It reports the device memory peak since the
process started (setup included), for the busiest GPU of a mesh.

## Performance notes

- `--backend cuda` is 1.6-1.7x faster than the previous CUDA kernels on
  electrostatic runs; against JAX it is 2.6-3.0x faster on electrostatic and
  2.9-4.2x on electromagnetic nonlinear runs at production grid sizes (H100;
  less on small or linear runs). JAX stays the reference for gradients and
  bitwise comparisons.
- The CUDA nonlinear bracket uses cuFFTDx (v6) when the library was built with
  `nvidia-mathdx`, and the cuFFT-only v5 pipeline for plane sizes v6 has no
  kernel for; `GYARADAX_BRACKET=v5` forces v5 everywhere.
- Grids close to one GPU's memory can fail on allocator fragmentation even when
  they fit: try `--mem-fraction 0.95`, or shard.
- `gyaradax info` shows the devices, whether the CUDA library and the v6
  bracket are available, and whether NCCL is installed.
- Details and measurements: `docs/NOTES.md` §13.

## Campaign scheduler

`scripts/gpu_scheduler.py` runs a list of jobs over a pool of GPU slots that may change while it
runs. It launches `gyaradax run` over ssh and follows each run through files on the shared
filesystem, so the scheduler can be stopped and restarted at any time.

```bash
S=scripts/gpu_scheduler.py
python $S run --pool pool.yaml --jobs jobs.yaml --state camp/state.json [--poll 30] [--dry-run]
python $S status --state camp/state.json          # jobs, slots, progress, ETAs
python $S cancel NAME --state camp/state.json     # stop and hold (--requeue: stop and queue again)
python $S resume NAME --state camp/state.json     # queue a cancelled / failed / diverged job again
```

**Pool.** One entry per slot: `host`, `gpus`, and optionally `speed` (relative throughput),
`python` (default: the gyaradax env), `env`, `name` and `drain`. The file is re-read on every poll:
an added slot gets work at the next poll, and a removed slot has its job stopped and queued again.
`drain: true` lets the slot finish its job and then gives it nothing new. A file that does not
parse, or is empty, is ignored and the previous pool kept. Multi-GPU slots on lizard must be an
NVLink pair (`[0, 1]`, `[2, 3]` or `[4, 5]`); `gpu_groups` sets this rule for other hosts.

```yaml
defaults: {python: /system/apps/userenv/galletti/gyaradax_env/bin/python}
slots:   # one slot per GPU; speeds are placeholders, measure with `gyaradax bench` and adjust
  - {host: scorpion, gpus: [0], speed: 2.0}   # 8 x B300
  - {host: scorpion, gpus: [1], speed: 2.0}
  - {host: scorpion, gpus: [2], speed: 2.0}
  - {host: scorpion, gpus: [3], speed: 2.0}
  - {host: scorpion, gpus: [4], speed: 2.0}
  - {host: scorpion, gpus: [5], speed: 2.0}
  - {host: scorpion, gpus: [6], speed: 2.0}
  - {host: scorpion, gpus: [7], speed: 2.0}
  - {host: shark, gpus: [0], speed: 1.0}      # 4 x A100 SXM
  - {host: shark, gpus: [1], speed: 1.0}
  - {host: shark, gpus: [2], speed: 1.0}
  - {host: shark, gpus: [3], speed: 1.0}
  - {host: lizard, gpus: [0], speed: 0.9}     # 6 x A100 PCIe
  - {host: lizard, gpus: [1], speed: 0.9}
  - {host: lizard, gpus: [2], speed: 0.9}
  - {host: lizard, gpus: [3], speed: 0.9}
  - {host: lizard, gpus: [4], speed: 0.9}
  - {host: lizard, gpus: [5], speed: 0.9}
```

For a two-GPU job, replace two lizard entries with one NVLink pair, e.g.
`{host: lizard, gpus: [0, 1], speed: 0.9}`.

Scorpion runs no sshd, and the other hosts cannot resolve its name. Slots on the machine the
scheduler runs on are launched locally, so a pool with scorpion slots needs the scheduler to run
on scorpion.

**Jobs.** `name`, then `config` (a new run, `output_dir` required) or `run_dir` (resume an
existing run in place), plus optional `args`, `gpus` (default 1), `cost` (default 1) and `env`.
Relative paths are taken from `workdir`, itself relative to the job file. `workdir` is also the
directory the run is launched from. `python -m gyaradax` imports the checkout in the launch
directory before the env's install, so use the repository root, or set `PYTHONPATH` in `env`.
Jobs appended to the file are picked up at the next poll. Removing a job from the file does not
cancel it; use `cancel` for that. `--device`, `--device-list` and `--output-dir` are refused:
the scheduler sets `CUDA_VISIBLE_DEVICES` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` itself.

```yaml
workdir: /system/user/galletti/git/gyaradax-dataset
defaults: {args: --backend cuda --dataset /restricteddata/ukaea/gyrokinetics/preprocessed/kinetic}
jobs:
  - {name: k603, config: configs/kinetic_603.yaml, output_dir: /data/runs/k603, cost: 40}
  - {name: k604_more, run_dir: /data/runs/k604, args: --n-steps 2000, cost: 10}
  - {name: big, config: configs/big.yaml, output_dir: /data/runs/big, gpus: 2, cost: 80}
```

**Placement.** At each poll, pending jobs are taken by decreasing `cost`. Each one gets the
fastest free slot with exactly its GPU count, or else the fastest larger slot. A larger slot is
kept free while a waiting job needs it. A job with `gpus: N > 1` gets `--n-gpus N`, unless its
args or config set a mesh. A job that failed prefers another slot for its retry.

**Launch and monitoring.** The scheduler starts each run detached, through
`gpu_scheduler.py wrap`. The wrapper writes `RUN_DIR/.sched/` (`pid`, `host`, `slot`, `gpus`,
`launch`, a `heartbeat` refreshed every `--heartbeat` seconds, then `exit`), and the run's output
goes to `.sched/log`. The scheduler reads these files and uses ssh only to launch, to signal,
and to check a process whose heartbeat has stopped changing (`kill -0` plus its command line).
What happens next:

| Outcome | Result |
|---|---|
| exit 0 | done |
| nonzero exit, or no heartbeat for `--heartbeat-timeout` and no process | retried up to `--max-retries` times, resuming from the run directory |
| exit 2 with a `DIVERGED` marker | diverged, not retried (a resume would diverge again) |
| `gyaradax run` cannot even start | the slot is paused for `--slot-backoff` |

A retry or a requeued job continues with `gyaradax run RUN_DIR`, dropping `--n-steps`,
`--n-blocks`, `--from-scratch`, `--resume-from` and `--overwrite`, so the leg runs to the target
stored in the run's `config.yaml`. A run stopped before its first snapshot starts over with
`--overwrite`; that flag is only added once the job has written to the directory itself.

**Preemption.** Removing a slot, `cancel` or `--requeue` sends SIGTERM to the wrapper, which
passes it on to `gyaradax`. After `--grace` seconds the scheduler sends SIGKILL to the whole
process group. gyaradax has no SIGTERM handler, so it dies at once: the steps since the last
snapshot are lost, which is up to `--snapshot-every` blocks plus the block in progress. Use a
small `--snapshot-every` for campaigns that get preempted. A kill during a snapshot write leaves
a truncated `step_*.npz`; the scheduler moves it to `.sched/broken/` and resumes from the
previous snapshot. A truncated diagnostics file (`fluxes.npz`, ...) makes gyaradax fail on
resume, so the job is marked failed with a note instead of using up its retries.

**Restarts.** The state file is rewritten atomically after every poll. A restarted scheduler
picks up its running jobs from the state file and their `.sched` files, without relaunching
them. Even without a state file, it reattaches to any live launch it finds in a job's `.sched`.
A lock file (`STATE.lock`) keeps a second scheduler off the same state. `cancel` and `resume`
drop request files next to the state file (`STATE.requests/`), which the scheduler applies at
its next poll.

## Legacy entry point

`python scripts/run.py CONFIG ...` still works and forwards to `gyaradax run`.
The retired `--kinetic` flag is accepted and ignored with a note.
