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
| `linear_gate.json`, `linear_mode.npz`, `outcome.json`, `monitor_series.npz` | run outcome, with the flags of "Run outcome" below |

SIGTERM stops a run after the block in progress and its snapshot, with exit code 75 (a second
SIGTERM exits at once); resuming the directory continues it from there. The dataset statistics
are then computed by the leg that completes the run.

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

### Equally spaced frames (time mode)

`--dump-dt DT` (config `solver.dump_dt`) puts the frames (diagnostics, dataset frames) on an
equally spaced time grid instead of every `--block-size` steps. Each block integrates to the next
frame time and shortens its last step to land on it exactly, so the frame times are
`t0 + k * DT` to round-off, whatever the adaptive time step does. `t0` is the time the run starts
from; snapshots and the dataset resume dump carry it (`dump_origin`), so resumed legs continue
the same grid. The run length is an end time:

```bash
gyaradax run my_case.yaml --dump-dt 1.4 --t-end 400     # frames at 1.4, 2.8, ..., ends at 400.4
gyaradax run outputs_kinetic_my_case --n-blocks 10      # 10 more frames
gyaradax run outputs_kinetic_my_case --t-end 600        # extend to t = 600
```

- The run ends on the first frame at or after `--t-end` (config `solver.t_end`), `--n-blocks N`
  runs N more frames, and `--n-steps N` (or `solver.n_steps` when there is no end time) ends on
  the first frame at or after N steps.
- `--snapshot-every K` counts frames. `--block-size` is unused: a compiled call takes at most 8x
  the steps of a frame at `solver.dt` and continues in another call when the time step is smaller.
- SIGTERM stops a time-mode run on the frame in progress, like a block.
- The stored kinetic trajectories were thinned to a minimum spacing of 1.4 (their native spacing
  was 2.18-2.27); `--dump-dt 1.4` matches it.
- Without `--dump-dt` nothing changes: the step-counted blocks are bitwise as before.

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
- The bf16 loss is tracked per frame: `quant_phi_err` is the relative L2 error of the stored phi
  frame, `quant_phi_df_err` that of phi solved from the stored df frame, and `quant_flux_err` the
  relative error of each flux (`flux_labels`) recomputed from the stored df. The reference is the
  same field solve and flux integral on the unrounded df, so the numbers hold the quantization
  alone; channels zero to round-off are nan. The run prints their medians when it ends.
- The df and phi normalisation statistics are computed from the stored frames when the run ends.
  `--dataset-stats-from T` limits them to frames from time T on. Once the transient is known,
  recompute them with:

  ```bash
  gyaradax dataset-stats /restricteddata/ukaea/gyrokinetics/preprocessed/kinetic/iteration_603_dp4_ifft_realpotens --from-time 40
  ```

- The settings are stored in the run's `config.yaml` (`dataset:` section): resuming the run
  directory keeps appending frames to the same trajectory. Frames past the snapshot a run
  resumes from are dropped first.
- Every leg replaces the trajectory's resume dump when it ends, finished or stopped by SIGTERM:
  `resume/state.npz` holds df and the solver state of the last frame (and the frame grid origin
  in time mode), and `resume/config.yaml` the leg's config. Each file is replaced atomically.
  `--resume-dtype f32` (default) stores df as complex64, `f64` as complex128; on the kinetic
  production grid (2 x 32 x 12 x 18 x 192 x 32) that is about 680 MB or 1.36 GB per trajectory.
  The setting is kept in the `dataset:` section.
- A trajectory can be extended from its resume dump without its run directory:

  ```bash
  gyaradax run DIR/iteration_603_dp4_ifft_realpotens --t-end 600 --output-dir outputs_603_more
  gyaradax run my_case.yaml --resume-from DIR/iteration_603_dp4_ifft_realpotens --n-blocks 50
  ```

  The first form runs the dump's own config; the second a config of your choice. Frames are
  appended to the same trajectory (on the same time grid in time mode), and the trajectory passes
  to the new run directory, so its previous run can no longer write to it. `--dataset` writes the
  continuation into a new trajectory instead. From an `f64` dump the continuation is bitwise the
  run that never stopped; from `f32` it starts from df rounded to complex64.
- Electromagnetic fields (A_par, B_par) are not written yet.
- With `--linear-gate` the trajectory also holds the linear eigenmode and the linear results, and
  with any outcome flag the full per-frame traces and the outcome (see "Run outcome").

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
| `--n-blocks` / `--block-size` | run length in blocks (frames in time mode), checkpoint cadence |
| `--dump-dt DT` / `--t-end T` | time mode: frames every DT time units, run to the first frame at or after T |
| `--snapshot-every K` | restart snapshot every K blocks (default 1) |
| `--from-scratch` | cold start: ignore K-files and the run directory's snapshots |
| `--resume-from K03` / `--resume-from DIR/step_000400.npz` / `--resume-from TRAJECTORY` | resume from a GKW K-dump, a gyaradax snapshot, or a dataset trajectory's resume dump |
| `--resume-dtype {f32,f64}` | df precision of the dataset trajectory's resume dump (default f32) |
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
counts, block or snapshot cadence, dump spacing, end time or start time in time
mode, sharded runs, or different static
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

## Linear growth check

A linear run of a config gives per-ky growth rates γ(kθρ). Run at a finer parallel resolution, it
is a ground truth for the production grid and dissipation: if γ at the production `ns` matches
γ at `ns 36`, the parallel direction is resolved for that parameter point; if a dissipation
change (e.g. `solver.disp_par`) moves γ away from it, the change alters the physics.

```bash
gyaradax growth my_case.yaml --ns 18 36                      # one linear run per resolution
gyaradax growth my_case.yaml --ns 18 36 --set solver.disp_par=4.0 --output growth.json
gyaradax run my_case.yaml --linear-check 36                  # check, then the nonlinear run
gyaradax run my_case.yaml --linear-check 24,36 --linear-check-tol 0.1
```

- **Linear run.** `solver.non_linear` is switched off, the geometry is computed from the
  config, and `--ns N` sets `grid.ns` (with `geometry.sgr_dist` scaled to keep the field-line
  length). `--set KEY=VALUE` overrides any other entry. The run is unsharded on one GPU.
- **Time step.** Capped at the linear stability estimate (`estimate_linear_timestep`), as GKW does
  for every explicit run; the input `solver.dt` is kept when it is below the estimate.
- **Stop.** After `t ≥ 15`, once the unstable ky (kθρ ≤ 1, γ > 0.02) vary by less than 0.005
  over 6 time units, or at `--t-end` (default 40). γ is the mean over those last 6 time units; a
  row marked "not settled" reached `--t-end` first.
- **Report.** Per resolution: the peak γ and its kθρ, γ at the dominant kθρ of the finest
  resolution (the reference), and the relative deviation from it.
- **`run --linear-check NS[,NS...]`** (off by default) runs this before the nonlinear run, at the
  run's own `ns` and the listed ones, writes `RUN_DIR/linear_check.json` and prints the table.
  A resumed run reuses the file while the physics of the config is unchanged (its name, length,
  backend and layout do not count). With `--linear-check-tol X` the run exits with code 3,
  without starting, when the deviation exceeds `X`; without it the check only reports.
  `--linear-check-t-end` caps each linear run.

## Run outcome: linear gate, decay and banding

For campaigns near the turbulence threshold, `gyaradax run` can skip linearly stable configs,
stop runs whose turbulence has died out, and flag (or stop) runs with the s-Nyquist banding
mode. All of it is off by default; without these flags a run is bitwise what it was.

```bash
gyaradax run my_case.yaml --linear-gate --stop-on-decay --dataset DIR    # the low-drive campaign
gyaradax run my_case.yaml --track-outcome                                # record only
gyaradax run my_case.yaml --stop-on-banding --banding-threshold 0.3
```

**Linear gate** (`--linear-gate`). Before the nonlinear run, a linear run of the config on its own
grid (`solver.non_linear` off, unsharded, time step capped at the linear stability estimate, as
`gyaradax growth`). It stops once the growth rates of every ky > 0 vary by less than 0.005 over
6 time units after t = 15, or at `--linear-gate-t-end` (default 60).

| Decision | Condition | Effect |
|---|---|---|
| `stable` | settled, and the largest γ is below `--linear-gate-stable` (default -0.01) | exit 4, no nonlinear run |
| `unstable` | the largest γ is above `--linear-gate-gamma-min` (default 0.02) | the nonlinear run goes ahead |
| `ambiguous` | anything else: between the two, or not settled | the nonlinear run goes ahead |

The gate only ever skips stable configs: linearly unstable does not mean turbulent (Dimits
shift), and the nonlinear run decides that.

- `RUN_DIR/linear_gate.json`: ky, γ(ky) and its spread over the last 6 time units, the real
  frequency ω(ky) and its spread, `settled`, `gamma_max`, `peak_ky`, `omega_peak`,
  `gamma_max_lowky`, `peak_ky_lowky`, the end time,
  time steps and wall time, and the decision. ω = d arg(Σ φ)/dt with GKW's sign
  (`diagnos_growth_freq`), the sum over s and the kx chain through kx = 0, measured over four
  10-step blocks after the growth rates settled.
- `RUN_DIR/linear_mode.npz`: the final linear eigenmode, df (complex64) and φ, normalized per
  connected kx chain as the linear solver keeps it.
- Both are reused on resume while the physics of the config is unchanged (as `--linear-check`);
  changed thresholds re-decide from the cached rates.
- With `--dataset`, the trajectory gets the eigenmode in the layout of the adiabatic dataset's
  paired linear runs (`data/linear.bin`, `data/linear_k.bin`): raw float32, `linear.bin` in real
  space with the shape of a df frame (`linear_shape`, here with the species axis), `linear_k.bin`
  the spectral df as re/im (`linear_k_bin_shape`), plus `data/linear_poten.bin`, φ in real space
  (x, s, y). The metadata carries `linear_bin`, `linear_space: real`, `linear_k_bin`, the
  `linear_rms*` moments of neugk's `linear_moments` (a species axis is reduced with the others),
  and `linear_ky`, `linear_gamma`, `linear_gamma_std`, `linear_omega`, `linear_omega_std`,
  `linear_settled`, `linear_peak_ky`, `linear_gamma_max`, `linear_gamma_max_lowky`,
  `linear_low_ky`, `linear_gate` (the decision), `linear_t` (end of the linear run),
  `linear_time` (time of the stored eigenmode), `linear_source`, `linear_dt` (the linear time
  step), and the fluxes of the final eigenmode below.
- **Eigenmode fluxes.** `linear_gate.json` also holds `mode_fluxes`: the flux integrals of the
  solver diagnostics on the final eigenmode (physical f, plus the A_par / B_par flutter parts),
  per species, channel (particle, heat, momentum) and ky. In the metadata, in the `flux_labels`
  layout: `linear_flux` (the sum over ky and kx), `linear_flux_ky` (`flux_labels` x ky, summed
  over kx), `linear_phi2_ky` (|φ|² averaged over the flux surface, summed over kx), the same over
  the kx chain through kx = 0 alone (`linear_flux_ky_chain0`, `linear_phi2_ky_chain0`: the mode
  whose γ and ω are reported), and the quasilinear weights `linear_weight_ky` =
  `linear_flux_ky_chain0 / linear_phi2_ky_chain0` and `linear_weight_ky_allkx` =
  `linear_flux_ky / linear_phi2_ky` (nan where |φ|² is 0). The linear run normalizes every kx
  chain to unit amplitude on its own, so fluxes and |φ|² have arbitrary size and only their
  ratios within a chain are physical; the all-kx weight averages the chains with equal weight.
- **Linear only** (`--linear-only`). The linear run of the gate alone, with no nonlinear run and
  no trajectory: `RUN_DIR/linear.json` (the `linear_gate.json` document plus the config, the
  eigenmode time and the file shapes) and the eigenmode as `RUN_DIR/data/linear.bin`,
  `linear_k.bin` and `linear_poten.bin` (the trajectory layout above; `linear_mode.npz` is removed
  once they are written). `outcome.json` gets class `linear_only` and the run exits 0, so the
  datagen scheduler marks it `done` with the gate results in its `outcomes` table. A repeat with
  the outputs present returns at once. `gyaradax.io.dataset.attach_linear` adds such a staged
  eigenmode to an existing trajectory (files and `linear_*` keys only, metadata backed up).
- **Low-ky peak.** `gamma_max` and the decision use every ky > 0 of the grid. `gamma_max_lowky`
  and `peak_ky_lowky` are the same over kθρ <= `--linear-gate-low-ky` (default 1.0), reported
  alongside (in `linear_gate.json`, the trajectory metadata and the `outcomes` table) to see
  whether a high-ky branch, possibly spurious on the coarse grid, is what keeps a case unstable.
- **Stable jobs still get a trajectory** (with `--dataset`): the static metadata of any
  trajectory, the linear results and eigenmode files above, `stable: True`, `frame_source:
  linear` and `outcome`, and one data frame (`timestep_00000` / `poten_00000` and the per-frame
  series) holding the final linear state: df and φ of `linear_mode.npz`, at the linear run's end
  time, solver step 0, its fluxes (of the normalized eigenmode, so their size is arbitrary) and
  γ(ky) as the growth rate. Like the stored stable runs (e.g. `iteration_502`, a single frame) it
  gives the loaders no training pairs; no zero-frame trajectory is written. It has no resume
  dump: a nonlinear leg should not start from the eigenmode.

**Outcome tracking.** Any of `--linear-gate`, `--stop-on-decay`, `--stop-on-banding`,
`--track-outcome` or `--dataset-from-flux` follows the run frame by frame:

- `RUN_DIR/monitor_series.npz`: per frame the time, step, fluxes, growth rates and the s-Nyquist
  shares of φ. A resumed leg continues it from its start step.
- `RUN_DIR/outcome.json`, rewritten every frame: the outcome class, a one-line `label`, the
  stop reason, frames and time reached, the heat-flux peak and last value, the banding flag and
  the time it was first flagged, the last and largest zonal s-Nyquist share, and the flux
  averages. The flags are command-line only, so the next leg needs them again (the datagen
  scheduler passes every job's arguments to every leg).

| Class | Meaning |
|---|---|
| `linear_stable` | skipped by the linear gate |
| `linear_only` | `--linear-only`: the linear run alone |
| `decayed` | stopped by `--stop-on-decay` |
| `banded` | banding flagged (stopped or not) |
| `flat` | mean heat flux over the averaging window below `--decay-floor` |
| `growing` / `decaying` | the heat flux drifts by more than 20% of its mean over the window, at two standard errors |
| `saturated` | none of the above |
| `running` | the run has not finished (an interrupted leg) |

**Flux averages.** Over the last 30% of the run time (the whole series, all legs): the mean of
every flux (per species: particle, heat, momentum) and of the total heat flux, the batch-means
error (5 consecutive batches, standard error of their means), and the trend: a line fitted to the
batch means, its change over the window relative to the mean (`drift`) and that change in
standard errors (`drift_sigma`).

**Stop on decay** (`--stop-on-decay`, exit 5). After `--decay-t-min` (default 30) the run stops
once the total heat flux has stayed below the floor for `--decay-window` frames (default 5) and
still falls over them (negative slope of log Q, and the last frame below the first). The floor is
the larger of `--decay-floor` (default 1e-3, gyro-Bohm) and `--decay-rel-floor` (default 1e-3)
times the peak heat flux. The falling condition keeps a slow linear phase, small but rising, from
counting. The stop frame gets a snapshot, so the run directory can still be extended (without the
flag). With `--dataset` the trajectory then keeps only that last frame (renumbered 0, as the
stored stable runs), unless `--decay-keep-frames`.

**Banding** (`--track-outcome`, `--stop-on-banding`). Per frame, φ in real space (x, s, y), the
stored-frame transform: zonal part = mean over y; its FFT along s; the zonal share is the power at
index ns // 2 summed over x, over the total zonal power. The non-zonal share is the same for φ
minus its zonal part. Banding is flagged once the zonal share exceeds `--banding-threshold`
(default 0.3) for `--banding-window` frames in a row (default 3). With `--stop-on-banding` the run
stops there (exit 6); otherwise it is only recorded.

**Deferred dataset frames** (`--dataset-from-flux Q`). Dataset frames are written only from the
first frame whose total heat flux reaches Q. A run that never does, or stops first, still writes
its last frame. The run directory diagnostics are complete either way.

**Trajectory metadata.** With any of these flags and `--dataset`, the metadata also holds the
full per-frame traces of the run (including frames not stored): `trace_timesteps`,
`trace_solver_steps`, `trace_flux` (frames x `flux_labels`), `trace_growth`, `trace_nyq_zonal`,
`trace_nyq_nonzonal`, plus `outcome` (the `outcome.json` document) and `stable` (class
`linear_stable`, `decayed` or `flat`).

| Exit code | Meaning |
|---|---|
| 4 | linearly stable, nonlinear run skipped (`--linear-gate`) |
| 5 | stopped on decay (`--stop-on-decay`) |
| 6 | stopped on banding (`--stop-on-banding`) |

These flags take one config: they are refused for batched runs.

## Data generation (`gyaradax datagen`)

`gyaradax datagen` runs a list of configs over an ssh GPU mesh and collects their trajectories
into a dataset. Everything else is derived: each config becomes a job with its own run directory,
trajectory name and cost, and the scheduler places, follows, retries and resumes the jobs until
they are done.

```bash
gyaradax datagen run configs/rerun/kinetic_*.yaml --campaign /system/user/publicwork/galletti/campaigns/rerun_dp4 \
    --targets targets.yaml --dataset /restricteddata/ukaea/gyrokinetics/preprocessed/kinetic \
    --name "iteration_{id}_dp4" --gpus all --detach
gyaradax datagen status --campaign DIR             # jobs, slots, progress, ETAs
gyaradax datagen add more/*.yaml --campaign DIR    # more configs, picked up at the next poll
gyaradax datagen cancel NAME --campaign DIR        # stop and hold (--requeue: stop and queue again)
gyaradax datagen resume NAME --campaign DIR        # queue a cancelled / failed / diverged job again
gyaradax datagen probe --campaign DIR              # the CUDA library on every target host
```

Run it from the checkout whose code the campaign should use: the jobs are launched from it.
`--detach` keeps the scheduler going after the terminal closes (log in `DIR/datagen.log`). The
scheduler exits once every job has finished; `--keep-running` keeps it polling for added jobs.
Running `gyaradax datagen run --campaign DIR` again (no configs) restarts the scheduler; it
reattaches to the running jobs.

**Campaign directory.** Created on first use:

| Path | Content |
|---|---|
| `campaign.yaml` | settings fixed at creation: `--dataset`, `--name`, `--args`, `--gpus`, the outcome switches, the code (checkout and commit) |
| `targets.yaml` | where jobs run (copied from `--targets`); edit it while the campaign runs |
| `configs/<name>.yaml` | each config as run, with `run.name` set to the job name |
| `runs/<name>/` | the run directories |
| `jobs.yaml`, `state.json` | the generated job list and the scheduler state |
| `state.sqlite` | the run database (below), rebuilt on every poll |

- **Names.** `--name` is a template over `{run}` (the config's `run.name`, the default), `{stem}`
  (the file name) and `{id}` (the last number in the file name). The name is used for the job, its
  run directory and its trajectory. `iteration_{id}_dp4` turns `kinetic_534.yaml` into
  `iteration_534_dp4`.
- **Dataset.** With `--dataset DIR`, every run writes `DIR/<name>_ifft_realpotens` (see
  "Preprocessed dataset output"). A name that already exists in DIR, and is not this campaign's,
  is refused before anything starts.
- **Cost.** Grid points times species times `solver.n_steps`, used to place the longest jobs first;
  in time mode the steps are `solver.t_end / solver.dt`.
- **Equally spaced frames.** Configs with `solver.dump_dt` and `solver.t_end` run in time mode (see
  "Equally spaced frames"); the end time stays in each run's `config.yaml`, so resumed legs run to
  it.
- **GPUs per job.** 1 by default. `--gpus all` gives every job all the GPUs of the slot it lands
  on, and each leg sizes its mesh (`--n-gpus`) to that slot, so a run can resume on a slot with a
  different GPU count. A config whose `sharding` asks for several GPUs is refused unless `--gpus`
  is given. Sharding is close to linear (kinetic, 192 × 32 × 18 × 32 × 12: 1.93× on 2 A100s,
  3.78× on 4), so wide slots return runs sooner at almost no cost in throughput.
- **Run database.** `state.sqlite` next to `state.json` mirrors the queue for SQL queries: `runs`
  (state, slot, legs, start and finish times, dataset frames written, simulation time, ETA, the
  config's `geometry`/`physics`/`grid` as JSON in `params`, the outcome class and banding flag),
  `legs` (one row per launch: slot, times, exit code, outcome), `slots` (drain, busy, current job),
  `outcomes` (below) and `meta`. It is rebuilt and swapped in whole on every poll, so a reader on
  any host sees a complete file; read it, do not write it. Example: `sqlite3 state.sqlite "SELECT
  name, state, frames, frames_total, eta_h, json_extract(params, '$.geometry.eps') FROM runs ORDER
  BY state"`.
- **Outcome switches.** `--linear-gate`, `--stop-on-decay`, `--stop-on-banding` and
  `--track-outcome` add the `gyaradax run` flag of the same name to every job (see "Run
  outcome"); their thresholds go in `--args`, e.g. `--args "--decay-t-min 40"`. A linearly
  stable job ends as `stable`, a decayed one as `decayed`, a run stopped on banding as `banded`;
  none is retried, and `resume` does not requeue them (continue such a run by hand with
  `gyaradax run RUN_DIR` and no stop flag). `status` adds an OUTCOME column with each run's
  `outcome.json` label once one exists.
- **Outcome table.** `outcomes` has one row per job, from the config and the run's
  `linear_gate.json` / `outcome.json` (filled as far as they exist, the scheduler reads them
  without numpy):

  | Columns | Content |
  |---|---|
  | `name`, `state`, `class`, `label` | job, scheduler state, outcome class (see "Run outcome") |
  | `rlt`, `rln`, `tmp`, `mas` | per-species lists (JSON) from `physics` |
  | `rlt_i`, `rlt_e`, `rln_i`, `temp_ratio` | ion and electron gradients, Te/Ti (species from `signz`) |
  | `q`, `shat`, `eps`, `grid` | geometry, and the `grid` section (JSON) |
  | `gate`, `settled`, `gamma_max`, `peak_ky`, `omega_peak`, `linear_t` | linear gate decision and results |
  | `gamma_max_lowky`, `peak_ky_lowky`, `low_ky` | the largest γ over kθρ <= `low_ky` and its kθρ |
  | `ky`, `gamma`, `gamma_std`, `omega`, `omega_std` | linear spectra (JSON lists) |
  | `t`, `frames`, `dataset_frames`, `stopped` | time and frames reached, frames in the trajectory, stop reason |
  | `banded`, `banded_t`, `nyq_zonal`, `nyq_nonzonal`, `nyq_zonal_max` | banding flag, its first time, last and largest s-Nyquist shares |
  | `q_peak`, `q_last` | total heat flux, peak and last frame |
  | `avg_t0`, `avg_t1`, `avg_frames` | averaging window (last 30% of the run) |
  | `qi`, `qi_err`, `qe`, `qe_err`, `gi`, `gi_err` | mean ion / electron heat flux and ion particle flux, batch-means errors |
  | `q_drift`, `q_drift_sigma`, `fluxes` | trend of the total heat flux; every average as JSON |
  | `cost`, `gpu_hours` | the job's cost estimate, GPU hours over all legs |

  Example: `sqlite3 state.sqlite "SELECT name, class, gamma_max, qi, qi_err, q_drift FROM
  outcomes ORDER BY class"`.
- **Unchanged configs.** Adding a config that is already in the campaign is a no-op. Adding a
  changed config under an existing name, or changing the settings, is refused: start another
  campaign for that.

### Targets: ssh mesh

One entry per slot: `host`, `gpus`, and optionally `speed` (throughput per GPU, relative), `python`
(default: the gyaradax env), `drain` and `busy_mib`. A `defaults` mapping applies to every slot,
and its `env` (a mapping of environment variables for every run) can only be set there. The file
is re-read on every poll:

- an added slot gets work at the next poll;
- a removed slot has its job stopped and queued again;
- `drain: true` lets the slot finish its job and then gives it nothing new;
- `busy_mib: N` (0, the default, turns it off) places nothing on a slot while one of its GPUs holds
  more than N MiB, i.e. while someone else uses it (`nvidia-smi` on the host, once per poll);
- a file that does not parse, or is empty, is ignored and the previous pool kept.

A `hosts` section switches whole machines; a host without an entry is `on`:

| Switch | Effect at the next poll |
|---|---|
| `on` | the host's slots take jobs |
| `drain` | its jobs run to the end; nothing new is placed there |
| `off` | its jobs get SIGTERM, write a snapshot and exit (up to `--grace`), then go back in the queue |
| `kill` | its jobs get SIGKILL at once and go back in the queue; the next leg resumes from the last snapshot |

`gyaradax datagen host HOST SWITCH --campaign DIR` sets the switch in the campaign's
`targets.yaml` (rewriting the file, so comments in it are dropped); a host name also matches its
short form (`shark` matches `shark.example.org`). A stopped or killed job does not count as a
failure. Restart snapshots keep df and φ in complex128 (`--snapshot-dtype`, default `c128`)
whatever the dataset's precision, so `kill` loses at most the frames since the last snapshot
(`--snapshot-every`).

```yaml
hosts: {shark: drain, coyote: kill}
```

Jobs with `--gpus all` go to the free slot with the largest `speed` × GPUs.

Multi-GPU slots on lizard must be an NVLink pair (`[0, 1]`, `[2, 3]` or `[4, 5]`).

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

For two-GPU runs (`--gpus all`), replace two lizard entries with one NVLink pair, e.g.
`{host: lizard, gpus: [0, 1], speed: 0.9}`.

Scorpion runs no sshd, and the other hosts cannot resolve its name. Slots on the machine the
scheduler runs on are launched locally, so a pool with scorpion slots needs the scheduler to run
on scorpion.

### CUDA library per environment

The CUDA library links the CUDA runtime, cuFFT and nvJitLink of the Python environment it is
built in, so it belongs to that environment rather than to a machine:

```bash
gyaradax build-cuda --archs 80,90,103     # once per environment and kernel version
```

- **Where it goes.** `$PREFIX/lib/gyaradax/<key>/`, where `key` hashes the kernel sources and the
  jaxlib version, so every checkout loads a library built from its own kernels. Without
  `--archs`, the GPUs of the build machine are used. The local machines share one environment:
  one build for sm 80, 90 and 103 covers the A100s and the B300s.
- **Lookup order.** `$GYARADAX_CUDA_LIB`, then the environment's build for the key, then a
  development build next to the sources (`cuda_kernels/libgyaradax_cuda.so`).
- **Missing architectures.** A build whose record lacks the GPU's architecture is not used, and
  `--backend auto` falls back to JAX.
- **Backend of a job.** Jobs whose `--args` do not set `--backend` run with `--backend auto`,
  overriding the config's `solver.backend`: CUDA where the host's library serves its GPU, JAX
  elsewhere. Before its first launch on a host, the scheduler probes that host's library. Where
  it is missing or lacks the GPU's architecture, the jobs there get `--backend jax`, and `status`
  lists the host. `gyaradax info` and `gyaradax datagen probe` show the same.

### Placement, monitoring and preemption

**Placement.** At each poll, pending jobs are taken by decreasing `cost`. A one-GPU job gets the
fastest free one-GPU slot, or else the fastest wider slot, where it runs on the slot's first GPU.
A `--gpus all` job gets the free slot with the largest `speed` × GPUs. A job that failed prefers
another slot for its retry.

**Launch and monitoring.** The scheduler starts each run detached, through the `wrap` command of
`gyaradax/datagen/scheduler.py`. The wrapper writes `RUN_DIR/.sched/`:

- `pid`, `host`, `slot`, `gpus` and `launch`;
- a `heartbeat`, refreshed every 30 seconds;
- then `exit`.

The run's output goes to `.sched/log`. The scheduler reads these files and uses ssh only to
launch, to signal, and to check a process whose heartbeat has stopped changing.

| Outcome | Result |
|---|---|
| exit 0 | done |
| exit 75 (stopped by SIGTERM after a snapshot) | queued again, not a failure |
| nonzero exit, or no heartbeat for 300 s and no process | retried up to `--max-retries` times, resuming from the run directory |
| exit 2 with a `DIVERGED` marker | diverged, not retried (a resume would diverge again) |
| exit 4 (`--linear-gate`: linearly stable) | stable, not retried |
| exit 5 (`--stop-on-decay`) | decayed, not retried |
| exit 6 (`--stop-on-banding`) | banded, not retried |
| `gyaradax run` cannot even start | the slot is paused for 300 s |

A retry or a requeued job continues with `gyaradax run RUN_DIR`, dropping `--n-steps`,
`--n-blocks`, `--from-scratch`, `--resume-from` and `--overwrite`, so the leg runs to the target
stored in the run's `config.yaml`. A run stopped before its first snapshot starts over with
`--overwrite`; that flag is only added once the job has written to the directory itself.

**Preemption.** Removing a slot, switching its host `off`, `cancel` and `--requeue` send SIGTERM.
`gyaradax run` finishes the running block, writes a snapshot and exits 75, so nothing is lost.
The dataset statistics are left to the leg that completes the run. The scheduler sends SIGKILL
only after `--grace` seconds (default 900); keep `--block-size` small enough for a block
to finish within that. A kill during a snapshot write leaves a truncated `step_*.npz`; the
scheduler moves it to `.sched/broken/` and resumes from the previous snapshot. A truncated
diagnostics file (`fluxes.npz`, ...) makes gyaradax fail on resume, so the job is marked failed
with a note instead of using up its retries.

**Restarts.** The state file is rewritten atomically after every poll. A restarted scheduler
picks up its running jobs from the state file and their `.sched` files, without relaunching
them. Even without a state file, it reattaches to any live launch it finds in a job's `.sched`.
A lock file (`state.json.lock`) keeps a second scheduler off the same campaign. `cancel` and
`resume` drop request files next to the state file, which the scheduler applies at its next poll.

## Legacy entry point

`python scripts/run.py CONFIG ...` still works and forwards to `gyaradax run`.
The retired `--kinetic` flag is accepted and ignored with a note.
