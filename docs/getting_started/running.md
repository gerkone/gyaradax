# Running simulations

`gyaradax` is the single entry point. Everything that defines a run is in the YAML config ({doc}`configuration`). Command-line flags only override it.

```bash
gyaradax run     CONFIG [CONFIG ...]   # run one or several configs
gyaradax run     RUN_DIR               # resume / extend a previous run
gyaradax bench   CONFIG                # measure solver throughput
gyaradax convert GKW_DIR OUT.yaml      # YAML config from a GKW run directory
gyaradax datagen CONFIG ... --out DIR  # run configs on the local GPUs into a dataset
gyaradax info                          # devices, backends, versions
```

`python -m gyaradax ...` is the same command. `--dry-run` builds everything and prints the run summary without stepping.

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

Every run writes into its output directory (`--output-dir`, default `outputs_<adiabatic|kinetic>_<run.name>`):

| File | Content |
|---|---|
| `config.yaml` | the input config with the command-line choices folded in |
| `geometry.pkl` | the geometry dict |
| `fluxes.npz`, `fluxes_em.npz`, `kxspec.npz`, `kyspec.npz`, `growth.npz`, `dt_history.npz` | diagnostic histories, one entry per block (`dt_history`: one per step) |
| `step_NNNNNN.npz` | restart snapshot (df, phi, state), refreshed every `--snapshot-every` blocks and after the last |
| `run_info.jsonl` | one line per invocation: command, host, git revision, start and target step |
| `DIVERGED` | written when df turns non-finite; the run stops at that block |

Diagnostics are appended every block. The snapshot, the large file, is written every `--snapshot-every K` blocks (default 1) and after the last block. `--save-dumps` keeps every snapshot instead of only the latest.

## Resuming

```bash
gyaradax run outputs_kinetic_my_case                  # continue up to solver.n_steps
gyaradax run outputs_kinetic_my_case --n-steps 2000   # or run 2000 more steps
gyaradax run my_case.yaml --resume-from outputs_kinetic_my_case/step_004000.npz
gyaradax run my_case.yaml --resume-from K03           # from a GKW K-file in run.data_dir
```

A resumed run continues the solver state bitwise as if it had not stopped, appends to the histories and rewrites `config.yaml` with the new total. Flags still override the stored config, e.g. `--n-gpus 8` on the next leg. A run never mixes its outputs with another run's: starting a new trajectory in a directory that holds run outputs, or resuming an older snapshot while later ones exist, stops with an error unless `--overwrite` is given. `--from-scratch` cold-starts and ignores K-files and snapshots.

## Batched runs

Configs that share the grid and the static parameters run batched under one `jax.vmap`. Each writes its own run directory exactly as a single run would:

```bash
gyaradax run configs/adiabatic/nonlinear/itg_high_shear.yaml configs/adiabatic/nonlinear/itg_weak_drive.yaml
gyaradax run configs/adiabatic/nonlinear/itg_high_shear.yaml configs/adiabatic/nonlinear/itg_weak_drive.yaml --output-dir scan   # scan/<run.name>
gyaradax run scan/itg_high_shear scan/itg_weak_drive --n-steps 2000                                     # resume both
```

Members may differ in array-valued settings (gradients, `beta`, ...). Configs that cannot share one solve (different grids, geometries, start steps, cadence, sharded runs, or different static parameters such as `dt`, the dissipation coefficients or the species masses) run one after another. A member that diverges stops writing while the others continue.

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

The automatic layout fills species first, then mu, then vpar (8 GPUs on a two-species grid with `nmu = 16` give `sp = 2, mu = 4`). Before JAX starts the CLI checks that every axis divides its grid dimension and that enough GPUs are visible, and suggests a valid layout otherwise. Per-axis flags `--n-gpus-{sp,vp,mu}` override the config. {doc}`../performance/index` discusses when sharding pays off.

## Debug output

Off by default, written into the run directory:

```bash
gyaradax run my_case.yaml --telemetry   # telemetry.jsonl: per-block wall time, dt, memory
gyaradax run my_case.yaml --profile     # profile/ trace of one block + profile_summary.json
gyaradax run my_case.yaml --debug       # both
```

`telemetry.jsonl` has a `start` record (host, git revision, devices, backend, bracket version, precision, mesh, grid, relevant environment), the compile time, one record per block and an `end` record. `--profile` traces the first compiled block with the JAX profiler (open it in Perfetto or TensorBoard) and writes the GPU time per kernel to `profile_summary.json`.

## Benchmarking

```bash
gyaradax bench configs/adiabatic/nonlinear/itg_high_shear.yaml --device 0 --backend cuda --steps 100 --blocks 5
```

`bench` always cold-starts, does no I/O and reports the steady-state median steps per second (the first block carries the compilation), plus the peak device memory. `--profile` adds the per-kernel breakdown.

## Equally spaced frames

`--dump-dt DT` (config `solver.dump_dt`) puts the frames on an equally spaced time grid instead of every `--block-size` steps: the last step before each frame is shortened to land on it, so frame times are `t0 + k * DT` to round-off. Snapshots carry `t0`, so resumed legs continue the same grid. The run ends on the first frame at or after `--t-end` (config `solver.t_end`); `--n-blocks N` runs N more frames, and `--snapshot-every K` counts frames. Without `--dump-dt` the step-counted blocks are unchanged.

## Dataset output

`--dataset DIR` also writes the run as a trajectory `DIR/<name>_ifft_realpotens/` in the layout the neugk loaders read (`--dataset-name` sets `<name>`, default `run.name`):

- `data/timestep_NNNNN.zstd16.bin`: df in real space, (re/im, species, vpar, mu, s, x, y), bf16, lossless zstd (`--dataset-bits bf16` stores it raw).
- `data/poten_NNNNN.zstd16.bin`: the legacy real phi (x, s, y), ky up to ny/2. `data/potenc_NNNNN.bin`: the complex phi (re/im, x, s, y) with every ky mode, float32.
- `metadata.pkl` and `metadata_light.pkl`: parameters, geometry and per frame the time, step, fluxes per species, ky/kx spectra, growth rate and the bf16 loss of the frame (`quant_*`). The df and phi statistics are computed when the run ends, from frames at or after `--dataset-stats-from T` (`gyaradax dataset-stats TRAJ --from-time T` recomputes them).
- `resume/state.npz` and `resume/config.yaml`: df and solver state of the last frame and the leg's config, replaced when each leg ends (`--resume-dtype f32`, the default, or `f64`). `gyaradax run TRAJ --t-end T --output-dir OUT` extends the trajectory in place from it.
- A linear config (`solver.non_linear: false`) writes no frames: its final eigenmode goes into `data/linear.bin` and `data/linear_poten.bin` (spectral, float32), so a linear run with the `--dataset-name` of a nonlinear trajectory adds its eigenmode to it.

A trajectory belongs to the run directory that created it; another run is refused and needs a new `--dataset-name`. Resuming the run directory appends to the same trajectory, dropping frames past the snapshot it resumes from. Electromagnetic fields are not written.

`--stop-on-banding` ends a run whose zonal phi piles up at the s-Nyquist mode (share of the zonal power above 0.3 for 3 frames), with its snapshot and exit code 6. SIGTERM ends a run after the running block and its snapshot, exit code 75.

## Data generation

`gyaradax datagen` runs a list of configs on the GPUs of this machine, one `gyaradax run` per config and one config per GPU at a time:

```bash
gyaradax datagen scan/*.yaml --out runs/scan --gpus 0,1 --dataset /data/kinetic --args "--backend cuda --stop-on-banding"
```

Each config runs in `OUT/<name>` (the file name without `.yaml`), writes the trajectory `<name>` and logs to `OUT/<name>.log`. Running the same command again skips the runs that completed, diverged or banded (exit code 0, 2 or 6, kept in `OUT/<name>/datagen_exit`) and resumes the others from their last snapshot. A config runs on as many GPUs as its `sharding:` section asks for (`n_gpus: 2`, or `n_gpus_sp` x `n_gpus_vp` x `n_gpus_mu`; one without it), so runs of different sizes share one `--gpus` list. A run of N GPUs takes an aligned block of N from that list (pairs: 1st+2nd, 3rd+4th, ...; quads: 1st-4th, 5th-8th), which follows the usual NVLink pairs and quads; list the GPUs so that the blocks match the machine. Smaller runs fill free GPUs while a larger one waits for its block. Scheduling over several machines is left to the cluster's own tools.

## Command reference

Every command and flag, generated from the argument parser
({func}`gyaradax.cli.build_parser`). The CUDA kernel flags (`--fused-linear`,
`--linear-arith`, `--fused-bracket`) are explained in {doc}`../performance/index`.

```{eval-rst}
.. argparse::
   :module: gyaradax.cli
   :func: build_parser
   :prog: gyaradax
```
