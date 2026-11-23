---
name: gyaradax-datagen
description: Plan and run a data-generation campaign of gyaradax runs (kinetic or adiabatic, training or evaluation), on one machine or over several machines with a shared filesystem, with the agent as the scheduler; then finalize, add converged linear solves and report. Use for any new campaign or to continue one.
---

# Data generation

References: `docs/getting_started/running.md` (Dataset output, Data generation), `gyaradax/datagen.py`. Configs are made and checked as in the `gyaradax-cli` skill; GPU flags per family in `docs/performance/index.md`.

## A campaign

A campaign is a directory, named by its run prefix, that holds everything about it:

```
CAMPAIGN/
  README.md           purpose, design, numerics, decision log (template: example/README.md)
  make_configs.py     the design: ranges in, configs out (template: example/make_configs.py)
  configs/NAME.yaml   one per run; design.csv lists the sampled parameters
  order.txt           names in priority order; reserves commented out below the main runs
  campaign.env        PY, REPO (a checkout of this branch), DATASET, FLAGS per GPU family (example/campaign.env)
  runs/ logs/ claims/ queue.log   written by the workers
```

`example/` is a complete starting point: `configs/kinetic.yaml` and `configs/adiabatic.yaml` are production configs (kinetic 2x32x12x18x192x32, adiabatic 32x8x16x85x32; adaptive dt with the eigenvalue cap, frames every 2.2 t to t 500), and `make_configs.py` samples the ranges with scrambled Sobol, sets ikxspace (kinetic floor(6.53 |shat|); adiabatic rescaled so dx matches the stored set at any eps), adds optional collisions and writes reserves.

## Steps

1. **Design.** State what the campaign adds (a gap in the dataset, a test set) in README.md, with the ranges. Compare the planned points with the existing dataset (the page's pair plot) before running anything. Use reserves to replace runs that band or diverge.
2. **Configs.** Copy `example/` into the campaign, edit the constants of `make_configs.py`, run it, and dry-run one config: `gyaradax run configs/NAME_000.yaml --dry-run`.
3. **Smoke test.** Run one config for a few hours on the target GPU; check fluxes are finite, the rate (t per hour) and the cost to t 500; the smoke run continues as part of the campaign.
4. **Launch.** One machine: `gyaradax datagen configs/*.yaml --out runs --gpus 0,1,2,3 --dataset DIR --args "$FLAGS"` (detached; `--per-gpu N` for small runs). Several machines: the worker below.
5. **Watch.** `python $SKILL/status.py CAMPAIGN` and the campaign page; check at the pace the runs move (hours), not in a tight loop.
6. **Outcomes.** done (exit 0): keep. diverged (2) or banded (6): keep the run directory, move the trajectory out of the dataset into a bad-runs area, queue a reserve. Stable runs (all modes decay): keep as stable. `failed (code)` in status: read `logs/NAME.log`, fix, `rm -r claims/NAME`.
7. **Finalize.** Training trajectories keep their frames; review head cuts on the page and apply the accepted ones. Evaluation trajectories keep their full history in the metadata and drop frame files beyond the last; keep the resume dump and the newest snapshot.
8. **Linear.** Every trajectory gets a converged linear solve of its config: the same config with `solver.non_linear: false`, run long enough that every growth rate in `growth.npz` has settled (up to t 5000, never a capped run), with `--dataset DIR --dataset-name NAME` so its eigenmode lands in the trajectory.
9. **Report.** README log, a per-trajectory README, and the page, rebuilt and republished at the end.

## Several machines: the agent is the scheduler

Requires one filesystem visible from every host.

1. **Place.** Find free GPUs (`rgpu --json --free` if available, else `nvidia-smi` per host). Never use a GPU someone else runs on or has reserved.
2. **Launch** one worker per GPU: `ssh HOST "nohup setsid bash $SKILL/worker.sh CAMPAIGN GPU > /dev/null 2>&1 < /dev/null &"` (`$SKILL` is this skill's directory). A worker waits for its GPU to be free, claims the next unfinished name of `order.txt` with an atomic `mkdir claims/NAME`, runs it with `gyaradax datagen`, and takes the next one until none is left. Interrupted runs (exit 75) release their claim, so any worker resumes them.
3. **Move or stop.** `touch CAMPAIGN/STOP.HOST.GPU` (or `STOP.HOST`, or `STOP` for all) so the worker takes nothing new, then `pkill -TERM -P PID` with the PID from `claims/NAME/who` (`HOST GPU PID`): the run finishes its block, writes its snapshot and exits 75, and its claim is released for another worker. Never kill by pattern (`pkill -f`).
4. **Record** every decision (moves, replacements, stops) as a line in `queue.log`.

## Campaign page

`page/build_page.py CAMPAIGN OUT.html --dataset DIR ... --existing "GLOB" [--eval]` fills `page/template.html` from disk: a lower-triangle pair plot of the campaign's parameters over an existing dataset, a progress table (20 rows per page), and per run the flux trace (linear or log10) with a head-cut cutter and an extend-by-Δt request. Publish it as an artifact with `capabilities: {db: {}}`. Decisions land in its database: `cuts/NAME` `{status: accept | keep, cut: t}` and `extend/NAME` `{dt, t_end}`; read them back with the artifact data tool, act on them (cut frames, `gyaradax run TRAJ --t-end T`), then rebuild and republish to the same URL. The cutter matters only for trajectories whose frames are kept; evaluation runs keep their full history (`--eval` shows plain traces). Adjust panels, parameters or collections to the campaign.

## Rules

- All runs of a campaign use the same code revision and flags, except GPU-specific kernel flags; say which host ran what.
- Production grids only; label any other grid as a test.
- Do not delete run directories, snapshots or dataset trajectories without the user's explicit OK.
- Re-running `gyaradax datagen` on a finished campaign is safe: it skips finished runs.
