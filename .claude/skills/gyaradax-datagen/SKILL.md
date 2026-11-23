---
name: gyaradax-datagen
description: Generate a dataset of gyaradax runs, on the GPUs of one machine or spread over several machines with a shared filesystem, and act as the scheduler (place, launch, watch, move, resume, replace runs). Use for any campaign of many simulations.
---

# Data generation

References: `docs/getting_started/running.md` (Dataset output, Data generation), `gyaradax/datagen.py`. Configs are made and checked as in the `gyaradax-cli` skill.

## One machine

```bash
gyaradax datagen configs/*.yaml --out runs --gpus 0,1,2,3 --dataset /data/kinetic --args "--backend cuda --fused-linear --stop-on-banding"
```

One run per config, one config per GPU (or `--per-gpu N` small runs per GPU; sharded configs take aligned blocks). Running the same command again skips finished runs (exit 0 done, 2 diverged, 6 banded; in `runs/NAME/datagen_exit`) and resumes the others from their last snapshot. Start it detached (`nohup setsid ... &`).

## Several machines: the agent is the scheduler

Requires one filesystem visible from every host. Lay out a campaign directory:

```
CAMPAIGN/
  configs/NAME.yaml   one per run
  order.txt           names in priority order; append reserves at the end at any time
  campaign.env        PY, REPO (a checkout of this branch), FLAGS, DATASET, optional BUSY_MB
  runs/ logs/ claims/ queue.log   written by the workers
```

`campaign.env` may set FLAGS per host, e.g. `case $(hostname -s) in b300*) FLAGS="$FLAGS --linear-arith ds" ;; esac`.

1. **Place.** Find free GPUs (`rgpu --json --free` if available, else `nvidia-smi` per host). Never use a GPU someone else runs on or has reserved.
2. **Launch** one worker per GPU: `ssh HOST "nohup setsid bash $SKILL/worker.sh CAMPAIGN GPU > /dev/null 2>&1 < /dev/null &"` (`$SKILL` is this skill's directory). A worker waits for its GPU to be free, claims the next unfinished name of `order.txt` with an atomic `mkdir claims/NAME`, runs it with `gyaradax datagen`, and takes the next one until none is left. Interrupted runs (exit 75) release their claim, so any worker resumes them.
3. **Watch** with `python $SKILL/status.py CAMPAIGN` (state, host and GPU, t, t per hour, hours left) and `queue.log`. Check at the pace the runs move (hours), not in a tight loop.
4. **Act on outcomes:**
   - done: nothing; the trajectory is in `DATASET`.
   - diverged (2) or banded (6): keep the run directory, report it, and append a reserve config to `order.txt` if the campaign has one.
   - `failed (code)` in status: read `logs/NAME.log`, fix the cause, `rm -r claims/NAME`; a worker picks it up and resumes it.
5. **Move or stop.** `touch CAMPAIGN/STOP.HOST.GPU` (or `STOP.HOST`, or `STOP` for all) so the worker takes nothing new, then `pkill -TERM -P PID` with the PID from `claims/NAME/who` (`HOST GPU PID`): the run finishes its block, writes its snapshot and exits 75, and its claim is released for another worker. Never kill by pattern (`pkill -f`).
6. **Record** every decision (moves, replacements, stops) as a line in `queue.log` and keep the newest snapshots of every run.

## Campaign page

`page/build_page.py CAMPAIGN OUT.html --dataset DIR ... --existing "GLOB" [--axes eps,rlt_i] [--eval]` fills `page/template.html` from disk: parameter distributions of the campaign over an existing dataset (a 2D scatter and one strip per parameter), a progress table, and per run the heat-flux trace with a head-cut cutter (the proposal is the first frame where the 5-frame mean of Q_i reaches its mean over the last 30 %; `--eval` shows plain traces instead). Publish it as an artifact with `capabilities: {db: {}}`; cut decisions go to its collection `cuts` (`{status: accept | keep, cut: t}`), which the agent reads back with the artifact data tool before cutting frames. Rebuild and republish to the same URL as the campaign moves. Start from this template and adjust the parameters, panels or database to the campaign; keep the cut rule and the collection layout unless the user asks.

## Rules

- All runs of a campaign use the same code revision and flags, except GPU-specific kernel flags; say which host ran what.
- Do not delete run directories, snapshots or dataset trajectories without the user's explicit OK.
- Re-running `gyaradax datagen` on a finished campaign is safe: it skips finished runs.
