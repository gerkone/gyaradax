"""``gyaradax datagen``: run a list of configs on the GPUs of this machine and collect a dataset.

One ``gyaradax run`` per config, one config per GPU at a time. Each config gets the run
directory ``OUT/<name>`` and, with ``--dataset``, the trajectory ``<name>`` (frames of a nonlinear
config, the eigenmode files of a linear one). A repeat skips the configs whose run finished,
diverged or banded and continues the others from their last snapshot.
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
import time

# exit code of the run's last leg, written by datagen
EXIT_FILE = "datagen_exit"
# completed, diverged or banded: a repeat skips them; any other code (75: interrupted) resumes
FINAL_CODES = (0, 2, 6)


def _finished(run_dir: str) -> bool:
    path = os.path.join(run_dir, EXIT_FILE)
    if not os.path.isfile(path):
        return False
    with open(path) as fh:
        return int(fh.read().strip() or -1) in FINAL_CODES


def _gpus(spec: str) -> list[str]:
    # "@FILE" re-reads the list from FILE at every check, so GPUs can be added or withdrawn while it runs
    if spec.startswith("@"):
        with open(spec[1:]) as fh:
            spec = fh.read()
    return [g.strip() for g in spec.replace("\n", ",").split(",") if g.strip()]


def _snapshot_in(run_dir: str) -> bool:
    return bool(glob.glob(os.path.join(glob.escape(run_dir), "step_*.npz")))


def _resume_time(traj: str) -> float:
    import numpy as np

    from gyaradax.io.dataset import resume_path

    with np.load(resume_path(traj)) as z:
        return float(np.ravel(z["time"])[0])


def _command(config: str, run_dir: str, gpu: str, args: argparse.Namespace) -> list[str]:
    from gyaradax.io.dataset import is_trajectory

    run = [sys.executable, "-m", "gyaradax", "run"]
    extra = shlex.split(args.args or "")
    if is_trajectory(config):
        # a trajectory is extended in place from its resume dump, never restarted from scratch
        if _snapshot_in(run_dir):
            return run + [run_dir, "--device", gpu] + extra
        if args.extend is None:
            raise SystemExit(f"error: {config} is a trajectory; extending it needs --extend DT")
        t_end = _resume_time(config) + args.extend
        return run + [config, "--device", gpu, "--output-dir", run_dir, "--t-end", f"{t_end:.6g}"] + extra
    started = os.path.isfile(os.path.join(run_dir, "config.yaml"))
    cmd = run + [run_dir if started else config, "--device", gpu]
    if not started:
        cmd += ["--output-dir", run_dir]
        if args.dataset:
            cmd += ["--dataset", args.dataset, "--dataset-name", os.path.basename(run_dir)]
    return cmd + extra


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gyaradax datagen", description=__doc__.splitlines()[0])
    p.add_argument("configs", nargs="+", help="YAML configs, one run each (named after the file), or dataset "
                   "trajectories to extend in place (named after the trajectory)")
    p.add_argument("--out", required=True, help="directory of the run directories and logs")
    p.add_argument("--gpus", default="0", help="comma-separated GPU indices to use (default 0), or @FILE to re-read them from FILE")
    p.add_argument("--dataset", default=None, help="write each run's trajectory under this directory")
    p.add_argument("--args", default=None, help="extra `gyaradax run` arguments, e.g. '--stop-on-banding'")
    p.add_argument("--extend", type=float, default=None, metavar="DT",
                   help="trajectories: run each DT time units past its resume dump (into the trajectory)")
    p.add_argument("--poll", type=float, default=10.0, help="seconds between checks (default 10)")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    todo = []
    for config in args.configs:
        name = os.path.splitext(os.path.basename(config.rstrip(os.sep)))[0].removesuffix("_ifft_realpotens")
        run_dir = os.path.join(args.out, name)
        if _finished(run_dir):
            print(f"{name}: finished before, skipped")
        else:
            todo.append((name, os.path.abspath(config.rstrip(os.sep)), run_dir))
    running: dict[str, tuple[str, subprocess.Popen]] = {}
    failed = 0
    while todo or running:
        busy = {gpu for gpu, _ in running.values()}
        free = [g for g in _gpus(args.gpus) if g not in busy]
        while todo and free:
            name, config, run_dir = todo.pop(0)
            gpu = free.pop(0)
            log = open(os.path.join(args.out, f"{name}.log"), "a")
            env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            proc = subprocess.Popen(_command(config, run_dir, gpu, args), stdout=log, stderr=subprocess.STDOUT, env=env)
            running[name] = (gpu, proc)
            print(f"{name}: started on GPU {gpu}", flush=True)
        time.sleep(args.poll)
        for name, (gpu, proc) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            run_dir = os.path.join(args.out, name)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, EXIT_FILE), "w") as fh:
                fh.write(f"{code}\n")
            failed += code != 0
            print(f"{name}: exit {code}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
