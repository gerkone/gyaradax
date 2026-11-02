"""``gyaradax datagen``: run a list of configs on the GPUs of this machine and collect a dataset.

One ``gyaradax run`` per config, one config per GPU at a time. Each config gets the run
directory ``OUT/<name>`` and, with ``--dataset``, the trajectory ``<name>`` (frames of a nonlinear
config, the eigenmode files of a linear one). A repeat skips the configs whose run finished,
diverged or banded and continues the others from their last snapshot.
"""

from __future__ import annotations

import argparse
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


def _command(config: str, run_dir: str, gpu: str, args: argparse.Namespace) -> list[str]:
    started = os.path.isfile(os.path.join(run_dir, "config.yaml"))
    cmd = [sys.executable, "-m", "gyaradax", "run", run_dir if started else config, "--device", gpu]
    if not started:
        cmd += ["--output-dir", run_dir]
        if args.dataset:
            cmd += ["--dataset", args.dataset, "--dataset-name", os.path.basename(run_dir)]
    return cmd + shlex.split(args.args or "")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gyaradax datagen", description=__doc__.splitlines()[0])
    p.add_argument("configs", nargs="+", help="YAML configs, one run each (named after the file)")
    p.add_argument("--out", required=True, help="directory of the run directories and logs")
    p.add_argument("--gpus", default="0", help="comma-separated GPU indices to use (default 0)")
    p.add_argument("--dataset", default=None, help="write each run's trajectory under this directory")
    p.add_argument("--args", default=None, help="extra `gyaradax run` arguments, e.g. '--stop-on-banding'")
    p.add_argument("--poll", type=float, default=10.0, help="seconds between checks (default 10)")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    todo = []
    for config in args.configs:
        name = os.path.splitext(os.path.basename(config))[0]
        run_dir = os.path.join(args.out, name)
        if _finished(run_dir):
            print(f"{name}: finished before, skipped")
        else:
            todo.append((name, os.path.abspath(config), run_dir))
    free = [g.strip() for g in args.gpus.split(",") if g.strip()]
    running: dict[str, tuple[str, subprocess.Popen]] = {}
    failed = 0
    while todo or running:
        while todo and free:
            name, config, run_dir = todo.pop(0)
            gpu = free.pop(0)
            log = open(os.path.join(args.out, f"{name}.log"), "a")
            proc = subprocess.Popen(_command(config, run_dir, gpu, args), stdout=log, stderr=subprocess.STDOUT)
            running[name] = (gpu, proc)
            print(f"{name}: started on GPU {gpu}", flush=True)
        time.sleep(args.poll)
        for name, (gpu, proc) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            free.append(gpu)
            run_dir = os.path.join(args.out, name)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, EXIT_FILE), "w") as fh:
                fh.write(f"{code}\n")
            failed += code != 0
            print(f"{name}: exit {code}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
