"""``gyaradax datagen``: run a list of configs over an ssh GPU mesh or Slurm and collect a dataset.

A campaign directory holds everything a campaign uses and produces::

    campaign.yaml   settings fixed at creation: dataset, name template, overrides, run arguments, code
    targets.yaml    where runs go: ssh slots or a ``slurm`` section, re-read while the campaign runs
    jobs.yaml       one job per config (generated)
    configs/        the configs as run, overrides applied
    runs/<name>/    run directories
    state.json      scheduler state; datagen.log is the scheduler log of a detached campaign

Each config becomes a job named from the ``--name`` template. Its run directory, dataset
trajectory and cost (grid size times steps) are derived from it, and an existing trajectory of
the same name in the dataset is refused before anything starts.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import yaml
from omegaconf import OmegaConf

from gyaradax.datagen import scheduler as sched

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = "campaign.yaml"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# set by the campaign for every job
OWN_FLAGS = (*sched.REFUSED_FLAGS, "--dataset", "--dataset-name")
# campaign switches passed to every job as `gyaradax run --<key with dashes>`
OUTCOME_SETTINGS = ("linear_gate", "stop_on_decay", "stop_on_banding", "track_outcome")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _parse_sets(items: Sequence[str]) -> dict[str, Any]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"error: --set expects KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


class Campaign:
    """A campaign directory: its settings, configs, jobs and the scheduler that runs them."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.settings_path = os.path.join(self.root, SETTINGS)
        self.targets = os.path.join(self.root, "targets.yaml")
        self.jobs_path = os.path.join(self.root, "jobs.yaml")
        self.state = os.path.join(self.root, "state.json")
        self.settings: dict[str, Any] = {}
        if os.path.isfile(self.settings_path):
            with open(self.settings_path) as fh:
                self.settings = yaml.safe_load(fh) or {}

    @property
    def exists(self) -> bool:
        return bool(self.settings)

    def create_or_check(self, args: argparse.Namespace) -> None:
        """Write the settings of a new campaign, or check that given flags match the existing ones."""
        given = {
            "dataset": os.path.abspath(args.dataset) if args.dataset else None,
            "name": args.name,
            "set": _parse_sets(args.set or []),
            "args": shlex.split(args.args) if args.args else [],
            "gpus": args.gpus,
            "linear_check": args.linear_check,
            "linear_check_tol": args.linear_check_tol,
            **{k: bool(getattr(args, k)) for k in OUTCOME_SETTINGS},
        }
        if self.exists:
            # only flags given on this call are compared
            explicit = {
                "dataset": args.dataset,
                "name": args.name,
                "set": args.set,
                "args": args.args,
                "gpus": args.gpus,
                "linear_check": args.linear_check,
                "linear_check_tol": args.linear_check_tol,
                **{k: getattr(args, k) for k in OUTCOME_SETTINGS},
            }
            saved = {**self.settings, **{k: bool(self.settings.get(k)) for k in OUTCOME_SETTINGS}}
            bad = [k for k, v in explicit.items() if v and given[k] != saved.get(k)]
            if bad:
                raise SystemExit(
                    f"error: {self.settings_path} fixes {', '.join(bad)}; drop the flags or start another campaign"
                )
            return
        if sched._has_flag(given["args"], OWN_FLAGS):
            raise SystemExit(
                f"error: --args may not contain {', '.join(OWN_FLAGS)}; the campaign sets them"
            )
        if given["linear_check"] and sched._has_flag(given["args"], ("--linear-check",)):
            raise SystemExit(
                "error: give the linear check either as --linear-check or in --args, not both"
            )
        given["name"] = given["name"] or "{run}"
        commit = _git("rev-parse", "HEAD")
        self.settings = {
            **given,
            "workdir": str(REPO_ROOT),
            "code": {"commit": commit, "dirty": bool(_git("status", "--porcelain"))}
            if commit
            else {},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.join(self.root, "configs"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "runs"), exist_ok=True)
        sched._atomic_write(self.settings_path, yaml.safe_dump(self.settings, sort_keys=False))
        print(f"campaign {self.root} created (code {REPO_ROOT} {commit[:10]})")

    def set_targets(self, path: str | None) -> None:
        if path:
            sched.load_targets(path)
            with open(path) as fh:
                sched._atomic_write(self.targets, fh.read())
        if not os.path.isfile(self.targets):
            raise SystemExit(f"error: {self.targets} missing; pass --targets FILE once")

    def job_name(self, cfg: Any, config_path: str) -> str:
        stem = Path(config_path).stem
        run = str(OmegaConf.select(cfg, "run.name") or stem)
        digits = re.findall(r"\d+", stem) or re.findall(r"\d+", run)
        fields = {"run": run, "stem": stem, "id": digits[-1] if digits else ""}
        try:
            name = str(self.settings["name"]).format(**fields)
        except (KeyError, IndexError) as exc:
            raise SystemExit(
                f"error: --name {self.settings['name']!r}: unknown field {exc}; use {{run}}, {{stem}}, {{id}}"
            )
        if "{id}" in self.settings["name"] and not fields["id"]:
            raise SystemExit(
                f"error: {config_path}: --name uses {{id}} but the file name has no number"
            )
        if not NAME.fullmatch(name):
            raise SystemExit(f"error: {config_path}: job name {name!r} is not a plain file name")
        return name

    def _job(self, name: str, config: str, cfg: Any) -> dict[str, Any]:
        grid = cfg.get("grid") or {}
        species = len(OmegaConf.select(cfg, "physics.signz") or [1])
        points = 1
        for key in ("nvpar", "nmu", "ns", "nkx", "nky"):
            points *= int(grid.get(key, 1) or 1)
        steps = int(OmegaConf.select(cfg, "solver.n_steps") or 1)
        t_end = OmegaConf.select(cfg, "solver.t_end")
        if t_end is not None and OmegaConf.select(cfg, "solver.dump_dt"):
            # time mode: steps at the largest time step
            steps = math.ceil(float(t_end) / float(OmegaConf.select(cfg, "solver.dt") or 1.0))
        shard = cfg.get("sharding") or {}
        gpus = self.settings.get("gpus") or int(shard.get("n_gpus", 0) or 0)
        gpus = gpus or max(
            1,
            int(shard.get("n_gpus_sp", 1) or 1)
            * int(shard.get("n_gpus_vp", 1) or 1)
            * int(shard.get("n_gpus_mu", 1) or 1),
        )
        run_dir = os.path.join(self.root, "runs", name)
        args = list(self.settings.get("args") or [])
        dataset = self.settings.get("dataset")
        if dataset:
            args += ["--dataset", dataset, "--dataset-name", name]
        if self.settings.get("linear_check"):
            args += ["--linear-check", str(self.settings["linear_check"])]
            if self.settings.get("linear_check_tol") is not None:
                args += ["--linear-check-tol", f"{float(self.settings['linear_check_tol']):g}"]
        for key in OUTCOME_SETTINGS:
            flag = "--" + key.replace("_", "-")
            if self.settings.get(key) and flag not in args:
                args.append(flag)
        return {
            "name": name,
            "config": config,
            "output_dir": run_dir,
            "args": args,
            "gpus": gpus,
            "cost": float(f"{points * species * steps / 1e12:.4g}"),
        }

    def _check_trajectory(self, name: str) -> None:
        dataset = self.settings.get("dataset")
        if not dataset:
            return
        from gyaradax.io.dataset import OWNER_FILE, trajectory_path

        traj = trajectory_path(dataset, name)
        if not os.path.lexists(traj):
            return
        owner = None
        with contextlib.suppress(OSError):
            owner = open(os.path.join(traj, OWNER_FILE)).read().strip()
        if os.path.islink(traj) or owner != os.path.join(self.root, "runs", name):
            raise SystemExit(
                f"error: {traj} already exists and is not this campaign's; choose another --name"
            )

    def add(self, config_paths: Sequence[str]) -> list[str]:
        """Materialise configs into jobs; a config already in the campaign unchanged is skipped."""
        lock = open(self.jobs_path + ".lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            doc = {"workdir": self.settings["workdir"], "jobs": []}
            if os.path.isfile(self.jobs_path):
                with open(self.jobs_path) as fh:
                    doc = yaml.safe_load(fh) or doc
            jobs = {j["name"]: j for j in doc.get("jobs") or []}
            added, staged = [], {}
            for path in config_paths:
                cfg = OmegaConf.load(path)
                for key, value in (self.settings.get("set") or {}).items():
                    OmegaConf.update(cfg, key, value, force_add=True)
                name = self.job_name(cfg, path)
                OmegaConf.update(cfg, "run.name", name, force_add=True)
                text = OmegaConf.to_yaml(cfg)
                target = os.path.join(self.root, "configs", f"{name}.yaml")
                if name in staged:
                    raise SystemExit(f"error: {path} and {staged[name]} both map to job {name}")
                staged[name] = path
                if name in jobs:
                    if os.path.isfile(target) and open(target).read() == text:
                        print(f"{name}: already in the campaign")
                        continue
                    raise SystemExit(f"error: job {name} exists with a different config ({target})")
                self._check_trajectory(name)
                sched._atomic_write(target, text)
                jobs[name] = self._job(name, target, cfg)
                added.append(name)
            doc["jobs"] = list(jobs.values())
            sched._atomic_write(self.jobs_path, yaml.safe_dump(doc, sort_keys=False))
        finally:
            lock.close()
        for name in added:
            j = jobs[name]
            print(f"{name}: added ({j['gpus']} GPU, cost {j['cost']:g}) -> {j['output_dir']}")
        return added

    def scheduler(self, args: argparse.Namespace) -> sched.Scheduler:
        return sched.Scheduler(
            self.targets,
            self.jobs_path,
            self.state,
            sched.SSHExecutor(timeout=args.ssh_timeout),
            grace=args.grace,
            max_retries=args.max_retries,
            dry_run=args.dry_run,
            deadline=args.deadline,
        )


def _detach(camp: Campaign, args: argparse.Namespace) -> int:
    argv = [
        sys.executable,
        "-m",
        "gyaradax",
        "datagen",
        "run",
        "--campaign",
        camp.root,
        *(["--keep-running"] if args.keep_running else []),
        "--poll",
        f"{args.poll:g}",
        "--grace",
        f"{args.grace:g}",
        "--max-retries",
        str(args.max_retries),
        *(["--deadline", repr(args.deadline)] if args.deadline else []),
    ]
    log = open(os.path.join(camp.root, "datagen.log"), "a")
    proc = subprocess.Popen(
        argv,
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"scheduler running in the background (pid {proc.pid}); log {log.name}")
    print(f"status: gyaradax datagen status --campaign {camp.root}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.time_limit:
        args.deadline = time.time() + args.time_limit
    camp = Campaign(args.campaign)
    camp.create_or_check(args)
    camp.set_targets(args.targets)
    if args.configs:
        camp.add(args.configs)
    if not os.path.isfile(camp.jobs_path):
        raise SystemExit("error: no jobs yet; pass config files")
    if args.deadline:
        stop_at = time.strftime("%Y-%m-%d %H:%M", time.localtime(args.deadline))
        print(f"time limit: jobs are stopped and requeued at {stop_at}")
    if args.detach and not args.dry_run:
        return _detach(camp, args)
    lock = None
    if not args.dry_run:
        lock = open(camp.state + ".lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"error: a scheduler already runs this campaign ({camp.state}.lock)")
    camp.scheduler(args).run(args.poll, once=args.once, until_idle=not args.keep_running)
    if lock is not None:
        lock.close()
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    camp = Campaign(args.campaign)
    camp.create_or_check(args)
    camp.add(args.configs)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    camp = Campaign(args.campaign)
    if not os.path.isfile(camp.state):
        raise SystemExit(f"error: {camp.state} not found; the campaign has not run yet")
    with open(camp.state) as fh:
        doc = json.load(fh)
    waiting = len(glob.glob(os.path.join(glob.escape(sched.requests_dir(camp.state)), "*.json")))
    s = camp.settings
    print(
        f"campaign {camp.root}: dataset {s.get('dataset') or '-'}, names {s.get('name')}, set {s.get('set') or {}}"
    )
    print(sched.format_status(doc, time.time(), waiting))
    checks = _linear_checks(doc)
    if checks:
        print("\nlinear check (γ at the reference's dominant ky):")
        width = max(len(n) for n in checks)
        for name, line in checks.items():
            print(f"  {name:<{width}}  {line}")
    return 0


def _linear_checks(doc: dict[str, Any]) -> dict[str, str]:
    """Summary per job of RUN_DIR/linear_check.json, where present."""
    from gyaradax.diagnostics.growth import summary

    out = {}
    for name, job in sorted(doc.get("jobs", {}).items()):
        path = os.path.join(job["spec"]["output_dir"], "linear_check.json")
        try:
            with open(path) as fh:
                out[name] = summary(json.load(fh))
        except (OSError, ValueError, KeyError):
            continue
    return out


def _cmd_request(args: argparse.Namespace) -> int:
    camp = Campaign(args.campaign)
    action = "requeue" if getattr(args, "requeue", False) else args.command
    for name in args.names:
        sched.submit_request(camp.state, action, name)
        print(f"{action} {name}: queued for the scheduler's next poll")
    return 0


def _gpus_arg(text: str) -> int | str:
    if text == "all":
        return text
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("--gpus must be >= 1 or 'all'")
    return value


def _cmd_host(args: argparse.Namespace) -> int:
    camp = Campaign(args.campaign)
    with open(camp.targets) as fh:
        doc = yaml.safe_load(fh) or {}
    hosts = doc.get("hosts") or {}
    if args.switch == "on":
        hosts.pop(args.host, None)
    else:
        hosts[args.host] = args.switch
    if hosts:
        doc["hosts"] = hosts
    else:
        doc.pop("hosts", None)
    sched._atomic_write(camp.targets, yaml.safe_dump(doc, sort_keys=False))
    effect = {
        "on": "takes jobs again",
        "drain": "finishes its jobs and takes no new ones",
        "off": "stops its jobs (SIGTERM, snapshot) and they go back in the queue",
        "kill": "kills its jobs now (SIGKILL) and they resume elsewhere from their last snapshot",
    }[args.switch]
    print(f"{args.host}: {args.switch} in {camp.targets}; at the next poll it {effect}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    camp = Campaign(args.campaign)
    path = args.targets or camp.targets
    slots, problems, slurm = sched.load_targets(path)
    for problem in problems:
        print(f"warning: {problem}")
    tool = REPO_ROOT / "gyaradax" / "backends" / "cuda_lib.py"
    executor = sched.SSHExecutor(timeout=args.ssh_timeout)
    hosts = {
        ("localhost (slurm login node)" if slurm else s.host): s.python for s in slots.values()
    }
    for host, python in sorted(hosts.items()):
        run_on = "localhost" if slurm else host
        try:
            res = (sched.LocalExecutor() if slurm else executor).run(
                run_on, f"{shlex.quote(python)} {shlex.quote(str(tool))} probe"
            )
            p = json.loads(res.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, OSError, ValueError, IndexError) as exc:
            print(f"{host}: probe failed ({exc})")
            continue
        state = "ok" if p["usable"] else f"JAX backend ({p['reason']})"
        print(
            f"{host}: GPUs sm {p['gpus'] or '?'}, library {p['lib']} (sm {p['archs'] or 'dev build'}): {state}"
        )
    if slurm:
        print(
            "slurm: compute nodes may differ from this node; the library must cover their sm (gyaradax build-cuda --archs)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gyaradax datagen",
        description="Run a list of configs over an ssh GPU mesh or Slurm and collect their dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def settings(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dataset", default=None, help="write each run as a trajectory into this directory"
        )
        p.add_argument(
            "--name",
            default=None,
            help="job and trajectory name template: {run} (run.name), {stem}, {id} (default {run})",
        )
        p.add_argument(
            "--set",
            action="append",
            default=None,
            metavar="KEY=VALUE",
            help="override a config entry, e.g. solver.disp_par=4.0 (repeatable)",
        )
        p.add_argument(
            "--args", default=None, help="extra `gyaradax run` arguments, e.g. '--backend cuda'"
        )
        p.add_argument(
            "--gpus",
            type=_gpus_arg,
            default=None,
            help="GPUs per job, or 'all': every GPU of the slot it lands on, mesh sized per "
            "leg (default: the config's sharding, else 1)",
        )
        p.add_argument(
            "--linear-check",
            default=None,
            metavar="NS[,NS...]",
            help="before each run, compare linear growth rates with these reference resolutions "
            "(gyaradax run --linear-check)",
        )
        p.add_argument(
            "--linear-check-tol",
            type=float,
            default=None,
            metavar="X",
            help="reject a job (state rejected, not retried) whose relative γ deviation exceeds X",
        )
        p.add_argument(
            "--linear-gate",
            action="store_true",
            help="linear run before each nonlinear run; a linearly stable job ends as `stable` "
            "(gyaradax run --linear-gate)",
        )
        p.add_argument(
            "--stop-on-decay",
            action="store_true",
            help="stop runs whose heat flux has decayed; they end as `decayed`",
        )
        p.add_argument(
            "--stop-on-banding",
            action="store_true",
            help="stop runs flagged for s-Nyquist banding; they end as `banded`",
        )
        p.add_argument(
            "--track-outcome",
            action="store_true",
            help="track every run's outcome (outcome.json, the outcomes table) without stopping",
        )

    def campaign(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--campaign", required=True, help="campaign directory (created on first use)"
        )

    run_p = sub.add_parser("run", help="add configs (optional) and run the campaign")
    run_p.add_argument("configs", nargs="*", help="config files")
    campaign(run_p)
    settings(run_p)
    run_p.add_argument(
        "--targets", default=None, help="targets YAML, copied to the campaign's targets.yaml"
    )
    run_p.add_argument(
        "--poll", type=float, default=30.0, help="seconds between polls (default 30)"
    )
    run_p.add_argument(
        "--grace",
        type=float,
        default=900.0,
        help="seconds from SIGTERM to SIGKILL; a run needs a block to snapshot (default 900)",
    )
    run_p.add_argument(
        "--max-retries", type=int, default=3, help="retries after failures (default 3)"
    )
    run_p.add_argument(
        "--ssh-timeout", type=float, default=30.0, help="timeout of each ssh call (default 30 s)"
    )
    run_p.add_argument("--once", action="store_true", help="poll once and exit")
    run_p.add_argument(
        "--keep-running",
        action="store_true",
        help="keep polling for added jobs once every job has finished (default: exit)",
    )
    run_p.add_argument(
        "--dry-run", action="store_true", help="print the launch commands of one round, run nothing"
    )
    run_p.add_argument("--detach", action="store_true", help="run the scheduler in the background")
    run_p.add_argument(
        "--time-limit",
        type=sched.parse_duration,
        default=None,
        metavar="DURATION",
        help="stop every job after this long (e.g. 10h, 90m): SIGTERM, snapshot, back in the queue; then exit",
    )
    # absolute deadline handed to a detached scheduler
    run_p.add_argument("--deadline", type=float, default=None, help=argparse.SUPPRESS)
    run_p.set_defaults(func=_cmd_run)

    add_p = sub.add_parser("add", help="add configs to a campaign (picked up at the next poll)")
    add_p.add_argument("configs", nargs="+", help="config files")
    campaign(add_p)
    settings(add_p)
    add_p.set_defaults(func=_cmd_add)

    status_p = sub.add_parser("status", help="jobs, slots, progress and ETAs")
    campaign(status_p)
    status_p.set_defaults(func=_cmd_status)

    cancel_p = sub.add_parser("cancel", help="stop jobs (SIGTERM, snapshot, exit) and hold them")
    cancel_p.add_argument("names", nargs="+")
    campaign(cancel_p)
    cancel_p.add_argument(
        "--requeue", action="store_true", help="put the jobs back in the queue instead"
    )
    cancel_p.set_defaults(func=_cmd_request)

    stable_p = sub.add_parser(
        "stable", help="stop jobs (SIGTERM, snapshot, exit) and record them as decayed"
    )
    stable_p.add_argument("names", nargs="+")
    campaign(stable_p)
    stable_p.set_defaults(func=_cmd_request)

    resume_p = sub.add_parser(
        "resume", help="put cancelled, failed, diverged or rejected jobs back in the queue"
    )
    resume_p.add_argument("names", nargs="+")
    campaign(resume_p)
    resume_p.set_defaults(func=_cmd_request)

    host_p = sub.add_parser(
        "host", help="switch a host on, to drain, off or kill while the campaign runs"
    )
    host_p.add_argument("host")
    host_p.add_argument("switch", choices=sched.HOST_SWITCHES)
    campaign(host_p)
    host_p.set_defaults(func=_cmd_host)

    probe_p = sub.add_parser("probe", help="check the CUDA library on every target host")
    campaign(probe_p)
    probe_p.add_argument("--targets", default=None, help="targets YAML (default: the campaign's)")
    probe_p.add_argument("--ssh-timeout", type=float, default=60.0)
    probe_p.set_defaults(func=_cmd_probe)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
