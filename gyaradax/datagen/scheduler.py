#!/usr/bin/env python
"""Scheduler for gyaradax runs over an ssh GPU mesh or a Slurm cluster.

``run`` re-reads the targets and job YAML files on every poll, places pending jobs longest-first
on free slots (ssh) or submits them (Slurm, one ``sbatch`` per leg), starts them through the
``wrap`` subcommand of this file and follows them through the ``RUN_DIR/.sched/`` files on the
shared filesystem. A run stopped by SIGTERM (a cancel, a removed slot, a Slurm walltime signal)
snapshots and exits 75, and its next leg resumes the run directory. ``status``, ``cancel`` and
``resume`` work on the state file. Standard library and PyYAML only, so ``wrap`` runs from the
file path without importing JAX. ``gyaradax datagen`` (campaign.py) builds the job file.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Sequence

import yaml

DEFAULT_PYTHON = "/system/apps/userenv/galletti/gyaradax_env/bin/python"
DEFAULT_GPU_GROUPS: dict[str, list[list[int]]] = {"lizard": [[0, 1], [2, 3], [4, 5]]}
DEFAULT_PROGRAM = ("{python}", "-m", "gyaradax")
SCHED = ".sched"
ACTIVE = ("starting", "running", "stopping")
STATE_ORDER = (
    "running",
    "starting",
    "stopping",
    "pending",
    "failed",
    "diverged",
    "rejected",
    "cancelled",
    "done",
)
DIAGNOSTICS = (
    "fluxes.npz",
    "fluxes_em.npz",
    "kxspec.npz",
    "kyspec.npz",
    "growth.npz",
    "dt_history.npz",
)
# flag -> number of values; a resumed leg drops these
FRESH_FLAGS = {
    "--from-scratch": 0,
    "--overwrite": 0,
    "--resume-from": 1,
    "--n-steps": 1,
    "--n-blocks": 1,
}
MESH_FLAGS = ("--n-gpus", "--n-gpus-sp", "--n-gpus-vp", "--n-gpus-mu")
REFUSED_FLAGS = ("--device", "--device-list", "--output-dir")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# exit code of a run stopped by SIGTERM after its snapshot (gyaradax.solver.driver.EXIT_INTERRUPTED)
INTERRUPTED = 75
# exit code of a run refused by its linear check (gyaradax.diagnostics.growth.EXIT_LINEAR_CHECK)
LINEAR_CHECK_FAILED = 3
SLURM_HOST = "slurm"
SLURM_QUEUED = ("PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "RESIZING", "SUSPENDED")
SLURM_RUNNING = ("RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT")
GPU_FLAGS = ("--gres", "--gpus", "--gpus-per-node", "--gpus-per-task", "-G")


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _int(text: str | None) -> int | None:
    try:
        return int(text.split()[0]) if text else None
    except ValueError:
        return None


@dataclass
class Attempt:
    launch: str | None
    pid: int | None
    host: str | None
    slot: str | None
    gpus: str | None
    heartbeat: str | None
    exit: int | None


def read_attempt(run_dir: str) -> Attempt:
    """The latest launch recorded in ``RUN_DIR/.sched``; ``exit`` is kept only if it is that launch's."""
    sched = os.path.join(run_dir, SCHED)

    def get(name: str) -> str | None:
        return _read(os.path.join(sched, name))

    launch = get("launch")
    exit_lines = (get("exit") or "").splitlines()
    code = _int(exit_lines[0]) if exit_lines else None
    if len(exit_lines) > 1 and exit_lines[1].strip() != launch:
        code = None
    return Attempt(
        launch, _int(get("pid")), get("host"), get("slot"), get("gpus"), get("heartbeat"), code
    )


def read_submission(run_dir: str) -> tuple[str | None, str | None]:
    """Slurm job id and launch id of the latest submission, from ``RUN_DIR/.sched/slurm``."""
    parts = (_read(os.path.join(run_dir, SCHED, "slurm")) or "").split()
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


def snapshot_steps(run_dir: str) -> dict[int, str]:
    steps = {}
    for path in glob.glob(os.path.join(glob.escape(run_dir), "step_*.npz")):
        m = re.search(r"step_(\d+)\.npz$", path)
        if m:
            steps[int(m.group(1))] = path
    return steps


def _valid_zip(path: str, member: str | None = None) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return member is None or member in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def latest_snapshot(run_dir: str, quarantine: bool = True) -> str | None:
    """The newest readable ``step_*.npz``; ``quarantine`` moves unreadable newer ones to ``.sched/broken``."""
    steps = snapshot_steps(run_dir)
    for step in sorted(steps, reverse=True):
        path = steps[step]
        if _valid_zip(path, "df.npy"):
            return path
        if quarantine:
            broken = os.path.join(run_dir, SCHED, "broken")
            os.makedirs(broken, exist_ok=True)
            os.replace(path, os.path.join(broken, os.path.basename(path)))
    return None


def run_legs(run_dir: str) -> list[dict[str, Any]]:
    legs = []
    for line in (_read(os.path.join(run_dir, "run_info.jsonl")) or "").splitlines():
        with contextlib.suppress(ValueError):
            legs.append(json.loads(line))
    return legs


def broken_diagnostics(run_dir: str) -> list[str]:
    paths = [os.path.join(run_dir, name) for name in DIAGNOSTICS]
    return [os.path.basename(p) for p in paths if os.path.exists(p) and not _valid_zip(p)]


def _has_flag(args: Sequence[str], flags: Sequence[str]) -> bool:
    return any(a == f or a.startswith(f + "=") for a in args for f in flags)


def strip_flags(args: Sequence[str], flags: dict[str, int]) -> list[str]:
    out: list[str] = []
    skip = 0
    for a in args:
        if skip:
            skip -= 1
            continue
        name = a.split("=", 1)[0]
        if name in flags:
            skip = 0 if "=" in a else flags[name]
            continue
        out.append(a)
    return out


def config_shards(path: str) -> bool:
    """Whether a config (or a run directory's config.yaml) asks for more than one GPU itself."""
    if os.path.isdir(path):
        path = os.path.join(path, "config.yaml")
    try:
        with open(path) as fh:
            shard = (yaml.safe_load(fh) or {}).get("sharding") or {}
        return any(int(v or 0) > 1 for v in shard.values())
    except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
        return False


@dataclass
class Slot:
    host: str
    gpus: list[int]
    speed: float = 1.0
    python: str = DEFAULT_PYTHON
    env: dict[str, str] = field(default_factory=dict)
    drain: bool = False
    name: str = ""

    @property
    def id(self) -> str:
        return self.name or f"{self.host}:{','.join(map(str, self.gpus))}"


@dataclass
class JobSpec:
    name: str
    output_dir: str
    config: str | None = None
    run_dir: str | None = None
    args: list[str] = field(default_factory=list)
    gpus: int = 1
    cost: float = 1.0
    workdir: str = "/"
    env: dict[str, str] = field(default_factory=dict)


def _env(value: Any, where: str) -> dict[str, str]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where}: env must be a mapping")
    bad = [k for k in value if not ENV_NAME.fullmatch(str(k))]
    if bad:
        raise ValueError(f"{where}: invalid environment variable names {bad}")
    return {str(k): str(v) for k, v in value.items()}


def _load_yaml(path: str, key: str) -> dict[str, Any]:
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    if doc is None:
        raise ValueError("empty file")
    if isinstance(doc, list):
        doc = {key: doc}
    if not isinstance(doc, dict):
        raise ValueError(f"expected a list or a mapping with '{key}'")
    return doc


def _slot(raw: Any, defaults: dict[str, Any], where: str) -> Slot:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a slot is a mapping")
    merged = {**defaults, **raw}
    host = str(merged.get("host") or "").strip()
    if not host or host.startswith("-") or " " in host:
        raise ValueError(f"{where}: host is required")
    gpus = merged.get("gpus")
    if isinstance(gpus, int):
        gpus = [gpus]
    if (
        not isinstance(gpus, list)
        or not gpus
        or len(set(gpus)) != len(gpus)
        or not all(isinstance(g, int) and not isinstance(g, bool) and g >= 0 for g in gpus)
    ):
        raise ValueError(f"{where}: gpus must be a list of distinct GPU indices")
    speed = float(merged.get("speed", 1.0))
    if speed <= 0:
        raise ValueError(f"{where}: speed must be positive")
    return Slot(
        host=host,
        gpus=list(gpus),
        speed=speed,
        python=str(merged.get("python") or DEFAULT_PYTHON),
        env={**_env(defaults.get("env"), where), **_env(raw.get("env"), where)},
        drain=bool(merged.get("drain", False)),
        name=str(raw.get("name") or ""),
    )


def load_pool(path: str) -> tuple[dict[str, Slot], list[str]]:
    """Parse the pool file into slots by id, plus the entries skipped and why.

    Raises on an unreadable, unparsable or empty file.
    """
    doc = _load_yaml(path, "slots")
    defaults = doc.get("defaults") or {}
    groups = {**DEFAULT_GPU_GROUPS, **(doc.get("gpu_groups") or {})}
    slots: dict[str, Slot] = {}
    used: dict[str, set[int]] = {}
    problems = []
    for i, raw in enumerate(doc.get("slots") or []):
        try:
            slot = _slot(raw, defaults, f"pool slot {i}")
        except (TypeError, ValueError) as exc:
            problems.append(str(exc))
            continue
        short = slot.host.split(".")[0]
        allowed = groups.get(slot.host, groups.get(short))
        if allowed and len(slot.gpus) > 1 and sorted(slot.gpus) not in [sorted(g) for g in allowed]:
            problems.append(
                f"pool slot {slot.id}: multi-GPU slots on {slot.host} must be one of {allowed}"
            )
            continue
        taken = used.setdefault(short, set())
        if slot.id in slots or taken & set(slot.gpus):
            problems.append(f"pool slot {slot.id}: overlaps another slot")
            continue
        taken.update(slot.gpus)
        slots[slot.id] = slot
    return slots, problems


@dataclass
class SlurmTarget:
    """A Slurm cluster as a target: at most ``max_jobs`` legs submitted at a time."""

    python: str = DEFAULT_PYTHON
    max_jobs: int = 8
    sbatch: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)
    signal_lead: int = 900
    env: dict[str, str] = field(default_factory=dict)


def _slurm_target(raw: Any) -> SlurmTarget:
    if not isinstance(raw, dict):
        raise ValueError("slurm: expected a mapping")
    known = {f.name for f in fields(SlurmTarget)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"slurm: unknown keys {unknown}")
    target = SlurmTarget(
        python=str(raw.get("python") or DEFAULT_PYTHON),
        max_jobs=int(raw.get("max_jobs", 8)),
        sbatch=_args(raw.get("sbatch"), "slurm.sbatch"),
        setup=[str(line) for line in (raw.get("setup") or [])],
        signal_lead=int(raw.get("signal_lead", 900)),
        env=_env(raw.get("env"), "slurm"),
    )
    if target.max_jobs < 1 or target.signal_lead < 0:
        raise ValueError("slurm: max_jobs must be >= 1 and signal_lead >= 0")
    return target


def load_targets(path: str) -> tuple[dict[str, Slot], list[str], SlurmTarget | None]:
    """The slots of a targets file: its ``slurm`` section as ``max_jobs`` submission slots, else the ssh pool."""
    doc = _load_yaml(path, "slots")
    if "slurm" not in doc:
        slots, problems = load_pool(path)
        return slots, problems, None
    target = _slurm_target(doc["slurm"])
    slots = {
        f"{SLURM_HOST}:{i}": Slot(
            host=SLURM_HOST, gpus=[], python=target.python, env=target.env, name=f"{SLURM_HOST}:{i}"
        )
        for i in range(target.max_jobs)
    }
    problems = ["targets: slots are ignored next to a slurm section"] if doc.get("slots") else []
    return slots, problems, target


def _args(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(f"{where}: args must be a string or a list")


def _job(raw: Any, defaults: dict[str, Any], base: str, where: str) -> JobSpec:
    if not isinstance(raw, dict) or not raw.get("name"):
        raise ValueError(f"{where}: a job is a mapping with a name")
    name = str(raw["name"])
    where = f"job {name}"
    merged = {**defaults, **raw}
    workdir = os.path.abspath(
        os.path.join(base, os.path.expanduser(str(merged.get("workdir", "."))))
    )

    def path(key: str) -> str | None:
        value = merged.get(key)
        return (
            os.path.abspath(os.path.join(workdir, os.path.expanduser(str(value))))
            if value
            else None
        )

    config, run_dir, output_dir = path("config"), path("run_dir"), path("output_dir")
    if (config is None) == (run_dir is None):
        raise ValueError(f"{where}: give exactly one of config and run_dir")
    if run_dir is not None:
        if output_dir not in (None, run_dir):
            raise ValueError(
                f"{where}: a run_dir job resumes in place; drop output_dir or set it to run_dir"
            )
        output_dir = run_dir
    if output_dir is None:
        raise ValueError(f"{where}: output_dir is required")
    args = _args(defaults.get("args"), where) + _args(raw.get("args"), where)
    if _has_flag(args, REFUSED_FLAGS):
        raise ValueError(
            f"{where}: --device, --device-list and --output-dir are set by the scheduler"
        )
    gpus = int(merged.get("gpus", 1))
    if gpus < 1:
        raise ValueError(f"{where}: gpus must be >= 1")
    return JobSpec(
        name=name,
        output_dir=output_dir,
        config=config,
        run_dir=run_dir,
        args=args,
        gpus=gpus,
        cost=float(merged.get("cost", 1.0)),
        workdir=workdir,
        env={**_env(defaults.get("env"), where), **_env(raw.get("env"), where)},
    )


def load_jobs(path: str) -> tuple[list[JobSpec], list[str]]:
    """Parse the job file into specs with absolute paths, plus the entries skipped and why.

    Relative paths are taken from ``workdir`` (top level and per job), itself relative to the
    job file's directory; it is also the directory the run is launched from.
    """
    doc = _load_yaml(path, "jobs")
    base = os.path.join(os.path.dirname(os.path.abspath(path)), str(doc.get("workdir", ".")))
    defaults = doc.get("defaults") or {}
    specs, names, problems = [], set(), []
    for i, raw in enumerate(doc.get("jobs") or []):
        try:
            spec = _job(raw, defaults, base, f"job {i}")
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(str(exc))
            continue
        if spec.name in names:
            problems.append(f"job {spec.name}: duplicate name")
            continue
        names.add(spec.name)
        specs.append(spec)
    return specs, problems


@dataclass
class Job:
    spec: JobSpec
    order: int = 0
    state: str = "pending"
    failures: int = 0
    attempts: int = 0
    base_legs: int | None = None
    slot: str | None = None
    host: str | None = None
    gpus: list[int] = field(default_factory=list)
    launch_id: str | None = None
    launched_at: float | None = None
    handle: str | None = None
    pid: int | None = None
    hb_value: str | None = None
    hb_seen: float | None = None
    stop_reason: str | None = None
    requeue: bool = True
    stop_started: float | None = None
    killed_at: float | None = None
    avoid: list[str] = field(default_factory=list)
    argv: list[str] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        data["spec"] = JobSpec(**data["spec"])
        return cls(**data)


class LaunchError(RuntimeError):
    pass


class Executor:
    """Runs shell commands on a host with bash; subclasses decide how the host is reached."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def argv(self, host: str, command: str) -> list[str]:
        return ["bash", "-c", command]

    def run(self, host: str, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            self.argv(host, command),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

    def describe(self, host: str, command: str) -> str:
        return shlex.join(self.argv(host, command))

    def launch(self, host: str, command: str) -> None:
        """Start a self-detaching command; raises LaunchError when it surely did not start.

        A timeout is not an error: the command may have started, and the scheduler's start
        timeout decides.
        """
        try:
            res = self.run(host, command)
        except subprocess.TimeoutExpired:
            return
        except OSError as exc:
            raise LaunchError(f"{host}: {exc}") from exc
        if res.returncode != 0:
            raise LaunchError(f"{host}: exit {res.returncode}: {(res.stderr or '').strip()[-300:]}")

    def alive(self, host: str, pid: int, token: str) -> bool | None:
        """Whether ``pid`` on ``host`` is still the wrapper of launch ``token``; None if unreachable."""
        q = shlex.quote(token)
        cmd = f"kill -0 {pid} 2>/dev/null && grep -qa -- {q} /proc/{pid}/cmdline 2>/dev/null && exit 0; exit 1"
        try:
            res = self.run(host, cmd)
        except (subprocess.TimeoutExpired, OSError):
            return None
        return {0: True, 1: False}.get(res.returncode)

    def signal(
        self, host: str, pid: int, sig: str, group: bool = False, token: str | None = None
    ) -> bool:
        if group:
            cmd = f"kill -{sig} -- -{pid}"
        elif token:
            cmd = f"grep -qa -- {shlex.quote(token)} /proc/{pid}/cmdline 2>/dev/null && kill -{sig} {pid}"
        else:
            cmd = f"kill -{sig} {pid}"
        try:
            return self.run(host, cmd).returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


class SSHExecutor(Executor):
    """Reaches hosts over ssh in batch mode with known host keys only; this machine runs locally."""

    options = (
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
    )  # fmt: skip

    def __init__(self, timeout: float = 30.0, local_hosts: Sequence[str] | None = None):
        super().__init__(timeout)
        name = socket.gethostname()
        self.local_hosts = set(
            local_hosts if local_hosts is not None else (name, name.split(".")[0])
        )

    def argv(self, host: str, command: str) -> list[str]:
        if host in self.local_hosts:
            return ["bash", "-c", command]
        return ["ssh", *self.options, host, "bash -c " + shlex.quote(command)]


class LocalExecutor(Executor):
    """Runs every command on this machine, whatever the host name."""


def requests_dir(state_path: str) -> str:
    return state_path + ".requests"


def submit_request(state_path: str, action: str, name: str) -> str:
    directory = requests_dir(state_path)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{time.time_ns()}-{os.getpid()}-{action}.json")
    _atomic_write(path, json.dumps({"action": action, "name": name}))
    return path


class Scheduler:
    """Places the jobs of a job file on the slots of a pool file and follows them to the end."""

    def __init__(
        self,
        pool_path: str,
        jobs_path: str,
        state_path: str,
        executor: Executor | None = None,
        *,
        heartbeat: float = 30.0,
        heartbeat_timeout: float = 300.0,
        grace: float = 120.0,
        start_timeout: float = 300.0,
        max_retries: int = 3,
        slot_backoff: float = 300.0,
        program: Sequence[str] = DEFAULT_PROGRAM,
        script: str | None = None,
        dry_run: bool = False,
        deadline: float | None = None,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] | None = None,
    ):
        self.pool_path = os.path.abspath(pool_path)
        self.jobs_path = os.path.abspath(jobs_path)
        self.state_path = os.path.abspath(state_path)
        self.executor = executor or SSHExecutor()
        self.heartbeat = heartbeat
        self.heartbeat_timeout = heartbeat_timeout
        self.grace = grace
        self.start_timeout = start_timeout
        self.max_retries = max_retries
        self.slot_backoff = slot_backoff
        self.program = list(program)
        self.script = os.path.abspath(script or __file__)
        self.dry_run = dry_run
        self.deadline = deadline
        self.clock = clock
        self.log = log or (
            lambda msg: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)
        )
        self.jobs: dict[str, Job] = {}
        self.pool: dict[str, Slot] = {}
        self.slurm: SlurmTarget | None = None
        self.local = LocalExecutor(timeout=self.executor.timeout)
        self.probes: dict[str, dict[str, Any]] = {}
        self._queue: dict[str, str] | None = None
        self.down_until: dict[str, float] = {}
        self.next_order = 0
        self._warned: set[str] = set()
        self._fresh = True
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path) as fh:
            doc = json.load(fh)
        self.jobs = {name: Job.from_dict(d) for name, d in doc.get("jobs", {}).items()}
        self.pool = {sid: Slot(**s) for sid, s in doc.get("slots", {}).items()}
        self.slurm = SlurmTarget(**doc["slurm"]) if doc.get("slurm") else None
        self.probes = doc.get("probes", {})
        self.down_until = {k: float(v) for k, v in doc.get("down_until", {}).items()}
        self.next_order = int(doc.get("next_order", len(self.jobs)))

    def save(self) -> None:
        if self.dry_run:
            return
        doc = {
            "version": 1,
            "updated": self.clock(),
            "scheduler": {"host": socket.gethostname(), "pid": os.getpid()},
            "pool_file": self.pool_path,
            "jobs_file": self.jobs_path,
            "deadline": self.deadline,
            "next_order": self.next_order,
            "slots": {sid: asdict(s) for sid, s in self.pool.items()},
            "slurm": asdict(self.slurm) if self.slurm else None,
            "probes": self.probes,
            "down_until": self.down_until,
            "jobs": {name: asdict(j) for name, j in self.jobs.items()},
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        _atomic_write(self.state_path, json.dumps(doc, indent=1, sort_keys=True))

    def _warn(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            self.log(f"warning: {msg}")

    def _active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.state in ACTIVE]

    def expired(self) -> bool:
        return self.deadline is not None and self.clock() >= self.deadline

    def _slurm_state(self, handle: str) -> str | None:
        """Queue state of a Slurm job ('' once it left the queue), None if squeue fails; one squeue per poll."""
        if self._queue is None:
            try:
                res = self.local.run("", 'squeue -h -u "$(id -un)" -o "%i %T"')
            except (subprocess.TimeoutExpired, OSError):
                return None
            if res.returncode != 0:
                return None
            self._queue = {}
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    self._queue[parts[0]] = parts[1]
        return self._queue.get(handle, "")

    def _alive(
        self, host: str | None, pid: int | None, token: str | None, handle: str | None
    ) -> bool | None:
        if handle:
            state = self._slurm_state(handle)
            return None if state is None else state in (*SLURM_QUEUED, *SLURM_RUNNING)
        if pid is None:
            return False
        return self.executor.alive(host or "", pid, token or "")

    def _signal(self, job: Job, sig: str, group: bool = False) -> bool:
        if job.handle:
            # the batch shell is the wrapper, which forwards TERM to the run; KILL cancels the job
            cmd = (
                f"scancel {job.handle}"
                if sig == "KILL"
                else f"scancel --batch --signal={sig} {job.handle}"
            )
            try:
                return self.local.run("", cmd).returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                return False
        if job.pid is None:
            return False
        if group:
            return self.executor.signal(job.host or "", job.pid, sig, group=True)
        return self.executor.signal(job.host or "", job.pid, sig, token=job.launch_id)

    def _probe(self, slot: Slot, workdir: str, now: float) -> dict[str, Any] | None:
        """The host's CUDA library state from ``gyaradax/backends/cuda_lib.py probe``, rechecked hourly."""
        tool = os.path.join(workdir, "gyaradax", "backends", "cuda_lib.py")
        if self.dry_run or slot.host == SLURM_HOST or not os.path.isfile(tool):
            return None
        key = f"{slot.host} {slot.python} {workdir}"
        cached = self.probes.get(key)
        if cached is None or now - float(cached.get("checked", 0.0)) > 3600.0:
            try:
                res = self.executor.run(
                    slot.host, f"{shlex.quote(slot.python)} {shlex.quote(tool)} probe"
                )
                found = json.loads(res.stdout.strip().splitlines()[-1])
            except (subprocess.TimeoutExpired, OSError, ValueError, IndexError):
                return cached
            found["checked"] = now
            if not found.get("usable") and (cached is None or cached.get("usable")):
                self.log(
                    f"{slot.host}: CUDA library unusable ({found.get('reason')}); runs there use the JAX "
                    "backend until `gyaradax build-cuda` builds one for that environment"
                )
            self.probes[key] = cached = found
        return cached

    def _backend_args(self, job: Job, slot: Slot, now: float) -> list[str]:
        """``--backend auto`` (CUDA where the library serves the GPU), ``jax`` where the probe failed."""
        if _has_flag(job.spec.args, ("--backend",)):
            return []
        probe = self._probe(slot, job.spec.workdir, now)
        usable = probe is None or probe.get("usable")
        return ["--backend", "auto" if usable else "jax"]

    def sbatch_script(self, job: Job, slot: Slot, argv: list[str], launch_id: str) -> str:
        """The batch script of one leg: Slurm options, the target's setup lines, then the wrapper."""
        target = self.slurm or SlurmTarget()
        spec = job.spec
        sched = os.path.join(spec.output_dir, SCHED)
        env = {**target.env, **spec.env, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        wrapper = [
            target.python,
            self.script,
            "wrap",
            "--dir",
            spec.output_dir,
            "--launch-id",
            launch_id,
            "--slot",
            slot.id,
            "--heartbeat",
            f"{self.heartbeat:g}",
            "--",
        ]
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.name}",
            f"#SBATCH --output={sched}/slurm-%j.log",
            f"#SBATCH --signal=B:TERM@{target.signal_lead}",
        ]
        if not _has_flag(target.sbatch, GPU_FLAGS):
            lines.append(f"#SBATCH --gres=gpu:{spec.gpus}")
        lines += [f"#SBATCH {a}" for a in target.sbatch]
        lines += target.setup
        lines.append(f"cd {shlex.quote(spec.workdir)}")
        lines.append(
            "exec " + shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *wrapper, *argv])
        )
        return "\n".join(lines) + "\n"

    def _submit(self, job: Job, slot: Slot, argv: list[str], launch_id: str, now: float) -> bool:
        spec = job.spec
        sched = os.path.join(spec.output_dir, SCHED)
        script = self.sbatch_script(job, slot, argv, launch_id)
        if self.dry_run:
            self.log(f"[dry-run] {spec.name} -> sbatch:\n{script}")
            return True
        os.makedirs(sched, exist_ok=True)
        path = os.path.join(sched, "job.sbatch")
        _atomic_write(path, script)
        try:
            res = self.local.run("", f"sbatch --parsable {shlex.quote(path)}")
            error = "" if res.returncode == 0 else (res.stderr or res.stdout or "").strip()[-300:]
        except (subprocess.TimeoutExpired, OSError) as exc:
            error = str(exc)
        if error:
            for sid in self.pool:
                self.down_until[sid] = now + self.slot_backoff
            self.log(
                f"{spec.name}: sbatch failed ({error}); no submissions for {self.slot_backoff:.0f} s"
            )
            return False
        handle = res.stdout.strip().split(";")[0]
        _atomic_write(os.path.join(sched, "slurm"), f"{handle} {launch_id}\n")
        job.state, job.slot, job.host, job.handle = "starting", slot.id, SLURM_HOST, handle
        job.gpus = []
        job.launch_id, job.launched_at, job.pid = launch_id, now, None
        job.hb_value, job.hb_seen, job.argv, job.note = None, None, argv, ""
        job.attempts += 1
        self.log(f"{spec.name}: submitted as Slurm job {handle}: {shlex.join(argv)}")
        return True

    def poll(self) -> None:
        """One round: re-read pool and jobs, apply requests, follow active jobs, launch pending."""
        now = self.clock()
        self._queue = None
        self._refresh_pool()
        self._refresh_jobs(now)
        if self.dry_run:
            self._place(now)
            return
        if self._fresh:
            for job in [j for j in self.jobs.values() if j.state == "pending"]:
                self._adopt(job, now)
            self._fresh = False
        self._apply_requests(now)
        for job in self._active():
            self._monitor(job, now)
        for job in self._active():
            if job.host != SLURM_HOST and job.slot not in self.pool:
                self._stop(job, "slot removed", True, now)
        if self.deadline is not None and now >= self.deadline:
            for job in self._active():
                self._stop(job, "time limit", True, now)
        else:
            self._place(now)
        for job in self._active():
            self._update_progress(job, now)
        self.save()

    def _refresh_pool(self) -> None:
        try:
            slots, problems, slurm = load_targets(self.pool_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self._warn(f"pool {self.pool_path} unreadable ({exc}); keeping the previous pool")
            return
        if (slurm is None) != (self.slurm is None) and self._active():
            self._warn(
                "targets switch between ssh and slurm while jobs run; keeping the previous targets"
            )
            return
        self.slurm = slurm
        for problem in problems:
            self._warn(problem)
        for sid in slots.keys() - self.pool.keys():
            self.log(f"slot {sid} added")
        for sid in self.pool.keys() - slots.keys():
            self.log(f"slot {sid} removed")
        self.pool = slots

    def _refresh_jobs(self, now: float) -> None:
        try:
            specs, problems = load_jobs(self.jobs_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self._warn(f"jobs {self.jobs_path} unreadable ({exc}); keeping the known jobs")
            return
        for problem in problems:
            self._warn(problem)
        claimed = {
            j.spec.output_dir: j.spec.name for j in self.jobs.values() if j.state not in ("done",)
        }
        for spec in specs:
            job = self.jobs.get(spec.name)
            if job is None:
                owner = claimed.get(spec.output_dir)
                if owner is not None:
                    self._warn(f"job {spec.name}: {spec.output_dir} belongs to job {owner}")
                    continue
                job = Job(spec=spec, order=self.next_order)
                self.next_order += 1
                self.jobs[spec.name] = job
                claimed[spec.output_dir] = spec.name
                self.log(f"job {spec.name} added (cost {spec.cost:g}, {spec.gpus} GPU)")
                if not self._fresh and not self.dry_run:
                    self._adopt(job, now)
            elif job.state == "pending" and job.spec != spec:
                if job.base_legs is not None and spec.output_dir != job.spec.output_dir:
                    self._warn(f"job {spec.name}: output_dir cannot change once the job has run")
                    continue
                job.spec = spec
                self.log(f"job {spec.name} updated")

    def _adopt(self, job: Job, now: float) -> None:
        """Take over a still-running launch, or a still-queued Slurm submission, of the job's ``.sched`` files."""
        att = read_attempt(job.spec.output_dir)
        handle, submitted = read_submission(job.spec.output_dir)
        if handle and submitted != att.launch:
            if self._slurm_state(handle) not in (*SLURM_QUEUED, *SLURM_RUNNING):
                return
            busy = {j.slot for j in self._active()}
            free = [sid for sid in self.pool if sid not in busy]
            job.state, job.slot, job.host, job.handle = (
                "starting",
                free[0] if free else f"{SLURM_HOST}:+",
                SLURM_HOST,
                handle,
            )
            job.launch_id, job.launched_at, job.pid = submitted, now, None
            if job.base_legs is None:
                job.base_legs = len(run_legs(job.spec.output_dir))
            self.log(f"{job.spec.name}: reattached to the queued Slurm job {handle}")
            return
        if not att.launch or att.exit is not None or att.pid is None or not att.host:
            return
        if handle and submitted == att.launch:
            job.handle = handle
        if self._alive(att.host, att.pid, att.launch, job.handle) is not True:
            job.handle = None
            return
        legs = len(run_legs(job.spec.output_dir))
        job.state, job.slot, job.host, job.pid = "running", att.slot, att.host, att.pid
        job.gpus = [int(g) for g in (att.gpus or "").split(",") if g.strip().isdigit()]
        job.launch_id, job.launched_at = att.launch, now
        job.hb_value, job.hb_seen = att.heartbeat, now
        if job.base_legs is None:
            job.base_legs = max(legs - 1, 0)
        self.log(f"{job.spec.name}: reattached to the running launch {att.launch} on {att.slot}")

    def _apply_requests(self, now: float) -> None:
        for path in sorted(
            glob.glob(os.path.join(glob.escape(requests_dir(self.state_path)), "*.json"))
        ):
            try:
                req = json.loads(_read(path) or "")
            except ValueError:
                req = {}
            with contextlib.suppress(OSError):
                os.remove(path)
            action, name = req.get("action"), str(req.get("name"))
            job = self.jobs.get(name)
            if job is None:
                self.log(f"request {action} {name}: no such job")
                continue
            if action in ("cancel", "requeue"):
                requeue = action == "requeue"
                if job.state in ACTIVE:
                    self._stop(job, "requeued" if requeue else "cancelled", requeue, now)
                elif job.state == "pending" and not requeue:
                    job.state = "cancelled"
                    self.log(f"{name}: cancelled")
                else:
                    self.log(f"request {action} {name}: nothing to do, the job is {job.state}")
            elif action == "resume":
                if job.state in ("cancelled", "failed", "diverged", "rejected"):
                    job.state, job.failures, job.note = "pending", 0, ""
                    self.log(f"{name}: back in the queue")
                elif job.state == "stopping":
                    job.requeue = True
                else:
                    self.log(f"request {action} {name}: nothing to do, the job is {job.state}")

    def _monitor(self, job: Job, now: float) -> None:
        att = read_attempt(job.spec.output_dir)
        if att.launch != job.launch_id:
            if job.handle:
                state = self._slurm_state(job.handle)
                if state is None or state in SLURM_QUEUED:
                    # time in the Slurm queue does not count toward the start timeout
                    job.launched_at = now
                    return
                if state == "":
                    log = os.path.join(job.spec.output_dir, SCHED, f"slurm-{job.handle}.log")
                    self._finish(
                        job,
                        None,
                        now,
                        f"the Slurm job ended before the run started; see {log}",
                        startup=True,
                    )
                    return
            if job.launched_at is not None and now - job.launched_at > self.start_timeout:
                self._finish(
                    job, None, now, "the wrapper did not start; see .sched/log", startup=True
                )
            return
        if job.pid is None:
            job.pid = att.pid
            job.hb_value, job.hb_seen = att.heartbeat, now
            if job.state == "starting":
                job.state = "running"
            elif job.pid is not None:
                self._signal(job, "TERM")
        if att.exit is not None:
            self._finish(job, att.exit, now)
            return
        if att.heartbeat != job.hb_value:
            job.hb_value, job.hb_seen = att.heartbeat, now
        if job.state == "stopping":
            self._escalate(job, now)
            return
        stale = now - (job.hb_seen if job.hb_seen is not None else now)
        if stale > self.heartbeat_timeout and job.pid is not None:
            alive = self._alive(job.host, job.pid, job.launch_id, job.handle)
            if alive is False or (alive is None and stale > 2 * self.heartbeat_timeout):
                self._finish(job, None, now, f"lost: no heartbeat for {stale:.0f} s and no process")

    def _escalate(self, job: Job, now: float) -> None:
        if job.pid is None:
            return
        if job.killed_at is None:
            if now - (job.stop_started or now) > self.grace:
                self._signal(job, "KILL", group=True)
                job.killed_at = now
                self.log(
                    f"{job.spec.name}: still running {self.grace:.0f} s after SIGTERM, sent SIGKILL"
                )
            return
        gone = self._alive(job.host, job.pid, job.launch_id, job.handle) is False
        if gone or now - job.killed_at > self.grace:
            self._finish(job, None, now, "killed")

    def _stop(self, job: Job, reason: str, requeue: bool, now: float) -> None:
        if job.state == "stopping":
            job.requeue = job.requeue and requeue
            return
        job.state, job.stop_reason, job.requeue = "stopping", reason, requeue
        job.stop_started, job.killed_at = now, None
        if job.handle and job.pid is None:
            self._signal(job, "KILL")
            self._finish(job, None, now, "removed from the Slurm queue before it started")
            return
        if job.pid is not None:
            self._signal(job, "TERM")
        self.log(
            f"{job.spec.name}: {reason}; stopping it on {job.slot} (SIGTERM, SIGKILL after {self.grace:.0f} s)"
        )

    def _finish(
        self, job: Job, code: int | None, now: float, note: str = "", startup: bool = False
    ) -> None:
        """Close the current launch of ``job`` and decide what comes next from its exit code."""
        self._update_progress(job, now)
        stopping = job.state == "stopping"
        diverged = code == 2 and os.path.exists(os.path.join(job.spec.output_dir, "DIVERGED"))
        if code == 0:
            state = "done"
        elif stopping:
            state = "pending" if job.requeue else "cancelled"
            note = f"{job.stop_reason}" + (f", {note}" if note else "")
        elif code == INTERRUPTED:
            state = "pending"
            note = "interrupted after a snapshot (walltime, preemption or SIGTERM); the next leg resumes it"
        elif diverged:
            state = "diverged"
            note = "diverged (DIVERGED marker); not retried"
        elif code == LINEAR_CHECK_FAILED:
            state = "rejected"
            note = "linear check outside its tolerance (linear_check.json); not retried"
        else:
            job.failures += 1
            job.avoid = [job.slot] if job.slot else []
            state = "pending" if job.failures <= self.max_retries else "failed"
            note = note or f"exit {code}; see {os.path.join(job.spec.output_dir, SCHED, 'log')}"
            if startup and job.slot:
                self.down_until[job.slot] = now + self.slot_backoff
        job.history.append(
            {
                "slot": job.slot,
                "host": job.host,
                "gpus": job.gpus,
                "launch": job.launch_id,
                "handle": job.handle,
                "start": job.launched_at,
                "end": now,
                "exit": code,
                "result": state,
                "note": note,
            }
        )
        self.log(f"{job.spec.name}: {state} (exit {code}{', ' + note if note else ''})")
        job.state, job.note = state, note
        job.slot, job.host, job.gpus, job.pid = None, None, [], None
        job.launch_id, job.launched_at, job.hb_value, job.hb_seen = None, None, None, None
        job.handle = None
        job.stop_reason, job.stop_started, job.killed_at, job.requeue = None, None, None, True

    def _update_progress(self, job: Job, now: float) -> None:
        steps = snapshot_steps(job.spec.output_dir)
        legs = run_legs(job.spec.output_dir)
        p = job.progress
        if legs and legs[-1].get("target_step") is not None:
            p["target"] = int(legs[-1]["target_step"])
        if not steps:
            return
        step = max(steps)
        if p.get("leg") != job.launch_id:
            p.update(leg=job.launch_id, first=[step, now], last=[step, now], step=step)
        elif step != p.get("step"):
            p.update(last=[step, now], step=step)

    @staticmethod
    def _choose(job: Job, free: list[Slot], waiting: list[Job]) -> Slot | None:
        """The fastest free slot with exactly the job's GPU count, else the fastest larger one.

        A larger slot is left alone while a waiting job needs more GPUs than this one and fits it.
        """
        need = job.spec.gpus

        def reserved(slot: Slot) -> bool:
            return len(slot.gpus) > need and any(
                need < k.spec.gpus <= len(slot.gpus) for k in waiting
            )

        if any(s.host == SLURM_HOST for s in free):
            return min(
                free,
                key=lambda s: (
                    int(s.id.rsplit(":", 1)[1]) if s.id.rsplit(":", 1)[1].isdigit() else 0
                ),
            )
        candidates = [s for s in free if len(s.gpus) >= need and not reserved(s)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda s: (s.id in job.avoid, len(s.gpus) != need, -s.speed, len(s.gpus), s.id),
        )

    def _place(self, now: float) -> None:
        busy = {j.slot for j in self._active()}
        free = [
            s
            for s in self.pool.values()
            if s.id not in busy and not s.drain and self.down_until.get(s.id, 0.0) <= now
        ]
        pending = sorted(
            (j for j in self.jobs.values() if j.state == "pending"),
            key=lambda j: (-j.spec.cost, j.order),
        )
        placed: set[str] = set()
        for job in pending:
            waiting = [k for k in pending if k is not job and k.spec.name not in placed]
            slot = self._choose(job, free, waiting)
            if slot is None:
                continue
            free.remove(slot)
            if self._launch(job, slot, now):
                placed.add(job.spec.name)

    def gyaradax_args(self, job: Job) -> list[str]:
        """The ``gyaradax`` arguments of the job's next leg: first launch, resume, or restart.

        A leg resumes the run directory once it holds a snapshot written by this job. Runs that
        stopped before their first snapshot start over with ``--overwrite``, which is never added
        to a directory this job has not written to.
        """
        spec = job.spec
        run_dir = spec.output_dir
        source = spec.config or spec.run_dir or ""
        first = ["run", source] + ([] if spec.run_dir else ["--output-dir", run_dir]) + spec.args
        ours = job.base_legs is not None and len(run_legs(run_dir)) > job.base_legs
        if not ours:
            args = first
        elif latest_snapshot(run_dir, quarantine=not self.dry_run) and os.path.isfile(
            os.path.join(run_dir, "config.yaml")
        ):
            args = ["run", run_dir] + strip_flags(spec.args, FRESH_FLAGS)
        else:
            args = first + ([] if "--overwrite" in first else ["--overwrite"])
        if spec.gpus > 1 and not _has_flag(args, MESH_FLAGS) and not config_shards(args[1]):
            args += ["--n-gpus", str(spec.gpus)]
        return args

    def launch_command(self, job: Job, slot: Slot, argv: list[str], launch_id: str) -> str:
        spec = job.spec
        sched = os.path.join(spec.output_dir, SCHED)
        env = {
            **slot.env,
            **spec.env,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, slot.gpus[: spec.gpus])),
        }
        wrapper = [
            slot.python,
            self.script,
            "wrap",
            "--dir",
            spec.output_dir,
            "--launch-id",
            launch_id,
            "--host",
            slot.host,
            "--slot",
            slot.id,
            "--heartbeat",
            f"{self.heartbeat:g}",
            "--",
        ]
        inner = shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *wrapper, *argv])
        q = shlex.quote
        return (
            f"test -x {q(slot.python)} && mkdir -p {q(sched)} && cd {q(spec.workdir)} && "
            f"{{ nohup {inner} >> {q(os.path.join(sched, 'log'))} 2>&1 < /dev/null & }}"
        )

    def _launch(self, job: Job, slot: Slot, now: float) -> bool:
        spec = job.spec
        if job.base_legs is None:
            job.base_legs = len(run_legs(spec.output_dir))
        broken = broken_diagnostics(spec.output_dir)
        if broken:
            job.state = "failed"
            job.note = (
                f"truncated {', '.join(broken)} in {spec.output_dir}; repair or remove, then resume"
            )
            self.log(f"{spec.name}: {job.note}")
            return False
        argv = [a.replace("{python}", slot.python) for a in self.program] + self.gyaradax_args(job)
        argv += self._backend_args(job, slot, now)
        launch_id = uuid.uuid4().hex[:16]
        if slot.host == SLURM_HOST:
            return self._submit(job, slot, argv, launch_id, now)
        command = self.launch_command(job, slot, argv, launch_id)
        if self.dry_run:
            self.log(
                f"[dry-run] {spec.name} -> {slot.id}: {self.executor.describe(slot.host, command)}"
            )
            return True
        try:
            self.executor.launch(slot.host, command)
        except LaunchError as exc:
            self.down_until[slot.id] = now + self.slot_backoff
            self.log(
                f"{spec.name}: launch on {slot.id} failed ({exc}); slot paused {self.slot_backoff:.0f} s"
            )
            return False
        job.state, job.slot, job.host = "starting", slot.id, slot.host
        job.gpus = slot.gpus[: spec.gpus]
        job.launch_id, job.launched_at, job.pid = launch_id, now, None
        job.hb_value, job.hb_seen, job.argv, job.note = None, None, argv, ""
        job.attempts += 1
        self.log(f"{spec.name}: launched on {slot.id}: {shlex.join(argv)}")
        return True

    def run(self, poll: float, once: bool = False, until_idle: bool = False) -> None:
        """Poll until a signal (or once); ``until_idle`` also stops once no job is pending or active.

        Past ``deadline`` the running jobs are stopped and requeued, and the loop ends once they are down.
        """
        stop: list[int] = []
        previous = {
            sig: signal.signal(sig, lambda signum, frame: stop.append(signum))
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            while True:
                self.poll()
                if once or self.dry_run or stop:
                    break
                if self.expired() and not self._active():
                    left = sum(j.state == "pending" for j in self.jobs.values())
                    self.log(
                        f"time limit reached; {left} job(s) pending, `gyaradax datagen run` resumes them"
                    )
                    break
                if (
                    until_idle
                    and self.jobs
                    and not any(
                        j.state == "pending" or j.state in ACTIVE for j in self.jobs.values()
                    )
                ):
                    self.log("every job has finished")
                    break
                wait = (
                    min(poll, 2.0)
                    if any(j.state == "stopping" for j in self.jobs.values())
                    else poll
                )
                deadline = time.monotonic() + wait
                while not stop and time.monotonic() < deadline:
                    time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))
                if stop:
                    break
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        self.save()
        if not self.dry_run:
            self.log("scheduler stopped; running jobs keep going and are reattached on restart")


def parse_duration(text: str) -> float:
    """Seconds in a duration such as ``10h``, ``90m``, ``1h30m``, ``2d`` or ``3600``."""
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1, "": 1}
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([dhms]?)", text.strip().lower())
    if not parts or re.sub(r"\d+(?:\.\d+)?\s*[dhms]?", "", text.strip().lower()).strip():
        raise argparse.ArgumentTypeError(f"not a duration: {text!r} (e.g. 10h, 90m, 3600)")
    seconds = sum(float(v) * units[u] for v, u in parts)
    if seconds <= 0:
        raise argparse.ArgumentTypeError(f"not a positive duration: {text!r}")
    return seconds


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    if s >= 86400:
        return f"{s // 86400}d{s % 86400 // 3600:02d}h"
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def eta_seconds(progress: dict[str, Any], now: float) -> float | None:
    """Time left for a running leg, from the snapshot steps seen during it."""
    target, first, last = progress.get("target"), progress.get("first"), progress.get("last")
    if not target or not first or not last:
        return None
    (s0, t0), (s1, t1) = first, last
    if s1 <= s0 or t1 <= t0:
        return None
    rate = (s1 - s0) / (t1 - t0)
    return max(0.0, (target - s1) / rate - (now - t1))


def _table(rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]


def format_status(doc: dict[str, Any], now: float, pending_requests: int = 0) -> str:
    """Tables of jobs (state, slot, progress, ETA) and slots from a state file's contents."""
    jobs = list(doc.get("jobs", {}).values())
    sched = doc.get("scheduler", {})
    lines = [
        f"scheduler {sched.get('host')}:{sched.get('pid')}, last poll "
        f"{_duration(now - doc.get('updated', now))} ago"
        + (f", {pending_requests} request(s) waiting" if pending_requests else "")
    ]
    if doc.get("deadline"):
        left = doc["deadline"] - now
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(doc["deadline"]))
        lines.append(
            f"time limit: jobs stop at {when} "
            + (f"(in {_duration(left)})" if left > 0 else f"(passed {_duration(-left)} ago)")
        )
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    lines.append("jobs: " + ", ".join(f"{counts[s]} {s}" for s in STATE_ORDER if s in counts))
    rows = [("JOB", "STATE", "SLOT", "GPUS", "COST", "FAIL", "STEP", "ETA", "NOTE")]
    rank = {s: i for i, s in enumerate(STATE_ORDER)}
    for j in sorted(jobs, key=lambda j: (rank.get(j["state"], 99), -j["spec"]["cost"], j["order"])):
        p = j.get("progress") or {}
        step = f"{p['step']}/{p.get('target', '?')}" if "step" in p else "-"
        eta = eta_seconds(p, now) if j["state"] in ACTIVE else None
        rows.append(
            (
                j["spec"]["name"],
                j["state"],
                (j.get("slot") or "-") + (f" (job {j['handle']})" if j.get("handle") else ""),
                ",".join(map(str, j.get("gpus") or [])) or str(j["spec"]["gpus"]),
                f"{j['spec']['cost']:g}",
                str(j.get("failures", 0)),
                step,
                _duration(eta),
                (j.get("note") or "")[:80],
            )
        )
    lines += [""] + _table(rows)
    for probe in (doc.get("probes") or {}).values():
        if not probe.get("usable"):
            lines.append(
                f"cuda: {probe.get('host')} cannot use {probe.get('lib')} ({probe.get('reason')}); JAX backend there"
            )
    by_slot = {j.get("slot"): j["spec"]["name"] for j in jobs if j["state"] in ACTIVE}
    srows = [("SLOT", "HOST", "GPUS", "SPEED", "STATUS", "JOB")]
    for sid, s in sorted(doc.get("slots", {}).items()):
        down = doc.get("down_until", {}).get(sid, 0.0)
        status = "busy" if sid in by_slot else "free"
        if s.get("drain"):
            status += ", draining"
        if down > now:
            status += f", paused {_duration(down - now)}"
        srows.append(
            (
                sid,
                s["host"],
                ",".join(map(str, s["gpus"])),
                f"{s['speed']:g}",
                status,
                by_slot.get(sid, "-"),
            )
        )
    lines += [""] + _table(srows)
    return "\n".join(lines)


def wrap(
    directory: str, launch_id: str, host: str, slot: str, heartbeat: float, command: list[str]
) -> int:
    """Run ``command`` and keep ``DIR/.sched`` up to date: pid, host, heartbeat, then the exit code.

    The wrapper leads its own process group; SIGTERM and SIGINT are forwarded to the command.
    """
    sched = os.path.join(directory, SCHED)
    os.makedirs(sched, exist_ok=True)
    with contextlib.suppress(OSError):
        os.setsid()

    def put(name: str, value: Any) -> None:
        _atomic_write(os.path.join(sched, name), f"{value}\n")

    for name in ("exit", "launch", "heartbeat", "pid", "child"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(sched, name))
    put("pid", os.getpid())
    put("host", host or socket.gethostname())
    put("slot", slot)
    put("gpus", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    put("heartbeat", f"{time.time():.3f}")
    put("launch", launch_id)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"=== {stamp} {socket.gethostname()} gpus={gpus} launch {launch_id}: {shlex.join(command)}",
        flush=True,
    )
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    except OSError as exc:
        print(f"=== cannot start: {exc}", flush=True)
        put("exit", f"127\n{launch_id}")
        return 127
    put("child", child.pid)

    def forward(signum: int, frame: Any) -> None:
        with contextlib.suppress(ProcessLookupError):
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    while True:
        try:
            rc = child.wait(timeout=heartbeat)
            break
        except subprocess.TimeoutExpired:
            put("heartbeat", f"{time.time():.3f}")
    code = rc if rc >= 0 else 128 - rc
    put("exit", f"{code}\n{launch_id}")
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} exit {code}", flush=True)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scheduler.py",
        description="Dispatch gyaradax runs over an ssh GPU mesh or Slurm (see gyaradax datagen).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the scheduler loop")
    run_p.add_argument("--pool", required=True, help="pool YAML (re-read every poll)")
    run_p.add_argument("--jobs", required=True, help="job YAML (re-read every poll)")
    run_p.add_argument("--state", required=True, help="scheduler state JSON")
    run_p.add_argument(
        "--poll", type=float, default=30.0, help="seconds between polls (default 30)"
    )
    run_p.add_argument(
        "--dry-run", action="store_true", help="print the launch commands of one round, run nothing"
    )
    run_p.add_argument("--once", action="store_true", help="poll once and exit")
    run_p.add_argument(
        "--grace", type=float, default=120.0, help="seconds from SIGTERM to SIGKILL (default 120)"
    )
    run_p.add_argument(
        "--heartbeat", type=float, default=30.0, help="wrapper heartbeat period (default 30 s)"
    )
    run_p.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=300.0,
        help="heartbeat age after which the process is checked over ssh (default 300 s)",
    )
    run_p.add_argument(
        "--start-timeout",
        type=float,
        default=300.0,
        help="seconds a launched wrapper has to report in (default 300)",
    )
    run_p.add_argument(
        "--max-retries", type=int, default=3, help="retries after failures (default 3)"
    )
    run_p.add_argument(
        "--slot-backoff",
        type=float,
        default=300.0,
        help="seconds a slot is skipped after a failed launch (default 300)",
    )
    run_p.add_argument(
        "--ssh-timeout", type=float, default=30.0, help="timeout of each ssh call (default 30 s)"
    )
    run_p.add_argument(
        "--time-limit",
        type=parse_duration,
        default=None,
        metavar="DURATION",
        help="stop and requeue every job after this long (e.g. 10h, 90m), then exit",
    )

    status_p = sub.add_parser("status", help="jobs, slots and ETAs from the state file")
    status_p.add_argument("--state", required=True)

    cancel_p = sub.add_parser("cancel", help="stop jobs (SIGTERM, grace, SIGKILL) and hold them")
    cancel_p.add_argument("names", nargs="+")
    cancel_p.add_argument("--state", required=True)
    cancel_p.add_argument(
        "--requeue", action="store_true", help="put the jobs back in the queue instead"
    )

    resume_p = sub.add_parser(
        "resume", help="put cancelled, failed, diverged or rejected jobs back in the queue"
    )
    resume_p.add_argument("names", nargs="+")
    resume_p.add_argument("--state", required=True)

    wrap_p = sub.add_parser("wrap", help="launch wrapper run on the GPU host (internal)")
    wrap_p.add_argument("--dir", required=True)
    wrap_p.add_argument("--launch-id", required=True)
    wrap_p.add_argument("--host", default="")
    wrap_p.add_argument("--slot", default="")
    wrap_p.add_argument("--heartbeat", type=float, default=30.0)
    wrap_p.add_argument("cmd", nargs=argparse.REMAINDER)
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    for path in (args.pool, args.jobs):
        if not os.path.isfile(path):
            raise SystemExit(f"error: {path} not found")
    lock = None
    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.state)), exist_ok=True)
        lock = open(args.state + ".lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"error: another scheduler holds {args.state}.lock")
    sched = Scheduler(
        args.pool,
        args.jobs,
        args.state,
        SSHExecutor(timeout=args.ssh_timeout),
        heartbeat=args.heartbeat,
        heartbeat_timeout=args.heartbeat_timeout,
        grace=args.grace,
        start_timeout=args.start_timeout,
        max_retries=args.max_retries,
        slot_backoff=args.slot_backoff,
        dry_run=args.dry_run,
        deadline=time.time() + args.time_limit if args.time_limit else None,
    )
    sched.run(args.poll, once=args.once)
    if lock is not None:
        lock.close()
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.state):
        raise SystemExit(f"error: {args.state} not found")
    with open(args.state) as fh:
        doc = json.load(fh)
    waiting = len(
        glob.glob(os.path.join(glob.escape(requests_dir(os.path.abspath(args.state))), "*.json"))
    )
    print(format_status(doc, time.time(), waiting))
    return 0


def _cmd_request(args: argparse.Namespace) -> int:
    action = "requeue" if getattr(args, "requeue", False) else args.command
    state = os.path.abspath(args.state)
    for name in args.names:
        submit_request(state, action, name)
        print(f"{action} {name}: queued for the scheduler's next poll")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "status":
        return _cmd_status(args)
    if args.command in ("cancel", "resume"):
        return _cmd_request(args)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        raise SystemExit("error: wrap needs a command after --")
    return wrap(args.dir, args.launch_id, args.host, args.slot, args.heartbeat, cmd)


if __name__ == "__main__":
    raise SystemExit(main())
