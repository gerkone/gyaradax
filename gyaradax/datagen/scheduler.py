#!/usr/bin/env python
"""Scheduler for gyaradax runs over an ssh GPU mesh.

``run`` re-reads the targets and job YAML files on every poll, places pending jobs longest-first
on free slots, starts them through the ``wrap`` subcommand of this file and follows them through
the ``RUN_DIR/.sched/`` files on the shared filesystem. A run stopped by SIGTERM (a cancel, a
removed slot) snapshots and exits 75, and its next leg resumes the run directory. ``status``,
``cancel`` and ``resume`` work on the state file. Standard library and PyYAML only, so ``wrap``
runs from the file path without importing JAX. ``gyaradax datagen`` (campaign.py) builds the job
file.
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
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from typing import IO, Any, Callable, Sequence

import yaml

# hosts share the scheduler's interpreter unless the pool names another
DEFAULT_PYTHON = sys.executable
DEFAULT_PROGRAM = ("{python}", "-m", "gyaradax")
SCHED = ".sched"
ACTIVE = ("starting", "running", "stopping")
STATE_ORDER = (
    "running", "starting", "stopping", "pending", "failed", "diverged", "cancelled", "banded",
    "decayed", "stable", "done",
)  # fmt: skip
# finished with a physics outcome, like done: not retried, not resumed
OUTCOME_STATES = ("stable", "decayed", "banded")
RESUMABLE = ("cancelled", "failed", "diverged", *OUTCOME_STATES)
DIAGNOSTICS = (
    "fluxes.npz", "fluxes_em.npz", "kxspec.npz", "kyspec.npz", "growth.npz", "dt_history.npz",
)  # fmt: skip
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
# exit codes of the run outcomes (gyaradax.diagnostics.outcome.EXIT_*)
OUTCOME_EXITS = {
    4: ("stable", "linearly stable (linear_gate.json); nonlinear run skipped"),
    5: ("decayed", "heat flux decayed (outcome.json); stopped"),
    6: ("banded", "s-Nyquist banding (outcome.json); stopped"),
}
STABLE_NOTE = "stopped by hand as stable (heat flux still decaying)"
GATE_FILE = "linear_gate.json"
OUTCOME_FILE = "outcome.json"
# job gpus value meaning every GPU of the slot it lands on ("all" in the job file)
ALL_GPUS = 0
# host switches: drain finishes jobs, off stops them with a snapshot, kill stops them now
HOST_SWITCHES = ("on", "drain", "off", "kill")
YAML_ERRORS = (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError)
CALL_ERRORS = (subprocess.TimeoutExpired, OSError)


RUNS_DB_SCHEMA = """
CREATE TABLE runs (
    name TEXT PRIMARY KEY, state TEXT, slot TEXT, host TEXT, gpus TEXT, legs INTEGER,
    attempts INTEGER, failures INTEGER, first_start TEXT, last_start TEXT, finished TEXT,
    frames INTEGER, frames_total INTEGER, t REAL, t_end REAL, dump_dt REAL, step INTEGER,
    target_step INTEGER, eta_h REAL, cost REAL, note TEXT, output_dir TEXT, dataset_dir TEXT,
    config TEXT, params TEXT, outcome TEXT, banded INTEGER
);
CREATE TABLE legs (
    name TEXT, leg INTEGER, launch TEXT, slot TEXT, host TEXT, gpus TEXT, start TEXT, end TEXT,
    hours REAL, exit INTEGER, result TEXT, note TEXT, PRIMARY KEY (name, leg)
);
CREATE TABLE slots (
    id TEXT PRIMARY KEY, host TEXT, gpus TEXT, speed REAL, drain INTEGER, busy INTEGER,
    paused_until TEXT, job TEXT
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE outcomes (
    name TEXT PRIMARY KEY, state TEXT, class TEXT, label TEXT,
    rlt TEXT, rln TEXT, tmp TEXT, mas TEXT, rlt_i REAL, rlt_e REAL, rln_i REAL, temp_ratio REAL,
    q REAL, shat REAL, eps REAL, grid TEXT,
    gate TEXT, settled INTEGER, gamma_max REAL, peak_ky REAL, omega_peak REAL, linear_t REAL,
    gamma_max_lowky REAL, peak_ky_lowky REAL, low_ky REAL,
    ky TEXT, gamma TEXT, gamma_std TEXT, omega TEXT, omega_std TEXT,
    t REAL, frames INTEGER, dataset_frames INTEGER, stopped TEXT, banded INTEGER, banded_t REAL,
    nyq_zonal REAL, nyq_nonzonal REAL, nyq_zonal_max REAL, q_peak REAL, q_last REAL,
    avg_t0 REAL, avg_t1 REAL, avg_frames INTEGER, qi REAL, qi_err REAL, qe REAL, qe_err REAL,
    gi REAL, gi_err REAL, q_drift REAL, q_drift_sigma REAL, fluxes TEXT,
    cost REAL, gpu_hours REAL
);
"""


def _when(epoch: float | None) -> str | None:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch)) if epoch else None


def _js(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _bit(doc: dict[str, Any], key: str) -> int | None:
    return None if key not in doc else int(bool(doc[key]))


def _csv(values: Sequence[int] | None) -> str:
    return ",".join(map(str, values or []))


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def job_meta(spec: JobSpec) -> dict[str, Any]:
    """Physics and geometry inputs, end time, frame spacing and dataset frame directory of a job."""
    meta: dict[str, Any] = {"params": {}}
    path = spec.config or (os.path.join(spec.run_dir, "config.yaml") if spec.run_dir else None)
    with contextlib.suppress(*YAML_ERRORS):
        with open(path or "") as fh:
            cfg = yaml.safe_load(fh) or {}
        meta["params"] = {
            k: cfg[k] for k in ("geometry", "physics", "grid") if isinstance(cfg.get(k), dict)
        }
        solver = cfg.get("solver") or {}
        if solver.get("t_end") is not None and solver.get("dump_dt"):
            meta["t_end"], meta["dump_dt"] = float(solver["t_end"]), float(solver["dump_dt"])
    root, name = _flag_value(spec.args, "--dataset"), _flag_value(spec.args, "--dataset-name")
    if root:
        meta["dataset_dir"] = os.path.join(root, f"{name or spec.name}_ifft_realpotens")
    return meta


def read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _species(values: Any) -> list[Any]:
    return list(values) if isinstance(values, list) else [values]


def outcome_row(
    name: str,
    state: str,
    meta: dict[str, Any],
    run_dir: str,
    cost: float,
    gpu_hours: float,
    frames: int | None,
) -> tuple[Any, ...]:
    """One row of the ``outcomes`` table from the config and the run's linear_gate.json / outcome.json."""
    params = meta.get("params") or {}
    physics, geometry, grid = (params.get(k) or {} for k in ("physics", "geometry", "grid"))
    signz = _species(physics.get("signz", 1.0))
    ion = next((i for i, z in enumerate(signz) if float(z) > 0), 0)
    electron = next((i for i, z in enumerate(signz) if float(z) < 0), None)

    def of(key: str, i: int | None) -> float | None:
        values = _species(physics.get(key)) if physics.get(key) is not None else []
        return float(values[i]) if i is not None and i < len(values) else None

    tmp_i, tmp_e = of("tmp", ion), of("tmp", electron)
    gate = read_json(os.path.join(run_dir, GATE_FILE)) or {}
    out = read_json(os.path.join(run_dir, OUTCOME_FILE)) or {}
    avg = out.get("average") or {}
    window = avg.get("window") or [None, None]

    def flux(i: int | None, channel: int, key: str = "mean") -> float | None:
        rows = avg.get(key) or []
        return rows[i][channel] if i is not None and i < len(rows) else None

    return (
        name, state, out.get("class"), out.get("label"),
        *(_js(physics.get(k)) for k in ("rlt", "rln", "tmp", "mas")),
        of("rlt", ion), of("rlt", electron), of("rln", ion),
        tmp_e / tmp_i if tmp_e is not None and tmp_i else None,
        *(geometry.get(k) for k in ("q", "shat", "eps")), _js(grid or None),
        gate.get("decision"), _bit(gate, "settled"),
        *(gate.get(k) for k in ("gamma_max", "peak_ky", "omega_peak", "t", "gamma_max_lowky",
                                "peak_ky_lowky", "low_ky")),
        *(_js(gate.get(k)) for k in ("ky", "gamma", "gamma_std", "omega", "omega_std")),
        out.get("t"), out.get("frames"), frames, out.get("stopped"), _bit(out, "banded"),
        *(out.get(k) for k in ("banded_t", "nyq_zonal", "nyq_nonzonal", "nyq_zonal_max",
                               "q_peak", "q_last")),
        window[0], window[1], avg.get("frames"),
        flux(ion, 1), flux(ion, 1, "err"), flux(electron, 1), flux(electron, 1, "err"),
        flux(ion, 0), flux(ion, 0, "err"),
        avg.get("q_drift"), avg.get("q_drift_sigma"), _js(avg or None), cost, gpu_hours,
    )  # fmt: skip


def dataset_frames(dataset_dir: str | None) -> int | None:
    """Frames a run has written to its dataset trajectory so far (``data/poten_*`` shards)."""
    try:
        return sum(
            n.startswith("poten_") for n in os.listdir(os.path.join(dataset_dir or "", "data"))
        )
    except OSError:
        return None


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
    names = ("launch", "exit", "pid", "host", "slot", "gpus", "heartbeat")
    get = {n: _read(os.path.join(run_dir, SCHED, n)) for n in names}
    exit_lines = (get["exit"] or "").splitlines()
    code = _int(exit_lines[0]) if exit_lines else None
    if len(exit_lines) > 1 and exit_lines[1].strip() != get["launch"]:
        code = None
    return Attempt(**{**get, "pid": _int(get["pid"]), "exit": code})


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


@dataclass
class Slot:
    host: str
    gpus: list[int]
    speed: float = 1.0
    python: str = DEFAULT_PYTHON
    env: dict[str, str] = field(default_factory=dict)
    drain: bool = False
    busy_mib: int = 0

    @property
    def id(self) -> str:
        return f"{self.host}:{_csv(self.gpus)}"


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


def _known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def job_gpus(spec: JobSpec, slot: Slot) -> list[int]:
    """The GPUs of ``slot`` a leg of ``spec`` runs on."""
    return list(slot.gpus) if spec.gpus == ALL_GPUS else slot.gpus[: spec.gpus]


def _env(value: Any, where: str) -> dict[str, str]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where}: env must be a mapping")
    bad = [k for k in value if not ENV_NAME.fullmatch(str(k))]
    if bad:
        raise ValueError(f"{where}: invalid environment variable names {bad}")
    return {str(k): str(v) for k, v in value.items()}


def _refuse(doc: dict[str, Any], keys: Sequence[str], where: str) -> None:
    used = [k for k in keys if doc.get(k)]
    if used:
        raise ValueError(f"{where}: {', '.join(used)} no longer supported")


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
    _refuse(raw, ("name", "env"), where)
    merged = {**defaults, **raw}
    host = str(merged.get("host") or "").strip()
    if not host or host.startswith("-") or " " in host:
        raise ValueError(f"{where}: host is required")
    gpus = merged.get("gpus")
    gpus = [gpus] if isinstance(gpus, int) else gpus
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
    busy_mib = merged.get("busy_mib", 0)
    if isinstance(busy_mib, bool) or not isinstance(busy_mib, int) or busy_mib < 0:
        raise ValueError(f"{where}: busy_mib must be a non-negative integer")
    return Slot(
        host=host,
        gpus=list(gpus),
        speed=speed,
        python=str(merged.get("python") or DEFAULT_PYTHON),
        env=_env(defaults.get("env"), where),
        drain=bool(merged.get("drain", False)),
        busy_mib=busy_mib,
    )


def host_switches(doc: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """The valid ``hosts`` switches of a targets document, plus the entries skipped and why."""
    raw = doc.get("hosts") or {}
    choices = ", ".join(HOST_SWITCHES)
    if not isinstance(raw, dict):
        return {}, [f"hosts: expected a mapping of host to {choices}"]
    switches = {str(h): v for h, v in raw.items() if v in HOST_SWITCHES}
    bad = [
        f"hosts: {h}: {v!r} is not one of {choices}"
        for h, v in raw.items()
        if v not in HOST_SWITCHES
    ]
    return switches, bad


def gpu_groups(doc: dict[str, Any]) -> tuple[dict[str, list[list[int]]], list[str]]:
    """The ``gpu_groups`` of a targets document (per host, the GPU sets a multi-GPU slot may span), plus problems."""
    raw = doc.get("gpu_groups") or {}
    if not isinstance(raw, dict):
        return {}, ["gpu_groups: expected a mapping of host to lists of GPU indices"]
    groups, bad = {}, []
    for host, sets in raw.items():
        if isinstance(sets, list) and all(
            isinstance(g, list) and g and all(isinstance(i, int) and not isinstance(i, bool) for i in g) for g in sets
        ):
            groups[str(host)] = [list(g) for g in sets]
        else:
            bad.append(f"gpu_groups: {host}: expected lists of GPU indices")
    return groups, bad


def load_pool(path: str) -> tuple[dict[str, Slot], list[str]]:
    """Parse the pool file into slots by id, plus the entries skipped and why; raises on an unreadable file."""
    doc = _load_yaml(path, "slots")
    _refuse(doc, ("slurm",), "targets")
    defaults = doc.get("defaults") or {}
    slots: dict[str, Slot] = {}
    used: dict[str, set[int]] = {}
    switches, problems = host_switches(doc)
    groups, bad = gpu_groups(doc)
    problems.extend(bad)
    for i, raw in enumerate(doc.get("slots") or []):
        try:
            slot = _slot(raw, defaults, f"pool slot {i}")
        except (TypeError, ValueError) as exc:
            problems.append(str(exc))
            continue
        short = slot.host.split(".")[0]
        switch = switches.get(slot.host, switches.get(short, "on"))
        if switch in ("off", "kill"):
            continue
        slot.drain = slot.drain or switch == "drain"
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


def _args(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(f"{where}: args must be a string or a list")


def _job(raw: Any, base: str, where: str) -> JobSpec:
    if not isinstance(raw, dict) or not raw.get("name"):
        raise ValueError(f"{where}: a job is a mapping with a name")
    name = str(raw["name"])
    where = f"job {name}"
    _refuse(raw, ("env", "workdir"), where)
    workdir = os.path.abspath(base)

    def path(key: str) -> str | None:
        value = raw.get(key)
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
    args = _args(raw.get("args"), where)
    if _has_flag(args, REFUSED_FLAGS):
        raise ValueError(
            f"{where}: --device, --device-list and --output-dir are set by the scheduler"
        )
    raw_gpus = raw.get("gpus", 1)
    gpus = ALL_GPUS if raw_gpus == "all" else int(raw_gpus)
    if gpus != 1 and raw_gpus != "all":
        raise ValueError(f"{where}: gpus must be 1 or 'all'")
    return JobSpec(
        name=name,
        output_dir=output_dir,
        config=config,
        run_dir=run_dir,
        args=args,
        gpus=gpus,
        cost=float(raw.get("cost", 1.0)),
        workdir=workdir,
    )


def load_jobs(path: str) -> tuple[list[JobSpec], list[str]]:
    """Parse the job file into specs with absolute paths, plus the entries skipped and why.

    Relative paths are taken from the top-level ``workdir``, itself relative to the job file's
    directory; it is also the directory the runs are launched from.
    """
    doc = _load_yaml(path, "jobs")
    _refuse(doc, ("defaults",), "jobs")
    base = os.path.join(os.path.dirname(os.path.abspath(path)), str(doc.get("workdir", ".")))
    specs, names, problems = [], set(), []
    for i, raw in enumerate(doc.get("jobs") or []):
        try:
            spec = _job(raw, base, f"job {i}")
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
    pid: int | None = None
    hb_value: str | None = None
    hb_seen: float | None = None
    stop_reason: str | None = None
    finish_as: str | None = None
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
        data = _known(cls, data)
        return cls(**{**data, "spec": JobSpec(**_known(JobSpec, data["spec"]))})

    def leg(self, **extra: Any) -> dict[str, Any]:
        return {
            "slot": self.slot, "host": self.host, "gpus": self.gpus, "launch": self.launch_id,
            "start": self.launched_at, **extra,
        }  # fmt: skip


# reset when a launch ends
LAUNCH_FIELDS = (
    "slot", "host", "gpus", "pid", "launch_id", "launched_at", "hb_value", "hb_seen", "stop_reason",
    "stop_started", "killed_at", "requeue", "finish_as",
)  # fmt: skip


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

    def ok(self, host: str, command: str) -> bool:
        try:
            return self.run(host, command).returncode == 0
        except CALL_ERRORS:
            return False

    def launch(self, host: str, command: str) -> None:
        """Start a self-detaching command; raises LaunchError when it surely did not start."""
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
        except CALL_ERRORS:
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
        return self.ok(host, cmd)


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


def request_files(state_path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(glob.escape(requests_dir(state_path)), "*.json")))


def submit_request(state_path: str, action: str, name: str) -> str:
    directory = requests_dir(state_path)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{time.time_ns()}-{os.getpid()}-{action}.json")
    _atomic_write(path, json.dumps({"action": action, "name": name}))
    return path


def lock_state(state_path: str, error: str) -> IO[str]:
    """Hold ``STATE.lock`` exclusively, or exit with ``error``."""
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    lock = open(state_path + ".lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(error)
    return lock


def _insert(con: sqlite3.Connection, table: str, row: Sequence[Any]) -> None:
    con.execute(f"INSERT INTO {table} VALUES ({','.join('?' * len(row))})", tuple(row))


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
        self.clock = clock
        self.log = log or (
            lambda msg: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)
        )
        self.jobs: dict[str, Job] = {}
        self.pool: dict[str, Slot] = {}
        self.probes: dict[str, dict[str, Any]] = {}
        self.down_until: dict[str, float] = {}
        self._gpu_mem: dict[str, dict[int, int] | None] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self.killed_hosts: set[str] = set()
        self._busy: set[str] = set()
        self.next_order = 0
        self._warned: set[str] = set()
        self._specs: dict[str, JobSpec] = {}
        self._fresh = True
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path) as fh:
            doc = json.load(fh)
        self.jobs = {name: Job.from_dict(d) for name, d in doc.get("jobs", {}).items()}
        self.pool = {sid: Slot(**_known(Slot, s)) for sid, s in doc.get("slots", {}).items()}
        self.probes = doc.get("probes", {})
        self.down_until = {k: float(v) for k, v in doc.get("down_until", {}).items()}
        self.next_order = int(doc.get("next_order", len(self.jobs)))

    def save(self) -> None:
        doc = {
            "version": 1,
            "updated": self.clock(),
            "scheduler": {"host": socket.gethostname(), "pid": os.getpid()},
            "pool_file": self.pool_path,
            "jobs_file": self.jobs_path,
            "next_order": self.next_order,
            "slots": {sid: asdict(s) for sid, s in self.pool.items()},
            "probes": self.probes,
            "down_until": self.down_until,
            "jobs": {name: asdict(j) for name, j in self.jobs.items()},
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        _atomic_write(self.state_path, json.dumps(doc, indent=1, sort_keys=True))
        try:
            self.write_db(doc["updated"])
        except (OSError, sqlite3.Error) as exc:
            self._warn(f"run database not written ({exc})")

    @property
    def db_path(self) -> str:
        return os.path.splitext(self.state_path)[0] + ".sqlite"

    def _meta_of(self, spec: JobSpec) -> dict[str, Any]:
        if spec.name not in self._meta:
            self._meta[spec.name] = job_meta(spec)
        return self._meta[spec.name]

    def write_db(self, now: float) -> None:
        """Rebuild the SQLite mirror of the queue and swap it in whole, for readers on any host."""
        tmp = f"{self.db_path}.{os.getpid()}.tmp"
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        with contextlib.closing(sqlite3.connect(tmp)) as con:
            con.executescript(RUNS_DB_SCHEMA)
            for name, job in self.jobs.items():
                self._db_job(con, name, job, now)
            running = {j.slot: n for n, j in self.jobs.items() if j.state in ACTIVE and j.slot}
            for sid, slot in self.pool.items():
                down = self.down_until.get(sid, 0)
                _insert(con, "slots", (
                    sid, slot.host, _csv(slot.gpus), slot.speed, int(slot.drain),
                    int(sid in self._busy), _when(self.down_until.get(sid)) if down > now else None,
                    running.get(sid),
                ))  # fmt: skip
            for row in (
                ("updated", _when(now)),
                ("scheduler", f"{socket.gethostname()} pid {os.getpid()}"),
                ("targets", self.pool_path),
                ("jobs", self.jobs_path),
                ("deadline", None),
            ):
                _insert(con, "meta", row)
            con.commit()
        os.replace(tmp, self.db_path)

    def _db_job(self, con: sqlite3.Connection, name: str, job: Job, now: float) -> None:
        spec, p, meta = job.spec, job.progress, self._meta_of(job.spec)
        active = job.state in ACTIVE
        starts = [h.get("start") for h in job.history if h.get("start")]
        starts += [job.launched_at] if job.launched_at else []
        finished = job.history[-1].get("end") if job.history and not active else None
        frames = p.get("frames")
        eta = eta_seconds(p, now) if active else None
        outcome = read_json(os.path.join(spec.output_dir, OUTCOME_FILE)) or {}
        _insert(con, "runs", (
            name, job.state, job.slot, job.host, _csv(job.gpus) or None,
            len(job.history) + active, job.attempts, job.failures,
            _when(min(starts) if starts else None), _when(max(starts) if starts else None),
            _when(finished), frames, p.get("frames_total"),
            frames * meta["dump_dt"] if frames is not None and "dump_dt" in meta else None,
            meta.get("t_end"), meta.get("dump_dt"), p.get("step"), p.get("target"),
            round(eta / 3600.0, 3) if eta is not None else None,
            spec.cost, job.note or None, spec.output_dir, meta.get("dataset_dir"),
            spec.config or spec.run_dir, json.dumps(meta.get("params") or {}),
            outcome.get("class"), _bit(outcome, "banded"),
        ))  # fmt: skip
        gpu_hours = 0.0
        for i, h in enumerate(job.history + ([job.leg(result=job.state)] if active else [])):
            start, end = h.get("start"), h.get("end")
            hours = ((end or now) - start) / 3600.0 if start else None
            if hours is not None:
                gpu_hours += hours * max(len(h.get("gpus") or []), 1)
            _insert(con, "legs", (
                name, i + 1, h.get("launch"), h.get("slot"), h.get("host"),
                _csv(h.get("gpus")) or None, _when(start), _when(end),
                round(hours, 3) if hours is not None else None,
                h.get("exit"), h.get("result"), h.get("note") or None,
            ))  # fmt: skip
        row = outcome_row(name, job.state, meta, spec.output_dir, spec.cost, gpu_hours, frames)
        _insert(con, "outcomes", row)

    def _warn(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            self.log(f"warning: {msg}")

    def _active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.state in ACTIVE]

    def _alive(self, host: str | None, pid: int | None, token: str | None) -> bool | None:
        if pid is None:
            return False
        return self.executor.alive(host or "", pid, token or "")

    def _signal(self, job: Job, sig: str, group: bool = False) -> bool:
        if job.pid is None:
            return False
        if group:
            return self.executor.signal(job.host or "", job.pid, sig, group=True)
        return self.executor.signal(job.host or "", job.pid, sig, token=job.launch_id)

    def _probe(self, slot: Slot, workdir: str, now: float) -> dict[str, Any] | None:
        """The host's CUDA library state from ``gyaradax/backends/cuda_lib.py probe``, rechecked hourly."""
        tool = os.path.join(workdir, "gyaradax", "backends", "cuda_lib.py")
        if not os.path.isfile(tool):
            return None
        key = f"{slot.host} {slot.python} {workdir}"
        cached = self.probes.get(key)
        if cached is None or now - float(cached.get("checked", 0.0)) > 3600.0:
            try:
                res = self.executor.run(
                    slot.host, f"{shlex.quote(slot.python)} {shlex.quote(tool)} probe"
                )
                found = json.loads(res.stdout.strip().splitlines()[-1])
            except (*CALL_ERRORS, ValueError, IndexError):
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
        return ["--backend", "auto" if probe is None or probe.get("usable") else "jax"]

    def launch_command(self, job: Job, slot: Slot, argv: list[str], launch_id: str) -> str:
        """``nohup env ... python scheduler.py wrap ... -- argv`` for one leg of ``job``, detached."""
        spec = job.spec
        sched = os.path.join(spec.output_dir, SCHED)
        env = {
            **slot.env,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "CUDA_VISIBLE_DEVICES": _csv(job_gpus(spec, slot)),
        }
        wrapper = [
            slot.python, self.script, "wrap", "--dir", spec.output_dir, "--launch-id", launch_id,
            "--host", slot.host, "--slot", slot.id, "--heartbeat", f"{self.heartbeat:g}", "--",
        ]  # fmt: skip
        inner = shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *wrapper, *argv])
        q = shlex.quote
        return (
            f"test -x {q(slot.python)} && mkdir -p {q(sched)} && cd {q(spec.workdir)} && "
            f"{{ nohup {inner} >> {q(os.path.join(sched, 'log'))} 2>&1 < /dev/null & }}"
        )

    def _started(self, job: Job, slot: Slot, argv: list[str], launch_id: str, now: float) -> None:
        job.state, job.slot, job.host = "starting", slot.id, slot.host
        job.gpus = job_gpus(job.spec, slot)
        job.launch_id, job.launched_at, job.pid = launch_id, now, None
        job.hb_value, job.hb_seen, job.argv, job.note = None, None, argv, ""
        job.attempts += 1

    def poll(self) -> None:
        """One round: re-read pool and jobs, apply requests, follow active jobs, launch pending."""
        now = self.clock()
        self._gpu_mem = {}
        self._refresh_pool()
        self._refresh_jobs(now)
        if self._fresh:
            for job in [j for j in self.jobs.values() if j.state == "pending"]:
                self._adopt(job, now)
            self._fresh = False
        self._apply_requests(now)
        for job in self._active():
            self._monitor(job, now)
        for job in self._active():
            if job.slot not in self.pool:
                if {job.host, (job.host or "").split(".")[0]} & self.killed_hosts:
                    self._kill(job, f"host {job.host} killed", now)
                else:
                    self._stop(job, "slot removed", True, now)
        # jobs resumed or requeued in this round launch with the current spec
        self._sync_pending()
        self._place(now)
        for job in self._active():
            self._update_progress(job, now)
        self.save()

    def _refresh_pool(self) -> None:
        try:
            slots, problems = load_pool(self.pool_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self._warn(f"pool {self.pool_path} unreadable ({exc}); keeping the previous pool")
            return
        with contextlib.suppress(OSError, ValueError, yaml.YAMLError):
            switches, _ = host_switches(_load_yaml(self.pool_path, "slots"))
            self.killed_hosts = {h for h, v in switches.items() if v == "kill"}
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
            j.spec.output_dir: j.spec.name
            for j in self.jobs.values()
            if j.state not in ("done", *OUTCOME_STATES)
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
                self.log(f"job {spec.name} added (cost {spec.cost:g}, {spec.gpus or 'all'} GPU)")
                if not self._fresh:
                    self._adopt(job, now)
        self._specs = {spec.name: spec for spec in specs}
        self._sync_pending()

    def _sync_pending(self) -> None:
        """Give pending jobs the spec last read from the jobs file."""
        for name, spec in self._specs.items():
            job = self.jobs.get(name)
            if job is None or job.state != "pending" or job.spec == spec:
                continue
            if job.base_legs is not None and spec.output_dir != job.spec.output_dir:
                self._warn(f"job {name}: output_dir cannot change once the job has run")
                continue
            job.spec = spec
            self.log(f"job {name} updated")

    def _adopt(self, job: Job, now: float) -> None:
        """Take over a still-running launch of the job's ``.sched`` files."""
        att = read_attempt(job.spec.output_dir)
        if not att.launch or att.exit is not None or att.pid is None or not att.host:
            return
        if self._alive(att.host, att.pid, att.launch) is not True:
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
        for path in request_files(self.state_path):
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
            requeue = action == "requeue"
            if action in ("cancel", "requeue") and job.state in ACTIVE:
                self._stop(job, "requeued" if requeue else "cancelled", requeue, now)
            elif action == "cancel" and job.state == "pending":
                job.state = "cancelled"
                self.log(f"{name}: cancelled")
            elif action == "stable" and job.state in ACTIVE:
                job.finish_as = "decayed"
                self._stop(job, STABLE_NOTE, False, now)
            elif action == "stable" and job.state in ("pending", "cancelled"):
                job.state, job.note = "decayed", STABLE_NOTE
                self.log(f"{name}: decayed ({STABLE_NOTE})")
            elif action == "resume" and job.state in RESUMABLE:
                job.state, job.failures, job.note = "pending", 0, ""
                self.log(f"{name}: back in the queue")
            elif action == "resume" and job.state == "stopping":
                job.requeue = True
            elif action in ("cancel", "requeue", "stable", "resume"):
                self.log(f"request {action} {name}: nothing to do, the job is {job.state}")

    def _monitor(self, job: Job, now: float) -> None:
        att = read_attempt(job.spec.output_dir)
        if att.launch != job.launch_id:
            if job.launched_at is not None and now - job.launched_at > self.start_timeout:
                note = "the wrapper did not start; see .sched/log"
                self._finish(job, None, now, note, startup=True)
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
            alive = self._alive(job.host, job.pid, job.launch_id)
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
        gone = self._alive(job.host, job.pid, job.launch_id) is False
        if gone or now - job.killed_at > self.grace:
            self._finish(job, None, now, "killed")

    def _stop(self, job: Job, reason: str, requeue: bool, now: float) -> None:
        if job.state == "stopping":
            job.requeue = job.requeue and requeue
            return
        job.state, job.stop_reason, job.requeue = "stopping", reason, requeue
        job.stop_started, job.killed_at = now, None
        if job.pid is not None:
            self._signal(job, "TERM")
        self.log(
            f"{job.spec.name}: {reason}; stopping it on {job.slot} (SIGTERM, SIGKILL after {self.grace:.0f} s)"
        )

    def _kill(self, job: Job, reason: str, now: float) -> None:
        """SIGKILL ``job`` at once and queue it again."""
        if job.state == "stopping" and job.killed_at is not None:
            return
        job.state, job.stop_reason, job.requeue = "stopping", reason, True
        job.stop_started, job.killed_at = now - self.grace - 1.0, None
        if job.pid is not None:
            self._signal(job, "KILL", group=True)
            job.killed_at = now
        self.log(
            f"{job.spec.name}: {reason}; SIGKILL on {job.slot}, it resumes from its last snapshot"
        )

    def _finish(
        self, job: Job, code: int | None, now: float, note: str = "", startup: bool = False
    ) -> None:
        """Close the current launch of ``job`` and decide what comes next from its exit code."""
        self._update_progress(job, now)
        diverged = code == 2 and os.path.exists(os.path.join(job.spec.output_dir, "DIVERGED"))
        if code == 0:
            state = "done"
        elif code in OUTCOME_EXITS:
            state, note = OUTCOME_EXITS[code]
            note += "; not retried"
        elif job.state == "stopping":
            state = job.finish_as or ("pending" if job.requeue else "cancelled")
            note = f"{job.stop_reason}" + (f", {note}" if note else "")
        elif code == INTERRUPTED:
            state = "pending"
            note = "interrupted after a snapshot (SIGTERM); the next leg resumes it"
        elif diverged:
            state = "diverged"
            note = "diverged (DIVERGED marker); not retried"
        else:
            job.failures += 1
            job.avoid = [job.slot] if job.slot else []
            state = "pending" if job.failures <= self.max_retries else "failed"
            note = note or f"exit {code}; see {os.path.join(job.spec.output_dir, SCHED, 'log')}"
            if startup and job.slot:
                self.down_until[job.slot] = now + self.slot_backoff
        job.history.append(job.leg(end=now, exit=code, result=state, note=note))
        self.log(f"{job.spec.name}: {state} (exit {code}{', ' + note if note else ''})")
        job.state, job.note = state, note
        blank = Job(job.spec)
        for key in LAUNCH_FIELDS:
            setattr(job, key, getattr(blank, key))

    def _update_progress(self, job: Job, now: float) -> None:
        meta, p = self._meta_of(job.spec), job.progress
        out = read_json(os.path.join(job.spec.output_dir, OUTCOME_FILE))
        if out is not None:
            p["outcome"] = out.get("label") or out.get("class")
        frames = dataset_frames(meta.get("dataset_dir"))
        if frames is not None:
            p["frames"] = frames
            if "t_end" in meta:
                p["frames_total"] = math.ceil(meta["t_end"] / meta["dump_dt"] - 1e-9)
            if p.get("frame_leg") != job.launch_id:
                p.update(
                    frame_leg=job.launch_id, frames_first=[frames, now], frames_last=[frames, now]
                )
            elif frames != p["frames_last"][0]:
                p["frames_last"] = [frames, now]
        steps = snapshot_steps(job.spec.output_dir)
        legs = run_legs(job.spec.output_dir)
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
    def _choose(job: Job, free: list[Slot]) -> Slot | None:
        """The free slot of most ``speed`` x GPUs for an all-GPU job, else the fastest one-GPU slot.

        A one-GPU job takes a wider slot only when no one-GPU slot is free.
        """

        def rank(s: Slot) -> tuple[Any, ...]:
            if job.spec.gpus == ALL_GPUS:
                return (s.id in job.avoid, -s.speed * len(s.gpus), s.id)
            return (s.id in job.avoid, len(s.gpus) != 1, -s.speed, len(s.gpus), s.id)

        return min(free, key=rank, default=None)

    def _gpu_memory(self, host: str) -> dict[int, int] | None:
        """Memory in use per GPU index on a host (MiB), read once per poll; None on failure."""
        if host not in self._gpu_mem:
            try:
                res = self.executor.run(
                    host, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
                )
                rows = (map(float, line.split(",")) for line in res.stdout.strip().splitlines())
                mem = {int(idx): int(used) for idx, used in rows}
                self._gpu_mem[host] = mem if res.returncode == 0 and mem else None
            except (*CALL_ERRORS, ValueError):
                self._gpu_mem[host] = None
        return self._gpu_mem[host]

    def _occupied(self, slot: Slot) -> bool:
        """True if another process holds more than ``busy_mib`` on one of the slot's GPUs."""
        if slot.busy_mib <= 0:
            return False
        mem = self._gpu_memory(slot.host)
        hot = None if mem is None else [g for g in slot.gpus if mem.get(g, 0) > slot.busy_mib]
        if mem is None or hot:
            if slot.id not in self._busy:
                why = (
                    "GPU memory unreadable"
                    if mem is None
                    else f"GPU {hot} in use by another process"
                )
                self.log(f"slot {slot.id}: {why}; nothing placed there until it is free")
                self._busy.add(slot.id)
            return True
        if slot.id in self._busy:
            self.log(f"slot {slot.id}: free again")
            self._busy.discard(slot.id)
        return False

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
        if pending:
            free = [s for s in free if not self._occupied(s)]
        for job in pending:
            slot = self._choose(job, free)
            if slot is None:
                continue
            free.remove(slot)
            self._launch(job, slot, now)

    def gyaradax_args(self, job: Job, slot: Slot | None = None) -> list[str]:
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
        elif latest_snapshot(run_dir) and os.path.isfile(os.path.join(run_dir, "config.yaml")):
            args = ["run", run_dir] + strip_flags(spec.args, FRESH_FLAGS)
        else:
            args = first + ([] if "--overwrite" in first else ["--overwrite"])
        if spec.gpus == ALL_GPUS and slot is not None and not _has_flag(args, MESH_FLAGS):
            # every leg sizes its mesh to its slot, over a layout saved by an earlier leg
            args += ["--n-gpus", str(len(slot.gpus))]
        return args

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
        argv = [a.replace("{python}", slot.python) for a in self.program]
        argv += self.gyaradax_args(job, slot) + self._backend_args(job, slot, now)
        launch_id = uuid.uuid4().hex[:16]
        command = self.launch_command(job, slot, argv, launch_id)
        try:
            self.executor.launch(slot.host, command)
        except LaunchError as exc:
            self.down_until[slot.id] = now + self.slot_backoff
            self.log(
                f"{spec.name}: launch on {slot.id} failed ({exc}); slot paused {self.slot_backoff:.0f} s"
            )
            return False
        self._started(job, slot, argv, launch_id, now)
        self.log(f"{spec.name}: launched on {slot.id}: {shlex.join(argv)}")
        return True

    def run(self, poll: float, once: bool = False, until_idle: bool = False) -> None:
        """Poll until a signal (or once); ``until_idle`` also stops once no job is pending or active."""
        stop: list[int] = []
        previous = {
            sig: signal.signal(sig, lambda signum, frame: stop.append(signum))
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            while True:
                self.poll()
                states = [j.state for j in self.jobs.values()]
                if once or stop:
                    break
                if until_idle and states and not set(states) & {"pending", *ACTIVE}:
                    self.log("every job has finished")
                    break
                end = time.monotonic() + (min(poll, 2.0) if "stopping" in states else poll)
                while not stop and time.monotonic() < end:
                    time.sleep(min(0.5, max(end - time.monotonic(), 0.0)))
                if stop:
                    break
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        self.save()
        self.log("scheduler stopped; running jobs keep going and are reattached on restart")


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
    """Time left for a running leg, from its dataset frames (time mode) or its snapshot steps."""
    p = progress
    if p.get("frames_total"):
        first, last, total = p.get("frames_first"), p.get("frames_last"), p["frames_total"]
    else:
        first, last, total = p.get("first"), p.get("last"), p.get("target")
    if not total or not first or not last or last[0] <= first[0] or last[1] <= first[1]:
        return None
    rate = (last[0] - first[0]) / (last[1] - first[1])
    return max(0.0, (total - last[0]) / rate - (now - last[1]))


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
    counts = Counter(j["state"] for j in jobs)
    lines.append("jobs: " + ", ".join(f"{counts[s]} {s}" for s in STATE_ORDER if s in counts))
    rows = [("JOB", "STATE", "SLOT", "GPUS", "COST", "FAIL", "STEP", "ETA", "OUTCOME", "NOTE")]
    rank = {s: i for i, s in enumerate(STATE_ORDER)}
    for j in sorted(jobs, key=lambda j: (rank.get(j["state"], 99), -j["spec"]["cost"], j["order"])):
        p = j.get("progress") or {}
        step = f"{p['step']}/{p.get('target', '?')}" if "step" in p else "-"
        if p.get("frames_total"):
            step = f"frame {p.get('frames', 0)}/{p['frames_total']}"
        rows.append((
            j["spec"]["name"],
            j["state"],
            j.get("slot") or "-",
            _csv(j.get("gpus")) or str(j["spec"]["gpus"] or "all"),
            f"{j['spec']['cost']:g}",
            str(j.get("failures", 0)),
            step,
            _duration(eta_seconds(p, now) if j["state"] in ACTIVE else None),
            (p.get("outcome") or "-")[:60],
            (j.get("note") or "")[:80],
        ))  # fmt: skip
    # the outcome column only once a run tracks its outcome
    if not any((j.get("progress") or {}).get("outcome") for j in jobs):
        rows = [r[:8] + r[9:] for r in rows]
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
        status = ("busy" if sid in by_slot else "free") + (", draining" if s.get("drain") else "")
        if down > now:
            status += f", paused {_duration(down - now)}"
        srows.append(
            (sid, s["host"], _csv(s["gpus"]), f"{s['speed']:g}", status, by_slot.get(sid, "-"))
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
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    put("pid", os.getpid())
    put("host", host or socket.gethostname())
    put("slot", slot)
    put("gpus", gpus)
    put("heartbeat", f"{time.time():.3f}")
    put("launch", launch_id)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
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


RUN_OPTIONS = (
    ("--poll", float, 30.0, "seconds between polls (default 30)"),
    ("--grace", float, 120.0, "seconds from SIGTERM to SIGKILL (default 120)"),
    ("--heartbeat", float, 30.0, "wrapper heartbeat period (default 30 s)"),
    ("--heartbeat-timeout", float, 300.0,
     "heartbeat age after which the process is checked over ssh (default 300 s)"),
    ("--start-timeout", float, 300.0, "seconds a launched wrapper has to report in (default 300)"),
    ("--max-retries", int, 3, "retries after failures (default 3)"),
    ("--slot-backoff", float, 300.0, "seconds a slot is skipped after a failed launch (default 300)"),
    ("--ssh-timeout", float, 30.0, "timeout of each ssh call (default 30 s)"),
)  # fmt: skip
REQUESTS = {
    "cancel": "stop jobs (SIGTERM, grace, SIGKILL) and hold them",
    "resume": "put cancelled, failed or diverged jobs back in the queue",
    "stable": "stop jobs (SIGTERM, grace, SIGKILL) and record them as decayed",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scheduler.py",
        description="Dispatch gyaradax runs over an ssh GPU mesh (see gyaradax datagen).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the scheduler loop")
    run_p.add_argument("--pool", required=True, help="pool YAML (re-read every poll)")
    run_p.add_argument("--jobs", required=True, help="job YAML (re-read every poll)")
    run_p.add_argument("--state", required=True, help="scheduler state JSON")
    run_p.add_argument("--once", action="store_true", help="poll once and exit")
    for flag, kind, default, text in RUN_OPTIONS:
        run_p.add_argument(flag, type=kind, default=default, help=text)

    status_p = sub.add_parser("status", help="jobs, slots and ETAs from the state file")
    status_p.add_argument("--state", required=True)

    for action, text in REQUESTS.items():
        request_p = sub.add_parser(action, help=text)
        request_p.add_argument("names", nargs="+")
        request_p.add_argument("--state", required=True)
        if action == "cancel":
            request_p.add_argument(
                "--requeue", action="store_true", help="put the jobs back in the queue instead"
            )

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
    keys = "heartbeat heartbeat_timeout grace start_timeout max_retries slot_backoff"
    options = {k: getattr(args, k) for k in keys.split()}
    with lock_state(args.state, f"error: another scheduler holds {args.state}.lock"):
        executor = SSHExecutor(timeout=args.ssh_timeout)
        Scheduler(args.pool, args.jobs, args.state, executor, **options).run(
            args.poll, once=args.once
        )
    return 0


def read_status(state_path: str, missing: str) -> tuple[dict[str, Any], str]:
    """The state document and its status tables, or exit with ``missing``."""
    if not os.path.isfile(state_path):
        raise SystemExit(missing)
    with open(state_path) as fh:
        doc = json.load(fh)
    waiting = len(request_files(os.path.abspath(state_path)))
    return doc, format_status(doc, time.time(), waiting)


def _cmd_request(args: argparse.Namespace) -> int:
    action = "requeue" if getattr(args, "requeue", False) else args.command
    for name in args.names:
        submit_request(os.path.abspath(args.state), action, name)
        print(f"{action} {name}: queued for the scheduler's next poll")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "status":
        print(read_status(args.state, f"error: {args.state} not found")[1])
        return 0
    if args.command in REQUESTS:
        return _cmd_request(args)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        raise SystemExit("error: wrap needs a command after --")
    return wrap(args.dir, args.launch_id, args.host, args.slot, args.heartbeat, cmd)


if __name__ == "__main__":
    raise SystemExit(main())
