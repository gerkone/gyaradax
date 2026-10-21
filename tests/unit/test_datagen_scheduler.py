"""gyaradax/datagen/scheduler.py driven with a local executor and a fake gyaradax."""

import contextlib
import glob
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
import yaml

from gyaradax.datagen import scheduler as gs

SCRIPT = gs.__file__

# mimics the run-directory behaviour of `gyaradax run`
FAKE_GYARADAX = r"""
import glob, json, os, re, signal, sys, time, zipfile
import yaml

def main(argv):
    assert argv[0] == "run", argv
    src, rest = argv[1], argv[2:]
    opts, flags, i = {}, set(), 0
    while i < len(rest):
        if rest[i] in ("--output-dir", "--n-steps", "--n-gpus"):
            opts[rest[i]] = rest[i + 1]
            i += 2
        else:
            flags.add(rest[i])
            i += 1
    resuming_dir = os.path.isdir(src)
    run_dir = opts.get("--output-dir", src if resuming_dir else None)
    with open(os.path.join(src, "config.yaml") if resuming_dir else src) as fh:
        cfg = yaml.safe_load(fh)
    os.makedirs(run_dir, exist_ok=True)
    snaps = sorted(int(re.search(r"step_(\d+)", p).group(1)) for p in glob.glob(os.path.join(run_dir, "step_*.npz")))
    resume = resuming_dir and bool(snaps) and "--from-scratch" not in flags
    info = os.path.join(run_dir, "run_info.jsonl")
    if not resume and os.path.exists(info) and "--overwrite" not in flags:
        print("error: holds the outputs of another run", file=sys.stderr)
        return 1
    start = snaps[-1] if resume else 0
    if not resume:
        for p in glob.glob(os.path.join(run_dir, "step_*.npz")):
            os.remove(p)
    if "--n-steps" in opts:
        target = start + int(opts["--n-steps"])
    else:
        target = cfg["steps"] if resume else start + cfg["steps"]
    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(dict(cfg, steps=target), fh)
    with open(info, "a") as fh:
        fh.write(json.dumps({
            "argv": argv, "start_step": start, "target_step": target,
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "prealloc": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        }) + "\n")
    # as gyaradax run --linear-gate / --stop-on-decay / --stop-on-banding / --track-outcome
    for name in ("linear_gate", "outcome"):
        if cfg.get(name):
            with open(os.path.join(run_dir, name + ".json"), "w") as fh:
                json.dump(cfg[name], fh)
    if cfg.get("outcome_exit"):
        return cfg["outcome_exit"]
    stop = []
    if cfg.get("ignore_term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum))
    step, prev = start, (os.path.join(run_dir, f"step_{start:06d}.npz") if resume else None)
    while step < target:
        time.sleep(cfg.get("step_time", 0.05))
        step += 1
        marker = os.path.join(run_dir, "failed_once")
        if step == cfg.get("fail_once_at") and not os.path.exists(marker):
            open(marker, "w").close()
            return 1
        if step == cfg.get("fail_at"):
            return 1
        if step == cfg.get("diverge_at"):
            open(os.path.join(run_dir, "DIVERGED"), "w").close()
            return 2
        path = os.path.join(run_dir, f"step_{step:06d}.npz")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("df.npy", b"df")
            zf.writestr("step.npy", str(step))
        if prev and prev != path and os.path.exists(prev):
            os.remove(prev)
        prev = path
        # as gyaradax run: SIGTERM stops after the step's snapshot with exit 75
        if stop and step < target:
            return 75
    return 0

sys.exit(main(sys.argv[1:]))
"""

FAKE_KEYS = (
    "steps", "step_time", "fail_once_at", "fail_at", "diverge_at", "ignore_term", "linear_gate",
    "outcome", "outcome_exit",
)


def slot(host, *gpus, speed=1.0, **extra):
    return {"host": host, "gpus": list(gpus), "speed": speed, **extra}


class Campaign:
    def __init__(self, root):
        self.root = root
        self.pool_path = str(root / "pool.yaml")
        self.jobs_path = str(root / "jobs.yaml")
        self.state_path = str(root / "state" / "state.json")
        self.fake = root / "fake_gyaradax.py"
        self.fake.write_text(FAKE_GYARADAX)
        self.entries = []
        self.logs = []

    def pool(self, *slots, hosts=None, env=None):
        defaults = {"python": sys.executable, **({"env": env} if env else {})}
        doc = {"defaults": defaults, "slots": list(slots), **({"hosts": hosts} if hosts else {})}
        Path(self.pool_path).write_text(yaml.safe_dump(doc))

    def add_job(self, name, steps=5, step_time=0.05, **job):
        cfg = {"steps": steps, "step_time": step_time, **{k: job.pop(k) for k in FAKE_KEYS if k in job}}
        (self.root / "configs").mkdir(exist_ok=True)
        (self.root / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
        entry = {"name": name, "config": f"configs/{name}.yaml", "output_dir": f"runs/{name}"}
        self.entries.append({**entry, **job})
        Path(self.jobs_path).write_text(yaml.safe_dump({"jobs": self.entries}))

    def scheduler(self, executor=None, **kw):
        opts = dict(heartbeat=0.2, heartbeat_timeout=2.0, grace=1.0, start_timeout=15.0, max_retries=2,
                    slot_backoff=60.0, program=[sys.executable, str(self.fake)], log=self.logs.append)
        executor = executor or gs.LocalExecutor()
        return gs.Scheduler(self.pool_path, self.jobs_path, self.state_path, executor, **{**opts, **kw})

    def run_dir(self, name):
        return str(self.root / "runs" / name)

    def legs(self, name):
        return gs.run_legs(self.run_dir(name))

    def drive(self, sched, until, timeout=30.0):
        drive(sched, until, timeout, self.logs)

    def cleanup(self):
        for pid_file in glob.glob(str(self.root / "runs" / "*" / ".sched" / "pid")):
            with contextlib.suppress(OSError):
                os.killpg(int(Path(pid_file).read_text().split()[0]), signal.SIGKILL)


@pytest.fixture
def camp(tmp_path):
    c = Campaign(tmp_path)
    yield c
    c.cleanup()


def drive(sched, until, timeout=30.0, logs=()):
    deadline = time.time() + timeout
    while time.time() < deadline:
        sched.poll()
        if until():
            return
        time.sleep(0.1)
    states = {n: (j.state, j.note) for n, j in sched.jobs.items()}
    raise AssertionError(f"condition not reached; jobs {states}; log {list(logs)[-10:]}")


def done(sched, *names):
    return lambda: all(sched.jobs[n].state == "done" for n in (names or sched.jobs))


def snapshot_at(run_dir, step):
    return lambda: max(gs.snapshot_steps(run_dir) or [0]) >= step


def write_run(run_dir, config=None, snapshot=None):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_info.jsonl"), "w") as fh:
        fh.write(json.dumps({"start_step": 0, "target_step": 30}) + "\n")
    if config is not None:
        with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
            fh.write(config)
    if snapshot is not None:
        with zipfile.ZipFile(os.path.join(run_dir, f"step_{snapshot:06d}.npz"), "w") as zf:
            zf.writestr("df.npy", b"df")


def sql(db, query):
    with contextlib.closing(sqlite3.connect(db)) as con:
        return con.execute(query).fetchall()


class Record(gs.LocalExecutor):
    """Records launch commands instead of running them."""

    def __init__(self):
        super().__init__()
        self.launched = []

    def launch(self, host, command):
        self.launched.append((host, command))


def test_lpt_puts_longest_jobs_on_fastest_slots(camp):
    camp.pool(slot("slow", 0, speed=1), slot("fast", 1, speed=3), slot("mid", 2, speed=2))
    for name, cost in (("short", 10), ("long", 30), ("medium", 20), ("tiny", 1)):
        camp.add_job(name, cost=cost, steps=3)
    sched = camp.scheduler()
    sched.poll()
    placed = {n: sched.jobs[n].slot for n in ("long", "medium", "short")}
    assert placed == {"long": "fast:1", "medium": "mid:2", "short": "slow:0"}
    assert sched.jobs["tiny"].state == "pending"
    camp.drive(sched, done(sched))
    leg = camp.legs("long")[0]
    assert leg["gpus"] == "1" and leg["prealloc"] == "false"
    assert leg["argv"][leg["argv"].index("--backend") + 1] == "auto"
    assert len(camp.legs("tiny")) == 1


def test_completion_frees_the_slot_and_new_jobs_are_picked_up(camp):
    camp.pool(slot("a", 0))
    camp.add_job("one", steps=3)
    sched = camp.scheduler()
    camp.drive(sched, done(sched, "one"))
    job = sched.jobs["one"]
    assert job.slot is None and job.history[-1]["exit"] == 0 and job.failures == 0
    with open(camp.state_path) as fh:
        doc = json.load(fh)
    assert doc["jobs"]["one"]["state"] == "done" and "one" in gs.format_status(doc, time.time())

    camp.add_job("two", steps=2)
    camp.drive(sched, done(sched, "two"))
    assert sched.jobs["two"].history[-1]["slot"] == "a:0"
    assert len(camp.legs("one")) == 1


def test_slot_added_at_runtime_gets_a_job(camp):
    camp.pool(slot("a", 0))
    camp.add_job("first", steps=60, cost=2)
    camp.add_job("second", steps=3, cost=1)
    sched = camp.scheduler()
    for _ in range(2):
        sched.poll()
        assert sched.jobs["first"].slot == "a:0" and sched.jobs["second"].state == "pending"

    camp.pool(slot("a", 0), slot("b", 5))
    sched.poll()
    assert sched.jobs["second"].slot == "b:5"
    camp.drive(sched, done(sched, "second"))
    assert camp.legs("second")[0]["gpus"] == "5"
    first = sched.jobs["first"]
    assert first.state in ("starting", "running") and first.attempts == 1 and first.slot == "a:0"


@pytest.mark.parametrize(
    "hosts, note",
    [(None, "slot removed"), ({"a": "off"}, "slot removed"), ({"a": "kill"}, "host a killed")],
    ids=["removed", "host-off", "host-kill"],
)
def test_lost_slot_moves_the_job_elsewhere_and_it_resumes_from_its_snapshot(camp, hosts, note):
    camp.pool(slot("a", 0, speed=2), slot("b", 1))
    camp.add_job("long", steps=1000, step_time=0.04, args=["--n-steps", "40"])
    sched = camp.scheduler(grace=60.0)
    sched.poll()
    job = sched.jobs["long"]
    assert job.slot == "a:0"
    run_dir = camp.run_dir("long")
    camp.drive(sched, snapshot_at(run_dir, 5))

    camp.pool(*([slot("a", 0, speed=2)] if hosts else []), slot("b", 1), hosts=hosts)
    started = time.time()
    sched.poll()
    assert job.state in ("stopping", "pending", "starting")
    camp.drive(sched, lambda: job.state in ("starting", "running") and job.slot == "b:1")
    assert time.time() - started < 10.0
    camp.drive(sched, done(sched, "long"))

    first, second = camp.legs("long")
    assert first["gpus"] == "0" and second["gpus"] == "1"
    assert second["argv"][:2] == ["run", run_dir] and "--n-steps" not in second["argv"]
    assert second["start_step"] >= 5 and second["target_step"] == first["target_step"] == 40
    assert max(gs.snapshot_steps(run_dir)) == 40
    stop = job.history[0]
    assert stop["note"].startswith(note) and stop["result"] == "pending" and job.failures == 0
    # removal and off stop with SIGTERM and a snapshot (exit 75), kill with SIGKILL
    assert (stop["exit"] == 75) == (note == "slot removed")


def test_cancel_escalates_to_sigkill_and_resume_requeues(camp):
    camp.pool(slot("a", 0))
    camp.add_job("stubborn", steps=50, step_time=0.05, ignore_term=True)
    sched = camp.scheduler()
    camp.drive(sched, snapshot_at(camp.run_dir("stubborn"), 3))
    pid = sched.jobs["stubborn"].pid

    assert gs.main(["cancel", "stubborn", "--state", camp.state_path]) == 0
    camp.drive(sched, lambda: sched.jobs["stubborn"].state == "cancelled")
    entry = sched.jobs["stubborn"].history[-1]
    assert entry["exit"] is None and "killed" in entry["note"]
    with contextlib.suppress(OSError):
        assert not open(f"/proc/{pid}/cmdline", "rb").read()
    for _ in range(3):
        sched.poll()
    assert sched.jobs["stubborn"].state == "cancelled"

    assert gs.main(["resume", "stubborn", "--state", camp.state_path]) == 0
    camp.drive(sched, done(sched, "stubborn"))
    legs = camp.legs("stubborn")
    assert len(legs) == 2 and legs[1]["start_step"] >= 3 and legs[1]["target_step"] == 50


def test_resumed_job_launches_with_the_args_edited_while_it_was_held(camp):
    camp.pool(slot("a", 0))
    camp.add_job("held", steps=40, args=["--tag-old"])
    sched = camp.scheduler()
    camp.drive(sched, snapshot_at(camp.run_dir("held"), 3))
    assert gs.main(["cancel", "held", "--state", camp.state_path]) == 0
    camp.drive(sched, lambda: sched.jobs["held"].state == "cancelled")
    camp.entries[0]["args"] = ["--tag-new"]
    Path(camp.jobs_path).write_text(yaml.safe_dump({"jobs": camp.entries}))
    assert gs.main(["resume", "held", "--state", camp.state_path]) == 0
    camp.drive(sched, lambda: len(camp.legs("held")) == 2)
    argv = camp.legs("held")[1]["argv"]
    assert "--tag-new" in argv and "--tag-old" not in argv


def test_failure_is_retried_from_the_run_directory_on_another_slot(camp):
    camp.pool(slot("a", 0, speed=2), slot("b", 1))
    camp.add_job("flaky", steps=10, fail_once_at=5)
    sched = camp.scheduler()
    camp.drive(sched, done(sched, "flaky"))
    job = sched.jobs["flaky"]
    assert job.failures == 1 and job.attempts == 2
    assert job.history[0]["exit"] == 1 and job.history[0]["result"] == "pending"
    first, second = camp.legs("flaky")
    assert second["argv"][:2] == ["run", camp.run_dir("flaky")]
    assert second["start_step"] == 4 and second["target_step"] == 10
    assert first["gpus"] == "0" and second["gpus"] == "1"


def test_lost_process_counts_as_failure_until_retries_run_out(camp):
    camp.pool(slot("a", 0))
    camp.add_job("doomed", steps=10_000)
    sched = camp.scheduler(max_retries=1, heartbeat_timeout=1.0)
    jobs = sched.jobs
    for attempt in (1, 2):
        camp.drive(
            sched, lambda: jobs["doomed"].state == "running" and jobs["doomed"].attempts == attempt
        )
        os.killpg(jobs["doomed"].pid, signal.SIGKILL)
        camp.drive(sched, lambda: jobs["doomed"].state != "running")
    assert jobs["doomed"].state == "failed" and jobs["doomed"].failures == 2
    assert all(h["note"].startswith("lost") for h in jobs["doomed"].history)


def test_divergence_is_not_retried(camp):
    camp.pool(slot("a", 0))
    camp.add_job("x", steps=10, diverge_at=3)
    sched = camp.scheduler()
    camp.drive(sched, lambda: sched.jobs["x"].state == "diverged")
    assert sched.jobs["x"].attempts == 1 and sched.jobs["x"].failures == 0
    assert gs.main(["resume", "x", "--state", camp.state_path]) == 0
    sched.poll()
    assert sched.jobs["x"].state in ("starting", "running", "diverged")


GATE = {
    "decision": "stable", "settled": True, "gamma_max": -0.03, "peak_ky": 0.35, "omega_peak": 0.1,
    "gamma_max_lowky": -0.04, "peak_ky_lowky": 0.2, "low_ky": 1.0,
    "t": 41.0, "ky": [0.0, 0.35], "gamma": [0.0, -0.03], "gamma_std": [0.0, 0.001],
    "omega": [0.0, 0.1], "omega_std": [0.0, 0.01],
}  # fmt: skip
SATURATED = {
    "class": "saturated", "label": "saturated t 150.0 Qi 3.2±0.4", "t": 150.0, "frames": 107,
    "stopped": None, "banded": False, "banded_t": None, "nyq_zonal": 0.01, "nyq_nonzonal": 0.002,
    "nyq_zonal_max": 0.05, "q_peak": 40.0, "q_last": 3.5,
    "average": {"window": [105.0, 150.0], "frames": 33, "mean": [[0.5, 3.2, 0.0], [0.4, 1.1, 0.0]],
                "err": [[0.1, 0.4, 0.0], [0.1, 0.2, 0.0]], "q_drift": 0.05, "q_drift_sigma": 0.7},
}  # fmt: skip


def test_outcome_exits_end_jobs_without_retries_and_fill_the_outcome_table(camp):
    camp.pool(slot("a", 0), slot("b", 1))
    decayed = {**SATURATED, "class": "decayed", "label": "decayed", "stopped": "decayed"}
    banded = {**SATURATED, "class": "banded", "label": "banded", "stopped": "banded", "banded": True}
    camp.add_job("stable", steps=3, linear_gate=GATE, outcome_exit=4, cost=3)
    camp.add_job("decay", steps=3, outcome=decayed, outcome_exit=5, cost=2)
    camp.add_job("band", steps=3, outcome=banded, outcome_exit=6, cost=1)
    camp.add_job("turb", steps=3, outcome={**SATURATED, "banded": True, "banded_t": 90.0})
    for name in ("stable", "decay", "band", "turb"):
        cfg_path = camp.root / "configs" / f"{name}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg.update(
            physics={"rlt": [5.0, 4.0], "rln": [2.0, 2.0], "tmp": [1.0, 1.2], "signz": [1, -1]},
            geometry={"q": 1.4, "shat": 0.8, "eps": 0.12},
            grid={"nvpar": 32, "nmu": 12, "ns": 18, "nkx": 192, "nky": 32},
        )
        cfg_path.write_text(yaml.safe_dump(cfg))
    sched = camp.scheduler()
    finished = ("stable", "decayed", "banded", "done")
    camp.drive(sched, lambda: all(j.state in finished for j in sched.jobs.values()))
    states = {n: j.state for n, j in sched.jobs.items()}
    assert states == {"stable": "stable", "decay": "decayed", "band": "banded", "turb": "done"}
    assert all(j.attempts == 1 and j.failures == 0 for j in sched.jobs.values())
    assert "not retried" in sched.jobs["stable"].note
    assert gs.main(["resume", "decay", "--state", camp.state_path]) == 0
    sched.poll()
    assert sched.jobs["decay"].state in ("pending", "starting", "running")
    assert gs.main(["stable", "decay", "--state", camp.state_path]) == 0
    active = ("pending", "starting", "running", "stopping")
    camp.drive(sched, lambda: sched.jobs["decay"].state not in active)
    assert sched.jobs["decay"].state == "decayed"
    assert "stopped by hand as stable" in sched.jobs["decay"].note

    rows = {r[0]: r[1:] for r in sql(sched.db_path, "SELECT name, state, outcome, banded FROM runs")}
    assert rows["turb"] == ("done", "saturated", 1) and rows["stable"] == ("stable", None, None)
    stable = sql(
        sched.db_path,
        "SELECT gate, settled, gamma_max, peak_ky, rlt_i, rlt_e, temp_ratio, q, shat, eps, "
        "json_extract(grid, '$.nkx'), json_extract(gamma, '$[1]'), gamma_max_lowky, "
        "peak_ky_lowky, low_ky FROM outcomes WHERE name = 'stable'",
    )
    assert stable == [
        ("stable", 1, -0.03, 0.35, 5.0, 4.0, 1.2, 1.4, 0.8, 0.12, 192, -0.03, -0.04, 0.2, 1.0)
    ]
    turb = sql(
        sched.db_path,
        "SELECT class, qi, qi_err, qe, qe_err, gi, avg_t0, avg_t1, banded, banded_t, "
        "nyq_zonal, q_drift, gpu_hours > 0 FROM outcomes WHERE name = 'turb'",
    )
    assert turb == [("saturated", 3.2, 0.4, 1.1, 0.2, 0.5, 105.0, 150.0, 1, 90.0, 0.01, 0.05, 1)]
    assert sql(sched.db_path, "SELECT class FROM outcomes WHERE name = 'band'") == [("banded",)]
    with open(camp.state_path) as fh:
        status = gs.format_status(json.load(fh), time.time())
    assert "OUTCOME" in status and "saturated t 150.0 Qi 3.2±0.4" in status
    assert "jobs: 1 banded, 1 decayed, 1 stable, 1 done" in status


@pytest.mark.parametrize("keep_state", [True, False], ids=["state", "sched-files"])
def test_restart_reattaches_to_the_running_job(camp, keep_state):
    camp.pool(slot("a", 0))
    camp.add_job("long", steps=80, step_time=0.05, cost=2)
    first = camp.scheduler()
    camp.drive(first, lambda: first.jobs["long"].state == "running")
    launch, pid = first.jobs["long"].launch_id, first.jobs["long"].pid
    if not keep_state:
        os.remove(camp.state_path)

    camp.add_job("other", steps=2, cost=1)
    second = camp.scheduler()
    second.poll()
    job = second.jobs["long"]
    assert (job.state, job.launch_id, job.pid, job.slot) == ("running", launch, pid, "a:0")
    assert second.jobs["other"].state == "pending"
    camp.drive(second, done(second))
    assert len(camp.legs("long")) == 1


def test_drained_slot_finishes_its_job_then_gets_nothing(camp):
    camp.pool(slot("a", 0))
    camp.add_job("first", steps=20, cost=2)
    sched = camp.scheduler()
    sched.poll()
    camp.pool(slot("a", 0, drain=True))
    camp.add_job("second", steps=2, cost=1)
    camp.drive(sched, done(sched, "first"))
    for _ in range(3):
        sched.poll()
        time.sleep(0.05)
    assert sched.jobs["second"].state == "pending" and len(camp.legs("first")) == 1

    camp.pool(slot("a", 0))
    camp.drive(sched, done(sched, "second"))


def test_busy_gpus_get_no_jobs_until_they_are_free(camp):
    class Smi(gs.LocalExecutor):
        mem = {0: 5000, 1: 0}

        def run(self, host, command):
            if command.startswith("nvidia-smi"):
                out = "\n".join(f"{i}, {m}" for i, m in self.mem.items())
                return subprocess.CompletedProcess(command, 0, out, "")
            return super().run(host, command)

    smi = Smi()
    camp.pool(slot("a", 0, speed=2, busy_mib=1000), slot("a", 1, busy_mib=1000))
    camp.add_job("first", steps=40, cost=2)
    sched = camp.scheduler(executor=smi)
    sched.poll()
    assert sched.jobs["first"].slot == "a:1"
    camp.add_job("second", steps=2, cost=1)
    for _ in range(3):
        sched.poll()
        time.sleep(0.05)
    assert sched.jobs["second"].state == "pending"
    assert sum("in use by another process" in line for line in camp.logs) == 1

    smi.mem[0] = 200
    camp.drive(sched, done(sched))
    assert camp.legs("second")[0]["gpus"] == "0"
    assert any("a:0: free again" in line for line in camp.logs)


def test_unreadable_pool_keeps_the_previous_one(camp):
    camp.pool(slot("a", 0))
    camp.add_job("long", steps=60)
    sched = camp.scheduler()
    camp.drive(sched, lambda: sched.jobs["long"].state == "running")
    for text in ("slots: [unterminated", ""):
        with open(camp.pool_path, "w") as fh:
            fh.write(text)
        sched.poll()
        assert sched.jobs["long"].state == "running" and list(sched.pool) == ["a:0"]
    assert any("unreadable" in line for line in camp.logs)
    camp.drive(sched, done(sched))


def test_launch_command_runs_the_wrapper_on_the_slot_gpus(camp):
    record = Record()
    camp.pool(slot("a", 3), env={"HDISP": "0.01"})
    camp.add_job("x", steps=2, args=["--backend", "cuda"])
    sched = camp.scheduler(executor=record)
    sched.poll()
    ((host, line),) = record.launched
    assert host == "a" and sched.jobs["x"].state == "starting"
    for part in ("CUDA_VISIBLE_DEVICES=3", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "--backend cuda"):
        assert part in line
    assert " wrap " in line and "HDISP=0.01" in line


def test_pool_validation(tmp_path):
    path = tmp_path / "pool.yaml"
    slots = [
        {"host": "node-a", "gpus": [0, 1]},
        {"host": "node-a", "gpus": [1, 2]},
        {"host": "node-a", "gpus": [1]},
        {"host": "node-b", "gpus": [0], "speed": 2, "env": {"X": 1}},
        {"gpus": [0]},
        {"host": "node-b", "gpus": [1], "drain": True, "name": None, "env": {}},
        {"host": "node-b", "gpus": [2], "name": "s2"},
    ]
    defaults = {"env": {"HDISP": "0.01"}}
    groups = {"node-a": [[0, 1], [2, 3], [4, 5]]}
    path.write_text(yaml.safe_dump({"defaults": defaults, "slots": slots, "gpu_groups": groups}))
    slots, problems = gs.load_pool(str(path))
    assert list(slots) == ["node-a:0,1", "node-b:1"]
    assert slots["node-b:1"].drain and slots["node-a:0,1"].python == gs.DEFAULT_PYTHON
    assert slots["node-b:1"].env == slots["node-a:0,1"].env == {"HDISP": "0.01"}
    assert len(problems) == 5 and any("must be one of" in p for p in problems)
    assert sum("no longer supported" in p for p in problems) == 2
    path.write_text(yaml.safe_dump({"defaults": {"env": {"BAD-NAME": 1}}, "slots": [{"host": "a", "gpus": [0]}]}))
    assert "invalid environment variable names" in gs.load_pool(str(path))[1][0]
    path.write_text(yaml.safe_dump({"gpu_groups": {"node-a": "0,1"}, "slots": [{"host": "node-a", "gpus": [0, 1]}]}))
    slots, problems = gs.load_pool(str(path))
    assert list(slots) == ["node-a:0,1"] and problems == ["gpu_groups: node-a: expected lists of GPU indices"]
    for doc in ("", "slurm: {max_jobs: 2}\n"):
        path.write_text(doc)
        with pytest.raises(ValueError):
            gs.load_pool(str(path))

    slots = [
        {"host": "node-b", "gpus": [0]},
        {"host": "node-b.local", "gpus": [1]},
        {"host": "node-c", "gpus": [0]},
        {"host": "node-a", "gpus": [0], "busy_mib": 1500},
        {"host": "node-d", "gpus": [0], "busy_mib": -1},
    ]
    hosts = {"node-b": "drain", "node-c": "off", "node-a": "maybe"}
    path.write_text(yaml.safe_dump({"hosts": hosts, "slots": slots}))
    slots, problems = gs.load_pool(str(path))
    assert list(slots) == ["node-b:0", "node-b.local:1", "node-a:0"]
    assert slots["node-b:0"].drain and slots["node-b.local:1"].drain
    assert not slots["node-a:0"].drain and slots["node-a:0"].busy_mib == 1500
    assert len(problems) == 2
    assert any("'maybe' is not one of" in p for p in problems)
    assert any("busy_mib" in p for p in problems)


def test_job_validation(tmp_path):
    path = tmp_path / "jobs.yaml"
    jobs = [
        {"name": "a", "config": "configs/a.yaml", "output_dir": "/abs/a", "args": "--n-steps 20"},
        {"name": "b", "run_dir": "runs/b", "cost": 5, "gpus": "all", "env": {}, "workdir": None},
        {"name": "c", "config": "x.yaml"},
        {"name": "d", "config": "x.yaml", "output_dir": "o", "args": ["--device", "1"]},
        {"name": "a", "run_dir": "r"},
        {"name": "e", "run_dir": "r", "gpus": 2},
        {"name": "f", "run_dir": "r", "env": {"X": "1"}},
        {"name": "g", "run_dir": "r", "workdir": "elsewhere"},
    ]
    path.write_text(yaml.safe_dump({"workdir": "repo", "defaults": None, "jobs": jobs}))
    specs, problems = gs.load_jobs(str(path))
    assert [s.name for s in specs] == ["a", "b"] and len(problems) == 6
    assert sum("no longer supported" in p for p in problems) == 2
    a, b = specs
    repo = str(tmp_path / "repo")
    assert a.config == os.path.join(repo, "configs", "a.yaml") and a.output_dir == "/abs/a"
    assert a.args == ["--n-steps", "20"] and a.workdir == repo and a.gpus == 1
    assert b.output_dir == b.run_dir == os.path.join(repo, "runs", "b") and b.gpus == gs.ALL_GPUS
    path.write_text(yaml.safe_dump({"defaults": {"args": ["--backend", "cuda"]}, "jobs": jobs}))
    with pytest.raises(ValueError, match="defaults no longer supported"):
        gs.load_jobs(str(path))


def test_next_leg_arguments(camp):
    camp.pool(slot("a", 0))
    camp.add_job("j", args=["--n-steps", "30", "--from-scratch", "--backend", "cuda"])
    sched = camp.scheduler()
    sched._refresh_jobs(time.time())
    job = sched.jobs["j"]
    run_dir = camp.run_dir("j")
    config = str(camp.root / "configs" / "j.yaml")
    fresh = ["run", config, "--output-dir", run_dir, "--n-steps", "30", "--from-scratch"]
    fresh += ["--backend", "cuda"]

    job.base_legs = 0
    assert sched.gyaradax_args(job) == fresh
    write_run(run_dir)
    assert sched.gyaradax_args(job) == fresh + ["--overwrite"]

    write_run(run_dir, config="steps: 30\n", snapshot=4)
    with open(os.path.join(run_dir, "step_000006.npz"), "wb") as fh:
        fh.write(b"PK\x03\x04truncated")
    assert sched.gyaradax_args(job) == ["run", run_dir, "--backend", "cuda"]
    assert sorted(gs.snapshot_steps(run_dir)) == [4]
    assert os.path.exists(os.path.join(run_dir, ".sched", "broken", "step_000006.npz"))

    job.base_legs = 1
    assert sched.gyaradax_args(job) == fresh

    args = ["--n-steps=5", "--mp", "--resume-from", "K03", "--overwrite"]
    assert gs.strip_flags(args, gs.FRESH_FLAGS) == ["--mp"]


def test_all_gpu_job_takes_the_whole_slot_and_sizes_its_mesh_per_leg(camp):
    record = Record()
    camp.pool(slot("a", 0, speed=2), slot("b", 1, 2, 3, 4), slot("c", 5, 6))
    camp.add_job("wide", gpus="all", args=["--backend", "cuda"])
    camp.add_job("narrow", gpus=1, cost=0.5)
    sched = camp.scheduler(executor=record)
    sched.poll()
    assert sched.jobs["wide"].slot == "b:1,2,3,4" and sched.jobs["narrow"].slot == "a:0"
    lines = {name: line for _, line in record.launched for name in sched.jobs if f"/runs/{name} " in line}
    assert "CUDA_VISIBLE_DEVICES=1,2,3,4" in lines["wide"] and "--n-gpus 4" in lines["wide"]
    assert "CUDA_VISIBLE_DEVICES=0" in lines["narrow"] and "--n-gpus" not in lines["narrow"]

    job = sched.jobs["wide"]
    job.base_legs = 0
    run_dir = camp.run_dir("wide")
    write_run(run_dir, config="sharding: {n_gpus_sp: 2, n_gpus_mu: 2}\n", snapshot=4)
    small = sched.pool["c:5,6"]
    assert sched.gyaradax_args(job, small) == ["run", run_dir, "--backend", "cuda", "--n-gpus", "2"]
    assert gs.job_gpus(job.spec, small) == [5, 6]
    # a one-GPU job on a wider slot takes its first GPU
    assert gs.job_gpus(sched.jobs["narrow"].spec, small) == [5]


def test_run_database_follows_the_queue(camp, tmp_path):
    dataset = tmp_path / "dataset"
    camp.pool(slot("a", 0))
    camp.add_job("first", steps=30, cost=2, args=["--dataset", str(dataset), "--dataset-name", "f"])
    camp.add_job("second", steps=2, cost=1)
    cfg_path = camp.root / "configs" / "first.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(solver={"t_end": 5.0, "dump_dt": 1.0}, geometry={"eps": 0.18, "q": 2.5})
    cfg_path.write_text(yaml.safe_dump(cfg))
    data = dataset / "f_ifft_realpotens" / "data"
    data.mkdir(parents=True)
    for k in range(3):
        (data / f"poten_{k:05d}.zstd16.bin").write_bytes(b"x")
    sched = camp.scheduler()
    db = sched.db_path

    camp.drive(sched, lambda: sched.jobs["first"].state == "running")
    assert dict(sql(db, "SELECT name, state FROM runs")) == {"first": "running", "second": "pending"}
    query = "SELECT frames, frames_total, t, json_extract(params, '$.geometry.eps') FROM runs"
    assert sql(db, query + " WHERE name = 'first'") == [(3, 5, 3.0, 0.18)]
    assert sql(db, "SELECT job FROM slots") == [("first",)]

    camp.drive(sched, done(sched))
    sched.save()
    assert dict(sql(db, "SELECT name, state FROM runs")) == {"first": "done", "second": "done"}
    leg = sql(db, "SELECT leg, slot, exit, result FROM legs WHERE name = 'first'")
    assert leg == [(1, "a:0", 0, "done")]
    assert sql(db, "SELECT value IS NOT NULL FROM meta WHERE key = 'updated'") == [(1,)]
    assert not glob.glob(db + ".*.tmp")


def test_truncated_diagnostics_fail_the_job_instead_of_burning_retries(camp):
    camp.pool(slot("a", 0))
    camp.add_job("j")
    os.makedirs(camp.run_dir("j"))
    with open(os.path.join(camp.run_dir("j"), "fluxes.npz"), "wb") as fh:
        fh.write(b"PK\x03\x04trunc")
    sched = camp.scheduler()
    sched.poll()
    assert sched.jobs["j"].state == "failed" and "fluxes.npz" in sched.jobs["j"].note


def test_ssh_executor_runs_strict_batch_ssh_under_a_timeout(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, fake_run.rc, "", "Connection refused")

    monkeypatch.setattr(gs.subprocess, "run", fake_run)
    ex = gs.SSHExecutor(timeout=7, local_hosts=["here"])
    fake_run.rc = 255
    with pytest.raises(gs.LaunchError, match="Connection refused"):
        ex.launch("node-d", "echo hi")
    argv, kw = calls[-1]
    assert argv[0] == "ssh" and "node-d" in argv
    assert "BatchMode=yes" in argv and "StrictHostKeyChecking=yes" in argv
    assert kw["timeout"] == 7 and kw["stdin"] == subprocess.DEVNULL
    assert argv[-1] == "bash -c 'echo hi'"
    for rc, alive in ((255, None), (1, False), (0, True)):
        fake_run.rc = rc
        assert ex.alive("node-d", 123, "tok") is alive
    assert ex.signal("node-d", 123, "KILL", group=True)
    assert calls[-1][0][-1] == "bash -c 'kill -KILL -- -123'"
    assert ex.argv("here", "true") == ["bash", "-c", "true"]
    assert gs.SSHExecutor().argv(socket.gethostname(), "true")[0] == "bash"


def test_scheduler_does_not_import_jax():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('gs', {SCRIPT!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['gs'] = m; spec.loader.exec_module(m)\n"
        "print(sorted(k for k in sys.modules if k.split('.')[0] in ('jax', 'jaxlib', 'numpy', 'gyaradax')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_until_idle_returns_once_every_job_has_finished(camp):
    camp.pool(slot("a", 0))
    camp.add_job("x", steps=2)
    sched = camp.scheduler()
    t0 = time.time()
    sched.run(poll=0.1, until_idle=True)
    assert sched.jobs["x"].state == "done" and time.time() - t0 < 30
    assert "every job has finished" in camp.logs
