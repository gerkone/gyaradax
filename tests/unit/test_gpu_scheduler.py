"""scripts/gpu_scheduler.py driven with a local executor and a fake gyaradax."""

import contextlib
import glob
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import zipfile

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "gpu_scheduler.py")


def _load():
    spec = importlib.util.spec_from_file_location("gpu_scheduler", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gs = _load()

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
    if cfg.get("ignore_term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
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
    return 0

sys.exit(main(sys.argv[1:]))
"""

FAKE_KEYS = ("steps", "step_time", "fail_once_at", "fail_at", "diverge_at", "ignore_term")


def slot(host, *gpus, speed=1.0, **extra):
    return {"host": host, "gpus": list(gpus), "speed": speed, **extra}


class Campaign:
    def __init__(self, root):
        self.root = root
        self.pool_path = str(root / "pool.yaml")
        self.jobs_path = str(root / "jobs.yaml")
        self.state_path = str(root / "state" / "state.json")
        self.fake = str(root / "fake_gyaradax.py")
        with open(self.fake, "w") as fh:
            fh.write(FAKE_GYARADAX)
        self.entries = []
        self.logs = []

    def pool(self, *slots):
        with open(self.pool_path, "w") as fh:
            yaml.safe_dump({"defaults": {"python": sys.executable}, "slots": list(slots)}, fh)

    def add_job(self, name, steps=5, step_time=0.05, **job):
        cfg = {
            "steps": steps,
            "step_time": step_time,
            **{k: job.pop(k) for k in FAKE_KEYS if k in job},
        }
        cfg_path = self.root / "configs" / f"{name}.yaml"
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg))
        self.entries.append(
            {"name": name, "config": f"configs/{name}.yaml", "output_dir": f"runs/{name}", **job}
        )
        with open(self.jobs_path, "w") as fh:
            yaml.safe_dump({"jobs": self.entries}, fh)

    def scheduler(self, **kw):
        opts = dict(
            heartbeat=0.2,
            heartbeat_timeout=2.0,
            grace=1.0,
            start_timeout=15.0,
            max_retries=2,
            slot_backoff=60.0,
            program=[sys.executable, self.fake],
            log=self.logs.append,
        )
        executor = kw.pop("executor", None) or gs.LocalExecutor()
        opts.update(kw)
        return gs.Scheduler(self.pool_path, self.jobs_path, self.state_path, executor, **opts)

    def run_dir(self, name):
        return str(self.root / "runs" / name)

    def legs(self, name):
        return gs.run_legs(self.run_dir(name))

    def drive(self, sched, until, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            sched.poll()
            if until():
                return
            time.sleep(0.1)
        states = {n: (j.state, j.note) for n, j in sched.jobs.items()}
        raise AssertionError(f"condition not reached; jobs {states}; log {self.logs[-10:]}")

    def cleanup(self):
        for pid_file in glob.glob(str(self.root / "runs" / "*" / ".sched" / "pid")):
            with open(pid_file) as fh:
                pid = int(fh.read().split()[0])
            with contextlib.suppress(OSError):
                os.killpg(pid, signal.SIGKILL)


@pytest.fixture
def camp(tmp_path):
    c = Campaign(tmp_path)
    yield c
    c.cleanup()


def _read_bytes(path):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def done(sched, *names):
    return lambda: all(sched.jobs[n].state == "done" for n in (names or sched.jobs))


def test_lpt_puts_longest_jobs_on_fastest_slots(camp):
    camp.pool(slot("slow", 0, speed=1), slot("fast", 1, speed=3), slot("mid", 2, speed=2))
    for name, cost in (("short", 10), ("long", 30), ("medium", 20), ("tiny", 1)):
        camp.add_job(name, cost=cost, steps=3)
    sched = camp.scheduler()
    sched.poll()
    assert {n: sched.jobs[n].slot for n in ("long", "medium", "short")} == {
        "long": "fast:1",
        "medium": "mid:2",
        "short": "slow:0",
    }
    assert sched.jobs["tiny"].state == "pending"
    camp.drive(sched, done(sched))
    assert camp.legs("long")[0]["gpus"] == "1"
    assert camp.legs("long")[0]["prealloc"] == "false"
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
    assert doc["jobs"]["one"]["state"] == "done"
    assert "one" in gs.format_status(doc, time.time())

    camp.add_job("two", steps=2)
    camp.drive(sched, done(sched, "two"))
    assert sched.jobs["two"].history[-1]["slot"] == "a:0"
    assert len(camp.legs("one")) == 1


def test_slot_added_at_runtime_gets_a_job(camp):
    camp.pool(slot("a", 0))
    camp.add_job("first", steps=60, cost=2)
    camp.add_job("second", steps=3, cost=1)
    sched = camp.scheduler()
    sched.poll()
    assert sched.jobs["first"].slot == "a:0"
    assert sched.jobs["second"].state == "pending"
    sched.poll()
    assert sched.jobs["second"].state == "pending"

    camp.pool(slot("a", 0), slot("b", 5))
    sched.poll()
    assert sched.jobs["second"].slot == "b:5"
    camp.drive(sched, done(sched, "second"))
    assert camp.legs("second")[0]["gpus"] == "5"
    assert sched.jobs["first"].state in ("starting", "running")
    assert sched.jobs["first"].attempts == 1 and sched.jobs["first"].slot == "a:0"


def test_removed_slot_preempts_and_the_job_resumes_elsewhere(camp):
    camp.pool(slot("a", 0, speed=2), slot("b", 1))
    camp.add_job("long", steps=1000, step_time=0.04, args=["--n-steps", "40"])
    sched = camp.scheduler()
    sched.poll()
    assert sched.jobs["long"].slot == "a:0"
    run_dir = camp.run_dir("long")
    camp.drive(sched, lambda: max(gs.snapshot_steps(run_dir) or [0]) >= 5)

    camp.pool(slot("b", 1))
    sched.poll()
    assert sched.jobs["long"].state in ("stopping", "pending", "starting")
    camp.drive(sched, done(sched, "long"))

    first, second = camp.legs("long")
    assert first["gpus"] == "0" and second["gpus"] == "1"
    assert second["argv"][:2] == ["run", run_dir]
    assert "--n-steps" not in second["argv"]
    assert second["start_step"] >= 5 and second["target_step"] == first["target_step"] == 40
    assert max(gs.snapshot_steps(run_dir)) == 40
    stop = sched.jobs["long"].history[0]
    assert (
        stop["exit"] == 143
        and stop["note"].startswith("slot removed")
        and stop["result"] == "pending"
    )
    assert sched.jobs["long"].failures == 0


def test_cancel_escalates_to_sigkill_and_resume_requeues(camp):
    camp.pool(slot("a", 0))
    camp.add_job("stubborn", steps=50, step_time=0.05, ignore_term=True)
    sched = camp.scheduler()
    camp.drive(sched, lambda: max(gs.snapshot_steps(camp.run_dir("stubborn")) or [0]) >= 3)
    pid = sched.jobs["stubborn"].pid

    assert gs.main(["cancel", "stubborn", "--state", camp.state_path]) == 0
    camp.drive(sched, lambda: sched.jobs["stubborn"].state == "cancelled")
    entry = sched.jobs["stubborn"].history[-1]
    assert entry["exit"] is None and "killed" in entry["note"]
    assert not (_read_bytes(f"/proc/{pid}/cmdline") or b"")
    for _ in range(3):
        sched.poll()
    assert sched.jobs["stubborn"].state == "cancelled"

    assert gs.main(["resume", "stubborn", "--state", camp.state_path]) == 0
    camp.drive(sched, done(sched, "stubborn"))
    legs = camp.legs("stubborn")
    assert len(legs) == 2 and legs[1]["start_step"] >= 3 and legs[1]["target_step"] == 50


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
    for attempt in (1, 2):
        camp.drive(
            sched,
            lambda: sched.jobs["doomed"].state == "running"
            and sched.jobs["doomed"].attempts == attempt,
        )
        os.killpg(sched.jobs["doomed"].pid, signal.SIGKILL)
        camp.drive(sched, lambda: sched.jobs["doomed"].state != "running")
    job = sched.jobs["doomed"]
    assert job.state == "failed" and job.failures == 2
    assert all(h["note"].startswith("lost") for h in job.history)


def test_divergence_is_not_retried(camp):
    camp.pool(slot("a", 0))
    camp.add_job("nan", steps=10, diverge_at=3)
    sched = camp.scheduler()
    camp.drive(sched, lambda: sched.jobs["nan"].state == "diverged")
    assert sched.jobs["nan"].attempts == 1


def test_restart_reattaches_to_the_running_job(camp):
    camp.pool(slot("a", 0))
    camp.add_job("long", steps=80, step_time=0.05, cost=2)
    first = camp.scheduler()
    camp.drive(first, lambda: first.jobs["long"].state == "running")
    launch, pid = first.jobs["long"].launch_id, first.jobs["long"].pid

    camp.add_job("other", steps=2, cost=1)
    second = camp.scheduler()
    second.poll()
    assert second.jobs["long"].state == "running"
    assert (second.jobs["long"].launch_id, second.jobs["long"].pid) == (launch, pid)
    assert second.jobs["other"].state == "pending"
    camp.drive(second, done(second))
    assert len(camp.legs("long")) == 1


def test_restart_without_state_adopts_the_run_from_its_sched_files(camp):
    camp.pool(slot("a", 0))
    camp.add_job("long", steps=80, step_time=0.05, cost=2)
    camp.add_job("other", steps=2, cost=1)
    first = camp.scheduler()
    camp.drive(first, lambda: first.jobs["long"].state == "running")
    launch = first.jobs["long"].launch_id
    os.remove(camp.state_path)

    second = camp.scheduler()
    second.poll()
    job = second.jobs["long"]
    assert job.state == "running" and job.launch_id == launch and job.slot == "a:0"
    assert second.jobs["other"].state == "pending"
    camp.drive(second, done(second))
    assert len(camp.legs("long")) == 1


def test_multi_gpu_jobs_only_go_to_slots_with_enough_gpus(camp):
    camp.pool(slot("one", 0, speed=10), slot("two", 0, speed=8), slot("pair", 2, 3, speed=1))
    camp.add_job("big", gpus=2, cost=1, steps=3)
    camp.add_job("small", cost=100, steps=3)
    camp.add_job("small2", cost=50, steps=3)
    sched = camp.scheduler()
    sched.poll()
    assert sched.jobs["big"].slot == "pair:2,3"
    assert sched.jobs["small"].slot == "one:0" and sched.jobs["small2"].slot == "two:0"
    camp.drive(sched, done(sched))
    leg = camp.legs("big")[0]
    assert leg["gpus"] == "2,3"
    assert leg["argv"][-2:] == ["--n-gpus", "2"]


def test_small_jobs_leave_the_pair_to_a_waiting_multi_gpu_job(camp):
    camp.pool(slot("pair", 0, 1))
    camp.add_job("small", cost=100, steps=2)
    camp.add_job("big", gpus=2, cost=1, steps=2)
    sched = camp.scheduler()
    sched.poll()
    assert sched.jobs["big"].slot == "pair:0,1" and sched.jobs["small"].state == "pending"
    camp.drive(sched, done(sched))
    assert camp.legs("small")[0]["gpus"] == "0"


def test_multi_gpu_job_waits_without_a_large_enough_slot(camp):
    camp.pool(slot("a", 0), slot("b", 1))
    camp.add_job("big", gpus=2, cost=100, steps=2)
    camp.add_job("small", cost=1, steps=2)
    sched = camp.scheduler()
    camp.drive(sched, done(sched, "small"))
    for _ in range(3):
        sched.poll()
    assert sched.jobs["big"].state == "pending" and sched.jobs["big"].attempts == 0


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
    assert sched.jobs["second"].state == "pending"
    assert len(camp.legs("first")) == 1

    camp.pool(slot("a", 0))
    camp.drive(sched, done(sched, "second"))


def test_unreadable_pool_keeps_the_previous_one(camp):
    camp.pool(slot("a", 0))
    camp.add_job("long", steps=60)
    sched = camp.scheduler()
    camp.drive(sched, lambda: sched.jobs["long"].state == "running")
    for text in ("slots: [unterminated", ""):
        with open(camp.pool_path, "w") as fh:
            fh.write(text)
        sched.poll()
        assert sched.jobs["long"].state == "running"
        assert list(sched.pool) == ["a:0"]
    assert any("unreadable" in line for line in camp.logs)
    camp.drive(sched, done(sched))


def test_dry_run_prints_commands_and_runs_nothing(camp, tmp_path):
    class Refuse(gs.LocalExecutor):
        def run(self, host, command):
            raise AssertionError("dry run executed a command")

    camp.pool(slot("a", 3))
    camp.add_job("x", steps=2, args=["--backend", "cuda"])
    sched = camp.scheduler(dry_run=True, executor=Refuse())
    sched.run(poll=0.1)
    lines = [line for line in camp.logs if line.startswith("[dry-run]")]
    assert len(lines) == 1
    assert (
        "CUDA_VISIBLE_DEVICES=3" in lines[0] and "XLA_PYTHON_CLIENT_PREALLOCATE=false" in lines[0]
    )
    assert "--backend cuda" in lines[0] and " wrap " in lines[0]
    assert not os.path.exists(camp.state_path)
    assert not os.path.exists(camp.run_dir("x"))


def test_pool_validation(tmp_path):
    path = tmp_path / "pool.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "slots": [
                    {"host": "lizard", "gpus": [0, 1]},
                    {"host": "lizard", "gpus": [1, 2]},
                    {"host": "lizard", "gpus": [1]},
                    {"host": "shark", "gpus": [0], "speed": 2, "env": {"BAD-NAME": 1}},
                    {"gpus": [0]},
                    {"host": "shark", "gpus": [1], "drain": True, "name": "s1"},
                ]
            }
        )
    )
    slots, problems = gs.load_pool(str(path))
    assert list(slots) == ["lizard:0,1", "s1"]
    assert slots["s1"].drain and slots["lizard:0,1"].python == gs.DEFAULT_PYTHON
    assert len(problems) == 4
    assert any("must be one of" in p for p in problems)
    path.write_text("")
    with pytest.raises(ValueError):
        gs.load_pool(str(path))


def test_job_validation(tmp_path):
    path = tmp_path / "jobs.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "workdir": "repo",
                "defaults": {"args": ["--backend", "cuda"]},
                "jobs": [
                    {
                        "name": "a",
                        "config": "configs/a.yaml",
                        "output_dir": "/abs/a",
                        "args": "--n-steps 20",
                    },
                    {"name": "b", "run_dir": "runs/b", "cost": 5, "gpus": 2},
                    {"name": "c", "config": "x.yaml"},
                    {"name": "d", "config": "x.yaml", "output_dir": "o", "args": ["--device", "1"]},
                    {"name": "a", "run_dir": "r"},
                ],
            }
        )
    )
    specs, problems = gs.load_jobs(str(path))
    assert [s.name for s in specs] == ["a", "b"]
    a, b = specs
    assert a.config == str(tmp_path / "repo" / "configs" / "a.yaml") and a.output_dir == "/abs/a"
    assert a.args == ["--backend", "cuda", "--n-steps", "20"] and a.workdir == str(
        tmp_path / "repo"
    )
    assert b.output_dir == b.run_dir == str(tmp_path / "repo" / "runs" / "b") and b.gpus == 2
    assert len(problems) == 3


def test_next_leg_arguments(camp):
    camp.pool(slot("a", 0))
    camp.add_job("j", args=["--n-steps", "30", "--from-scratch", "--backend", "cuda"])
    sched = camp.scheduler()
    sched._refresh_jobs(time.time())
    job = sched.jobs["j"]
    run_dir = camp.run_dir("j")
    config = str(camp.root / "configs" / "j.yaml")
    fresh = [
        "run",
        config,
        "--output-dir",
        run_dir,
        "--n-steps",
        "30",
        "--from-scratch",
        "--backend",
        "cuda",
    ]

    job.base_legs = 0
    assert sched.gyaradax_args(job) == fresh
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "run_info.jsonl"), "w") as fh:
        fh.write(json.dumps({"start_step": 0, "target_step": 30}) + "\n")
    assert sched.gyaradax_args(job) == fresh + ["--overwrite"]

    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        fh.write("steps: 30\n")
    with zipfile.ZipFile(os.path.join(run_dir, "step_000004.npz"), "w") as zf:
        zf.writestr("df.npy", b"df")
    with open(os.path.join(run_dir, "step_000006.npz"), "wb") as fh:
        fh.write(b"PK\x03\x04truncated")
    assert sched.gyaradax_args(job) == ["run", run_dir, "--backend", "cuda"]
    assert sorted(gs.snapshot_steps(run_dir)) == [4]
    assert os.path.exists(os.path.join(run_dir, ".sched", "broken", "step_000006.npz"))

    job.base_legs = 1
    assert sched.gyaradax_args(job) == fresh

    assert gs.strip_flags(
        ["--n-steps=5", "--mp", "--resume-from", "K03", "--overwrite"], gs.FRESH_FLAGS
    ) == ["--mp"]


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
        ex.launch("scorpion", "echo hi")
    argv, kw = calls[-1]
    assert argv[0] == "ssh" and "scorpion" in argv
    assert "BatchMode=yes" in argv and "StrictHostKeyChecking=yes" in argv
    assert kw["timeout"] == 7 and kw["stdin"] == subprocess.DEVNULL
    assert argv[-1] == "bash -c 'echo hi'"
    assert ex.alive("scorpion", 123, "tok") is None
    fake_run.rc = 1
    assert ex.alive("scorpion", 123, "tok") is False
    fake_run.rc = 0
    assert ex.alive("scorpion", 123, "tok") is True
    assert ex.signal("scorpion", 123, "KILL", group=True)
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
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"
