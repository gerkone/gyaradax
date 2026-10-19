"""gyaradax datagen: campaign directories, job names, overrides, collisions, and a full run."""

import argparse
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

from gyaradax.datagen import campaign as cg
from gyaradax.datagen import scheduler as gs


def _config(path, name, n_steps=100, ns=8, **extra):
    cfg = {
        "run": {"name": name},
        "solver": {"n_steps": n_steps, "disp_par": 1.0},
        "physics": {"signz": [1.0, -1.0]},
        "grid": {"nvpar": 8, "nmu": 2, "ns": ns, "nkx": 11, "nky": 5},
        **extra,
    }
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _jobs(camp_dir):
    with open(os.path.join(camp_dir, "jobs.yaml")) as fh:
        return {j["name"]: j for j in yaml.safe_load(fh)["jobs"]}


def test_configs_become_named_jobs_with_overrides(tmp_path):
    a = _config(tmp_path / "kinetic_534.yaml", "kinetic_534", n_steps=200)
    b = _config(tmp_path / "kinetic_614.yaml", "kinetic_614", n_steps=100, ns=16)
    camp, data = tmp_path / "camp", tmp_path / "data"
    argv = ["add", a, b, "--campaign", str(camp), "--dataset", str(data), "--name", "iteration_{id}_dp4"]
    assert cg.main([*argv, "--set", "solver.disp_par=4.0", "--args", "--backend cuda"]) == 0

    jobs = _jobs(camp)
    assert set(jobs) == {"iteration_534_dp4", "iteration_614_dp4"}
    job = jobs["iteration_534_dp4"]
    assert job["output_dir"] == str(camp / "runs" / "iteration_534_dp4") and job["gpus"] == 1
    assert job["args"] == ["--backend", "cuda", "--dataset", str(data), "--dataset-name", "iteration_534_dp4"]
    # cost is grid points x species x steps: twice the steps on half the grid costs the same
    assert job["cost"] == pytest.approx(jobs["iteration_614_dp4"]["cost"]) and job["cost"] > 0
    with open(job["config"]) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["solver"]["disp_par"] == 4.0 and cfg["run"]["name"] == "iteration_534_dp4"
    with open(camp / "campaign.yaml") as fh:
        settings = yaml.safe_load(fh)
    assert settings["workdir"] == str(cg.REPO_ROOT) and settings["set"] == {"solver.disp_par": 4.0}
    # the job file is the scheduler's, launched from the campaign's code
    specs, problems = gs.load_jobs(str(camp / "jobs.yaml"))
    assert not problems and {s.name for s in specs} == set(jobs)
    assert all(s.workdir == str(cg.REPO_ROOT) for s in specs)


def test_collisions_changed_configs_and_settings_are_refused(tmp_path):
    a = _config(tmp_path / "kinetic_1.yaml", "kinetic_1")
    camp, data = str(tmp_path / "camp"), tmp_path / "data"
    base = ["--campaign", camp, "--dataset", str(data)]
    # a stored trajectory of the same name, linked from another copy
    (tmp_path / "stored").mkdir()
    data.mkdir()
    os.symlink(tmp_path / "stored", data / "kinetic_1_ifft_realpotens")
    with pytest.raises(SystemExit, match="already exists"):
        cg.main(["add", a, *base])
    assert not os.path.exists(os.path.join(camp, "configs", "kinetic_1.yaml"))
    os.remove(data / "kinetic_1_ifft_realpotens")

    assert cg.main(["add", a, *base]) == 0
    assert cg.main(["add", a, "--campaign", camp]) == 0
    assert list(_jobs(camp)) == ["kinetic_1"]
    _config(tmp_path / "kinetic_1.yaml", "kinetic_1", n_steps=999)
    with pytest.raises(SystemExit, match="different config"):
        cg.main(["add", a, "--campaign", camp])
    with pytest.raises(SystemExit, match="fixes set"):
        cg.main(["add", a, "--campaign", camp, "--set", "solver.disp_par=2"])
    with pytest.raises(SystemExit, match="may not contain"):
        cg.main(["add", a, "--campaign", str(tmp_path / "c2"), "--args", "--dataset x"])
    plain = _config(tmp_path / "plain.yaml", "plain")
    with pytest.raises(SystemExit, match="has no number"):
        cg.main(["add", plain, "--campaign", str(tmp_path / "c3"), "--name", "x_{id}"])


def _targets(path):
    slots = [{"host": "here", "gpus": [g]} for g in (0, 1)]
    path.write_text(yaml.safe_dump({"defaults": {"python": sys.executable}, "slots": slots}))
    return str(path)


def test_campaign_jobs_run_to_completion(tmp_path):
    from test_datagen_scheduler import FAKE_GYARADAX, done, drive  # type: ignore[import-not-found]

    fake = tmp_path / "fake_gyaradax.py"
    fake.write_text(FAKE_GYARADAX)
    configs = [_config(tmp_path / f"k_{i}.yaml", f"k_{i}", steps=3, step_time=0.02) for i in (1, 2, 3)]
    camp = cg.Campaign(str(tmp_path / "camp"))
    assert cg.main(["add", *configs, "--campaign", camp.root, "--dataset", str(tmp_path / "data")]) == 0
    _targets(Path(camp.targets))
    logs = []
    sched = gs.Scheduler(
        camp.targets, camp.jobs_path, camp.state, gs.LocalExecutor(),
        heartbeat=0.2, grace=1.0, program=[sys.executable, str(fake)], log=logs.append,
    )
    drive(sched, lambda: len(sched.jobs) == 3 and done(sched)(), timeout=60, logs=logs)
    assert sorted(sched.jobs) == ["k_1", "k_2", "k_3"], logs
    for name, job in sched.jobs.items():
        legs = gs.run_legs(job.spec.output_dir)
        assert len(legs) == 1 and legs[0]["argv"][:2] == ["run", os.path.join(camp.root, "configs", f"{name}.yaml")]
        assert legs[0]["argv"][legs[0]["argv"].index("--dataset-name") + 1] == name
    # the host's CUDA library was probed once
    assert len(sched.probes) == 1


def test_time_limit_becomes_a_deadline_the_detached_scheduler_keeps(tmp_path, monkeypatch):
    assert [gs.parse_duration(t) for t in ("10h", "90m", "1h30m", "2d", "3600", "45s", "1.5h")] == [
        36000, 5400, 5400, 172800, 3600, 45, 5400
    ]
    for bad in ("", "10x", "h", "0", "-5m"):
        with pytest.raises(argparse.ArgumentTypeError):
            gs.parse_duration(bad)

    launched = []

    class Proc:
        pid = 1

    def popen(argv, **kw):
        if "datagen" not in argv:
            return real_popen(argv, **kw)
        launched.append(argv)
        return Proc()

    real_popen = cg.subprocess.Popen
    monkeypatch.setattr(cg.subprocess, "Popen", popen)
    targets = _targets(tmp_path / "targets.yaml")
    cfg = _config(tmp_path / "k_1.yaml", "k_1")
    t0 = time.time()
    argv = ["run", cfg, "--campaign", str(tmp_path / "camp"), "--targets", targets]
    assert cg.main([*argv, "--time-limit", "10h", "--detach"]) == 0
    (child,) = launched
    deadline = float(child[child.index("--deadline") + 1])
    assert t0 + 36000 <= deadline <= time.time() + 36000
    assert "--time-limit" not in child


def test_time_mode_cost_follows_the_end_time(tmp_path):
    timed = tmp_path / "k_1.yaml"
    _config(timed, "k_1", n_steps=10)
    cfg = yaml.safe_load(timed.read_text())
    cfg["solver"].update(dt=0.004, dump_dt=1.4, t_end=400.0)
    timed.write_text(yaml.safe_dump(cfg))
    stepped = _config(tmp_path / "k_2.yaml", "k_2", n_steps=100000)
    assert cg.main(["add", str(timed), stepped, "--campaign", str(tmp_path / "camp")]) == 0
    jobs = _jobs(tmp_path / "camp")
    assert jobs["k_1"]["cost"] == jobs["k_2"]["cost"] > 0


def test_outcome_switches_go_to_every_job(tmp_path):
    a = _config(tmp_path / "kinetic_5.yaml", "kinetic_5")
    camp = str(tmp_path / "camp")
    argv = ["add", a, "--campaign", camp, "--linear-gate", "--stop-on-decay"]
    assert cg.main([*argv, "--args", "--decay-t-min 40"]) == 0
    job = _jobs(camp)["kinetic_5"]
    assert job["args"] == ["--decay-t-min", "40", "--linear-gate", "--stop-on-decay"]
    with open(os.path.join(camp, "campaign.yaml")) as fh:
        settings = yaml.safe_load(fh)
    assert settings["linear_gate"] and settings["stop_on_decay"] and not settings["stop_on_banding"]
    assert cg.main(["add", a, "--campaign", camp, "--linear-gate"]) == 0
    with pytest.raises(SystemExit, match="fixes stop_on_banding"):
        cg.main(["add", a, "--campaign", camp, "--stop-on-banding"])
