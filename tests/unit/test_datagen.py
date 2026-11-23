"""gyaradax datagen: configs on the local GPUs, one run each, finished runs skipped."""

import os
import subprocess

import pytest

from gyaradax import datagen


class _Proc:
    def __init__(self, cmd, **kw):
        self.cmd = cmd

    def poll(self):
        return 0


def test_configs_run_one_per_gpu_and_finished_runs_are_skipped(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _Proc(cmd))
    configs = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("run: {}\n")
        configs.append(str(path))
    out = tmp_path / "out"
    (out / "c").mkdir(parents=True)
    (out / "c" / datagen.EXIT_FILE).write_text("0\n")
    (out / "b").mkdir()
    (out / "b" / "config.yaml").write_text("run: {}\n")
    argv = [*configs, "--out", str(out), "--gpus", "0,1", "--dataset", str(tmp_path / "data"), "--poll", "0"]
    assert datagen.main(argv) == 0
    by_name = {os.path.basename(c[c.index("--device") - 1].removesuffix(".yaml")): c for c in calls}
    assert sorted(by_name) == ["a", "b"]
    # a starts from its config into its run directory and trajectory; b continues its run directory
    assert by_name["a"][4] == configs[0] and by_name["a"][by_name["a"].index("--dataset-name") + 1] == "a"
    assert by_name["b"][4] == str(out / "b") and "--dataset" not in by_name["b"]
    assert {c[c.index("--device") + 1] for c in calls} <= {"0", "1"}
    # every run leaves its exit code; a finished run is skipped next time
    assert (out / "a" / datagen.EXIT_FILE).read_text() == "0\n"




def test_runs_take_aligned_gpu_blocks_from_their_sharding(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _Proc(cmd))
    sizes = {"one": None, "two": "n_gpus: 2", "quad": "n_gpus_sp: 2\n  n_gpus_mu: 2"}
    configs = []
    for name, shard in sizes.items():
        path = tmp_path / f"{name}.yaml"
        path.write_text("run: {}\n" + (f"sharding:\n  {shard}\n" if shard else ""))
        configs.append(str(path))
    assert datagen.main([*configs, "--out", str(tmp_path / "out"), "--gpus", "0,1,2,3,4,5,6,7", "--poll", "0"]) == 0
    where = {}
    for cmd in calls:
        name = os.path.basename(cmd[4]).removesuffix(".yaml")
        where[name] = cmd[cmd.index("--device") + 1] if "--device" in cmd else cmd[cmd.index("--device-list") + 1]
    # one GPU for "one"; "two" gets the first free aligned pair, "quad" the first free aligned quad
    assert where == {"one": "0", "two": "2,3", "quad": "4,5,6,7"}


def test_a_run_wider_than_the_pool_is_refused(tmp_path):
    path = tmp_path / "wide.yaml"
    path.write_text("sharding:\n  n_gpus: 4\n")
    with pytest.raises(SystemExit):
        datagen.main([str(path), "--out", str(tmp_path / "out"), "--gpus", "0,1", "--poll", "0"])


def test_one_gpu_runs_share_a_gpu_up_to_per_gpu(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _Running(cmd))
    configs = []
    for name in ("a", "b", "c", "d", "e"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("run: {}\n")
        configs.append(str(path))
    monkeypatch.setattr(datagen.time, "sleep", lambda s: (_ for _ in ()).throw(StopIteration))
    with pytest.raises(StopIteration):
        datagen.main([*configs, "--out", str(tmp_path / "out"), "--gpus", "0,1", "--per-gpu", "2", "--poll", "0"])
    # four runs start at once, two per GPU; the fifth waits
    assert [cmd[cmd.index("--device") + 1] for cmd in calls] == ["0", "0", "1", "1"]


class _Running(_Proc):
    def poll(self):
        return None
