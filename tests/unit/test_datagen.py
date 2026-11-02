"""gyaradax datagen: configs on the local GPUs, one run each, finished runs skipped."""

import os
import subprocess

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


def test_trajectories_are_extended_in_place_and_resumed_from_their_run_directory(tmp_path, monkeypatch):
    import numpy as np

    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or _Proc(cmd))
    trajs = []
    for name in ("x", "y"):
        traj = tmp_path / "data" / f"{name}_ifft_realpotens"
        (traj / "resume").mkdir(parents=True)
        (traj / ".gyaradax_run").write_text("run\n")
        np.savez(traj / "resume" / "state.npz", time=np.array(10.0))
        trajs.append(str(traj))
    out = tmp_path / "out"
    (out / "y").mkdir(parents=True)
    (out / "y" / "step_000100.npz").write_bytes(b"")
    assert datagen.main([*trajs, "--out", str(out), "--gpus", "0", "--extend", "200", "--poll", "0"]) == 0
    x, y = calls
    # x starts from its resume dump into its own trajectory; y continues the run directory it has
    assert x[4] == trajs[0] and x[x.index("--t-end") + 1] == "210" and "--dataset" not in x
    assert x[x.index("--output-dir") + 1] == str(out / "x")
    assert y[4] == str(out / "y") and "--t-end" not in y
