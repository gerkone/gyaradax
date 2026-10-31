"""gyaradax datagen: configs on the local GPUs, one run each, finished runs skipped."""

import json
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
    (out / "c" / "outcome.json").write_text(json.dumps({"final": True}))
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
