"""gyaradax run: run-directory contents, resume from a run directory, debug outputs."""

import json
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from gyaradax.cli import main

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATE_KEYS = (
    "df",
    "time",
    "step",
    "accumulated_norm_factor",
    "window_start_amp",
    "last_growth_rate",
)


@pytest.fixture
def tiny_config(tmp_path):
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "nl_em_waltz_b01.yaml"))
    for key, value in dict(nvpar=8, nmu=2, ns=8, nkx=11, nky=5).items():
        cfg.grid[key] = value
    cfg.run.pop("data_dir", None)
    path = tmp_path / "tiny.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def _run(*argv):
    assert main(["run", *argv]) == 0


def test_resume_from_run_directory_is_bitwise(tmp_path, tiny_config):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    common = ["--from-scratch", "--block-size", "2", "--backend", "jax"]
    _run(tiny_config, *common, "--n-steps", "4", "--output-dir", a)
    _run(tiny_config, *common, "--n-steps", "2", "--output-dir", b)
    _run(b, "--n-steps", "2")

    with (
        np.load(os.path.join(a, "step_000004.npz")) as ca,
        np.load(os.path.join(b, "step_000004.npz")) as cb,
    ):
        for key in STATE_KEYS:
            np.testing.assert_array_equal(cb[key], ca[key])
    with np.load(os.path.join(a, "fluxes.npz")) as fa, np.load(os.path.join(b, "fluxes.npz")) as fb:
        np.testing.assert_array_equal(fb["step"], fa["step"])
        np.testing.assert_array_equal(fb["fluxes"], fa["fluxes"])

    # only the latest snapshot is kept, the resumed one included
    for run_dir in (a, b):
        assert sorted(f for f in os.listdir(run_dir) if f.startswith("step_")) == ["step_000004.npz"]
    eff = OmegaConf.load(os.path.join(b, "config.yaml"))
    assert eff.solver.n_steps == 4 and eff.run.block_size == 2 and eff.solver.backend == "jax"
    assert os.path.exists(os.path.join(b, "geometry.pkl"))
    with open(os.path.join(b, "run_info.jsonl")) as fh:
        assert [json.loads(line)["start_step"] for line in fh] == [0, 2]

    # the run directory already reached solver.n_steps
    assert main(["run", b]) == 0
    assert not os.path.exists(os.path.join(b, "step_000006.npz"))


def test_debug_writes_telemetry_and_profile(tmp_path, tiny_config):
    out = str(tmp_path / "c")
    _run(
        tiny_config,
        "--from-scratch",
        "--n-steps",
        "4",
        "--block-size",
        "2",
        "--output-dir",
        out,
        "--debug",
    )
    with open(os.path.join(out, "telemetry.jsonl")) as fh:
        records = [json.loads(line) for line in fh]
    assert [r["event"] for r in records] == ["start", "compile", "block", "block", "end"]
    assert [r["profiled"] for r in records if r["event"] == "block"] == [True, False]
    with open(os.path.join(out, "profile_summary.json")) as fh:
        assert json.load(fh)["kernels"]
    assert os.path.isdir(os.path.join(out, "profile"))


def _snapshots(run_dir):
    return sorted(f for f in os.listdir(run_dir) if f.startswith("step_"))


def test_snapshot_cadence(tmp_path, tiny_config):
    common = ["--from-scratch", "--block-size", "1", "--n-steps", "5", "--snapshot-every", "2"]
    every, rolling = str(tmp_path / "every"), str(tmp_path / "rolling")
    _run(tiny_config, *common, "--save-dumps", "--output-dir", every)
    _run(tiny_config, *common, "--output-dir", rolling)

    # every second block and the last one; diagnostics still every block
    assert _snapshots(every) == [f"step_{k:06d}.npz" for k in (0, 2, 4, 5)]
    assert _snapshots(rolling) == ["step_000005.npz"]
    with np.load(os.path.join(rolling, "fluxes.npz")) as fh:
        assert fh["step"].tolist() == [0, 1, 2, 3, 4, 5]
    assert OmegaConf.load(os.path.join(rolling, "config.yaml")).run.snapshot_every == 2


def test_batched_run_matches_single_runs(tmp_path, tiny_config):
    cfg = OmegaConf.load(tiny_config)
    paths = []
    for name, rlt in (("ma", 9.0), ("mb", 6.0)):
        cfg.run.name = name
        cfg.physics.rlt = [rlt, rlt]
        path = str(tmp_path / f"{name}.yaml")
        OmegaConf.save(cfg, path)
        paths.append(path)
    batch = str(tmp_path / "batch")
    common = ["--from-scratch", "--block-size", "2", "--backend", "jax"]
    _run(*paths, *common, "--n-steps", "4", "--output-dir", batch)
    _run(os.path.join(batch, "ma"), os.path.join(batch, "mb"), "--n-steps", "2")

    for name, path in zip(("ma", "mb"), paths):
        single = str(tmp_path / f"single_{name}")
        _run(path, *common, "--n-steps", "6", "--output-dir", single)
        member = os.path.join(batch, name)
        assert _snapshots(member) == ["step_000006.npz"]
        with (
            np.load(os.path.join(member, "step_000006.npz")) as cb,
            np.load(os.path.join(single, "step_000006.npz")) as cs,
        ):
            for key in STATE_KEYS:
                np.testing.assert_array_equal(cb[key], cs[key])
        with (
            np.load(os.path.join(member, "fluxes.npz")) as fb,
            np.load(os.path.join(single, "fluxes.npz")) as fs,
        ):
            np.testing.assert_array_equal(fb["step"], fs["step"])
            np.testing.assert_array_equal(fb["fluxes"], fs["fluxes"])
        with open(os.path.join(member, "run_info.jsonl")) as fh:
            assert [json.loads(line)["start_step"] for line in fh] == [0, 4]
