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

    # only the latest restart snapshot is kept; the config records the resolved run
    assert sorted(f for f in os.listdir(a) if f.startswith("step_")) == ["step_000004.npz"]
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
