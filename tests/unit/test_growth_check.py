"""Linear growth check: growth rates per resolution, `gyaradax growth`, `run --linear-check`, datagen."""

import json
import os

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from gyaradax.cli import main
from gyaradax.datagen import campaign as cg
from gyaradax.diagnostics import growth

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRID = dict(nvpar=8, nmu=2, ns=8, nkx=11, nky=5)


def _cfg(**solver):
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "kinetic.yaml"))
    for key, value in GRID.items():
        cfg.grid[key] = value
    for key, value in solver.items():
        cfg.solver[key] = value
    cfg.run.pop("data_dir", None)
    return cfg


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "kinetic_7.yaml"
    OmegaConf.save(_cfg(), path)
    return str(path)


def test_adjusted_config_runs_linearly_at_another_resolution():
    cfg = _cfg()
    with_ds = _cfg()
    with_ds.geometry.sgr_dist = 1.0 / 8
    assert growth.adjusted_config(with_ds, 16).geometry.sgr_dist == pytest.approx(1.0 / 16)
    out = growth.adjusted_config(cfg, 16, {"solver.disp_par": 4.0})
    assert out.grid.ns == 16
    assert out.solver.non_linear is False and out.solver.disp_par == 4.0
    assert cfg.grid.ns == 8 and cfg.solver.non_linear is True
    # the key ignores the run's name, length and backend, not its physics
    same = _cfg(n_steps=5, backend="cuda")
    same.run.name = "other"
    assert growth.config_key(growth.adjusted_config(same)) == growth.config_key(
        growth.adjusted_config(cfg)
    )
    assert growth.config_key(out) != growth.config_key(growth.adjusted_config(cfg))


def test_growth_rates_cap_dt_at_the_linear_estimate():
    r = growth.growth_rates(_cfg(dt=1.0), backend="jax", t_end=0.5, t_min=1.0, block=10)
    assert r["dt_input"] == 1.0 and r["dt"] == r["dt_linear_estimate"] < 1.0
    assert r["t"] >= 0.5 and len(r["gamma"]) == len(r["ky"]) == GRID["nky"]
    assert np.all(np.isfinite(r["gamma"])) and r["ns"] == 8


def test_compare_against_the_reference_dominant_ky():
    ky = [0.0, 0.2, 0.4, 0.6]
    ref = {"ns": 36, "ky": ky, "gamma": [0.0, 0.05, 0.10, 0.02]}
    run = {"ns": 18, "ky": ky, "gamma": [0.0, 0.06, 0.125, 0.0]}
    c = growth.compare(run, ref)
    assert c["ky"] == 0.4 and c["gamma"] == 0.125 and c["rel"] == pytest.approx(0.25)
    report = growth.assemble([dict(run, config_key="k"), dict(ref, config_key="r")], 18)
    assert report["reference_ns"] == 36 and report["config_key"] == "k"
    assert growth.summary(report) == "γ ns18 0.125 / ns36 0.100 (+25%)"
    stable = {"ns": 36, "ky": ky, "gamma": [0.0, -0.1, -0.2, -0.3]}
    assert growth.compare(run, stable)["rel"] is None


def test_growth_command_writes_each_resolution(tmp_path, config):
    out = str(tmp_path / "growth.json")
    argv = [
        "growth",
        config,
        "--ns",
        "8",
        "12",
        "--t-end",
        "0.5",
        "--backend",
        "jax",
        "--output",
        out,
    ]
    assert main(argv) == 0
    with open(out) as fh:
        report = json.load(fh)
    assert [r["ns"] for r in report["runs"]] == [8, 12] and report["reference_ns"] == 12
    assert set(report["comparison"]) == {"8", "12"} and report["config"] == os.path.abspath(config)


def test_run_linear_check_is_written_and_reused_on_resume(tmp_path, config, capsys):
    out = str(tmp_path / "run")
    common = [
        "--linear-check",
        "12",
        "--linear-check-t-end",
        "0.5",
        "--backend",
        "jax",
        "--n-steps",
        "2",
    ]
    assert main(["run", config, "--from-scratch", "--output-dir", out, *common]) == 0
    path = os.path.join(out, "linear_check.json")
    with open(path) as fh:
        report = json.load(fh)
    assert report["ns"] == 8 and [r["ns"] for r in report["runs"]] == [8, 12]
    mtime = os.path.getmtime(path)
    capsys.readouterr()
    assert main(["run", out, *common]) == 0
    assert "(computed before)" in capsys.readouterr().out and os.path.getmtime(path) == mtime


def test_tolerance_refuses_the_run_with_its_own_exit_code(tmp_path, config, capsys):
    out = tmp_path / "run"
    out.mkdir()
    cfg = OmegaConf.load(config)
    key = growth.config_key(growth.adjusted_config(cfg))
    ky = [0.0, 0.35, 0.7, 1.05, 1.4]
    runs = [
        {
            "ns": 8,
            "ky": ky,
            "gamma": [0.0, 0.15, 0.05, 0.0, 0.0],
            "converged": True,
            "t": 20.0,
            "config_key": key,
        },
        {
            "ns": 12,
            "ky": ky,
            "gamma": [0.0, 0.10, 0.04, 0.0, 0.0],
            "converged": True,
            "t": 20.0,
            "config_key": "x",
        },
    ]
    for r in runs:
        r["peak"] = growth.peak(r)
    with open(out / "linear_check.json", "w") as fh:
        json.dump(growth.assemble(runs, 8), fh)
    argv = [
        "run",
        config,
        "--from-scratch",
        "--output-dir",
        str(out),
        "--backend",
        "jax",
        "--n-steps",
        "2",
    ]
    assert (
        main([*argv, "--linear-check", "12", "--linear-check-tol", "0.1"])
        == growth.EXIT_LINEAR_CHECK
    )
    assert "+50.0%, tolerance 10.0%" in capsys.readouterr().out
    assert not any(p.name.startswith("step_") for p in out.iterdir())
    assert main([*argv, "--linear-check", "12", "--linear-check-tol", "0.6"]) == 0


def test_datagen_passes_the_check_to_every_job_and_reports_it(tmp_path, config):
    camp = str(tmp_path / "camp")
    assert (
        cg.main(
            [
                "add",
                config,
                "--campaign",
                camp,
                "--linear-check",
                "24,36",
                "--linear-check-tol",
                "0.2",
            ]
        )
        == 0
    )
    with open(os.path.join(camp, "jobs.yaml")) as fh:
        job = yaml.safe_load(fh)["jobs"][0]
    assert job["args"][-4:] == ["--linear-check", "24,36", "--linear-check-tol", "0.2"]
    with pytest.raises(SystemExit, match="not both"):
        cg.main(
            [
                "add",
                config,
                "--campaign",
                str(tmp_path / "c2"),
                "--linear-check",
                "36",
                "--args",
                "--linear-check 24",
            ]
        )
    with pytest.raises(SystemExit, match="fixes linear_check"):
        cg.main(["add", config, "--campaign", camp, "--linear-check", "48"])
    ky = [0.0, 0.5]
    report = growth.assemble(
        [
            {"ns": 8, "ky": ky, "gamma": [0.0, 0.064], "config_key": "k"},
            {"ns": 36, "ky": ky, "gamma": [0.0, 0.051], "config_key": "r"},
        ],
        8,
    )
    os.makedirs(job["output_dir"], exist_ok=True)
    with open(os.path.join(job["output_dir"], "linear_check.json"), "w") as fh:
        json.dump(report, fh)
    doc = {"jobs": {job["name"]: {"spec": {"output_dir": job["output_dir"]}}}}
    assert cg._linear_checks(doc) == {job["name"]: "γ ns8 0.064 / ns36 0.051 (+25%)"}
