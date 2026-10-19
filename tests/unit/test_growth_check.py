"""Linear growth check: growth rates per resolution, `gyaradax growth`, `run --linear-check`."""

import json
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from gyaradax.cli import main
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


@pytest.mark.parametrize("adiabatic", [False, True])
def test_mode_fluxes_sum_to_the_flux_and_weights_ignore_the_normalization(adiabatic):
    import jax.numpy as jnp

    from gyaradax.core.params import gkparams_from_config
    from gyaradax.geometry import compute_geometry_from_config
    from gyaradax.operators.precompute import linear_precompute
    from gyaradax.solver.driver import _diagnostics, gk_init

    cfg = growth.adjusted_config(_cfg())
    if adiabatic:
        cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "iteration_13.yaml"))
        for key, value in GRID.items():
            cfg.grid[key] = value
    params = gkparams_from_config(cfg, n_gpus_sp=1, n_gpus_vp=1, n_gpus_mu=1)
    n_species = 1 if params.adiabatic_electrons else len(params.mas)
    df, geometry, _ = gk_init(compute_geometry_from_config(cfg), params, n_species=n_species)
    rng = np.random.default_rng(1)
    df = jnp.asarray(rng.standard_normal(df.shape) + 1j * rng.standard_normal(df.shape))
    pre = linear_precompute(geometry, params)
    out = growth.mode_fluxes(df, geometry, params, pre)
    _, ref = _diagnostics(df, geometry, params, pre)
    ref = np.asarray(ref, np.float64).reshape(-1, 3)
    assert out["flux_ky"].shape == (ref.shape[0], 3, GRID["nky"])
    atol = 1e-12 * np.abs(ref).max()
    np.testing.assert_allclose(out["flux_ky"].sum(-1), ref, rtol=1e-10, atol=atol)
    np.testing.assert_allclose(out["flux"], ref, rtol=1e-10, atol=atol)
    np.testing.assert_allclose(out["weight_ky_allkx"] * out["phi2_ky"], out["flux_ky"], rtol=1e-12)
    chain = out["weight_ky"] * out["phi2_ky_chain0"]
    np.testing.assert_allclose(chain, out["flux_ky_chain0"], rtol=1e-12)
    # ky = 0 has one kx per chain; the chain of a ky > 0 holds part of its kx
    assert np.all(out["phi2_ky_chain0"] <= out["phi2_ky"] * (1 + 1e-12))
    assert np.all(out["phi2_ky_chain0"][1:] < out["phi2_ky"][1:])
    # rescaling one ky (as the per-ky normalization does) keeps its weight
    scaled = df.at[..., 2].multiply(3.0 - 2.0j)
    again = growth.mode_fluxes(scaled, geometry, params, pre)
    np.testing.assert_allclose(again["phi2_ky"][2], 13.0 * out["phi2_ky"][2], rtol=1e-10)
    for key in ("weight_ky", "weight_ky_allkx"):
        np.testing.assert_allclose(again[key], out[key], rtol=1e-8, atol=1e-14)
