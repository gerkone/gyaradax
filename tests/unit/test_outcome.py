"""Run outcome: linear gate, decay and banding detectors, outcome summary, `run` flags."""

import json
import os
import pickle

import numpy as np
import pytest
from omegaconf import OmegaConf

from gyaradax.cli import main
from gyaradax.diagnostics import growth, outcome
from gyaradax.io import dataset as ds

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRID = dict(nvpar=8, nmu=2, ns=8, nkx=11, nky=5)
KY = [0.0, 0.35, 0.7, 1.05, 1.4]


def test_gate_decision_only_calls_settled_damped_rates_stable():
    def r(gamma, converged=True):
        return {"ky": KY, "gamma": gamma, "converged": converged}

    assert outcome.gate_decision(r([0.3, -0.05, -0.02, -0.1, -0.2])) == "stable"
    assert outcome.gate_decision(r([0.3, -0.05, -0.02, -0.1, -0.2], False)) == "ambiguous"
    # ky = 0 does not count; a weakly damped or marginal mode is ambiguous
    assert outcome.gate_decision(r([0.3, -0.05, -0.005, -0.1, -0.2])) == "ambiguous"
    assert outcome.gate_decision(r([0.0, 0.01, -0.05, -0.1, -0.2])) == "ambiguous"
    assert outcome.gate_decision(r([0.0, 0.01, 0.05, -0.1, -0.2], False)) == "unstable"
    assert outcome.gate_decision(r([0.0, -0.03, -0.05, -0.1, -0.2]), stable=-0.04) == "ambiguous"
    rep = outcome.gate_report(
        {**r([0.0, -0.03, -0.02, -0.1, -0.2]), "t": 30.0, "omega": [0, 0.1, 0.2, 0.3, 0.4]},
        -0.01,
        0.02,
    )
    assert rep["decision"] == "stable" and rep["gamma_max"] == -0.02
    assert rep["peak_ky"] == 0.7 and rep["omega_peak"] == 0.2 and rep["settled"] is True


def test_nyquist_shares_of_an_odd_even_zonal_mode():
    nx, ns, ny = 12, 18, 16
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)[:, None, None]
    s = np.arange(ns)[None, :, None]
    y = np.linspace(0, 2 * np.pi, ny, endpoint=False)[None, None, :]
    envelope = np.exp(-(((s - 8.5) / 3) ** 2))
    smooth = np.cos(x) * np.cos(2 * np.pi * s / ns) * np.sin(3 * y)
    zonal_band = 2.0 * np.sin(x) * envelope * (-1.0) ** s
    zonal, nonzonal = outcome.nyquist_shares(smooth + zonal_band + 0 * y)
    assert zonal > 0.4 and nonzonal < 1e-12
    zonal, nonzonal = outcome.nyquist_shares(smooth + 0.3 * np.cos(x) + 0 * y)
    assert zonal < 1e-12 and nonzonal < 1e-12
    _, nonzonal = outcome.nyquist_shares(np.sin(3 * y) * (-1.0) ** s + 0 * x)
    assert nonzonal == pytest.approx(1.0)


def test_decay_needs_the_minimum_time_the_floor_and_a_falling_flux():
    t = np.arange(0.0, 60.0, 2.0)
    falling = 1e-2 * np.exp(-0.2 * t)
    kw = dict(t_min=30.0, floor=1e-3, rel_floor=1e-3, window=5)
    assert outcome.decayed(t, falling, **kw)
    assert not outcome.decayed(t[:15], falling[:15], **kw)
    # rising from noise (a slow linear phase) is not decay, however small
    assert not outcome.decayed(t, 1e-9 * np.exp(0.05 * t), **kw)
    # below the floor but flat or rebounding
    assert not outcome.decayed(t, np.full_like(t, 1e-4), **kw)
    rebound = falling.copy()
    rebound[-2:] *= 5
    assert not outcome.decayed(t, rebound, **kw)
    # the relative floor: decayed from a peak of 100 down to 0.05
    burst = 100.0 * np.exp(-0.25 * t)
    assert outcome.decayed(t, burst, **{**kw, "rel_floor": 1e-3})
    assert not outcome.decayed(t, burst + 0.5, **kw)


def test_banding_needs_consecutive_frames_over_the_threshold():
    assert outcome.banded([0.0, 0.1, 0.4, 0.5, 0.6], 0.3, 3)
    assert not outcome.banded([0.0, 0.4, 0.5, 0.2, 0.6], 0.3, 3)
    assert not outcome.banded([0.5, 0.6], 0.3, 3)


def test_flux_average_window_error_and_trend():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 100.0, 201)
    flat = 3.0 + 0.1 * rng.standard_normal(t.size)
    avg = outcome.flux_average(t, flat[:, None])
    assert avg["window"][0] == pytest.approx(70.0) and avg["frames"] == 61
    assert avg["mean"][0] == pytest.approx(3.0, abs=0.05) and 0 < avg["err"][0] < 0.05
    assert abs(avg["drift"][0]) < 0.2
    rising = outcome.flux_average(t, (1.0 + 0.05 * t)[:, None])
    assert rising["drift"][0] > 0.2 and rising["drift_sigma"][0] > 2
    cls = outcome.classify
    assert cls(None, False, 3.0, 0.01, 0.5, floor=1e-3) == "saturated"
    assert cls(None, False, 3.0, 0.5, 9.0, floor=1e-3) == "growing"
    assert cls(None, False, 3.0, -0.5, -9.0, floor=1e-3) == "decaying"
    assert cls(None, False, 1e-5, 0.0, 0.0, floor=1e-3) == "flat"
    assert cls(None, True, 3.0, 0.0, 0.0, floor=1e-3) == "banded"
    assert cls("decayed", True, 3.0, 0.0, 0.0, floor=1e-3) == "decayed"


class _State:
    def __init__(self, t, step, nky=5):
        self.time, self.step = np.float64(t), np.int32(step)
        self.last_growth_rate = np.zeros(nky)


def _spectral_phi(real):
    # inverse of dataset.phi_to_real for a (x, s, y) field
    spec = np.fft.fftshift(np.fft.rfftn(real, axes=(0, 2), norm="forward"), axes=0)
    return np.moveaxis(spec, 1, 0)


def test_monitor_follows_frames_and_resumes_its_series(tmp_path):
    nx, ns, ny = 8, 8, 8
    s = np.arange(ns)[None, :, None]
    x = np.arange(nx)[:, None, None]
    banded_phi = _spectral_phi(
        np.cos(2 * np.pi * x / nx) * (-1.0) ** s + 0.0 * np.zeros((1, 1, ny))
    )
    settings = outcome.MonitorSettings(stop_on_banding=True, banding_window=2)
    mon = outcome.RunMonitor(str(tmp_path), settings, ion=0, electron=1)
    fluxes = np.array([[0.1, 2.0, 0.0], [0.1, 0.5, 0.0]])
    assert mon.observe(_State(1.0, 10), fluxes, banded_phi) is None
    with open(tmp_path / outcome.OUTCOME_FILE) as fh:
        doc = json.load(fh)
    assert doc["class"] == "running" and doc["nyq_zonal"] > 0.9 and doc["frames"] == 1
    assert mon.observe(_State(2.0, 20), fluxes, banded_phi) == "banded"
    assert mon.exit_code == outcome.EXIT_BANDED
    doc = mon.finish()
    assert doc["class"] == "banded" and doc["banded_t"] == 2.0
    assert doc["average"]["mean"][0][1] == 2.0 and doc["average"]["mean"][1][1] == 0.5
    # a resumed run keeps the frames up to its start step
    again = outcome.RunMonitor(str(tmp_path), settings, start_step=10)
    assert again.series["step"] == [10] and again.stopped is None


def test_monitor_holds_dataset_frames_until_the_flux_threshold(tmp_path):
    mon = outcome.RunMonitor(str(tmp_path), outcome.MonitorSettings(dataset_from_flux=1.0))
    phi = np.zeros((8, 8, 5), complex)
    mon.observe(_State(1.0, 1), np.array([0.0, 0.2, 0.0]), phi)
    assert not mon.write_frame(0) and mon.write_frame(0, final=True)
    mon.observe(_State(2.0, 2), np.array([0.0, 1.5, 0.0]), phi)
    assert mon.write_frame(0)
    mon.observe(_State(3.0, 3), np.array([0.0, 0.2, 0.0]), phi)
    assert mon.write_frame(1)


# end-to-end on a tiny kinetic grid (JAX backend)


@pytest.fixture
def config(tmp_path):
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "kinetic.yaml"))
    for key, value in GRID.items():
        cfg.grid[key] = value
    cfg.run.name = "tiny"
    path = tmp_path / "tiny.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def _meta(traj):
    with open(os.path.join(traj, "metadata.pkl"), "rb") as fh:
        return pickle.load(fh)


COMMON = ["--from-scratch", "--backend", "jax", "--block-size", "2"]


def test_linear_gate_keeps_the_eigenmode_and_runs_the_ambiguous_case(tmp_path, config, capsys):
    run, root = str(tmp_path / "run"), str(tmp_path / "data")
    argv = ["run", config, *COMMON, "--n-steps", "4", "--output-dir", run, "--dataset", root]
    assert main([*argv, "--linear-gate", "--linear-gate-t-end", "0.5"]) == 0
    with open(os.path.join(run, outcome.GATE_FILE)) as fh:
        gate = json.load(fh)
    assert gate["decision"] in ("ambiguous", "unstable") and not gate["settled"]
    assert len(gate["omega"]) == len(gate["gamma"]) == GRID["nky"]
    assert np.all(np.isfinite(gate["omega"])) and gate["peak_ky"] > 0
    with np.load(os.path.join(run, outcome.MODE_FILE)) as mode:
        shape = mode["df"].shape
        assert mode["df"].dtype == np.complex64 and mode["phi"].shape == (8, 11, 5)
    assert shape == (2, 8, 2, 8, 11, 5)

    traj = ds.trajectory_path(root, "tiny")
    meta = _meta(traj)
    assert tuple(meta["linear_shape"]) == tuple(meta["df_shape"]) == (2, *shape)
    real = np.fromfile(os.path.join(traj, meta["linear_bin"]), np.float32)
    k = np.fromfile(os.path.join(traj, meta["linear_k_bin"]), np.float32)
    assert real.size == k.size == 2 * np.prod(shape) and meta["linear_space"] == "real"
    assert meta["linear_rms_ckxky"].shape == (2, 11, 5) and meta["linear_rms"] > 0
    assert meta["linear_gamma"].tolist() == pytest.approx(gate["gamma"])
    assert meta["linear_gate"] == gate["decision"]
    with open(os.path.join(run, outcome.OUTCOME_FILE)) as fh:
        doc = json.load(fh)
    assert doc["final"] and doc["frames"] == 2 and doc["gate"] == gate["decision"]
    assert meta["trace_flux"].shape == (2, 6) and meta["outcome"]["class"] == doc["class"]

    # the next leg reuses the gate
    capsys.readouterr()
    assert main(["run", run, "--n-steps", "2", "--linear-gate", "--linear-gate-t-end", "0.5"]) == 0
    assert "(computed before)" in capsys.readouterr().out
    assert _meta(traj)["solver_steps"].tolist() == [2, 4, 6]


def test_cached_stable_gate_skips_the_nonlinear_run(tmp_path, config, capsys):
    run = tmp_path / "run"
    run.mkdir()
    key = growth.config_key(growth.adjusted_config(OmegaConf.load(config)))
    report = {
        "ns": 8, "ky": KY, "gamma": [0.0, -0.05, -0.03, -0.1, -0.2], "gamma_std": [0.0] * 5,
        "t": 40.0, "converged": True, "config_key": key,
    }  # fmt: skip
    with open(run / outcome.GATE_FILE, "w") as fh:
        json.dump(report, fh)
    np.savez(run / outcome.MODE_FILE, df=np.zeros(1, np.complex64), phi=np.zeros(1), time=40.0)
    argv = ["run", config, *COMMON, "--n-steps", "2", "--output-dir", str(run), "--linear-gate"]
    assert main(argv) == outcome.EXIT_STABLE
    assert "(computed before)" in capsys.readouterr().out
    assert not any(p.name.startswith("step_") for p in run.iterdir())
    with open(run / outcome.OUTCOME_FILE) as fh:
        doc = json.load(fh)
    assert doc["class"] == "linear_stable" and doc["frames"] == 0


def test_stop_on_decay_keeps_one_frame_and_the_full_trace(tmp_path, config, monkeypatch):
    monkeypatch.setattr(outcome, "decayed", lambda t, q, **kw: len(t) >= 3)
    run, root = str(tmp_path / "run"), str(tmp_path / "data")
    argv = ["run", config, *COMMON, "--n-steps", "20", "--output-dir", run, "--dataset", root]
    assert main([*argv, "--stop-on-decay"]) == outcome.EXIT_DECAYED
    assert os.path.exists(os.path.join(run, "step_000006.npz"))
    traj = ds.trajectory_path(root, "tiny")
    meta = _meta(traj)
    assert meta["solver_steps"].tolist() == [6] and meta["stable"]
    assert meta["trace_solver_steps"].tolist() == [2, 4, 6]
    assert sorted(os.listdir(os.path.join(traj, "data"))) == [
        "poten_00000.zstd16.bin",
        "timestep_00000.zstd16.bin",
    ]
    assert meta["outcome"]["class"] == "decayed" and "phi_mean" in meta
    with np.load(ds.resume_path(traj)) as dump:
        assert int(dump["step"]) == 6


def test_stop_on_banding_and_deferred_dataset_frames(tmp_path, config):
    run, root = str(tmp_path / "run"), str(tmp_path / "data")
    argv = ["run", config, *COMMON, "--n-steps", "20", "--output-dir", run, "--dataset", root]
    flags = ["--stop-on-banding", "--banding-threshold", "-1", "--banding-window", "2"]
    assert main([*argv, *flags, "--dataset-from-flux", "1e9"]) == outcome.EXIT_BANDED
    with open(os.path.join(run, outcome.OUTCOME_FILE)) as fh:
        doc = json.load(fh)
    assert doc["class"] == "banded" and doc["stopped"] == "banded" and doc["banded_t"] > 0
    # the frames waited for a flux never reached; the stop frame is kept
    meta = _meta(ds.trajectory_path(root, "tiny"))
    assert meta["solver_steps"].tolist() == [4] and meta["trace_nyq_zonal"].shape == (2,)


def test_deferred_frames_of_a_quiet_run_keep_the_last(tmp_path, config):
    run, root = str(tmp_path / "run"), str(tmp_path / "data")
    argv = ["run", config, *COMMON, "--n-steps", "4", "--output-dir", run, "--dataset", root]
    assert main([*argv, "--dataset-from-flux", "1e9"]) == 0
    meta = _meta(ds.trajectory_path(root, "tiny"))
    assert meta["solver_steps"].tolist() == [4] and meta["trace_solver_steps"].tolist() == [2, 4]
