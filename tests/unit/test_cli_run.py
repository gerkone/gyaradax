"""gyaradax run: run-directory contents, resume from a run directory, debug outputs."""

import json
import os
import signal
import subprocess
import sys
import time

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


def _flux_steps(run_dir):
    try:
        with np.load(os.path.join(run_dir, "fluxes.npz")) as f:
            return np.asarray(f["step"])
    except (OSError, ValueError, KeyError):
        return np.zeros(0)


def test_sigterm_snapshots_and_the_run_directory_resumes_bitwise(tmp_path, tiny_config):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    common = ["--from-scratch", "--block-size", "2", "--backend", "jax", "--n-steps", "400"]
    _run(tiny_config, *common, "--output-dir", a)
    proc = subprocess.Popen(
        [sys.executable, "-m", "gyaradax", "run", tiny_config, *common, "--output-dir", b],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 600
    while time.time() < deadline and proc.poll() is None and _flux_steps(b).max(initial=0) < 2:
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=600)
    assert proc.returncode == 75, out[-2000:]
    snaps = [f for f in os.listdir(b) if f.startswith("step_")]
    assert len(snaps) == 1 and 2 <= int(snaps[0][5:11]) < 400, snaps
    _run(b)
    with np.load(os.path.join(a, "step_000400.npz")) as ca, np.load(os.path.join(b, "step_000400.npz")) as cb:
        for key in STATE_KEYS:
            np.testing.assert_array_equal(cb[key], ca[key])


def _grid_times(run_dir):
    with np.load(os.path.join(run_dir, "fluxes.npz")) as f:
        return np.asarray(f["time"]), np.asarray(f["step"])


def test_dump_dt_puts_frames_on_an_equally_spaced_time_grid(tmp_path, tiny_config):
    out = str(tmp_path / "t")
    # dt 0.005: each frame takes two full steps and one shortened to 0.003
    _run(tiny_config, "--from-scratch", "--backend", "jax", "--dump-dt", "0.013", "--t-end", "0.06", "--output-dir", out)
    times, steps = _grid_times(out)
    np.testing.assert_array_equal(times, np.arange(6) * 0.013)
    assert steps.tolist() == [0, 3, 6, 9, 12, 15]
    with np.load(os.path.join(out, "dt_history.npz")) as h:
        np.testing.assert_allclose(h["dt_used"][:3], [0.005, 0.005, 0.003], rtol=1e-12)
    with np.load(os.path.join(out, "step_000015.npz")) as ck:
        assert float(ck["dump_origin"]) == 0.0 and float(ck["time"]) == 5 * 0.013
    eff = OmegaConf.load(os.path.join(out, "config.yaml"))
    assert eff.solver.dump_dt == 0.013 and eff.solver.t_end == 0.06
    with open(os.path.join(out, "run_info.jsonl")) as fh:
        assert json.loads(fh.readline())["target_step"] is None
    # the run directory already reached t_end
    assert main(["run", out]) == 0
    assert _grid_times(out)[1].tolist() == steps.tolist()


def test_time_mode_resume_continues_the_grid_bitwise(tmp_path, tiny_config):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    common = ["--from-scratch", "--backend", "jax", "--dump-dt", "0.013"]
    _run(tiny_config, *common, "--t-end", "0.065", "--output-dir", a)
    _run(tiny_config, *common, "--t-end", "0.026", "--output-dir", b)
    # two more frames: --n-blocks counts frames in time mode
    _run(b, "--n-blocks", "2")
    _run(b, "--t-end", "0.065")
    (ta, sa), (tb, sb) = _grid_times(a), _grid_times(b)
    np.testing.assert_array_equal(tb, ta)
    np.testing.assert_array_equal(sb, sa)
    with np.load(os.path.join(a, "step_000015.npz")) as ca, np.load(os.path.join(b, "step_000015.npz")) as cb:
        for key in STATE_KEYS:
            np.testing.assert_array_equal(cb[key], ca[key])
    with open(os.path.join(b, "run_info.jsonl")) as fh:
        assert [json.loads(line)["start_step"] for line in fh] == [0, 6, 12]


def test_sigterm_in_time_mode_stops_on_a_frame_and_resumes_the_grid(tmp_path, tiny_config):
    from gyaradax.io import dataset as ds

    cfg = OmegaConf.load(tiny_config)
    cfg.solver.nlapar = False
    config = str(tmp_path / "es.yaml")
    OmegaConf.save(cfg, config)
    run, root = str(tmp_path / "b"), str(tmp_path / "data")
    common = ["--from-scratch", "--backend", "jax", "--dump-dt", "0.013", "--t-end", "1.3", "--dataset", root]
    proc = subprocess.Popen(
        [sys.executable, "-m", "gyaradax", "run", config, *common, "--output-dir", run],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 600
    while time.time() < deadline and proc.poll() is None and _flux_steps(run).max(initial=0) < 6:
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=600)
    assert proc.returncode == 75, out[-2000:]
    snaps = [f for f in os.listdir(run) if f.startswith("step_")]
    with np.load(os.path.join(run, snaps[0])) as ck:
        t_stop = float(ck["time"])
    k = round(t_stop / 0.013)
    assert len(snaps) == 1 and 2 <= k < 100 and t_stop == k * 0.013, (snaps, t_stop)
    # the trajectory's resume dump holds the state of its last frame
    traj = ds.trajectory_path(root, cfg.run.name)
    meta = ds.load_metadata(traj)
    with np.load(ds.resume_path(traj)) as r:
        assert r["df"].dtype == np.complex64 and float(r["time"]) == t_stop
        assert int(r["frame"]) == len(meta["timesteps"]) - 1 and int(r["step"]) == int(meta["solver_steps"][-1])

    _run(run, "--t-end", str(t_stop + 3 * 0.013))
    times, _ = _grid_times(run)
    np.testing.assert_allclose(times, np.arange(k + 4) * 0.013, rtol=0, atol=1e-12)
    meta = ds.load_metadata(traj)
    np.testing.assert_allclose(meta["timesteps"], np.arange(1, k + 4) * 0.013, rtol=0, atol=1e-12)
