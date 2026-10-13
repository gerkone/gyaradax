"""Fixed-dt runs: an input dt above the linear stability estimate is reduced to it, as in GKW."""

import os
from dataclasses import replace

import jax
import numpy as np
import pytest
from conftest import noisy_case  # type: ignore[import-not-found]
from omegaconf import OmegaConf

from gyaradax.cfl import estimate_linear_timestep
from gyaradax.cli import main
from gyaradax.precompute import linear_precompute
from gyaradax.solver import gksolve

GRID = dict(nvpar=8, nmu=2, ns=8, nkx=11, nky=5)
N_STEPS = 6
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _case(dt_factor):
    df, geometry, params, state = noisy_case(
        "kinetic.yaml", GRID, dict(non_linear=False, adaptive_dt=False, backend="jax")
    )
    pre = linear_precompute(geometry, params)
    dt_lin = float(estimate_linear_timestep(pre, params=params))
    return df, geometry, replace(params, dt=dt_factor * dt_lin), state, pre, dt_lin


def test_dt_above_the_estimate_runs_as_if_the_estimate_were_given():
    df, geometry, params, state, pre, dt_lin = _case(3.0)
    out = gksolve(df, geometry, params, state, N_STEPS, pre=pre, return_dt_info=True)
    ref = gksolve(df, geometry, replace(params, dt=dt_lin), state, N_STEPS, pre=pre)
    np.testing.assert_array_equal(np.asarray(out[3]["dt_used"]), dt_lin)
    np.testing.assert_array_equal(np.asarray(out[3]["dt_lin"]), dt_lin)
    assert float(out[2].time) == pytest.approx(N_STEPS * dt_lin, rel=1e-12)
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(ref[0]))
    # growth rates are taken over the time actually stepped
    np.testing.assert_array_equal(
        np.asarray(out[2].last_growth_rate), np.asarray(ref[2].last_growth_rate)
    )


def test_dt_below_the_estimate_is_kept():
    df, geometry, params, state, pre, _ = _case(0.5)
    out = gksolve(df, geometry, params, state, N_STEPS, pre=pre, return_dt_info=True)
    np.testing.assert_array_equal(np.asarray(out[3]["dt_used"]), params.dt)
    assert float(out[2].time) == pytest.approx(N_STEPS * params.dt, rel=1e-12)


def test_traced_precompute_caps_the_step_dt():
    df, geometry, params, state, pre, dt_lin = _case(3.0)
    run = jax.jit(
        lambda d, p: gksolve(d, geometry, params, state, N_STEPS, pre=p, return_dt_info=True)
    )
    traced = run(df, pre)
    concrete = gksolve(df, geometry, params, state, N_STEPS, pre=pre)
    np.testing.assert_allclose(np.asarray(traced[3]["dt_used"]), dt_lin, rtol=1e-14)
    assert float(traced[2].time) == pytest.approx(N_STEPS * dt_lin, rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(traced[0]), np.asarray(concrete[0]), rtol=1e-10, atol=1e-20
    )
    np.testing.assert_allclose(
        np.asarray(traced[2].last_growth_rate), np.asarray(concrete[2].last_growth_rate), rtol=1e-8
    )


def test_cli_reports_and_uses_the_reduced_dt(tmp_path, capsys):
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "kinetic.yaml"))
    for key, value in GRID.items():
        cfg.grid[key] = value
    cfg.solver.non_linear = False
    cfg.solver.adaptive_dt = False
    cfg.solver.dt = 1.0
    path = tmp_path / "linear.yaml"
    OmegaConf.save(cfg, path)
    out = str(tmp_path / "run")
    argv = [
        "run",
        str(path),
        "--from-scratch",
        "--backend",
        "jax",
        "--n-steps",
        "4",
        "--output-dir",
        out,
    ]
    assert main(argv) == 0
    assert "linear timestep automatically reduced: dt 1.0000e+00 ->" in capsys.readouterr().out
    snap = max(f for f in os.listdir(out) if f.startswith("step_") and f.endswith(".npz"))
    with np.load(os.path.join(out, snap)) as ck:
        assert 0.0 < float(ck["time"]) < 4 * 1.0 * 0.5
