"""gyaradax run --dataset: preprocessed frames, metadata, statistics and resume."""

import os
import pickle

import numpy as np
import pytest
from omegaconf import OmegaConf

from gyaradax.io import dataset as ds
from gyaradax.cli import main

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CASES = {
    "kinetic": ("nl_em_waltz_b01.yaml", dict(nvpar=8, nmu=2, ns=8, nkx=11, nky=5)),
    "adiabatic": ("iteration_13.yaml", dict(nvpar=8, nmu=2, ns=8, nkx=11, nky=6)),
}


def test_zstd16_roundtrip_over_several_chunks():
    bits = np.random.default_rng(0).integers(0, 2**16, size=3 * ds.CHUNK_BYTES // 4 + 7, dtype=np.uint16)
    buf = ds.zstd16_encode(bits)
    assert buf[:4] == ds.MAGIC and len(buf) > ds.ALIGN
    np.testing.assert_array_equal(ds.zstd16_decode(buf), bits)


@pytest.mark.parametrize("nkx", [11, 12])
def test_real_to_df_inverts_df_to_real(nkx):
    rng = np.random.default_rng(nkx)
    df = rng.standard_normal((2, 3, nkx, 5)) + 1j * rng.standard_normal((2, 3, nkx, 5))
    np.testing.assert_allclose(np.asarray(ds.real_to_df(ds.df_to_real(df))), df, atol=1e-5)


@pytest.mark.parametrize("kind", list(CASES))
def test_run_writes_preprocessed_trajectory(tmp_path, kind):
    src, grid = CASES[kind]
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", src))
    for key, value in grid.items():
        cfg.grid[key] = value
    cfg.run.pop("data_dir", None)
    cfg.run.name = f"tiny_{kind}"
    if kind == "kinetic":
        cfg.solver.nlapar = False
    config = str(tmp_path / "tiny.yaml")
    OmegaConf.save(cfg, config)
    run, root = str(tmp_path / "run"), str(tmp_path / "data")
    common = ["--from-scratch", "--backend", "jax", "--block-size", "2", "--dataset", root]
    assert main(["run", config, *common, "--n-steps", "4", "--output-dir", run]) == 0

    traj = ds.trajectory_path(root, f"tiny_{kind}")
    with open(os.path.join(traj, "metadata.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    assert meta["solver_steps"].tolist() == [2, 4]
    assert meta["n_species"] == (2 if kind == "kinetic" else 1)
    assert meta["df_mean"].ndim == len(meta["df_shape"]) and meta["phi_mean"].shape == meta["phi_shape"]
    assert meta["quantization"] == "bf16"
    assert set(meta["geometry"]) == {*ds.GEOMETRY_KEYS, "adiabatic", "beta", "nlapar", "nlbpar"}
    with open(os.path.join(traj, "metadata_light.pkl"), "rb") as fh:
        assert not {"df_mean", "phi_var"} & set(pickle.load(fh))
    # bf16 loss tracker: phi frame rounding, and phi / heat fluxes recomputed from the stored df
    assert meta["quant_flux_err"].shape == meta["flux"].shape
    assert np.all((meta["quant_phi_err"] > 1e-4) & (meta["quant_phi_err"] < 1e-2))
    assert np.all((meta["quant_phi_df_err"] > 1e-5) & (meta["quant_phi_df_err"] < 1e-1))
    eflux = [i for i, label in enumerate(meta["flux_labels"]) if label.startswith("eflux")]
    assert np.all(np.abs(meta["quant_flux_err"][:, eflux]) < 5e-2)

    # the last frame is the bf16 real-space transform of the final state
    with np.load(os.path.join(run, "step_000004.npz")) as ck:
        want_df, want_phi = ds.df_frame_bits(ck["df"]), ds._to_bf16_bits(ds.phi_to_real(ck["phi"]))
    got_df = ds.read_frame(ds.frame_path(traj, "timestep", 1, "zstd16"), meta["df_shape"])
    got_phi = ds.read_frame(ds.frame_path(traj, "poten", 1, "zstd16"), meta["phi_shape"])
    np.testing.assert_array_equal(got_df.view(np.uint32).ravel() >> 16, want_df)
    np.testing.assert_array_equal(got_phi.view(np.uint32).ravel() >> 16, want_phi)

    # resuming the run directory appends to the same trajectory
    assert main(["run", run, "--n-steps", "2"]) == 0
    with open(os.path.join(traj, "metadata.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    assert meta["solver_steps"].tolist() == [2, 4, 6]
    assert OmegaConf.load(os.path.join(run, "config.yaml")).dataset.name == f"tiny_{kind}"

    # statistics can be recomputed over a later window: frames 1 and 2 only
    assert main(["dataset-stats", traj, "--from-time", str(float(meta["timesteps"][1]))]) == 0
    with open(os.path.join(traj, "metadata.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    phis = [ds.read_frame(ds.frame_path(traj, "poten", i, "zstd16"), meta["phi_shape"]) for i in (1, 2)]
    np.testing.assert_allclose(meta["phi_mean"], np.mean(phis, axis=0), rtol=1e-3, atol=1e-6)
    assert meta["phi_mean"].dtype == np.float32


def test_writer_refuses_trajectories_of_other_runs(tmp_path):
    root, mine, other = tmp_path / "data", str(tmp_path / "run_a"), str(tmp_path / "run_b")
    traj = ds.trajectory_path(str(root), "case")
    ds._claim(traj, mine)
    ds._claim(traj, mine)
    with pytest.raises(FileExistsError, match="belongs to the run"):
        ds._claim(traj, other)
    # a stored trajectory without an owner, and a symlink to one
    stored = ds.trajectory_path(str(root), "stored")
    os.makedirs(os.path.join(stored, "data"))
    open(os.path.join(stored, "metadata.pkl"), "wb").close()
    with pytest.raises(FileExistsError, match="not written by a gyaradax run"):
        ds._claim(stored, mine)
    os.symlink(stored, ds.trajectory_path(str(root), "linked"))
    with pytest.raises(FileExistsError, match="symlink"):
        ds._claim(ds.trajectory_path(str(root), "linked"), mine)
    assert sorted(os.listdir(stored)) == ["data", "metadata.pkl"]
