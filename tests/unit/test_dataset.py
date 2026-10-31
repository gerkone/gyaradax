"""gyaradax run --dataset: preprocessed frames, metadata, statistics and resume."""

import os
import pickle
import shutil

import numpy as np
import pytest
from omegaconf import OmegaConf

from gyaradax.io import dataset as ds
from gyaradax.io.schema import TREE_PATHS, SCHEMA_VERSION, flatten_metadata, nest_metadata
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
        tree = pickle.load(fh)
    assert tree["schema"] == SCHEMA_VERSION and "extra" not in tree
    assert tree["frames"]["solver_steps"].tolist() == [2, 4] and tree["params"]["n_species"] in (1, 2)
    meta = flatten_metadata(tree)
    assert meta.keys() == ds.load_metadata(traj).keys()
    assert meta["solver_steps"].tolist() == [2, 4]
    n_species = 2 if kind == "kinetic" else 1
    assert meta["n_species"] == n_species and len(meta["species"]) == n_species
    assert meta["df_shape"][:2] == (2, n_species) and meta["flux"].shape == (2, n_species, 3)
    assert meta["electron_species_index"] == (1 if kind == "kinetic" else -1)
    for key in ("ion_temp_grad", "density_grad", "electron_temp_grad", "temp_ratio", "eps", "coll_freq"):
        assert meta[key].shape == (1,)
    assert meta["df_mean"].ndim == len(meta["df_shape"]) and meta["phi_mean"].shape == meta["phi_shape"]
    assert meta["quantization"] == "bf16"
    assert set(meta["geometry"]) == {*ds.GEOMETRY_KEYS, "adiabatic", "beta", "nlapar", "nlbpar"}
    with open(os.path.join(traj, "metadata_light.pkl"), "rb") as fh:
        assert not {"df_mean", "phi_var"} & set(flatten_metadata(pickle.load(fh)))
    # bf16 loss tracker: phi frame rounding, and phi / heat fluxes recomputed from the stored df
    assert meta["quant_flux_err"].shape == meta["flux"].shape
    assert np.all((meta["quant_phi_err"] > 1e-4) & (meta["quant_phi_err"] < 1e-2))
    assert np.all((meta["quant_phi_df_err"] > 1e-5) & (meta["quant_phi_df_err"] < 1e-1))
    eflux = list(meta["flux_labels"]).index("eflux")
    assert np.all(np.abs(meta["quant_flux_err"][..., eflux]) < 5e-2)

    # the last frame is the bf16 real-space transform of the final state
    with np.load(os.path.join(run, "step_000004.npz")) as ck:
        want_df, want_phi = ds.df_frame_bits(ck["df"]), ds._to_bf16_bits(ds.phi_to_real(ck["phi"]))
        want_phic = ds.phi_to_reim(ck["phi"])
    got_df = ds.read_frame(ds.frame_path(traj, "timestep", 1, "zstd16"), meta["df_shape"])
    got_phi = ds.read_frame(ds.frame_path(traj, "poten", 1, "bf16"), meta["phi_shape"])
    got_phic = np.fromfile(ds.potenc_path(traj, 1), np.float32).reshape(2, *meta["phi_shape"])
    np.testing.assert_array_equal(got_df.view(np.uint32).ravel() >> 16, want_df)
    np.testing.assert_array_equal(got_phi.view(np.uint32).ravel() >> 16, want_phi)
    np.testing.assert_array_equal(got_phic, want_phic)
    assert meta["phic_source"] == "solver_exact" and meta["phic_mean"].shape == want_phic.shape

    # resuming the run directory appends to the same trajectory
    assert main(["run", run, "--n-steps", "2"]) == 0
    with open(os.path.join(traj, "metadata.pkl"), "rb") as fh:
        meta = flatten_metadata(pickle.load(fh))
    assert meta["solver_steps"].tolist() == [2, 4, 6]
    assert OmegaConf.load(os.path.join(run, "config.yaml")).dataset.name == f"tiny_{kind}"

    # statistics can be recomputed over a later window: frames 1 and 2 only
    assert main(["dataset-stats", traj, "--from-time", str(float(meta["timesteps"][1]))]) == 0
    with open(os.path.join(traj, "metadata.pkl"), "rb") as fh:
        meta = flatten_metadata(pickle.load(fh))
    phis = [ds.read_frame(ds.stored_frame(traj, "poten", i), meta["phi_shape"]) for i in (1, 2)]
    np.testing.assert_allclose(meta["phi_mean"], np.mean(phis, axis=0), rtol=1e-3, atol=1e-6)
    assert meta["phi_mean"].dtype == np.float32
    phics = [np.fromfile(ds.potenc_path(traj, i), np.float32).reshape(2, *meta["phi_shape"]) for i in (1, 2)]
    np.testing.assert_allclose(meta["phic_mean"], np.mean(phics, axis=0), rtol=1e-3, atol=1e-6)


def test_reim_phi_frames_keep_every_ky_mode():
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((3, 7, 6)) + 1j * rng.standard_normal((3, 7, 6))
    frame = ds.phi_to_reim(phi)
    assert frame.shape == (2, 7, 3, 6)
    np.testing.assert_allclose(ds.reim_to_phi(frame), phi, atol=1e-6)
    # the real frame keeps ky <= nky / 2 only
    real = np.moveaxis(np.fft.fftshift(np.fft.rfftn(ds.phi_to_real(phi), axes=(0, 2), norm="forward"), axes=0), 1, 0)
    np.testing.assert_allclose(real[..., 1:3], phi[..., 1:3], atol=1e-6)
    assert real.shape[-1] == 4


def test_writer_adds_potenc_and_refuses_old_layouts(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "_static_metadata", lambda *args: {"phi_shape": (6, 3, 5)})
    config = tmp_path / "config.yaml"
    config.write_text("run: {}\n")
    root = str(tmp_path / "data")
    plain = ds.DatasetWriter(root, "plain", {}, None, str(config))
    assert plain.potenc and plain.static["phic_source"] == "solver_exact"
    bare = ds.DatasetWriter(root, "bare", {}, None, str(config), potenc=False)
    assert not bare.potenc and "phic_source" not in bare.static
    ds._claim(ds.trajectory_path(root, "old"), str(tmp_path))
    old = {"phi_shape": (2, 6, 3, 5), "phi_layout": "reim"}
    ds.write_metadata(ds.trajectory_path(root, "old"), old)
    with pytest.raises(ValueError, match="predates the schema"):
        ds.DatasetWriter(root, "old", {}, None, str(config))
    ds.write_metadata(ds.trajectory_path(root, "old"), {**old, "species": ("ion",)})
    with pytest.raises(ValueError, match="complex poten"):
        ds.DatasetWriter(root, "old", {}, None, str(config))


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


def _tiny_kinetic(tmp_path, name):
    src, grid = CASES["kinetic"]
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", src))
    for key, value in grid.items():
        cfg.grid[key] = value
    cfg.run.pop("data_dir", None)
    cfg.run.name = name
    cfg.solver.nlapar = False
    path = str(tmp_path / f"{name}.yaml")
    OmegaConf.save(cfg, path)
    return path


def test_trajectory_extends_from_its_resume_dump(tmp_path):
    root = str(tmp_path / "data")
    common = ["--from-scratch", "--backend", "jax", "--dump-dt", "0.013", "--dataset", root]
    # reference: one leg to t = 0.065
    ref = _tiny_kinetic(tmp_path, "ref")
    assert main(["run", ref, *common, "--t-end", "0.065", "--output-dir", str(tmp_path / "ref_run")]) == 0
    # two legs: to t = 0.026 with an f64 resume dump, then from the trajectory alone
    case = _tiny_kinetic(tmp_path, "case")
    first = str(tmp_path / "first_run")
    assert main(["run", case, *common, "--t-end", "0.026", "--resume-dtype", "f64", "--output-dir", first]) == 0
    traj = ds.trajectory_path(root, "case")
    with np.load(ds.resume_path(traj)) as r:
        assert r["df"].dtype == np.complex128 and float(r["time"]) == 2 * 0.013 and int(r["frame"]) == 1
    shutil.rmtree(first)
    second = str(tmp_path / "second_run")
    assert main(["run", traj, "--t-end", "0.065", "--output-dir", second]) == 0

    meta, want = ds.load_metadata(traj), ds.load_metadata(ds.trajectory_path(root, "ref"))
    np.testing.assert_array_equal(meta["timesteps"], want["timesteps"])
    np.testing.assert_array_equal(meta["solver_steps"], want["solver_steps"])
    # f64 resume: the extension equals the uninterrupted run bitwise
    np.testing.assert_array_equal(meta["flux"], want["flux"])
    for i in range(len(meta["timesteps"])):
        for kind, shape in (("timestep", meta["df_shape"]), ("poten", meta["phi_shape"])):
            np.testing.assert_array_equal(
                ds.read_frame(ds.stored_frame(traj, kind, i), shape),
                ds.read_frame(ds.stored_frame(ds.trajectory_path(root, "ref"), kind, i), shape),
            )
        np.testing.assert_array_equal(
            np.fromfile(ds.potenc_path(traj, i), np.float32),
            np.fromfile(ds.potenc_path(ds.trajectory_path(root, "ref"), i), np.float32),
        )
    with np.load(ds.resume_path(traj)) as r, np.load(os.path.join(second, "step_000015.npz")) as ck:
        # the leg's own resume dump keeps the dtype its config carries
        assert r["df"].dtype == np.complex128 and int(r["frame"]) == 4
        np.testing.assert_array_equal(r["df"], ck["df"])
    assert "df_mean" in meta
    # the trajectory now belongs to the extending run
    with open(os.path.join(traj, ds.OWNER_FILE)) as fh:
        assert fh.read().strip() == second


def test_f32_resume_dump_extends_the_trajectory_on_the_same_grid(tmp_path):
    root = str(tmp_path / "data")
    case = _tiny_kinetic(tmp_path, "case")
    run = str(tmp_path / "run")
    common = ["--from-scratch", "--backend", "jax", "--dump-dt", "0.013", "--dataset", root]
    assert main(["run", case, *common, "--t-end", "0.026", "--output-dir", run]) == 0
    traj = ds.trajectory_path(root, "case")
    with np.load(ds.resume_path(traj)) as r, np.load(os.path.join(run, "step_000006.npz")) as ck:
        assert r["df"].dtype == np.complex64
        np.testing.assert_array_equal(r["df"], ck["df"].astype(np.complex64))
    assert main(["run", "--resume-from", traj, os.path.join(traj, "resume", "config.yaml"),
                 "--n-blocks", "2", "--output-dir", str(tmp_path / "more")]) == 0
    meta = ds.load_metadata(traj)
    np.testing.assert_array_equal(meta["timesteps"], np.arange(1, 5) * 0.013)
    # the original run directory no longer owns the trajectory
    with pytest.raises(FileExistsError, match="belongs to the run"):
        main(["run", run, "--t-end", "0.1"])


def test_attach_linear_adds_only_the_linear_keys_and_files(tmp_path):
    rng = np.random.default_rng(0)
    df_shape, phi_shape, nky = (2, 2, 4, 2, 3, 6, 5), (6, 3, 5), 5
    traj, staged = tmp_path / "iteration_1_ifft_realpotens", tmp_path / "staged"
    (traj / "data").mkdir(parents=True)
    frame = traj / "data" / "timestep_00000.zstd16.bin"
    frame.write_bytes(b"frame")
    meta = {"df_shape": df_shape, "phi_shape": phi_shape, "timesteps": np.arange(3.0), "flux": 1.0}
    stale = {"linear_k_bin": "data/linear_k.bin", "linear_k_bin_shape": np.asarray(df_shape)}
    original = {}
    for name in ("metadata.pkl", "metadata_light.pkl"):
        original[name] = pickle.dumps({**meta, **stale, "name": name})
        (traj / name).write_bytes(original[name])
    spec = rng.standard_normal(df_shape[1:]) + 1j * rng.standard_normal(df_shape[1:])
    phi = rng.standard_normal((3, 6, 5)) + 1j * rng.standard_normal((3, 6, 5))
    assert ds.linear_frames(spec[0], phi)[0].shape == (2, 1, *df_shape[2:])
    frames = ds.linear_frames(spec, phi)
    ds.write_linear_frames(str(staged / "data"), frames)
    flux_ky = rng.standard_normal((2, 3, nky))
    phi2 = rng.uniform(1, 2, nky)
    gate = {
        "t": 30.0, "ky": list(range(nky)), "gamma": [0.1] * nky, "gamma_std": [0.0] * nky,
        "omega": [0.2] * nky, "omega_std": [0.0] * nky, "settled": True, "peak_ky": 2.0,
        "gamma_max": 0.1, "decision": "unstable", "dt": 0.01,
        "mode_fluxes": {"flux": flux_ky.sum(-1).tolist(), "flux_ky": flux_ky.tolist(),
                        "phi2_ky": phi2.tolist(), "weight_ky_allkx": (flux_ky / phi2).tolist(),
                        "flux_ky_chain0": (flux_ky / 2).tolist(), "phi2_ky_chain0": phi2.tolist(),
                        "weight_ky": (flux_ky / 2 / phi2).tolist()},
    }  # fmt: skip

    entries = ds.attach_linear(str(traj), str(staged), gate, 30.5, "20260101")
    again = ds.attach_linear(str(traj), str(staged), gate, 30.5, "20260101")
    assert sorted(entries) == sorted(again) and all(k.startswith("linear_") for k in entries)
    for name in ("metadata.pkl", "metadata_light.pkl"):
        with open(traj / name, "rb") as fh:
            doc = flatten_metadata(pickle.load(fh))
        assert (traj / f"{name}.bak-linear-20260101").read_bytes() == original[name]
        assert set(doc) == set(meta) | {"name"} | set(entries) and not set(stale) & set(doc)
        assert doc["timesteps"].tolist() == [0.0, 1.0, 2.0] and doc["name"] == name
        assert doc["linear_flux_ky"].shape == (2, 3, nky) and doc["linear_dt"] == 0.01
        assert doc["linear_space"] == "spectral" and doc["linear_schema"] == ds.LINEAR_SCHEMA
        np.testing.assert_allclose(doc["linear_flux_ky"].sum(-1), doc["linear_flux"])
        np.testing.assert_allclose(doc["linear_weight_ky_allkx"] * phi2, doc["linear_flux_ky"])
        np.testing.assert_allclose(doc["linear_weight_ky"] * phi2, doc["linear_flux_ky_chain0"])
    assert frame.read_bytes() == b"frame"
    k = np.fromfile(traj / "data" / "linear.bin", np.float32).reshape(df_shape)
    np.testing.assert_array_equal(k[0] + 1j * k[1], spec.astype(np.complex64))
    pk = np.fromfile(traj / "data" / "linear_poten.bin", np.float32).reshape(2, 3, 6, 5)
    np.testing.assert_array_equal(pk[0] + 1j * pk[1], phi.astype(np.complex64))
    assert sorted(os.listdir(traj / "data")) == [
        "linear.bin", "linear_poten.bin", "timestep_00000.zstd16.bin"
    ]  # fmt: skip


def test_schema_tree_round_trips_and_matches_neugk():
    flat = {"timesteps": np.arange(2.0), "ion_temp_grad": np.ones(1), "ds": 0.1, "geometry": {"kxrh": np.ones(3)},
            "linear_hist_t": np.zeros(0), "linear_rms": np.float64(1.0), "df_mean": np.zeros(2),
            "df_stats_agg_axes": (2, 4), "sus_underresolved": True, "odd_key": 1}
    tree = nest_metadata(flat)
    assert tree["schema"] == SCHEMA_VERSION and tree["geometry"]["ds"] == 0.1 and "ds" not in flat["geometry"]
    assert tree["linear"]["history"]["t"].size == 0 and tree["stats"]["df"]["agg_axes"] == (2, 4)
    assert tree["flags"]["sus_underresolved"] and tree["extra"] == {"odd_key": 1}
    back = flatten_metadata(tree)
    assert back.keys() == flat.keys() and back["geometry"].keys() == flat["geometry"].keys()
    neugk = pytest.importorskip("neugk_jax.dataset.schema")
    assert neugk.SCHEMA_VERSION == SCHEMA_VERSION and neugk.TREE_PATHS == TREE_PATHS
