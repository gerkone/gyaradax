"""Characterization tests for current geometry construction contracts.

These tests freeze existing behavior before introducing a GeometrySpec /
GeometryModel abstraction.  They intentionally describe today's public wrapper
semantics, including historical default differences between direct Python,
GKW input.dat, YAML config, and loaded GKW reference geometry paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from gyaradax.geometry import (
    GeometrySpec,
    compute_geometry,
    compute_geometry_from_input,
    create_geometry,
    geometry_spec_from_compute_kwargs,
    geometry_spec_from_config,
    geometry_spec_from_input_dat,
    get_geometry_model,
    list_geometry_models,
)
from gyaradax.geometry.circular import CircularGeometryModel, SAlphaGeometryModel
from gyaradax.geometry.miller import MillerGeometryModel
from gyaradax.simulate import _geometry_from_config
from gyaradax.utils import load_geometry

jax.config.update("jax_enable_x64", True)


_BASE_KWARGS: dict[str, Any] = dict(
    q=1.4,
    shat=0.78,
    eps=0.19,
    ns=8,
    nkx=5,
    nky=3,
    nvpar=6,
    nmu=4,
    vpar_max=3.0,
    nperiod=1,
    krhomax=1.0,
)

_MILLER_KWARGS: dict[str, Any] = dict(
    kappa=1.4,
    delta=-0.3,
    square=0.2,
    Zmil=0.1,
    dRmil=-0.22,
    dZmil=-0.2,
    skappa=0.4,
    sdelta=0.8,
    ssquare=0.4,
    gradp=-0.2,
    gradp_type="alpha",
)

_FLOAT_CONTRACT: dict[str, tuple[int, ...]] = {
    "sgrid": (8,),
    "ints": (8,),
    "intvp": (6,),
    "vpgr": (6,),
    "intmu": (4,),
    "mugr": (4,),
    "kxrh": (5,),
    "krho": (3,),
    "parseval": (3,),
    "bn": (8,),
    "ffun": (8,),
    "gfun": (8,),
    "efun": (8,),
    "rfun": (8,),
    "bt_frac": (8,),
    "little_g": (8, 3),
    "dfun": (8, 3),
    "hfun": (8, 3),
    "ifun": (8, 3),
}

_INT_CONTRACT: dict[str, tuple[int, ...]] = {
    "mode_label": (5, 3),
    "ixplus": (5, 3),
    "ixminus": (5, 3),
    "ixzero": (),
    "iyzero": (),
    "s_shift": (9, 8, 5, 3),
    "kx_shift": (9, 8, 5, 3),
}


def _assert_arrays_equal(left: Any, right: Any, keys: tuple[str, ...]) -> None:
    for key in keys:
        np.testing.assert_allclose(np.asarray(left[key]), np.asarray(right[key]), err_msg=key)


def _minimal_input_dat(geom_body: str = "") -> str:
    return f"""
 &GRIDSIZE
 NX = 5
 N_s_grid = 8
 N_mu_grid = 4
 N_vpar_grid = 6
 NMOD = 3
 nperiod = 1
 /
 &MODE
 mode_box = .true.
 krhomax = 1.0
 ikxspace = 5
 /
 &GEOM
 SHAT = 0.78
 Q = 1.4
 EPS = 0.19
 {geom_body}
 /
"""


def test_geometry_spec_from_compute_kwargs_preserves_direct_api_shape() -> None:
    """GeometrySpec is a neutral copy of direct compute_geometry arguments."""
    spec = geometry_spec_from_compute_kwargs(**_BASE_KWARGS, geom_type="miller", **_MILLER_KWARGS)

    assert isinstance(spec, GeometrySpec)
    assert spec.model == "miller"
    assert spec.geom_type == "miller"
    assert spec.q == _BASE_KWARGS["q"]
    assert spec.model_params["kappa"] == _MILLER_KWARGS["kappa"]
    kwargs = spec.compute_kwargs()
    assert kwargs["geom_type"] == "miller"
    assert kwargs["nkx"] == _BASE_KWARGS["nkx"]
    assert kwargs["kappa"] == _MILLER_KWARGS["kappa"]


def test_geometry_registry_has_current_analytic_models() -> None:
    """Batch-B registry exposes current models without changing behavior."""
    assert list_geometry_models() == ("circ", "miller", "s-alpha")


def test_registry_entries_use_dedicated_model_adapters() -> None:
    """Current analytic geometries are registered through distinct adapters."""
    circ_model = get_geometry_model("circ")
    salpha_model = get_geometry_model("s-alpha")
    assert isinstance(circ_model, CircularGeometryModel)
    assert isinstance(salpha_model, SAlphaGeometryModel)
    assert isinstance(get_geometry_model("miller"), MillerGeometryModel)
    assert hasattr(circ_model, "continuous_geometry")
    assert hasattr(salpha_model, "continuous_geometry")


def test_create_geometry_from_spec_matches_compute_geometry() -> None:
    """Spec/factory path is behavior-equivalent to direct compute_geometry."""
    spec = geometry_spec_from_compute_kwargs(**_BASE_KWARGS, geom_type="circ")
    from_spec = create_geometry(spec)
    direct = compute_geometry(**_BASE_KWARGS, geom_type="circ")

    _assert_arrays_equal(from_spec, direct, ("bn", "efun", "little_g", "krho", "kxrh"))


def test_compute_geometry_default_geom_type_is_circ() -> None:
    """Direct Python compute_geometry defaults to Lapillonne circular geometry."""
    default = compute_geometry(**_BASE_KWARGS)
    explicit_circ = compute_geometry(**_BASE_KWARGS, geom_type="circ")
    explicit_salpha = compute_geometry(**_BASE_KWARGS, geom_type="s-alpha")

    _assert_arrays_equal(default, explicit_circ, ("bn", "efun", "little_g", "krho", "kxrh"))
    assert not np.allclose(np.asarray(default["krho"]), np.asarray(explicit_salpha["krho"]))


def test_compute_geometry_from_input_absent_geom_type_defaults_to_s_alpha(tmp_path: Path) -> None:
    """GKW input.dat path defaults missing geom_type to s-alpha, not circ."""
    input_dat = tmp_path / "input.dat"
    input_dat.write_text(_minimal_input_dat(), encoding="utf-8")

    spec = geometry_spec_from_input_dat(str(input_dat))
    assert spec.model == "s-alpha"
    from_input = compute_geometry_from_input(str(input_dat))
    from_spec = create_geometry(spec)
    explicit_salpha = compute_geometry(**_BASE_KWARGS, geom_type="s-alpha")
    explicit_circ = compute_geometry(**_BASE_KWARGS, geom_type="circ")

    _assert_arrays_equal(from_input, from_spec, ("bn", "efun", "little_g", "krho", "kxrh"))
    _assert_arrays_equal(from_input, explicit_salpha, ("bn", "efun", "little_g", "krho", "kxrh"))
    assert not np.allclose(np.asarray(from_input["krho"]), np.asarray(explicit_circ["krho"]))


def test_config_geometry_absent_geometry_model_defaults_to_circ() -> None:
    """YAML/config geometry path currently inherits compute_geometry's circ default."""
    cfg = OmegaConf.create(
        {
            "geometry": {"q": 1.4, "shat": 0.78, "eps": 0.19},
            "grid": {
                "ns": 8,
                "nkx": 5,
                "nky": 3,
                "nvpar": 6,
                "nmu": 4,
                "vpar_max": 3.0,
                "nperiod": 1,
                "krhomax": 1.0,
            },
        }
    )

    spec = geometry_spec_from_config(cfg)
    assert spec.model == "circ"
    from_config = _geometry_from_config(cfg)
    from_spec = create_geometry(spec)
    explicit_circ = compute_geometry(**_BASE_KWARGS, geom_type="circ")
    explicit_salpha = compute_geometry(**_BASE_KWARGS, geom_type="s-alpha")

    _assert_arrays_equal(from_config, from_spec, ("bn", "efun", "little_g", "krho", "kxrh"))
    _assert_arrays_equal(from_config, explicit_circ, ("bn", "efun", "little_g", "krho", "kxrh"))
    assert not np.allclose(np.asarray(from_config["krho"]), np.asarray(explicit_salpha["krho"]))


def test_load_geometry_reads_reference_geom_dat_not_analytic_geometry(tmp_path: Path) -> None:
    """Loaded GKW geometry and analytic compute-from-input remain distinct sources.

    load_geometry must honor geom.dat contents.  compute_geometry_from_input may
    read some grid files, but must not source tensor profiles from geom.dat.
    """
    repo_root = Path(__file__).resolve().parents[2]
    case_root = repo_root / "tests" / "data" / "gkw_cases" / "zonal_flow"
    reference_dir = case_root / "reference"
    run_dir = tmp_path / "gkw_run"
    shutil.copytree(reference_dir, run_dir)
    shutil.copy(case_root / "input.dat", run_dir / "input.dat")

    geom_dat = run_dir / "geom.dat"
    text = geom_dat.read_text(encoding="utf-8")
    original_bn0 = "1.05157E+00"
    assert original_bn0 in text, (
        "fixture geom.dat format changed; update this source-semantics test"
    )
    geom_dat.write_text(text.replace(original_bn0, "9.87654E+00", 1), encoding="utf-8")

    loaded = load_geometry(str(run_dir))
    computed = compute_geometry_from_input(str(run_dir / "input.dat"))

    assert np.isclose(float(loaded["bn"][0]), 9.87654)
    assert not np.isclose(float(computed["bn"][0]), 9.87654)


def test_compute_geometry_representative_keys_shapes_dtypes_for_models() -> None:
    """Analytic geometry models return the current shared dict contract."""
    cases: list[tuple[str, dict[str, Any]]] = [
        ("circ", {}),
        ("s-alpha", {}),
        ("miller", _MILLER_KWARGS),
    ]

    for geom_type, extra_kwargs in cases:
        geom = compute_geometry(**_BASE_KWARGS, geom_type=geom_type, **extra_kwargs)

        for key, shape in _FLOAT_CONTRACT.items():
            assert key in geom, f"{geom_type}: missing {key}"
            assert geom[key].shape == shape, f"{geom_type}: {key} shape"
            assert geom[key].dtype == jnp.float64, f"{geom_type}: {key} dtype"

        for key, shape in _INT_CONTRACT.items():
            assert key in geom, f"{geom_type}: missing {key}"
            assert geom[key].shape == shape, f"{geom_type}: {key} shape"
            assert geom[key].dtype == jnp.int32, f"{geom_type}: {key} dtype"

        assert geom["pos_par_grid_class"].shape == (8, 5, 3)
        assert geom["pos_par_grid_class"].dtype == jnp.int8
        assert geom["valid_shift"].shape == (9, 8, 5, 3)
        assert geom["valid_shift"].dtype == jnp.bool_
