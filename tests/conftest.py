import os
import re
import pytest
import numpy as np
from gyaradax import load_geometry
from gyaradax.io import reference_data as refdata
from gyaradax.core.jax_config import enable_x64

enable_x64()

# ── Backend registry ─────────────────────────────────────────────────────────
# Centralised backend lists used by all test files via:
#   from conftest import JAX_BACKENDS, CUDA_BACKENDS, ALL_BACKENDS
# Tuple format: (backend, use_z2z, mixed_precision)

try:
    from gyaradax.backends._cuda import is_available as _cuda_available

    HAS_CUDA = _cuda_available()
except ImportError:
    HAS_CUDA = False

JAX_BACKENDS = [
    ("jax", False, False),  # JAX R2C FP64
    ("jax", False, True),  # JAX R2C MP
    ("jax", True, False),  # JAX Z2Z FP64
    ("jax", True, True),  # JAX Z2Z MP
]

CUDA_BACKENDS = [
    ("cuda", False, False),  # CUDA Z2Z FP64
    ("cuda", False, True),  # CUDA Z2Z MP
]

ALL_BACKENDS = JAX_BACKENDS + (CUDA_BACKENDS if HAS_CUDA else [])


def rel_l2(pred, ref, eps=1e-30):
    """relative l2 error between two arrays."""
    return float(
        np.linalg.norm(np.asarray(pred) - np.asarray(ref)) / (np.linalg.norm(np.asarray(ref)) + eps)
    )


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def noisy_case(cfg_name, grid, overrides=None, seed=0, **param_kw):
    """(df, geometry, params, state) for a repo config on a reduced grid with analytic geometry.

    ``param_kw`` goes to ``gkparams_from_config``, ``overrides`` replaces fields afterwards;
    df gets a seeded 1e-4 complex perturbation so every mode and species is exercised.
    """
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    from gyaradax.geometry import compute_geometry_from_config
    from gyaradax.core.params import gkparams_from_config, load_config
    from gyaradax.solver.driver import gk_init

    cfg = load_config(os.path.join(REPO_ROOT, "configs", cfg_name))
    for key, value in grid.items():
        cfg.grid[key] = value
    params = replace(gkparams_from_config(cfg, **param_kw), **(overrides or {}))
    geometry = compute_geometry_from_config(cfg)
    nsp = 1 if params.adiabatic_electrons else int(np.asarray(params.mas).shape[0])
    df, geometry, state = gk_init(geometry, params, n_species=nsp)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    noise = jax.random.normal(k1, df.shape) + 1j * jax.random.normal(k2, df.shape)
    return (df + 1e-4 * noise).astype(jnp.complex128), geometry, params, state


def read_dump_time(dat_path):
    """read simulation TIME from a gkw .dat metadata file."""
    with open(dat_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"TIME\s*=\s*([0-9eE+\-.]+)", text)
    if m is None:
        raise ValueError(f"TIME not found in {dat_path}")
    return float(m.group(1))


def read_dump_dtim(dat_path):
    """read the actual DTIM from a gkw dump .dat metadata file."""
    with open(dat_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"DTIM\s*=\s*([0-9eE+\-.]+)", text)
    if m is None:
        raise ValueError(f"DTIM not found in {dat_path}")
    return float(m.group(1))


# ── GKW reference runs ───────────────────────────────────────────────────────
# Cases and dump pairs come from tests/data/gkw_reference.json, a copy of the
# manifest of the Hugging Face dataset (gyaradax.io.reference_data). They are
# read from $GYARADAX_TEST_DATA and downloaded only with --download-test-data,
# GYARADAX_DOWNLOAD_TEST_DATA=1 or download_test_data = true in the pytest ini
# options; otherwise the tests that need them skip.

REFERENCE = refdata.load_manifest(os.path.join(os.path.dirname(__file__), "data", "gkw_reference.json"))
NONLINEAR_CASES = sorted(REFERENCE["cases"]["nonlinear"])
LINEAR_CASES = sorted(REFERENCE["cases"]["linear"])
KINETIC_CASES = sorted(REFERENCE["cases"]["kinetic"])
PHASES = ("early", "mid", "late")


def pytest_addoption(parser):
    parser.addoption(
        "--download-test-data",
        action="store_true",
        help=f"download missing GKW reference cases from {refdata.REPO_ID} into ${refdata.ENV_ROOT}",
    )
    parser.addini(
        "download_test_data",
        type="bool",
        default=False,
        help="download missing GKW reference cases, as --download-test-data",
    )


def _reference_dir(request, kind):
    config = request.config
    download = (
        config.getoption("--download-test-data")
        or config.getini("download_test_data")
        or refdata.download_requested()
    )
    path = refdata.case_dir(REFERENCE, kind, request.param, download=download)
    if path is None:
        pytest.skip(
            f"GKW reference case {kind}/{request.param} not found under {refdata.data_root()}; "
            "pass --download-test-data to fetch it"
        )
    return path


def reference_case(path):
    """Manifest entry of a reference case directory."""
    kind, name = os.path.split(os.path.normpath(path))[-2:]
    kind = os.path.basename(kind)
    return REFERENCE["cases"][kind][name]


def reference_pair(path, phase):
    """(start, end) dump names of the early, mid or late pair of a case."""
    pair = reference_case(path)["pairs"][phase]
    return pair["start"], pair["end"]


def reference_dumps(path):
    """All dump names of a case, in time order."""
    return [name for phase in PHASES for name in reference_pair(path, phase)]


@pytest.fixture(params=NONLINEAR_CASES)
def adiabatic_dir(request):
    """nonlinear adiabatic-electron reference case."""
    return _reference_dir(request, "nonlinear")


@pytest.fixture(params=LINEAR_CASES)
def lin_dir(request):
    """linear adiabatic-electron reference case."""
    return _reference_dir(request, "linear")


@pytest.fixture(params=NONLINEAR_CASES)
def nonlin_dir(request):
    """nonlinear adiabatic-electron reference case."""
    return _reference_dir(request, "nonlinear")


@pytest.fixture
def adiabatic_geom(adiabatic_dir):
    return load_geometry(adiabatic_dir)


@pytest.fixture
def lin_geom(lin_dir):
    return load_geometry(lin_dir)


@pytest.fixture
def nonlin_geom(nonlin_dir):
    return load_geometry(nonlin_dir)


def _get_shape(geom):
    return (
        len(geom["intvp"]),
        len(geom["intmu"]),
        len(geom["ints"]),
        len(geom["kxrh"]),
        len(geom["krho"]),
    )


@pytest.fixture
def adiabatic_shape(adiabatic_geom):
    return _get_shape(adiabatic_geom)


@pytest.fixture
def lin_shape(lin_geom):
    return _get_shape(lin_geom)


@pytest.fixture
def nonlin_shape(nonlin_geom):
    return _get_shape(nonlin_geom)


@pytest.fixture(params=KINETIC_CASES)
def kinetic_dir(request):
    """kinetic-electron reference case."""
    return _reference_dir(request, "kinetic")


@pytest.fixture
def kinetic_geom(kinetic_dir):
    return load_geometry(kinetic_dir)


@pytest.fixture
def kinetic_shape(kinetic_geom):
    return _get_shape(kinetic_geom)
