import os
import re
import pytest
import numpy as np
from gyaradax import load_geometry
from gyaradax.jax_config import enable_x64

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
    from gyaradax.params import gkparams_from_config, load_config
    from gyaradax.simulate import gk_init

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


GKW_DATA_ROOT = os.environ.get(
    "GKW_DATA_ROOT", os.path.join(os.path.dirname(__file__), "data", "gkw_raw")
)
ITERATIONS = [8, 13, 131, 200]


@pytest.fixture(params=ITERATIONS)
def adiabatic_dir(request):
    """base directory for adiabatic electron simulations."""
    path = os.path.join(GKW_DATA_ROOT, f"iteration_{request.param}")
    if not os.path.exists(path):
        pytest.skip(f"adiabatic reference data not found at {path}")
    return path


@pytest.fixture(params=ITERATIONS)
def lin_dir(request):
    """directory for linear-only adiabatic simulations."""
    path = os.path.join(GKW_DATA_ROOT, f"iteration_{request.param}_Lin")
    if not os.path.exists(path):
        pytest.skip(f"linear reference data not found at {path}")
    return path


@pytest.fixture(params=ITERATIONS)
def nonlin_dir(request):
    """directory for nonlinear adiabatic simulations."""
    path = os.path.join(GKW_DATA_ROOT, f"iteration_{request.param}")
    if not os.path.exists(path):
        pytest.skip(f"nonlinear reference data not found at {path}")
    return path


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


KINETIC_CASES = [
    "v3_kiteration_991_half_rlt",
    "v3_kiteration_991_ntsks128",
    "v3_kiteration_991_double_rlt",
]


@pytest.fixture(params=KINETIC_CASES)
def kinetic_dir(request):
    """Directory for kinetic electron simulations."""
    path = os.path.join(GKW_DATA_ROOT, "kinetic_electrons", request.param)
    if not os.path.exists(path):
        pytest.skip(f"kinetic reference data not found at {path}")
    return path


@pytest.fixture
def kinetic_geom(kinetic_dir):
    return load_geometry(kinetic_dir)


@pytest.fixture
def kinetic_shape(kinetic_geom):
    return _get_shape(kinetic_geom)
