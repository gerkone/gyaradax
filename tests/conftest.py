import os
import re
import pytest
import jax
import numpy as np
from gyaradax import load_geometry

jax.config.update("jax_enable_x64", True)


def rel_l2(pred, ref, eps=1e-30):
    """relative l2 error between two arrays."""
    return float(
        np.linalg.norm(np.asarray(pred) - np.asarray(ref))
        / (np.linalg.norm(np.asarray(ref)) + eps)
    )


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


# standard iterations for verification across different parameter regimes
ITERATIONS = [8, 13, 131, 200]


@pytest.fixture(params=ITERATIONS)
def adiabatic_dir(request):
    """base directory for adiabatic electron simulations."""
    path = f"/restricteddata/ukaea/gyrokinetics/raw/iteration_{request.param}"
    if not os.path.exists(path):
        pytest.skip(f"adiabatic reference data not found at {path}")
    return path


@pytest.fixture(params=ITERATIONS)
def lin_dir(request):
    """directory for linear-only adiabatic simulations."""
    path = f"/restricteddata/ukaea/gyrokinetics/raw/iteration_{request.param}_Lin"
    if not os.path.exists(path):
        pytest.skip(f"linear reference data not found at {path}")
    return path


@pytest.fixture(params=ITERATIONS)
def nonlin_dir(request):
    """directory for nonlinear adiabatic simulations."""
    path = f"/restricteddata/ukaea/gyrokinetics/raw/iteration_{request.param}"
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
    path = f"/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons/{request.param}"
    if not os.path.exists(path):
        pytest.skip(f"kinetic reference data not found at {path}")
    return path


@pytest.fixture
def kinetic_geom(kinetic_dir):
    return load_geometry(kinetic_dir)


@pytest.fixture
def kinetic_shape(kinetic_geom):
    return _get_shape(kinetic_geom)
