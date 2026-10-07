"""Device-mesh resolution of the gyaradax CLI (JAX-free)."""

import argparse

import pytest

from gyaradax.cli import _auto_mesh, _check_mesh, _mesh_request

KINETIC = dict(
    adiabatic=False,
    nsp=2,
    nvpar=64,
    nmu=16,
    n_gpus=0,
    n_gpus_sp=1,
    n_gpus_vp=1,
    n_gpus_mu=1,
)
ADIABATIC = dict(KINETIC, adiabatic=True, nsp=1, nvpar=32, nmu=8)


def _args(**kw):
    base = dict(n_gpus=0, n_gpus_sp=0, n_gpus_vp=0, n_gpus_mu=0, device_list=None)
    return argparse.Namespace(**{**base, **kw})


@pytest.mark.parametrize(
    "n, facts, expected",
    [
        (2, KINETIC, (2, 1, 1)),
        (8, KINETIC, (2, 1, 4)),
        (32, KINETIC, (2, 1, 16)),
        (64, KINETIC, (2, 2, 16)),
        (4, ADIABATIC, (1, 1, 4)),
        (16, ADIABATIC, (1, 2, 8)),
    ],
)
def test_auto_mesh_fills_species_then_mu_then_vpar(n, facts, expected):
    assert _auto_mesh(n, facts) == expected


def test_auto_mesh_rejects_counts_the_grid_cannot_take():
    with pytest.raises(SystemExit, match="cannot split 3 GPUs"):
        _auto_mesh(3, ADIABATIC)


def test_mesh_request_precedence():
    facts = dict(KINETIC, n_gpus_mu=4)
    assert _mesh_request(_args(n_gpus_vp=2), facts) == (1, 2, 4)
    assert _mesh_request(_args(n_gpus=8), facts) == (2, 1, 4)
    assert _mesh_request(_args(), facts) == (1, 1, 4)
    assert _mesh_request(_args(), dict(KINETIC, n_gpus=4)) == (2, 1, 2)
    assert _mesh_request(_args(device_list="0,1,2,3"), KINETIC) == (2, 1, 2)
    assert _mesh_request(_args(device_list="3"), KINETIC) == (1, 1, 1)


def test_check_mesh_explains_the_problem():
    with pytest.raises(SystemExit, match="does not divide nmu"):
        _check_mesh((1, 1, 3), KINETIC, None)
    with pytest.raises(SystemExit, match="only 2 are visible"):
        _check_mesh((2, 1, 2), KINETIC, 2)
    with pytest.raises(SystemExit, match="vpar points"):
        _check_mesh((1, 64, 1), KINETIC, None)
    _check_mesh((2, 2, 4), KINETIC, 16)
