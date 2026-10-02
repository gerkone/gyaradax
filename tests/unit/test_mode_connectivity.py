"""gyaradax's generated mode connectivity vs GKW's own mode_label dumps.

The spectral parallel ("twist-shift") boundary condition connects kx to
kx + (2*nperiod-1)*|q*shat*ky/eps| after one pass through the parallel domain
(GKW mode.f90:776), over a radial grid of spacing
|q*shat*ky_min/(eps*ikxspace)| (mode.f90:698). The kx-index stride is
therefore (2*nperiod-1)*ikxspace*iy and grows with the toroidal mode number.

Regression guard: a fixed ikxspace stride at every ky makes mode iy see an
effective shear of shat/iy and gives it an over-long extended ballooning
domain. That under-damps the high-ky growth-rate tail (~0.16 in gamma against
GKW, up to ~47% of the peak) while leaving ky<=15 untouched, so it survives
any spot check on the peak growth rate.

Nothing else in the suite covers this. The other tests that touch mode_label
read GKW's file rather than gyaradax's, and the geometry parity test compares
against a pre-refactor gyaradax snapshot that shares the same bug. geom.dat,
where the metric and drift geometry is validated, carries no connectivity at
all: it is written by geom.f90, the connectivity by mode.f90.
"""

import os

import numpy as np
import pytest

from gyaradax.geometry.grids import _build_mode_label


GKW_CASES = os.path.join(os.path.dirname(__file__), "..", "data", "gkw_cases")

# (case dir, nkx, nky, ikxspace) from each case's input.dat
_CASES = [
    ("nl_em_b005_minrepro", 3, 2, 1),
    ("nl_em_b005_minrepro_5_3", 5, 3, 1),
    ("nl_em_b005_minrepro_7_4", 7, 4, 1),
    ("lin_em_waltz_b005_multikx", 11, 8, 5),
    ("nl_em_waltz_b005_es", 11, 8, 5),
]


def _chain_partition(mode_label):
    """Set of kx-index chains per ky, independent of how labels are numbered."""
    ml = np.asarray(mode_label)
    return [
        frozenset(
            frozenset(np.flatnonzero(ml[:, iy] == lbl).tolist())
            for lbl in np.unique(ml[:, iy])
        )
        for iy in range(ml.shape[1])
    ]


@pytest.mark.parametrize("case,nkx,nky,ikxspace", _CASES)
def test_mode_label_matches_gkw(case, nkx, nky, ikxspace):
    """Generated connectivity partitions kx exactly as GKW does, for every ky."""
    path = os.path.join(GKW_CASES, case, "mode_label")
    if not os.path.exists(path):
        pytest.skip(f"{case}/mode_label not available")

    gkw = np.loadtxt(path)
    if gkw.shape == (nky, nkx):
        gkw = gkw.T
    assert gkw.shape == (nkx, nky), f"{case}: unexpected mode_label shape {gkw.shape}"

    got = _chain_partition(_build_mode_label(nkx, nky, ikxspace, 1))
    want = _chain_partition(gkw)
    for iy in range(nky):
        assert got[iy] == want[iy], (
            f"{case}: ky index {iy} connectivity differs\n"
            f"  gyaradax: {sorted(sorted(c) for c in got[iy])}\n"
            f"  GKW     : {sorted(sorted(c) for c in want[iy])}"
        )


def test_chain_stride_scales_with_ky():
    """Stride between connected kx grows linearly with the ky mode index."""
    nkx, nky, ikxspace = 85, 32, 7
    ml = _build_mode_label(nkx, nky, ikxspace, 1)
    ixzero = (nkx - 1) // 2

    for iy in range(1, nky):
        chain = np.flatnonzero(ml[:, iy] == ml[ixzero, iy])
        expected = ikxspace * iy
        if expected >= nkx:
            assert chain.tolist() == [ixzero], f"ky index {iy}: chain must be isolated"
        else:
            assert np.all(np.diff(chain) == expected), (
                f"ky index {iy}: stride {np.unique(np.diff(chain))} != {expected}"
            )


def test_zonal_mode_is_unconnected():
    """ky=0 gets one label per kx: GKW treats it as periodic, not twist-shifted."""
    ml = _build_mode_label(21, 4, 3, 1)
    assert len(np.unique(ml[:, 0])) == 21


def test_nperiod_defaults_to_one_when_config_omits_it():
    """Omitting nperiod everywhere gives a single poloidal turn, as GKW does."""
    from gyaradax.geometry.spec import geometry_spec_from_config

    cfg = {
        "geometry": {"q": 2.0, "shat": 1.0, "eps": 0.19},
        "grid": {"ns": 16, "nkx": 21, "nky": 4, "nvpar": 16, "nmu": 8, "ikxspace": 3},
    }
    assert geometry_spec_from_config(cfg).nperiod == 1

    nkx, nky, ikxspace = 21, 4, 3
    explicit = _build_mode_label(nkx, nky, ikxspace, 1)
    implicit = _build_mode_label(nkx, nky, ikxspace)
    assert np.array_equal(explicit, implicit)


def test_nperiod_scales_the_stride():
    """nperiod>1 spans (2*nperiod-1) poloidal turns, scaling the shift to match."""
    nkx, nky, ikxspace = 61, 6, 2
    base = _build_mode_label(nkx, nky, ikxspace, 1)
    wide = _build_mode_label(nkx, nky, ikxspace, 2)
    ixzero = (nkx - 1) // 2

    for iy in range(1, nky):
        cb = np.flatnonzero(base[:, iy] == base[ixzero, iy])
        cw = np.flatnonzero(wide[:, iy] == wide[ixzero, iy])
        if len(cb) > 1 and len(cw) > 1:
            assert np.diff(cw)[0] == 3 * np.diff(cb)[0]
