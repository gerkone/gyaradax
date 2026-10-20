"""Check the sat2 rule against GACODE TGLF SAT2's own output.

Requires a reference dump produced by scripts/gen_tglf_sat2_ref.py; see the
skip message and that script's header.
"""
import json
import os

import numpy as np
import pytest

import jax.numpy as jnp

from gyaradax.quasilinear.rules import _gamma_kymix, _zonal_mixing, dlnpdr_tglf, ql_flux_sat2

# tglf_multiscale_spectrum.f90: b2 drops to 3.55 once nmodes_in > 1
_B0, _B1, _B2 = 0.76, 1.22, 3.74
_KYMIN = 0.173 * np.sqrt(2.0)

_SKIP = (
    "TGLF_SAT2_REF is not set. It must point at the JSON written by "
    "scripts/gen_tglf_sat2_ref.py, which runs GACODE TGLF SAT2 through an "
    "f2py build that exposes the per-ky spectra (see that script's header)."
)


@pytest.fixture(scope="module")
def tglf_ref():
    path = os.environ.get("TGLF_SAT2_REF")
    if not path:
        pytest.skip(_SKIP)
    if not os.path.isfile(path):
        pytest.skip(f"{_SKIP} Current value does not exist: {path}")
    with open(path) as fh:
        ref = json.load(fh)
    if not ref:
        pytest.skip(f"{_SKIP} The file at {path} holds no cases.")
    return ref


def _tglf_kx_width(ky, kymax, grad_r0, b_geo0):
    kycut = _B0 * kymax
    gq = b_geo0 / grad_r0
    return np.where(
        ky < kycut, kycut / grad_r0, kycut / grad_r0 + _B1 * (ky - kycut) * gq
    )


def _tglf_sat_geo(ky, kymax, grad_r0, b_geo0, bt0, g0, g1, g2):
    kycut = _B0 * kymax
    d1 = (bt0 / b_geo0) ** 4 / grad_r0
    d2 = 1.0 / (b_geo0 / grad_r0) ** 2
    return np.where(
        ky < kycut,
        g0 * d1 * g1,
        g0 * (d1 * g1 * kycut + (ky - kycut) * d2 * g2) / np.maximum(ky, 1e-30),
    )


def _our_intensity(c):
    """SAT2 intensity from our helpers, fed TGLF's gamma and kx_width model."""
    ky = np.asarray(c["ky"])
    gamma = np.asarray(c["gamma"])
    kymin = _KYMIN * c["grad_r0"] if c["alpha_zf"] < 0 else 0.0

    kymax, vzf, jmax = _zonal_mixing(ky, gamma, 1e-30, kymin=kymin)
    gmix = np.asarray(_gamma_kymix(ky, gamma, kymax, vzf, jmax, 1.0, 1e-30))

    b2 = 3.55 if c["nmodes"] > 1 else _B2
    cnorm = b2 * (12.0 / c["dlnpdr"])
    kxw = _tglf_kx_width(ky, c["kymax"], c["grad_r0"], c["b_geo0"])
    geo = _tglf_sat_geo(
        ky, c["kymax"], c["grad_r0"], c["b_geo0"], c["bt0"],
        c["sat_geo0"], c["sat_geo1"], c["sat_geo2"],
    )
    return (1.0 / c["kymax"]) * cnorm * (gmix / (kxw * ky)) ** 2 * geo, kymax, vzf


def _case_ids(ref):
    return sorted(ref)


def test_zonal_mixing_matches_tglf(tglf_ref):
    for name in _case_ids(tglf_ref):
        c = tglf_ref[name]
        _, kymax, vzf = _our_intensity(c)
        assert float(kymax) == pytest.approx(c["kymax"], rel=1e-10), name
        assert float(vzf) == pytest.approx(c["vzf"], rel=1e-10), name


def test_sat2_intensity_matches_tglf(tglf_ref):
    for name in _case_ids(tglf_ref):
        c = tglf_ref[name]
        ours, _, _ = _our_intensity(c)
        ref = np.asarray(c["phinorm"])
        ok = ref > 0
        assert ok.sum() > 4, f"{name}: TGLF returned no saturated intensity"
        rel = np.abs(ours[ok] - ref[ok]) / ref[ok]
        assert rel.max() < 1e-10, f"{name}: max rel {rel.max():.3e}"


def test_rule_matches_tglf_intensity(tglf_ref):
    """ql_flux_sat2 itself, one ky at a time, against TGLF's intensity.

    The rule drops sat_geo and folds b2 into cn, so those two factors are
    divided out of TGLF's spectrum; the gate is made a hard relu, as in TGLF.
    """
    for name in _case_ids(tglf_ref):
        c = tglf_ref[name]
        ky, gamma = np.asarray(c["ky"]), np.asarray(c["gamma"])
        n = ky.size
        kymin = _KYMIN * c["grad_r0"] if c["alpha_zf"] < 0 else 0.0
        ours = np.array([
            float(ql_flux_sat2(
                jnp.asarray(gamma), None, jnp.ones((1, n)), jnp.eye(n)[k][None, :], jnp.asarray(ky),
                jnp.zeros(1), None, None, cn=1.0, dlnpdr=c["dlnpdr"], kymin=kymin, kthnorm=1.0,
                grad_r0=c["grad_r0"], gq=c["b_geo0"] / c["grad_r0"], flux_exponent=1.0,
                gate_sharpness=1e8,
            ))
            for k in range(n)
        ])
        b2 = 3.55 if c["nmodes"] > 1 else _B2
        geo = _tglf_sat_geo(
            ky, c["kymax"], c["grad_r0"], c["b_geo0"], c["bt0"],
            c["sat_geo0"], c["sat_geo1"], c["sat_geo2"],
        )
        ref = np.asarray(c["phinorm"]) / (b2 * geo)
        ok = ref > 0
        rel = np.abs(ours[ok] - ref[ok]) / ref[ok]
        assert rel.max() < 1e-8, f"{name}: max rel {rel.max():.3e}"


def test_dlnpdr_matches_tglf(tglf_ref):
    # base case: a/L_n = 1, a/L_T = 3 for both species, R/a = 3, Te = Ti
    ours = float(dlnpdr_tglf(np.array([9.0, 9.0]), np.array([3.0, 3.0]), te_ti=1.0))
    assert ours == pytest.approx(tglf_ref["base_azf-1"]["dlnpdr"], rel=1e-12)


def test_reference_covers_both_kymin_settings(tglf_ref):
    azf = {tglf_ref[n]["alpha_zf"] < 0 for n in tglf_ref}
    assert azf == {True, False}, "reference must cover ALPHA_ZF of both signs"
