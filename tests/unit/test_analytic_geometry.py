"""tests for analytic circular geometry computation."""

import os
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gyaradax.utils import load_geometry
from gyaradax.geometry import compute_geometry, compute_geometry_from_input


GKW_DATA_ROOT = os.environ.get("GKW_DATA_ROOT", "/restricteddata/ukaea/gyrokinetics/raw")
ITERATIONS = [8, 13, 131, 200]
KINETIC_CASES = [
    "v3_kiteration_991_half_rlt",
    "v3_kiteration_991_ntsks128",
    "v3_kiteration_991_double_rlt",
]


@pytest.fixture(params=ITERATIONS)
def gkw_dir(request):
    path = os.path.join(GKW_DATA_ROOT, f"iteration_{request.param}")
    if not os.path.exists(path):
        pytest.skip(f"reference data not found at {path}")
    return path


@pytest.fixture(params=ITERATIONS + KINETIC_CASES)
def gkw_dir_all(request):
    """fixture covering both adiabatic and kinetic electron cases."""
    param = request.param
    if isinstance(param, int):
        path = os.path.join(GKW_DATA_ROOT, f"iteration_{param}")
    else:
        path = os.path.join(GKW_DATA_ROOT, "kinetic_electrons", param)
    if not os.path.exists(path):
        pytest.skip(f"reference data not found at {path}")
    return path


def test_basic_fields(gkw_dir):
    """bn, ffun, bt_frac, rfun match GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    for name in ["sgrid", "bn", "ffun", "bt_frac", "rfun"]:
        r = np.asarray(ref[name])
        c = np.asarray(comp[name])
        assert r.shape == c.shape, f"{name} shape mismatch"
        np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg=name)


def test_metric_tensor(gkw_dir):
    """little_g (metric components) match GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    r = np.asarray(ref["little_g"])
    c = np.asarray(comp["little_g"])
    assert r.shape == c.shape
    np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg="little_g")


def test_gfun(gkw_dir):
    """mirror force function matches GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    r = np.asarray(ref["gfun"])
    c = np.asarray(comp["gfun"])
    np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg="gfun")


def test_efun(gkw_dir):
    """ExB function matches GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    r = np.asarray(ref["efun"])
    c = np.asarray(comp["efun"])
    np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg="efun")


def test_dfun_eps(gkw_dir):
    """radial drift D_eps matches GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    r = np.asarray(ref["dfun"])[:, 0]
    c = np.asarray(comp["dfun"])[:, 0]
    np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg="D_eps")


def test_dfun_zeta(gkw_dir):
    """binormal drift D_zeta matches GKW to rtol=2e-3, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    r = np.asarray(ref["dfun"])[:, 1]
    c = np.asarray(comp["dfun"])[:, 1]
    np.testing.assert_allclose(c, r, rtol=2e-3, atol=1e-6, err_msg="D_zeta")


def test_velocity_grids(gkw_dir):
    """vpgr, mugr, intvp, intmu match GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    for name in ["vpgr", "mugr", "intvp", "intmu"]:
        r = np.asarray(ref[name])
        c = np.asarray(comp[name])
        assert r.shape == c.shape, f"{name} shape mismatch: {r.shape} vs {c.shape}"
        np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg=name)


def test_wavenumber_grids(gkw_dir):
    """kxrh and krho match GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    for name in ["kxrh", "krho"]:
        r = np.asarray(ref[name])
        c = np.asarray(comp[name])
        assert r.shape == c.shape, f"{name} shape mismatch: {r.shape} vs {c.shape}"
        np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg=name)


def test_mode_connectivity_scalars(gkw_dir):
    """ixzero and iyzero match GKW."""
    ref = load_geometry(gkw_dir)
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))

    for name in ["ixzero", "iyzero"]:
        r = np.asarray(ref[name])
        c = np.asarray(comp[name])
        np.testing.assert_array_equal(c, r, err_msg=name)


def test_mode_connectivity_structure(gkw_dir):
    """ixplus/ixminus have correct chain structure (spacing = ikxspace)."""
    comp = compute_geometry_from_input(os.path.join(gkw_dir, "input.dat"))
    ixplus = np.asarray(comp["ixplus"])
    nkx, nky = ixplus.shape

    # for ky>0: connected modes should be ikxspace apart
    for iy in range(1, min(3, nky)):
        for ix in range(nkx):
            if ixplus[ix, iy] >= 0:
                assert ixplus[ix, iy] - ix == 5, f"ixplus spacing wrong at ix={ix}, iy={iy}"


# --- hfun / ifun tests (adiabatic + kinetic cases) ---


def test_hfun_eps(gkw_dir_all):
    """H_eps matches GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir_all)
    comp = compute_geometry_from_input(os.path.join(gkw_dir_all, "input.dat"))

    if "hfun" not in ref:
        pytest.skip("hfun not in reference geometry")

    r = np.asarray(ref["hfun"])[:, 0]
    c = np.asarray(comp["hfun"])[:, 0]
    np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg="H_eps")


def test_hfun_zeta(gkw_dir_all):
    """H_zeta matches GKW to rtol=2e-3, atol=1e-6."""
    ref = load_geometry(gkw_dir_all)
    comp = compute_geometry_from_input(os.path.join(gkw_dir_all, "input.dat"))

    if "hfun" not in ref:
        pytest.skip("hfun not in reference geometry")

    r = np.asarray(ref["hfun"])[:, 1]
    c = np.asarray(comp["hfun"])[:, 1]
    np.testing.assert_allclose(c, r, rtol=2e-3, atol=1e-6, err_msg="H_zeta")


def test_ifun_eps(gkw_dir_all):
    """I_eps matches GKW to rtol=1e-4, atol=1e-6."""
    ref = load_geometry(gkw_dir_all)
    comp = compute_geometry_from_input(os.path.join(gkw_dir_all, "input.dat"))

    if "ifun" not in ref:
        pytest.skip("ifun not in reference geometry")

    r = np.asarray(ref["ifun"])[:, 0]
    c = np.asarray(comp["ifun"])[:, 0]
    np.testing.assert_allclose(c, r, rtol=1e-4, atol=1e-6, err_msg="I_eps")


def test_ifun_zeta(gkw_dir_all):
    """I_zeta matches GKW to rtol=2e-3, atol=1e-6."""
    ref = load_geometry(gkw_dir_all)
    comp = compute_geometry_from_input(os.path.join(gkw_dir_all, "input.dat"))

    if "ifun" not in ref:
        pytest.skip("ifun not in reference geometry")

    r = np.asarray(ref["ifun"])[:, 1]
    c = np.asarray(comp["ifun"])[:, 1]
    np.testing.assert_allclose(c, r, rtol=2e-3, atol=1e-6, err_msg="I_zeta")


# ---------------------------------------------------------------------------
# Differentiability tests: AD gradients vs finite differences
# ---------------------------------------------------------------------------

_GEOM_KWARGS = dict(
    ns=16,
    nkx=9,
    nky=8,
    nvpar=24,
    nmu=8,
    vpar_max=3.0,
    nperiod=1,
    krhomax=1.4,
)


def _geom_scalar(q_val, shat_val, eps_val, field="bn"):
    """Scalar loss from geometry for AD testing."""
    geom = compute_geometry(
        q=q_val,
        shat=shat_val,
        eps=eps_val,
        kxmax=2.0 * 9 * shat_val,
        **_GEOM_KWARGS,
    )
    return jnp.sum(geom[field] ** 2)


@pytest.mark.parametrize("field", ["bn", "ffun", "gfun", "efun", "little_g"])
def test_geometry_grad_q(field):
    """AD gradient of geometry w.r.t. q matches finite differences."""
    q0, s0, e0 = jnp.array(4.57), jnp.array(3.08), jnp.array(0.19)
    grad_fn = jax.grad(lambda q: _geom_scalar(q, s0, e0, field))
    ad = float(grad_fn(q0))
    eps_fd = 1e-5
    fd = float(
        (_geom_scalar(q0 + eps_fd, s0, e0, field) - _geom_scalar(q0 - eps_fd, s0, e0, field))
        / (2 * eps_fd)
    )
    err = abs(ad - fd)
    rel = err / max(abs(ad), abs(fd), 1.0)
    assert np.isfinite(ad), f"AD gradient is not finite for {field}"
    assert rel < 1e-4, f"AD vs FD mismatch for d({field})/dq: rel={rel:.2e}"


@pytest.mark.parametrize("field", ["bn", "ffun", "gfun", "efun", "little_g"])
def test_geometry_grad_shat(field):
    """AD gradient of geometry w.r.t. shat matches finite differences."""
    q0, s0, e0 = jnp.array(4.57), jnp.array(3.08), jnp.array(0.19)
    grad_fn = jax.grad(lambda s: _geom_scalar(q0, s, e0, field))
    ad = float(grad_fn(s0))
    eps_fd = 1e-5
    fd = float(
        (_geom_scalar(q0, s0 + eps_fd, e0, field) - _geom_scalar(q0, s0 - eps_fd, e0, field))
        / (2 * eps_fd)
    )
    err = abs(ad - fd)
    rel = err / max(abs(ad), abs(fd), 1.0)
    assert np.isfinite(ad), f"AD gradient is not finite for {field}"
    assert rel < 1e-4, f"AD vs FD mismatch for d({field})/dshat: rel={rel:.2e}"


@pytest.mark.parametrize("field", ["bn", "ffun", "gfun", "efun", "little_g"])
def test_geometry_grad_eps(field):
    """AD gradient of geometry w.r.t. eps matches finite differences."""
    q0, s0, e0 = jnp.array(4.57), jnp.array(3.08), jnp.array(0.19)
    grad_fn = jax.grad(lambda e: _geom_scalar(q0, s0, e, field))
    ad = float(grad_fn(e0))
    eps_fd = 1e-5
    fd = float(
        (_geom_scalar(q0, s0, e0 + eps_fd, field) - _geom_scalar(q0, s0, e0 - eps_fd, field))
        / (2 * eps_fd)
    )
    err = abs(ad - fd)
    rel = err / max(abs(ad), abs(fd), 1.0)
    assert np.isfinite(ad), f"AD gradient is not finite for {field}"
    assert rel < 1e-4, f"AD vs FD mismatch for d({field})/deps: rel={rel:.2e}"


def test_geometry_grad_drift_tensors():
    """AD gradients of drift tensors (dfun, hfun, ifun) w.r.t. all params."""
    q0, s0, e0 = jnp.array(4.57), jnp.array(3.08), jnp.array(0.19)
    for field in ["dfun", "hfun", "ifun"]:
        grad_fn = jax.grad(_geom_scalar, argnums=(0, 1, 2))
        dq, ds, de = grad_fn(q0, s0, e0, field)
        assert np.isfinite(float(dq)), f"dq not finite for {field}"
        assert np.isfinite(float(ds)), f"dshat not finite for {field}"
        assert np.isfinite(float(de)), f"deps not finite for {field}"

        # FD check on q only (representative)
        eps_fd = 1e-5
        fd = float(
            (_geom_scalar(q0 + eps_fd, s0, e0, field) - _geom_scalar(q0 - eps_fd, s0, e0, field))
            / (2 * eps_fd)
        )
        rel = abs(float(dq) - fd) / (abs(float(dq)) + 1e-30)
        assert rel < 1e-4, f"AD vs FD mismatch for d({field})/dq: rel={rel:.2e}"
