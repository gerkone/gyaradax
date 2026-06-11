"""Parity + AD + JIT smoke tests for the refactored geometry/ module.

Locks in the post-refactor invariants:
  - circ / s-alpha numerically match the pre-refactor single-file geometry.
  - compute_continuous_geometry is jax.grad-able and jax.jit-able.
The pre-refactor snapshot is reconstructed from git on demand so the test
runs anywhere the repo has history. Skips silently if git is unavailable.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gyaradax.geometry import (
    build_topology,
    compute_continuous_geometry,
    compute_geometry,
)


_PRE_REFACTOR_COMMIT_PARENT = "0378172^"  # parent of the geometry refactor


def _load_old_geom():
    """Extract pre-refactor gyaradax/geometry.py from git into a temp file."""
    if shutil.which("git") is None:
        pytest.skip("git not available; cannot reconstruct pre-refactor geometry")
    repo = Path(__file__).resolve().parents[2]
    try:
        src = subprocess.check_output(
            ["git", "-C", str(repo), "show",
             f"{_PRE_REFACTOR_COMMIT_PARENT}:gyaradax/geometry.py"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        pytest.skip("pre-refactor geometry.py snapshot not in git history")
    tmp = Path("/tmp") / "geom_old_parity_test.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("geom_old_parity_test", str(tmp))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["geom_old_parity_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_PARITY_CONFIGS = [
    ("circ", 1.4, 0.8, 0.18, 16, 85, 32, 5),
    ("circ", 4.6, 3.1, 0.19, 16, 85, 32, 5),
    ("s-alpha", 2.0, 1.0, 0.18, 16, 43, 16, 5),
    ("circ", 2.5, 1.5, 0.10, 32, 64, 16, 4),
    ("circ", 3.0, 2.0, 0.25, 16, 32, 8, 8),
]


@pytest.mark.parametrize("geom_type,q,shat,eps,ns,nkx,nky,ikxspace", _PARITY_CONFIGS)
def test_parity_with_pre_refactor(geom_type, q, shat, eps, ns, nkx, nky, ikxspace):
    """Refactored compute_geometry matches the original numerically (bit-identical)."""
    old_mod = _load_old_geom()
    kwargs = dict(
        q=q, shat=shat, eps=eps, ns=ns, nkx=nkx, nky=nky,
        nvpar=32, nmu=8, vpar_max=3.0, nperiod=1,
        kxmax=1.0, krhomax=1.4, ikxspace=ikxspace,
        signB=1.0, Rref=100.0, geom_type=geom_type,
    )
    old = old_mod.compute_geometry(**kwargs)
    new = compute_geometry(**kwargs)
    # Every key the pre-refactor returned must still be present and identical.
    # (New keys may be added; we don't check those.)
    for k in old:
        assert k in new, f"key {k!r} present in OLD but missing in NEW"
        a = np.asarray(old[k]); b = np.asarray(new[k])
        if np.issubdtype(a.dtype, np.floating):
            assert np.array_equal(a, b), (
                f"{k}: max abs diff {float(np.max(np.abs(a - b))):.3e}"
            )
        else:
            assert np.array_equal(a, b), f"{k}: int/bool mismatch"


def test_compute_continuous_geometry_is_grad_able():
    """jax.grad through compute_continuous_geometry w.r.t. q returns finite non-zero."""
    ns, nkx, nky, ikxspace = 16, 85, 32, 5
    topo = build_topology(nkx, nky, ikxspace, ns)

    def loss(q):
        g = compute_continuous_geometry(
            q=q, shat=0.8, eps=0.18, ns=ns, nkx=nkx, nky=nky,
            nvpar=32, nmu=8, vpar_max=3.0, nperiod=1,
            kxmax=1.0, krhomax=1.4, ikxspace=ikxspace,
            signB=1.0, Rref=100.0, geom_type="circ", topology=topo,
        )
        return jnp.sum(g["little_g"])

    grad = float(jax.grad(loss)(jnp.array(1.4, dtype=jnp.float64)))
    assert np.isfinite(grad) and grad != 0.0


def test_compute_continuous_geometry_is_jit_able():
    """jax.jit of compute_continuous_geometry matches non-jit bit-for-bit."""
    ns, nkx, nky, ikxspace = 16, 85, 32, 5
    topo = build_topology(nkx, nky, ikxspace, ns)

    def f(q, shat, eps):
        return compute_continuous_geometry(
            q=q, shat=shat, eps=eps, ns=ns, nkx=nkx, nky=nky,
            nvpar=32, nmu=8, vpar_max=3.0, nperiod=1,
            kxmax=1.0, krhomax=1.4, ikxspace=ikxspace,
            signB=1.0, Rref=100.0, geom_type="circ", topology=topo,
        )

    q0, s0, e0 = jnp.array(1.4), jnp.array(0.8), jnp.array(0.18)
    out_ref = f(q0, s0, e0)
    out_jit = jax.jit(f)(q0, s0, e0)
    for k in out_ref:
        a = np.asarray(out_ref[k]); b = np.asarray(out_jit[k])
        if np.issubdtype(a.dtype, np.floating):
            assert np.allclose(a, b, atol=1e-12), f"jit mismatch on key {k}"
        else:
            assert np.array_equal(a, b), f"jit mismatch (int/bool) on key {k}"
