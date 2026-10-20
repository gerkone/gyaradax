"""AD, JIT and traced-topology tests for compute_continuous_geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gyaradax.geometry import build_topology, compute_continuous_geometry


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


@pytest.mark.parametrize("shat", [0.8, -0.8])
def test_prebuilt_topology_follows_traced_shear(shat):
    """Under jit, a both-orientation topology connects kx as the concrete topology=None path does."""
    ns, nkx, nky, ikxspace = 16, 43, 16, 5
    topo = build_topology(nkx, nky, ikxspace, ns)
    kw = dict(ns=ns, nkx=nkx, nky=nky, nvpar=16, nmu=4, vpar_max=3.0, nperiod=1, kxmax=1.0,
              krhomax=1.4, ikxspace=ikxspace, signB=1.0, Rref=100.0, geom_type="circ")
    ref = compute_continuous_geometry(q=1.4, shat=shat, eps=0.18, **kw)
    got = jax.jit(lambda s: compute_continuous_geometry(q=1.4, shat=s, eps=0.18, topology=topo, **kw))(
        jnp.array(shat)
    )
    for k in ("ixplus", "ixminus", "pos_par_grid_class", "s_shift", "kx_shift", "valid_shift"):
        assert np.array_equal(np.asarray(got[k]), np.asarray(ref[k])), k
