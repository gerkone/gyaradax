import os
import jax
import jax.numpy as jnp
import numpy as np

from gyaradax.diag import (
    term_iii_fft_pack_roundtrip,
    term_iii_rhs,
)
from gyaradax.geometry import load_runtime_params


def test_runtime_params_types_and_values(nonlin_dir):
    """verify that runtime parameters are parsed with correct types."""
    runtime = load_runtime_params(os.path.join(nonlin_dir, "input.dat"))

    assert isinstance(runtime["dtim"], float)
    assert isinstance(runtime["naverage"], int)
    assert isinstance(runtime["non_linear"], bool)
    assert isinstance(runtime["method"], str)


def _harden_parity(spec_kxky, ixzero):
    """ensure spectral data represents a real signal (conjugate symmetry at ky=0)."""
    # for ky=0, we need f(kx) = conj(f(-kx))
    # our array has kx indexed by jind.
    # index 0 is kx=0.
    # indices 1..nkx/2 are positive kx.
    # indices nkx-1.. are negative kx.

    # but spec_kxky is indexed by ix (0..nkx-1).
    # ixzero is the index where kx=0.

    # let's just make the ky=0 part satisfy the symmetry in ix space around ixzero.
    # wait, kx connection is complex.
    # actually, for a simple roundtrip test, we can just set ky=0 to zero
    # and check if the other modes (ky > 0) are preserved.
    # for ky > 0, there are no constraints on the half-spectrum.

    # However, irfft also expects the imaginary part of the Nyquist frequency to be zero if it's even.
    # To be safe, we just set the whole ky=0 column to zero for the roundtrip test.
    return spec_kxky.at[:, 0].set(0.0)


def test_term_iii_fft_roundtrip(nonlin_geom, nonlin_shape):
    """verify pseudospectral fft roundtrip preserves physical modes."""
    key = jax.random.PRNGKey(123)
    nkx, nky = nonlin_shape[3], nonlin_shape[4]
    spec_kxky = jax.random.normal(
        key, (nkx, nky), dtype=jnp.float64
    ) + 1j * jax.random.normal(key, (nkx, nky), dtype=jnp.float64)

    # zero out ky=0 to avoid parity issues at the DC component for the roundtrip identity
    spec_kxky = spec_kxky.at[:, 0].set(0.0)

    # roundtrip through dealiased grids
    repacked = term_iii_fft_pack_roundtrip(spec_kxky, nonlin_geom)

    assert repacked.shape == spec_kxky.shape
    # modes should be preserved (modulo floating point error)
    # we use a slightly more relaxed tolerance for the full complex roundtrip
    np.testing.assert_allclose(
        np.asarray(repacked), np.asarray(spec_kxky), rtol=1e-10, atol=1e-10
    )


def test_term_iii_rhs_shapes(nonlin_geom, nonlin_shape):
    """verify nonlinear term iii output shape."""
    df = jnp.zeros(nonlin_shape, dtype=jnp.complex128)
    rhs_nl = term_iii_rhs(df, nonlin_geom)
    assert rhs_nl.shape == nonlin_shape
