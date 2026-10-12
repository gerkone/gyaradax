import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import noisy_case  # type: ignore[import-not-found]

from gyaradax.integrals import _phi_kinetic, charge_density, precompute_phi_kinetic

GRID = dict(nvpar=16, nmu=4, ns=16, nkx=21, nky=5)


# frozen copy of the kinetic phi solve before charge_density
def _phi_kinetic_ref(geometry, df, phi_weight=None, phi_diag=None):
    if phi_weight is None or phi_diag is None:
        phi_weight, phi_diag = precompute_phi_kinetic(geometry)
    phi_num = jnp.einsum("avmjkl,avmjkl->jkl", phi_weight, df)
    return -phi_num / phi_diag


def _close(actual, desired):
    desired = np.asarray(desired)
    np.testing.assert_allclose(
        np.asarray(actual), desired, rtol=1e-14, atol=1e-14 * np.abs(desired).max()
    )


@pytest.fixture(scope="module")
def case():
    df, geometry, params, _ = noisy_case("nl_em_waltz_b01.yaml", GRID, dict(nlapar=False), seed=3)
    geom = dict(geometry)
    for key in ("mas", "signz", "tmp", "de", "vthrat"):
        geom[key] = jnp.asarray(getattr(params, key), dtype=jnp.float64).reshape(-1)
    weight, diag = precompute_phi_kinetic(geom)
    return df, geom, weight, diag


def test_charge_density_shape(case):
    df, geom, weight, _ = case
    rho = charge_density(geom, df)
    nsp, _, _, ns, nkx, nky = df.shape
    assert nsp == 2
    assert rho.shape == (nsp, ns, nkx, nky)
    np.testing.assert_array_equal(np.asarray(rho), np.asarray(charge_density(geom, df, weight)))


@pytest.mark.parametrize("jit", [False, True])
def test_phi_kinetic_bitwise_unchanged(case, jit):
    df, geom, weight, diag = case
    wrap = jax.jit if jit else (lambda fn: fn)
    phi = wrap(lambda d: _phi_kinetic(geom, d, weight, diag))(df)
    ref = wrap(lambda d: _phi_kinetic_ref(geom, d, weight, diag))(df)
    np.testing.assert_array_equal(np.asarray(phi), np.asarray(ref))
    phi_default = wrap(lambda d: _phi_kinetic(geom, d))(df)
    ref_default = wrap(lambda d: _phi_kinetic_ref(geom, d))(df)
    np.testing.assert_array_equal(np.asarray(phi_default), np.asarray(ref_default))


def test_species_sum_is_phi_numerator(case):
    df, geom, weight, diag = case
    phi = jax.jit(lambda d: _phi_kinetic(geom, d, weight, diag))(df)
    rho_sum = jax.jit(lambda d: charge_density(geom, d, weight).sum(0) / diag)(df)
    np.testing.assert_array_equal(np.asarray(rho_sum), -np.asarray(phi))
    _close(charge_density(geom, df, weight).sum(0) / diag, -phi)


def test_factored_weight_matches_dense(case):
    df, geom, weight, _ = case
    factored = weight[:, :1]
    np.testing.assert_array_equal(np.broadcast_to(factored, weight.shape), np.asarray(weight))
    dense = charge_density(geom, df, weight)
    _close(charge_density(geom, df, factored), dense)
    _close(jax.jit(lambda d: charge_density(geom, d, factored))(df), dense)
