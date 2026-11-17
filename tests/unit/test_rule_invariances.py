"""Properties every saturation rule must satisfy, whatever its form.

These are written to catch the class of bug this code kept producing: a
quantity used in the wrong units, a convention applied to one input and not
its partner, a branch on a traced value. Each is a statement about the
PHYSICS that holds for any correct rule, so they keep their value as the
rules change.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyaradax.quasilinear.rules import RULES

RNG = np.random.default_rng(0)
NS, NKX, NKY = 8, 21, 12


def spectrum(seed=0):
    """A synthetic eigenmode with the shapes the rules expect."""
    rng = np.random.default_rng(seed)
    ky = np.linspace(0.0, 1.4, NKY)
    kx = np.linspace(-3.0, 3.0, NKX)
    gam = np.concatenate([[0.0], 0.3 * np.exp(-((ky[1:] - 0.4) ** 2) / 0.1)])
    phi2 = rng.uniform(0.1, 1.0, (NS, NKX, NKY)) * np.exp(-(kx[None, :, None] ** 2) / 4)
    little_g = np.stack([np.full(NS, 16.0), np.full(NS, 0.5), np.ones(NS)])
    ints = np.full(NS, 1.0 / NS)
    return dict(
        growth_rate=jnp.asarray(gam), phi2=jnp.asarray(phi2),
        phi2_kxy=jnp.asarray((phi2 * ints[:, None, None]).sum(axis=0)),
        flux_kxy=jnp.asarray(rng.uniform(0.05, 1.0, (NKX, NKY))),
        krho=jnp.asarray(ky), kxrh=jnp.asarray(kx),
        little_g=jnp.asarray(little_g), ds=float(ints.mean()),
    )


def call(name, spec, **kw):
    fn = RULES[name]
    ok = set(inspect.signature(fn).parameters)
    extra = {k: v for k, v in dict(
        dlnpdr=8.0, kthnorm=4.0, grad_r0=1.0, gq=0.83).items() if k in ok}
    return float(fn(cn=1.0, **{k: v for k, v in spec.items() if k in ok}, **extra, **kw))


@pytest.mark.parametrize("name", sorted(RULES))
def test_invariant_to_overall_eigenmode_amplitude(name):
    """A linear eigenmode's amplitude is set by the initial condition and is
    physically meaningless. Scaling it scales |phi|^2 AND the flux together,
    both being quadratic in the fields, and the rule must not notice -- W is
    a ratio of the two and <kperp2> is an intensity-weighted mean."""
    s = spectrum()
    base = call(name, s)
    for factor in (1e-6, 1e3):
        t = dict(s)
        t["phi2"] = s["phi2"] * factor
        t["phi2_kxy"] = s["phi2_kxy"] * factor
        t["flux_kxy"] = s["flux_kxy"] * factor
        assert np.isclose(call(name, t), base, rtol=1e-6), f"{name} scaled by {factor}"


@pytest.mark.parametrize("name", sorted(RULES))
def test_linear_in_the_flux(name):
    """The QL weight enters linearly, so doubling the linear flux doubles Q.
    This is what lets ql_flux_per_species decompose by species and channel,
    and it must hold at the CALIBRATED exponent, not just at 1 -- which is
    why sat2 applies its exponent per mode rather than to the summed flux."""
    s = spectrum()
    t = dict(s)
    t["flux_kxy"] = s["flux_kxy"] * 2.0
    assert np.isclose(call(name, t), 2.0 * call(name, s), rtol=1e-6)


@pytest.mark.parametrize("name", sorted(RULES))
def test_stable_spectrum_gives_exactly_zero(name):
    s = dict(spectrum())
    s["growth_rate"] = jnp.full_like(s["growth_rate"], -0.5)
    assert call(name, s) == 0.0


@pytest.mark.parametrize("name", sorted(RULES))
def test_adding_a_dead_ky_changes_nothing(name):
    """A ky column with no amplitude and no growth must not contribute. It
    would if a rule averaged over ky instead of summing, or divided by nky."""
    s = spectrum()
    t = {k: v for k, v in s.items()}
    t["growth_rate"] = jnp.concatenate([s["growth_rate"], jnp.array([-1.0])])
    t["krho"] = jnp.concatenate([s["krho"], jnp.array([1.6])])
    t["phi2"] = jnp.concatenate([s["phi2"], jnp.zeros((NS, NKX, 1))], axis=2)
    t["phi2_kxy"] = jnp.concatenate([s["phi2_kxy"], jnp.zeros((NKX, 1))], axis=1)
    t["flux_kxy"] = jnp.concatenate([s["flux_kxy"], jnp.zeros((NKX, 1))], axis=1)
    assert np.isclose(call(name, t), call(name, s), rtol=1e-6), name


@pytest.mark.parametrize("name", sorted(RULES))
def test_finite_and_traceable_under_vmap(name):
    """The TORAX plugin vmaps the rule over radii; nothing may branch on a
    traced value, and the result must stay finite."""
    s = spectrum()
    ok = set(inspect.signature(RULES[name]).parameters)
    extra = {k: v for k, v in dict(
        dlnpdr=8.0, kthnorm=4.0, grad_r0=1.0, gq=0.83).items() if k in ok}
    kw = {k: v for k, v in s.items() if k in ok}

    def one(scale):
        return RULES[name](cn=1.0, **{**kw, "growth_rate": s["growth_rate"] * scale},
                           **extra)

    out = np.asarray(jax.vmap(one)(jnp.array([0.5, 1.0, 2.0])))
    assert out.shape == (3,) and np.all(np.isfinite(out)) and np.all(out >= 0)


@pytest.mark.parametrize("name", sorted(RULES))
def test_monotone_in_growth_rate(name):
    """More unstable must mean more flux. Not guaranteed by construction once
    the zonal-mixing clamp and the gate are involved, so worth pinning."""
    s = spectrum()
    q = [call(name, {**s, "growth_rate": s["growth_rate"] * f})
         for f in (0.5, 1.0, 2.0, 4.0)]
    assert all(b > a for a, b in zip(q, q[1:])), f"{name}: {q}"


@pytest.mark.parametrize("name", sorted(RULES))
def test_per_species_decomposition_sums(name):
    """The quasilinear postulate: one saturated intensity, species and
    channels differing only through their weights. Broken the moment a rule
    stops being linear in the flux."""
    from gyaradax.quasilinear.rules import ql_flux_per_species

    s = spectrum()
    ok = set(inspect.signature(RULES[name]).parameters)
    extra = {k: v for k, v in dict(
        dlnpdr=8.0, kthnorm=4.0, grad_r0=1.0, gq=0.83).items() if k in ok}
    shared = {k: v for k, v in s.items() if k in ok and k != "flux_kxy"}
    sp = jnp.stack([jnp.stack([s["flux_kxy"] * f for f in (0.3, 1.0, 0.1)]),
                    jnp.stack([s["flux_kxy"] * f for f in (0.2, 0.6, 0.05)])])
    per = np.asarray(ql_flux_per_species(name, flux_kxy_sp=sp, cn=1.0,
                                         **shared, **extra))
    whole = float(RULES[name](cn=1.0, flux_kxy=sp.sum(axis=(0, 1)), **shared, **extra))
    assert np.isclose(per.sum(), whole, rtol=1e-6), f"{name}: {per.sum()} vs {whole}"
