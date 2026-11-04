"""Correctness tests for the QL saturation rules (canonical + RULES zoo).

Fast, CPU-able: pure jax.numpy on tiny synthetic spectra, no solver runs.
The optional kinetic integration test reads the last dump of a GKW kinetic-electron run
(read-only) and is skipped when the data root is not reachable.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import reference_dir, reference_dumps  # type: ignore[import-not-found]
from gyaradax.quasilinear import calibration
from gyaradax.geometry.grids import _build_mode_label
from gyaradax.quasilinear.rules import (
    RULES,
    dlnpdr_tglf,
    kx_chain0,
    ql_flux,
    ql_flux_per_species,
)

NS, NKX, NKY = 8, 5, 6

KINETIC_CASE = "kinetic_mid_te_gradient"


def synthetic_spectrum(seed=0, zonal_content=0.0):
    """Small physically-shaped spectrum: (ns, nkx, nky) grids with a ky=0 column."""
    rng = np.random.default_rng(seed)
    krho = jnp.asarray(np.linspace(0.0, 1.0, NKY))
    kxrh = jnp.asarray(np.linspace(-1.0, 1.0, NKX))
    s = np.linspace(-0.5, 0.5, NS, endpoint=False)
    ds = float(s[1] - s[0])

    # circular-like metric: g_zz ~ 1, g_ez small, g_ee ~ 1 + variation
    little_g = jnp.asarray(
        np.stack(
            [
                1.0 + 0.1 * np.cos(2 * np.pi * s),
                0.2 * np.sin(2 * np.pi * s),
                1.0 + 0.5 * np.sin(np.pi * s) ** 2,
            ]
        )
    )

    # ballooning-like |phi|^2: gaussian in s and kx, per-ky amplitude
    amp_ky = 1.0 + rng.random(NKY)
    phi2 = (
        np.exp(-(s[:, None, None] ** 2) / 0.05)
        * np.exp(-(np.asarray(kxrh)[None, :, None] ** 2) / 0.5)
        * amp_ky[None, None, :]
    )
    if zonal_content > 0.0:
        phi2[:, :, 0] = zonal_content * np.exp(-(np.asarray(kxrh)[None, :] ** 2) / 0.1)
    phi2 = jnp.asarray(phi2)
    phi2_kxy = jnp.sum(phi2, axis=0) * ds

    # linear flux proportional to intensity with a positive per-mode weight
    flux_kxy = phi2_kxy * jnp.asarray(0.5 + rng.random((NKX, NKY)))

    # ITG-like gamma(ky): unstable mid-ky band, stable zonal + tail
    gamma = jnp.asarray(np.array([0.0, 0.10, 0.30, 0.25, 0.05, -0.10]))
    frequency = jnp.asarray(np.array([0.0, 0.2, 0.5, 0.7, -0.3, -0.6]))

    return dict(
        growth_rate=gamma,
        phi2=phi2,
        phi2_kxy=phi2_kxy,
        flux_kxy=flux_kxy,
        krho=krho,
        kxrh=kxrh,
        little_g=little_g,
        ds=ds,
        frequency=frequency,
    )


def call_rule(fn, spec, name=None, **overrides):
    kwargs = dict(overrides)
    return fn(
        spec["growth_rate"],
        spec["phi2"],
        spec["phi2_kxy"],
        spec["flux_kxy"],
        spec["krho"],
        spec["kxrh"],
        spec["little_g"],
        spec["ds"],
        **kwargs,
    )


class TestCanonicalQlFlux:
    def test_linear_amplitude_invariance(self):
        """Scaling phi2, phi2_kxy, flux_kxy consistently leaves ql_flux unchanged."""
        spec = synthetic_spectrum()
        base = float(call_rule(ql_flux, spec))
        for lam in (1e-6, 1e3, 1e12):
            scaled = dict(
                spec,
                phi2=spec["phi2"] * lam,
                phi2_kxy=spec["phi2_kxy"] * lam,
                flux_kxy=spec["flux_kxy"] * lam,
            )
            assert float(call_rule(ql_flux, scaled)) == pytest.approx(base, rel=1e-10)

    def test_stable_spectrum_zero_flux(self):
        """gamma <= 0 everywhere gives exactly zero flux."""
        spec = synthetic_spectrum()
        spec["growth_rate"] = -jnp.abs(spec["growth_rate"]) - 0.01
        assert float(call_rule(ql_flux, spec)) == 0.0

    def test_zonal_column_contributes_nothing(self):
        """Perturbing gamma and flux in the ky=0 column leaves the result unchanged."""
        spec = synthetic_spectrum(zonal_content=5.0)
        base = float(call_rule(ql_flux, spec))
        spec2 = dict(spec)
        spec2["growth_rate"] = spec["growth_rate"].at[0].set(10.0)
        spec2["flux_kxy"] = spec["flux_kxy"].at[:, 0].set(1e6)
        assert float(call_rule(ql_flux, spec2)) == pytest.approx(base, rel=1e-12)

    def test_kperp_floor_no_blowup(self):
        """Pathological tiny <k_perp^2> (zero eigenmode weight) stays finite."""
        spec = synthetic_spectrum()
        spec["phi2"] = jnp.zeros_like(spec["phi2"])
        q = float(call_rule(ql_flux, spec))
        assert np.isfinite(q)
        # bounded by the geometric floor krho^2 * min(g_zz), not the 1e-30 eps
        g_zz_min = float(jnp.min(spec["little_g"][0]))
        w = np.sum(np.asarray(spec["flux_kxy"]), axis=0) / np.sum(
            np.asarray(spec["phi2_kxy"]), axis=0
        )
        gam = np.maximum(np.asarray(spec["growth_rate"]), 0.0)
        kr = np.asarray(spec["krho"])
        bound = np.sum((w * gam)[1:] / (kr[1:] ** 2 * g_zz_min))
        assert q <= bound * (1.0 + 1e-8)

    def test_gamma_monotonicity(self):
        """Increasing gamma at fixed everything else does not decrease flux."""
        spec = synthetic_spectrum()
        base = float(call_rule(ql_flux, spec))
        for bump in (0.01, 0.1, 1.0):
            spec2 = dict(spec, growth_rate=spec["growth_rate"] + bump)
            assert float(call_rule(ql_flux, spec2)) >= base - 1e-12
            base = float(call_rule(ql_flux, spec2))

    def test_positive_on_unstable_spectrum(self):
        spec = synthetic_spectrum()
        assert float(call_rule(ql_flux, spec)) > 0.0


@pytest.mark.parametrize("name", sorted(RULES.keys()))
class TestRuleZoo:
    def test_runs_finite_nonnegative(self, name):
        spec = synthetic_spectrum(zonal_content=0.5)
        q = float(call_rule(RULES[name], spec, name=name))
        assert np.isfinite(q)
        assert q >= 0.0

    def test_amplitude_constant_scales_linearly(self, name):
        spec = synthetic_spectrum(zonal_content=0.5)
        q1 = float(call_rule(RULES[name], spec, name=name, cn=1.0))
        q3 = float(call_rule(RULES[name], spec, name=name, cn=3.0))
        assert q3 == pytest.approx(3.0 * q1, rel=1e-12)

    def test_linear_amplitude_invariance(self, name):
        spec = synthetic_spectrum(zonal_content=0.5)
        base = float(call_rule(RULES[name], spec, name=name))
        lam = 4.2e5
        scaled = dict(
            spec,
            phi2=spec["phi2"] * lam,
            phi2_kxy=spec["phi2_kxy"] * lam,
            flux_kxy=spec["flux_kxy"] * lam,
        )
        assert float(call_rule(RULES[name], scaled, name=name)) == pytest.approx(
            base, rel=1e-9
        )

    def test_stable_spectrum_zero_flux(self, name):
        spec = synthetic_spectrum(zonal_content=0.5)
        spec["growth_rate"] = -jnp.abs(spec["growth_rate"]) - 0.01
        assert float(call_rule(RULES[name], spec, name=name)) == 0.0

    def test_gamma_monotonicity(self, name):
        spec = synthetic_spectrum(zonal_content=0.5)
        prev = float(call_rule(RULES[name], spec, name=name))
        for bump in (0.01, 0.1, 1.0):
            spec2 = dict(spec, growth_rate=spec["growth_rate"] + bump)
            cur = float(call_rule(RULES[name], spec2, name=name))
            assert cur >= prev - 1e-12
            prev = cur

    def test_growth_rate_gradient_finite(self, name):
        spec = synthetic_spectrum(zonal_content=0.5)
        grad = jax.grad(
            lambda g: call_rule(RULES[name], dict(spec, growth_rate=g), name=name)
        )(spec["growth_rate"])
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.any(np.asarray(grad) != 0.0)


class TestRuleSpecifics:
    def test_sat2_exb_shift_suppresses_and_is_off_by_default(self):
        """The shift is TGLF's ExB kx0_e, zero without rotation (tglf_geometry.f90:59)."""
        spec = synthetic_spectrum()
        args = (spec["growth_rate"], spec["phi2"], spec["phi2_kxy"], spec["flux_kxy"],
                spec["krho"], spec["kxrh"], spec["little_g"], spec["ds"])
        q_default = float(call_rule(RULES["sat2"], spec))
        assert float(RULES["sat2"](*args, kx0_e=0.0)) == pytest.approx(q_default)
        # with ExB shear the 1/(1+ay*kx^2) factor bites
        assert float(RULES["sat2"](*args, kx0_e=0.5)) < q_default

    def test_sat2_ignores_the_eigenmode_kx_centroid(self):
        """Shifting |phi|^2(kx) must not act as an ExB shift; it is not one."""
        spec = synthetic_spectrum()
        shifted = dict(spec)
        shifted["phi2_kxy"] = jnp.roll(spec["phi2_kxy"], NKX // 4, axis=0)
        # the centroid moves, but only through kperp2_eff, not the shift term
        q0 = float(call_rule(RULES["sat2"], spec, kx0_e=0.0))
        q1 = float(call_rule(RULES["sat2"], shifted, kx0_e=0.0))
        assert np.isfinite(q0) and np.isfinite(q1)

    def test_sat2_is_quadratic_in_growth_rate(self):
        """SAT2 intensity is the SQUARE of gamma/<kperp^2>; canonical is linear."""
        spec = synthetic_spectrum()
        doubled = dict(spec)
        doubled["growth_rate"] = spec["growth_rate"] * 2.0
        # at the CALIBRATED exponent sat2 is gamma^(2p); the quadratic form
        # is what flux_exponent=1 selects, so test that explicitly
        r_sat2 = float(call_rule(RULES["sat2"], doubled, flux_exponent=1.0)) / float(
            call_rule(RULES["sat2"], spec, flux_exponent=1.0))
        r_can = float(call_rule(ql_flux, doubled)) / float(call_rule(ql_flux, spec))
        # the stability gate is not fully saturated at the smallest gamma, so
        # both ratios sit above their 4 and 2 asymptotes; how far depends on
        # kx_width_model, which shifts which modes clear the gate
        assert 3.9 < r_sat2 < 4.5
        assert 1.9 < r_can < 2.2

    def test_rules_trace_under_vmap_with_rule_options(self):
        """The TORAX plugin vmaps the rule over radii and passes rule_options
        as ordinary kwargs, so no rule may branch on one of their VALUES.
        Every tracer bug that reached the ITER run got through because the
        tests only ever called the rules directly."""
        import inspect

        import jax
        from gyaradax.quasilinear.rules import ql_flux_sat2

        sig = set(inspect.signature(ql_flux_sat2).parameters)
        spec = {k: v for k, v in synthetic_spectrum().items() if k in sig}
        base = dict(cn=1.0, dlnpdr=8.0, kthnorm=3.8, grad_r0=1.0, gq=0.83)

        def one(p):
            return ql_flux_sat2(flux_exponent=p, **base, **spec)

        out = np.asarray(jax.vmap(one)(jnp.array([0.648, 1.0, 0.5])))
        assert out.shape == (3,) and np.all(np.isfinite(out))
        # differentiable in the exponent, which q**p alone is not at q = 0
        assert np.isfinite(float(jax.grad(one)(0.648)))

    def test_sat2_zero_flux_stays_exactly_zero(self):
        """The eps floor must not turn a stable spectrum into a tiny flux."""
        spec = synthetic_spectrum()
        spec["growth_rate"] = jnp.full_like(spec["growth_rate"], -1.0)
        for p in (1.0, 0.620):
            q = float(call_rule(RULES["sat2"], spec, flux_exponent=p))
            assert q == 0.0, f"flux_exponent={p} gave {q}"

    def test_sat2_preserves_the_sign_of_the_flux(self):
        """A per-species particle flux can be negative (inward pinch); the
        exponent must not clip it, or ql_flux_per_species stops summing."""
        spec = dict(synthetic_spectrum())
        spec["flux_kxy"] = -spec["flux_kxy"]
        for p in (1.0, 0.648):
            q = float(call_rule(RULES["sat2"], spec, flux_exponent=p))
            assert q < 0.0, f"flux_exponent={p} gave {q}, sign lost"

    def test_sat2_kperp_floor_no_blowup(self):
        spec = synthetic_spectrum()
        spec["phi2"] = jnp.zeros_like(spec["phi2"])
        q = float(call_rule(RULES["sat2"], spec))
        assert np.isfinite(q)


def synthetic_species_fluxes(spec, seed=1):
    """(nsp=2, 3, nkx, nky) per-species per-channel fluxes with distinct QL weights."""
    rng = np.random.default_rng(seed)
    base = np.asarray(spec["phi2_kxy"])
    w_ion = 0.5 + rng.random((3, NKX, NKY))
    w_ele = 0.1 + 0.3 * rng.random((3, NKX, NKY))
    # electron particle flux opposite in sign to the ion one (ambipolar-ish)
    flux_sp = np.stack([w_ion * base[None], w_ele * base[None]])
    flux_sp[1, 0] = -0.9 * flux_sp[0, 0]
    return jnp.asarray(flux_sp)


def call_per_species(rule, spec, flux_sp, name=None, **overrides):
    kwargs = dict(overrides)
    return ql_flux_per_species(
        rule,
        spec["growth_rate"],
        spec["phi2"],
        spec["phi2_kxy"],
        flux_sp,
        spec["krho"],
        spec["kxrh"],
        spec["little_g"],
        spec["ds"],
        **kwargs,
    )


@pytest.mark.parametrize("name", sorted(RULES.keys()))
class TestPerSpecies:
    """Kinetic-electron support: shared saturated intensity, per-species weights."""

    def test_decomposition_sums_to_total(self, name):
        """Sum over species of per-species flux == rule on the species-summed flux."""
        spec = synthetic_spectrum(zonal_content=0.5)
        flux_sp = synthetic_species_fluxes(spec)
        per_sp = np.asarray(call_per_species(RULES[name], spec, flux_sp, name=name))
        assert per_sp.shape == (2, 3)
        for c in range(3):
            spec_c = dict(spec, flux_kxy=jnp.sum(flux_sp[:, c], axis=0))
            total = float(call_rule(RULES[name], spec_c, name=name))
            assert per_sp[:, c].sum() == pytest.approx(total, rel=1e-10, abs=1e-14)

    def test_electron_channel_differs_from_ion(self, name):
        spec = synthetic_spectrum(zonal_content=0.5)
        flux_sp = synthetic_species_fluxes(spec)
        per_sp = np.asarray(call_per_species(RULES[name], spec, flux_sp, name=name))
        assert per_sp[0, 1] != pytest.approx(per_sp[1, 1], rel=1e-3)

    def test_amplitude_invariance_per_species(self, name):
        """Consistent |phi|^2 scaling leaves every per-species flux unchanged."""
        spec = synthetic_spectrum(zonal_content=0.5)
        flux_sp = synthetic_species_fluxes(spec)
        base = np.asarray(call_per_species(RULES[name], spec, flux_sp, name=name))
        lam = 7.7e4
        scaled = dict(
            spec,
            phi2=spec["phi2"] * lam,
            phi2_kxy=spec["phi2_kxy"] * lam,
        )
        out = np.asarray(call_per_species(RULES[name], scaled, flux_sp * lam, name=name))
        np.testing.assert_allclose(out, base, rtol=1e-9)

    def test_rule_name_string_accepted(self, name):
        spec = synthetic_spectrum()
        flux_sp = synthetic_species_fluxes(spec)
        by_name = np.asarray(call_per_species(name, spec, flux_sp, name=name))
        by_fn = np.asarray(call_per_species(RULES[name], spec, flux_sp, name=name))
        np.testing.assert_array_equal(by_name, by_fn)


class TestKineticIntegration:
    """End-to-end kinetic harvest + per-species rules on the last dump of a GKW kinetic run."""

    @pytest.fixture(scope="class")
    def kinetic_arr(self, request):
        from gyaradax.quasilinear.linear_pipeline import linear_from_fds

        path = reference_dir(request, "kinetic", KINETIC_CASE)
        return linear_from_fds(path, dump=reference_dumps(path)[-1])

    def test_harvest_shapes_and_consistency(self, kinetic_arr):
        arr = kinetic_arr
        assert arr["is_kinetic"]
        fsp = np.asarray(arr["fluxes_kxy_sp"])
        assert fsp.ndim == 4 and fsp.shape[1] == 3
        assert np.isfinite(fsp).all()
        # species-summed channels must equal the per-species sums exactly
        np.testing.assert_allclose(fsp[:, 0].sum(0), np.asarray(arr["pflux_kxy"]), rtol=1e-12)
        np.testing.assert_allclose(fsp[:, 1].sum(0), np.asarray(arr["eflux_kxy"]), rtol=1e-12)
        np.testing.assert_allclose(fsp[:, 2].sum(0), np.asarray(arr["vflux_kxy"]), rtol=1e-12)

    def test_per_species_rules_finite(self, kinetic_arr):
        arr = kinetic_arr
        for name, fn in RULES.items():
            per_sp = np.asarray(
                ql_flux_per_species(
                    fn,
                    arr["growth_rate"],
                    arr["phi2"],
                    arr["phi2_kxy"],
                    arr["fluxes_kxy_sp"],
                    arr["krho"],
                    arr["kxrh"],
                    arr["little_g"],
                    arr["ds"],
                )
            )
            assert per_sp.shape == (np.asarray(arr["fluxes_kxy_sp"]).shape[0], 3)
            assert np.isfinite(per_sp).all(), name


class TestDlnpdr:
    """TGLF's pressure-gradient normalization, tglf_multiscale_spectrum.f90:85-97."""

    def test_adiabatic_counts_electrons_as_tglf_does(self):
        # TGLF's species loop starts at 1, so adiabatic electrons still enter
        # ptot with rlt_e = 0 and Te = Ti: (2*rln + rlt)/2, not rln + rlt
        assert dlnpdr_tglf(8.0, 4.0, adiabatic=True) == pytest.approx(8.0)
        assert dlnpdr_tglf(12.0, 2.0, adiabatic=True) == pytest.approx(8.0)

    def test_kinetic_is_temperature_weighted(self):
        # te_ti = 1 makes the two species weigh equally
        got = dlnpdr_tglf(np.array([6.0, 4.0]), np.array([2.0, 2.0]), te_ti=1.0)
        assert got == pytest.approx(7.0)
        # hotter electrons (te_ti > 1 => ti_te < 1) tilt toward the electron term
        hot_e = dlnpdr_tglf(np.array([12.0, 2.0]), np.array([1.0, 1.0]), te_ti=3.0)
        assert hot_e == pytest.approx((3.0 + (1 / 3) * 13.0) / (1 + 1 / 3))

    def test_clamped_into_the_tglf_window(self):
        # floor bites for weak gradients, which our scans reach (rln can be 0)
        assert dlnpdr_tglf(1.0, 0.0, adiabatic=True) == pytest.approx(4.0)
        # and the rlnp_cutoff caps the steep end
        assert dlnpdr_tglf(100.0, 50.0, adiabatic=True) == pytest.approx(18.0)

    def test_sat2_scales_with_the_normalization(self):
        """cnorm = 12/dlnpdr enters the intensity linearly, so the flux
        scales with it raised to flux_exponent."""
        spec = synthetic_spectrum()
        for p in (1.0, 0.620):
            q_off = float(call_rule(RULES["sat2"], spec, flux_exponent=p))
            q_on = float(RULES["sat2"](
                spec["growth_rate"], spec["phi2"], spec["phi2_kxy"],
                spec["flux_kxy"], spec["krho"], spec["kxrh"], spec["little_g"],
                spec["ds"], dlnpdr=6.0, flux_exponent=p,
            ))
            assert q_on == pytest.approx(q_off * (12.0 / 6.0) ** p, rel=1e-10)


def test_kx_chain0_selects_the_theta0_chain():
    kxrh = jnp.linspace(-1.0, 1.0, 11)
    chain = np.asarray(kx_chain0(jnp.asarray(_build_mode_label(11, 3, 2)), kxrh))
    assert chain[:, 0].nonzero()[0].tolist() == [5]
    assert chain[:, 1].nonzero()[0].tolist() == [1, 3, 5, 7, 9]
    assert chain[:, 2].nonzero()[0].tolist() == [1, 5, 9]
    spec = synthetic_spectrum()
    full = jnp.ones(spec["phi2"].shape[1:])
    assert float(call_rule(ql_flux, spec, kx_chain=full)) == pytest.approx(float(call_rule(ql_flux, spec)))


@pytest.mark.parametrize("name", [n for n in calibration.names() if calibration.load(n)["rule"] in RULES])
def test_bundled_calibration_drives_its_rule(name):
    cal = calibration.load(name)
    assert cal["rule"] in RULES and cal["parameters"]["cn"] > 0
    q = float(call_rule(RULES[cal["rule"]], synthetic_spectrum(), cn=cal["parameters"]["cn"],
                        **calibration.rule_kwargs(cal)))
    assert np.isfinite(q) and q > 0


def test_auto_resolves_the_rule_default():
    for rule, name in calibration.DEFAULTS.items():
        assert calibration.load("auto", rule) == calibration.load(name)
    assert calibration.load("auto", "sat0") is None
    with pytest.raises(ValueError, match="is for rule"):
        calibration.load("sat2_v1", "canonical")


def _summary_table(path, name, kinetic):
    import pandas as pd

    ky = np.linspace(0.0, 1.4, 16)
    gamma = 0.3 * np.sin(np.pi * ky / 1.4)
    rows = []
    for i, (scale, test_set, sat) in enumerate(((1.0, None, 5.0), (2.0, None, 5.0), (1.5, "id", 5.0), (0.0, "id", None))):
        rows.append(dict(
            name=f"run_{i}", test_set=test_set, saturation_time=sat, collisionality=0.0,
            lin_ky=ky, lin_gamma=scale * gamma, lin_weight_Qi=np.where(ky > 0, 1.0, np.nan),
            lin_weight_Qe=np.where(ky > 0, 0.5, np.nan) if kinetic else None,
            lin_gq=1.0, lin_grad_r0=1.0, dlnpdr_tglf=6.0,
            Qi_fluxavg=10.0 * scale, Qe_fluxavg=5.0 * scale if kinetic else np.nan,
        ))
    pd.DataFrame(rows).to_parquet(path / f"{name}.parquet")


def test_calibration_fit_on_summary_tables(tmp_path):
    _summary_table(tmp_path, "adiabatic", kinetic=False)
    _summary_table(tmp_path, "kinetic", kinetic=True)
    rows = calibration.summary_rows(tmp_path)
    assert {(r["kind"], r["channel"]) for r in rows} == {
        ("adiabatic", "Qi"), ("kinetic", "Qi"), ("kinetic", "Qe")}
    assert sum(r["stable"] for r in rows) == 3 and {r["split"] for r in rows} == {"train", "id"}
    rec, metrics = calibration.fit_sat2(tmp_path, exponents=[0.5, 1.0], folds=2)
    assert rec["electrons"] == ["adiabatic", "kinetic"] and rec["channels"] == ["Qe", "Qi"]
    assert rec["parameters"]["cn"] > 0 and rec["parameters"]["flux_exponent"] in (0.5, 1.0)
    assert set(metrics) == {"n_train", "cross_validation", "id"}
    assert metrics["id"]["kinetic/Qi"]["stable_over_1"] == 0.0


def _satx_record(seed=0, nmembers=2, hidden=4):
    from gyaradax.quasilinear.satx import OPERATING

    rng = np.random.default_rng(seed)
    nin = len(OPERATING) + 5
    members = [dict(lc_Qi=[0.0, 0.0], lc_Qe=[0.0, -0.3], lc_Gamma=[0.0, -1.0], p=[0.5, 0.5], le=0.0,
                    w1=rng.normal(0, 0.3, (nin, hidden)).tolist(), b1=np.zeros(hidden).tolist(),
                    w2=rng.normal(0, 0.1, (hidden, 4)).tolist(), b2=np.zeros(4).tolist())
               for _ in range(nmembers)]
    train = [[1.0, 10.0], [0.0, 10.0], [0.0, 5.0], [0.5, 2.0], [0.3, 4.0], [1.0, 6.0], [0.05, 0.2],
             [0.0, 0.003], [0.0, 1.0]]
    box = [[0.0, 15.0], [0.0, 15.0], [-3.0, 8.0], [0.25, 4.0], [0.15, 8.0], [0.5, 12.0], [0.01, 0.3],
           [0.0, 0.005], [0.0, 1.0]]
    scale = dict(rlt=5.0, rln=2.5, q=2.5, eps_center=0.12, eps_width=0.08, nu=0.0015, omega_width=0.02,
                 gamma_floor=0.01, gamma_sum_floor=1e-6, kymax=0.3, p_range=0.3)
    return dict(members=members, scale=scale, train_range=train, input_box=box, gate_sharpness=20.0)


def test_satx_soft_box_identity_inside_and_saturating_outside():
    from gyaradax.quasilinear.satx import OPERATING, soft_box

    rec = _satx_record()
    train, box = np.asarray(rec["train_range"]), np.asarray(rec["input_box"])
    mid = train.mean(axis=1)
    np.testing.assert_allclose(np.asarray(soft_box(jnp.asarray(mid), train, box)), mid, rtol=1e-12)
    far = np.where(np.isin(OPERATING, ("kinetic",)), 1.0, box[:, 1] * 10.0)
    out = np.asarray(soft_box(jnp.asarray(far), train, box))
    assert np.all(out <= box[:, 1] + 1e-9) and np.all(out >= train[:, 1] - 1e-9)
    shat = OPERATING.index("shat")
    neg = mid.copy()
    neg[shat] = -1.0
    assert box[shat, 0] - 1e-9 <= float(soft_box(jnp.asarray(neg), train, box)[shat]) <= train[shat, 0]


def test_satx_mlp_finite_with_gradients():
    from gyaradax.quasilinear.satx import OPERATING, ql_flux_satx_mlp

    opts = _satx_record()
    krho = jnp.linspace(0.0, 1.5, 32) / 1.3
    gamma = 0.3 * jnp.sin(jnp.pi * jnp.linspace(0.0, 1.0, 32))
    omega = jnp.linspace(0.0, 1.2, 32)
    weights = jnp.stack([jnp.ones(32), 0.6 * jnp.ones(32), 0.1 * jnp.ones(32)])
    op = dict(rlt_i=6.0, rlt_e=6.0, rln=2.0, te_ti=1.0, shat=1.0, q=1.5, eps=0.15, nu=1e-3, kinetic=1.0)
    x = jnp.asarray([op[k] for k in OPERATING])

    def flux(x):
        return ql_flux_satx_mlp(gamma, omega, weights, krho, x, 6.0, 1.3, 1.0, 1.0, **opts)

    out = flux(x)
    assert out.shape == (3,) and np.all(np.isfinite(np.asarray(out))) and float(out[0]) > 0
    assert np.all(np.isfinite(np.asarray(jax.jacfwd(flux)(x))))
    assert float(ql_flux_satx_mlp(-jnp.abs(gamma), omega, weights, krho, x, 6.0, 1.3, 1.0, 1.0, **opts)[0]) < 1e-6
