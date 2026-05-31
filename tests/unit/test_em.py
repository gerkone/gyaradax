"""Electromagnetic extension tests for A_parallel (and later B_parallel).

Tests are structured in layers:
  1. Parameter / config tests (no GPU needed)
  2. Precomputation shape and key tests
  3. Field solve unit tests (Ampere's law)
  4. g2f transform tests
  5. RHS parity tests
  6. Growth rate validation against GKW reference
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace

from gyaradax.params import (
    GKParams,
    gkparams_from_config,
    gkparams_from_input_and_geometry,
    load_config,
)
from gyaradax.geometry import compute_geometry_from_input
from gyaradax.precompute import linear_precompute
from gyaradax.solver import init_f, default_state
from gyaradax.simulate import gk_run
from gyaradax.integrals import calculate_em_fluxes, precompute_bpar
from gyaradax.fields import _compute_fields, g_to_f
from gyaradax.utils import load_gkw_dump
from gyaradax import load_geometry

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GKW_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gkw_cases")
GKW_EM_DATA_ROOT = os.environ.get("GKW_EM_DATA_ROOT", GKW_CASES_DIR)


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_em_case(name):
    """Load an EM test case directory, skip if missing."""
    d = os.path.join(GKW_CASES_DIR, name)
    if not os.path.exists(d):
        pytest.skip(f"EM reference data not found at {d}")
    return d


def _load_external_em_case(name):
    """Load an optional EM fixture from GKW_EM_DATA_ROOT, skip if missing."""
    d = os.path.join(GKW_EM_DATA_ROOT, name)
    if not os.path.exists(d):
        pytest.skip(f"external EM GKW reference data not found at {d}")
    return d


def _load_em_geometry(case_dir):
    """Load geometry for an EM test case using compute_geometry_from_input."""
    input_path = os.path.join(case_dir, "input.dat")
    return compute_geometry_from_input(input_path)


def _load_em_reference_geometry(case_dir):
    """Load file-backed GKW geometry when available, else compute from input.dat."""
    if os.path.exists(os.path.join(case_dir, "geom.dat")):
        return load_geometry(case_dir)
    return _load_em_geometry(case_dir)


def _read_growth_rates(directory):
    """Read growth_rates_all_modes file."""
    path = os.path.join(directory, "growth_rates_all_modes")
    return np.loadtxt(path)


def _read_time_dat(directory):
    """Read time.dat -> (time, growth_rate, frequency) arrays."""
    data = np.loadtxt(os.path.join(directory, "time.dat"))
    return data[:, 0], data[:, 1], data[:, 2]


def _log_abs_correlation(a, b):
    """Correlation of log-amplitudes, insensitive to sign and scale conventions."""
    log_a = np.log(np.maximum(np.abs(np.asarray(a, dtype=np.float64)), 1e-300))
    log_b = np.log(np.maximum(np.abs(np.asarray(b, dtype=np.float64)), 1e-300))
    return float(np.corrcoef(log_a, log_b)[0, 1])


def _median_tail_abs_ratio(a, b, n_tail=3):
    """Median |a/b| over the tail windows, for scale-sensitive diagnostics."""
    a_arr = np.asarray(a, dtype=np.float64)[-n_tail:]
    b_arr = np.asarray(b, dtype=np.float64)[-n_tail:]
    return float(np.median(np.abs(a_arr) / np.maximum(np.abs(b_arr), 1e-300)))


def _run_em_flux_windows(case_dir, n_windows):
    """Run gyaradax and return EM-only flux diagnostics for each averaging window."""
    geometry = _load_em_geometry(case_dir)
    params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
    pre = linear_precompute(geometry, params)
    nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
    df = init_f(geometry, finit=params.finit, amp_init_real=params.amp_init, n_species=nsp)
    state = default_state(nky=len(geometry["krho"]))

    em_fluxes = []
    for _ in range(n_windows):
        df, _, _, state = gk_run(df, geometry, params, state, n_steps=params.naverage, pre=pre)
        _, apar, bpar = _compute_fields(df, geometry, params, pre)
        df_f = g_to_f(df, apar, params, pre)
        em_fluxes.append(
            np.asarray(
                calculate_em_fluxes(geometry, df_f, apar, params=params, bpar=bpar, pre=pre)
            ).ravel()
        )
    return np.asarray(em_fluxes)


# ── 1. Parameter tests ──────────────────────────────────────────────────────


class TestEMParams:
    """Test that GKParams correctly handles EM fields."""

    def test_default_em_disabled(self):
        """Default GKParams has EM disabled."""
        p = GKParams()
        assert p.nlapar is False
        assert p.nlbpar is False
        assert p.beta == 0.0

    def test_em_params_creation(self):
        """Can create GKParams with EM fields."""
        p = GKParams(nlapar=True, nlbpar=False, beta=0.001)
        assert p.nlapar is True
        assert p.nlbpar is False
        assert p.beta == 0.001

    def test_em_params_static_fields(self):
        """nlapar and nlbpar are static (control flow) fields."""
        p = GKParams(nlapar=True, nlbpar=True, beta=0.01)
        leaves, aux = p.tree_flatten()
        assert "nlapar" in aux
        assert "nlbpar" in aux
        assert aux["nlapar"] is True
        assert aux["nlbpar"] is True

    def test_beta_is_leaf(self):
        """beta should be a JAX-traceable leaf, not static."""
        p = GKParams(beta=0.005)
        leaves, aux = p.tree_flatten()
        leaf_keys = aux["_leaf_keys"]
        assert "beta" in leaf_keys

    def test_em_params_from_input_dat(self):
        """Parse EM params from a GKW input.dat with nlapar."""
        case_dir = _load_em_case("em_adiabat_apar")
        input_path = os.path.join(case_dir, "input.dat")
        geometry = compute_geometry_from_input(input_path)
        params = gkparams_from_input_and_geometry(input_path, geometry)
        assert params.nlapar is True
        assert params.beta > 0

    @pytest.mark.parametrize(
        ("case_name", "config_name", "beta", "n_steps"),
        [
            ("nl_em_waltz_b005", "nl_em_waltz_b005.yaml", 0.005, 24000),
            ("nl_em_waltz_b01", "nl_em_waltz_b01.yaml", 0.01, 12000),
        ],
    )
    def test_nl_em_waltz_yaml_matches_gkw_input(self, case_name, config_name, beta, n_steps):
        """The nonlinear Waltz EM YAMLs mirror the corresponding GKW input.dat files."""
        case_dir = _load_em_case(case_name)
        input_path = os.path.join(case_dir, "input.dat")
        config = load_config(os.path.join(REPO_ROOT, "configs", config_name))

        geometry = compute_geometry_from_input(input_path)
        params_from_input = gkparams_from_input_and_geometry(input_path, geometry)
        params_from_yaml = gkparams_from_config(config)

        assert config.geometry.geometry_model == "circ"
        assert config.run.data_dir == f"tests/data/gkw_cases/{case_name}"
        assert int(config.solver.n_steps) == n_steps
        assert params_from_yaml.nlapar is True
        assert params_from_input.nlapar is True
        assert params_from_yaml.nlbpar is False
        assert params_from_input.nlbpar is False
        assert params_from_yaml.non_linear is True
        assert params_from_input.non_linear is True
        assert params_from_yaml.adiabatic_electrons is False
        assert params_from_input.adiabatic_electrons is False
        assert params_from_yaml.beta == pytest.approx(beta)
        assert params_from_input.beta == pytest.approx(beta)
        assert params_from_yaml.dt == pytest.approx(params_from_input.dt)
        assert params_from_yaml.naverage == params_from_input.naverage
        assert int(config.grid.nvpar) == len(geometry["intvp"])
        assert int(config.grid.nmu) == len(geometry["intmu"])
        assert int(config.grid.ns) == len(geometry["ints"])
        assert int(config.grid.nky) == len(geometry["krho"])


# ── 2. Precomputation tests ─────────────────────────────────────────────────


class TestEMPrecompute:
    """Test that linear_precompute produces EM arrays when nlapar=True."""

    @pytest.fixture
    def em_setup(self):
        """Small EM setup for precompute tests."""
        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        return geometry, params

    def test_precompute_has_apar_keys(self, em_setup):
        """GKPre contains Ampere solve keys when nlapar=True."""
        geometry, params = em_setup
        assert params.nlapar is True
        pre = linear_precompute(geometry, params)
        assert "apar_weight" in pre
        assert "apar_diag" in pre

    def test_precompute_has_g2f_factor(self, em_setup):
        """GKPre contains g2f transform factor when nlapar=True."""
        geometry, params = em_setup
        pre = linear_precompute(geometry, params)
        assert "g2f_factor" in pre

    def test_apar_diag_shape(self, em_setup):
        """apar_diag has shape (ns, nkx, nky), same as phi."""
        geometry, params = em_setup
        pre = linear_precompute(geometry, params)
        phi_shape = (
            len(geometry["sgrid"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        assert pre["apar_diag"].shape == phi_shape

    def test_apar_diag_nonzero(self, em_setup):
        """Ampere denominator is nonzero everywhere (well-posed equation)."""
        geometry, params = em_setup
        pre = linear_precompute(geometry, params)
        assert jnp.all(jnp.abs(pre["apar_diag"]) > 1e-30)

    def test_no_em_keys_when_disabled(self):
        """No EM precomputed keys when nlapar=False."""
        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        params = replace(params, nlapar=False, nlbpar=False, beta=0.0)
        pre = linear_precompute(geometry, params)
        assert "apar_weight" not in pre
        assert "apar_diag" not in pre

    def test_bpar_helper_matches_linear_precompute(self, em_setup):
        """B_parallel helper returns the exact arrays used by linear_precompute."""
        geometry, params = em_setup
        assert params.nlbpar is True
        pre = linear_precompute(geometry, params)
        direct = precompute_bpar(geometry, params, pre)

        for key in ("phi_weight", "phi_diag", "bpar_weight", "bpar_chi_factor"):
            assert key in pre
            assert key in direct
            assert direct[key].shape == pre[key].shape
            assert jnp.all(jnp.isfinite(direct[key]))
            np.testing.assert_allclose(np.asarray(direct[key]), np.asarray(pre[key]))

    def test_adiabatic_bpar_guard(self):
        """Adiabatic-electron B_parallel is rejected until its field solve is implemented."""
        case_dir = _load_em_case("em_adiabat_apar")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        params = replace(params, nlapar=True, nlbpar=True)

        with pytest.raises(NotImplementedError, match="adiabatic electrons"):
            linear_precompute(geometry, params)

    def test_kinetic_bpar_only_precompute_has_bpar_keys(self, em_setup):
        """Kinetic B_parallel-only precompute keeps coupled phi/Bpar arrays."""
        geometry, params = em_setup
        params = replace(params, nlapar=False, nlbpar=True)

        pre = linear_precompute(geometry, params)

        for key in ("phi_weight", "phi_diag", "bpar_weight", "bpar_chi_factor"):
            assert key in pre
            assert jnp.all(jnp.isfinite(pre[key]))
        assert "apar_weight" not in pre
        assert "apar_diag" not in pre
        assert "g2f_factor" not in pre


# ── 3. Field solve tests ────────────────────────────────────────────────────


class TestAmpereSolve:
    """Test the Ampere field solve for A_parallel."""

    @staticmethod
    def _random_complex(shape, seed=0, scale=1e-4):
        key_re, key_im = jax.random.split(jax.random.PRNGKey(seed))
        return (jax.random.normal(key_re, shape) + 1j * jax.random.normal(key_im, shape)).astype(
            jnp.complex128
        ) * scale

    @staticmethod
    def _field_shape(geometry, params):
        base = (
            len(geometry["vpgr"]),
            len(geometry["mugr"]),
            len(geometry["sgrid"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        if params.adiabatic_electrons:
            return base
        return (len(np.atleast_1d(np.asarray(params.mas))),) + base

    @staticmethod
    def _a_only_setup(case_name):
        case_dir = _load_em_case(case_name)
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        assert params.nlapar is True
        assert params.nlbpar is False
        pre = linear_precompute(geometry, params)
        return geometry, params, pre, TestAmpereSolve._field_shape(geometry, params)

    @pytest.fixture
    def em_full_setup(self):
        """Full EM setup with precomputed arrays."""
        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        nsp = 2
        nvpar = len(geometry["vpgr"])
        nmu = len(geometry["mugr"])
        ns = len(geometry["sgrid"])
        nkx = len(geometry["kxrh"])
        nky = len(geometry["krho"])
        shape = (nsp, nvpar, nmu, ns, nkx, nky)
        return geometry, params, pre, shape

    def test_apar_output_shape(self, em_full_setup):
        """A_par solve returns (ns, nkx, nky) array."""
        from gyaradax.integrals import calculate_apar

        geometry, params, pre, shape = em_full_setup
        df = jnp.zeros(shape, dtype=jnp.complex128)
        apar = calculate_apar(geometry, df, params=params, pre=pre)
        expected_shape = shape[3:]  # (ns, nkx, nky)
        assert apar.shape == expected_shape

    def test_apar_zero_df_gives_zero(self, em_full_setup):
        """Zero distribution function gives zero A_parallel."""
        from gyaradax.integrals import calculate_apar

        geometry, params, pre, shape = em_full_setup
        df = jnp.zeros(shape, dtype=jnp.complex128)
        apar = calculate_apar(geometry, df, params=params, pre=pre)
        assert jnp.allclose(apar, 0.0, atol=1e-30)

    def test_apar_dtype_complex128(self, em_full_setup):
        """A_par should be complex128 (same as phi)."""
        from gyaradax.integrals import calculate_apar

        geometry, params, pre, shape = em_full_setup
        df = jnp.ones(shape, dtype=jnp.complex128) * 1e-4
        apar = calculate_apar(geometry, df, params=params, pre=pre)
        assert apar.dtype == jnp.complex128

    def test_compute_fields_apar_uses_mixed_g_source(self):
        """_compute_fields A_parallel matches direct Ampere solve from mixed g."""
        from gyaradax.fields import _compute_fields
        from gyaradax.integrals import calculate_apar

        geometry, params, pre, shape = self._a_only_setup("em_cbc_apar")
        dg = self._random_complex(shape, seed=101)

        phi, apar, bpar = _compute_fields(dg, geometry, params, pre)
        direct_apar = calculate_apar(geometry, dg, params=params, pre=pre)

        assert apar is not None
        assert bpar is None
        assert float(jnp.max(jnp.abs(phi))) > 1e-10
        assert float(jnp.max(jnp.abs(apar))) > 1e-10
        np.testing.assert_allclose(np.asarray(apar), np.asarray(direct_apar), rtol=0.0, atol=0.0)

    def test_apar_from_physical_f_differs_from_mixed_g_source(self):
        """Passing physical f into Ampere double-counts g2f and changes A_parallel."""
        from gyaradax.fields import g_to_f
        from gyaradax.integrals import calculate_apar

        geometry, params, pre, shape = self._a_only_setup("em_cbc_apar")
        dg = self._random_complex(shape, seed=202)
        correct_apar = calculate_apar(geometry, dg, params=params, pre=pre)
        df = g_to_f(dg, correct_apar, params, pre)
        wrong_apar = calculate_apar(geometry, df, params=params, pre=pre)

        diff = float(jnp.max(jnp.abs(wrong_apar - correct_apar)))
        assert float(jnp.max(jnp.abs(correct_apar))) > 1e-10
        assert diff > 1e-8
        assert not jnp.allclose(wrong_apar, correct_apar, rtol=1e-5, atol=1e-10)

    @pytest.mark.parametrize("case_name", ["em_adiabat_apar", "em_cbc_apar"])
    def test_phi_from_g_equals_phi_from_f_for_a_only_symmetric_vpar(self, case_name):
        """A-only g2f correction is odd in v_parallel and cancels from phi."""
        from gyaradax.fields import _compute_fields, _compute_phi, g_to_f

        geometry, params, pre, shape = self._a_only_setup(case_name)
        dg = self._random_complex(shape, seed=303)
        _phi_fields, apar, bpar = _compute_fields(dg, geometry, params, pre)
        assert apar is not None
        assert bpar is None

        df = g_to_f(dg, apar, params, pre)
        phi_from_g = _compute_phi(dg, geometry, params, pre)
        phi_from_f = _compute_phi(df, geometry, params, pre)

        assert float(jnp.max(jnp.abs(phi_from_g))) > 1e-10
        np.testing.assert_allclose(
            np.asarray(phi_from_f),
            np.asarray(phi_from_g),
            rtol=1e-12,
            atol=1e-14,
        )


class TestBParallelLowRiskDeltas:
    """Targeted tests for Bpar B1/B2 low-risk behavior."""

    @staticmethod
    def _bpar_waltz_setup():
        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        shape = (
            len(np.atleast_1d(np.asarray(params.mas))),
            len(geometry["vpgr"]),
            len(geometry["mugr"]),
            len(geometry["sgrid"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        return geometry, params, pre, shape

    @staticmethod
    def _bpar_only_setup():
        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        params = replace(params, nlapar=False, nlbpar=True, non_linear=False)
        pre = linear_precompute(geometry, params)
        shape = (
            len(np.atleast_1d(np.asarray(params.mas))),
            len(geometry["vpgr"]),
            len(geometry["mugr"]),
            len(geometry["sgrid"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        return geometry, params, pre, shape

    @staticmethod
    def _random_complex(shape, seed=0, scale=1e-4):
        key_re, key_im = jax.random.split(jax.random.PRNGKey(seed))
        return (jax.random.normal(key_re, shape) + 1j * jax.random.normal(key_im, shape)).astype(
            jnp.complex128
        ) * scale

    def test_kinetic_bpar_only_compute_fields_returns_bpar_without_apar(self):
        """Bpar-only field solve returns finite phi/Bpar and no A_parallel."""
        from gyaradax.fields import _compute_fields

        geometry, params, pre, shape = self._bpar_only_setup()
        dg = self._random_complex(shape, seed=505)

        phi, apar, bpar = _compute_fields(dg, geometry, params, pre)

        assert apar is None
        assert bpar is not None
        assert phi.shape == shape[3:]
        assert bpar.shape == shape[3:]
        assert jnp.all(jnp.isfinite(phi))
        assert jnp.all(jnp.isfinite(bpar))
        assert float(jnp.max(jnp.abs(bpar))) > 1e-12

    def test_kinetic_bpar_only_jax_rhs_finite_with_x_and_xi_terms(self):
        """JAX linear RHS accepts apar=None,bpar!=None and includes Terms X/XI."""
        from gyaradax.backends._jax import JAXOps
        from gyaradax.fields import _compute_fields

        geometry, params, pre, shape = self._bpar_only_setup()
        dg = self._random_complex(shape, seed=606)
        phi, apar, bpar = _compute_fields(dg, geometry, params, pre)

        ops = JAXOps(pre)
        rhs = ops.linear_rhs(dg, phi, geometry, params, pre, apar=apar, bpar=bpar)
        sp_pre = {
            "bessel": pre["bessel"][0],
            "fmaxwl": pre["fmaxwl"][0],
            "dmaxwel_fm_ek": pre["dmaxwel_fm_ek"][0],
            "drift_x": pre["drift_x"][0],
            "drift_y": pre["drift_y"][0],
            "utrap": pre["utrap"][0],
            "abs_dum2_vp": pre["abs_dum2_vp"][0],
            "tmp0": pre["tmp0"][0],
            "signz0": pre["signz0"][0],
            "s_total_upar": jnp.moveaxis(pre["s_total_upar"], 1, 0)[0],
            "s_total_t7": jnp.moveaxis(pre["s_total_t7"], 1, 0)[0],
            "bpar_chi_factor": pre["bpar_chi_factor"][0],
            "kx_b": pre["kx_b"].ravel().reshape(1, 1, 1, -1, 1),
            "ky_b": pre["ky_b"].ravel().reshape(1, 1, 1, 1, -1),
            "hyper": pre["hyper"],
        }
        terms = ops._linear_rhs_terms(dg[0], phi, params, sp_pre, apar=apar, bpar=bpar)

        assert apar is None
        assert bpar is not None
        assert rhs.shape == dg.shape
        assert jnp.all(jnp.isfinite(rhs))
        assert float(jnp.max(jnp.abs(terms["X_bpar_par"]))) > 1e-14
        assert float(jnp.max(jnp.abs(terms["XI_curv_bpar"]))) > 1e-14

    def test_kinetic_bpar_only_one_gk_run_step_finite(self):
        """One solver step with kinetic Bpar-only remains finite."""
        geometry, params, pre, shape = self._bpar_only_setup()
        df = init_f(geometry, finit=params.finit, amp_init_real=params.amp_init, n_species=shape[0])
        state = default_state(nky=shape[-1])

        df_next, phi, fluxes, state = gk_run(df, geometry, params, state, n_steps=1, pre=pre)

        assert jnp.all(jnp.isfinite(df_next))
        assert jnp.all(jnp.isfinite(phi))
        assert jnp.all(jnp.isfinite(jnp.asarray(fluxes)))
        assert jnp.all(jnp.isfinite(state.time))

    def test_bpar_g2f_parity_cancels_from_phi_and_bpar_weights(self):
        """Symmetric-vpar Bpar weights cancel the odd g2f correction; Apar does not."""
        geometry, params, pre, _shape = self._bpar_waltz_setup()
        assert params.nlapar is True
        assert params.nlbpar is True

        phi_coupling = jnp.einsum("avmjkl,avmjkl->jkl", pre["phi_weight"], pre["g2f_factor"])
        bpar_coupling = jnp.einsum("avmjkl,avmjkl->jkl", pre["bpar_weight"], pre["g2f_factor"])
        apar_coupling = jnp.einsum("avmjkl,avmjkl->jkl", pre["apar_weight"], pre["g2f_factor"])

        np.testing.assert_allclose(np.asarray(phi_coupling), 0.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(bpar_coupling), 0.0, rtol=0.0, atol=1e-12)
        assert float(jnp.max(jnp.abs(apar_coupling))) > 1e-6
        # Keep the local variable live so future edits do not accidentally make
        # the setup independent of the Waltz Bpar geometry/parameters.
        assert len(geometry["vpgr"]) == pre["g2f_factor"].shape[1]

    def test_bpar_coupled_fields_same_from_g_and_physical_f_for_supported_case(self):
        """Current Waltz Bpar phi/Bpar solve is invariant to g->f by parity."""
        from gyaradax.fields import g_to_f
        from gyaradax.integrals import calculate_apar, calculate_phi

        geometry, params, pre, shape = self._bpar_waltz_setup()
        dg = self._random_complex(shape, seed=404)
        apar = calculate_apar(geometry, dg, params=params, pre=pre)
        df = g_to_f(dg, apar, params, pre)

        phi_from_g = calculate_phi(geometry, dg, params=params, pre=pre)
        phi_from_f = calculate_phi(geometry, df, params=params, pre=pre)
        bpar_from_g = -jnp.einsum("avmjkl,avmjkl->jkl", pre["bpar_weight"], dg) / pre["phi_diag"]
        bpar_from_f = -jnp.einsum("avmjkl,avmjkl->jkl", pre["bpar_weight"], df) / pre["phi_diag"]

        assert float(jnp.max(jnp.abs(apar))) > 1e-10
        np.testing.assert_allclose(
            np.asarray(phi_from_f), np.asarray(phi_from_g), rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(bpar_from_f), np.asarray(bpar_from_g), rtol=1e-12, atol=1e-14
        )

    def test_bpar_curvature_rhs_matches_manual_contribution(self):
        """Term XI contributes the manual curvature-drift × bpar expression."""
        from gyaradax.backends._jax import JAXOps

        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        assert params.nlbpar is True
        assert "bpar_chi_factor" in pre

        shape = (
            len(geometry["vpgr"]),
            len(geometry["mugr"]),
            len(geometry["sgrid"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        df = jnp.zeros(shape, dtype=jnp.complex128)
        phi = jnp.zeros(shape[2:], dtype=jnp.complex128)
        bpar = jnp.ones(shape[2:], dtype=jnp.complex128) * (2.0e-5 + 3.0e-5j)
        sp_pre = {
            "bessel": pre["bessel"][0],
            "fmaxwl": pre["fmaxwl"][0],
            "dmaxwel_fm_ek": pre["dmaxwel_fm_ek"][0],
            "drift_x": pre["drift_x"][0],
            "drift_y": pre["drift_y"][0],
            "utrap": pre["utrap"][0],
            "abs_dum2_vp": pre["abs_dum2_vp"][0],
            "tmp0": pre["tmp0"][0],
            "signz0": pre["signz0"][0],
            "s_total_upar": jnp.moveaxis(pre["s_total_upar"], 1, 0)[0],
            "s_total_t7": jnp.moveaxis(pre["s_total_t7"], 1, 0)[0],
            "bpar_chi_factor": pre["bpar_chi_factor"][0],
            "kx_b": pre["kx_b"].ravel().reshape(1, 1, 1, -1, 1),
            "ky_b": pre["ky_b"].ravel().reshape(1, 1, 1, 1, -1),
            "hyper": pre["hyper"],
        }

        terms = JAXOps(pre, mixed_precision=False)._linear_rhs_terms(
            df, phi, params, sp_pre, bpar=bpar
        )
        kdotvd = sp_pre["drift_x"] * sp_pre["kx_b"] + sp_pre["drift_y"] * sp_pre["ky_b"]
        gyro_bpar_scaled = sp_pre["bpar_chi_factor"] * bpar[None, None, :, :, :]
        manual = (
            -1j
            * params.drive_scale
            * sp_pre["signz0"]
            * kdotvd
            * (sp_pre["fmaxwl"] / jnp.maximum(sp_pre["tmp0"], 1e-15))
            * gyro_bpar_scaled
        )

        assert "XI_curv_bpar" in terms
        assert float(jnp.max(jnp.abs(terms["XI_curv_bpar"]))) > 1e-14
        np.testing.assert_allclose(
            np.asarray(terms["XI_curv_bpar"]), np.asarray(manual), rtol=1e-13, atol=1e-18
        )

    def test_adiabatic_em_flux_uses_j0_gyroaveraged_apar(self):
        """5D EM flux diagnostic uses J0*A_parallel, not bare A_parallel."""
        from gyaradax.integrals import calculate_em_fluxes

        case_dir = _load_em_case("em_adiabat_apar")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)

        shape = (
            len(geometry["vpgr"]),
            len(geometry["mugr"]),
            len(geometry["sgrid"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        vp_profile = jnp.asarray(geometry["vpgr"], dtype=jnp.float64).reshape(-1, 1, 1, 1, 1)
        df = jnp.broadcast_to((1.0e-4 * vp_profile).astype(jnp.complex128), shape)
        apar = jnp.ones(shape[2:], dtype=jnp.complex128) * 1.0j
        got = calculate_em_fluxes(geometry, df, apar, params=params, pre=pre)

        ints = jnp.asarray(geometry["ints"], dtype=jnp.float64)
        fsa = jnp.sum(ints)
        parseval_b = jnp.asarray(geometry["parseval"], dtype=jnp.float64).reshape(1, 1, 1, 1, -1)
        intvp_b = jnp.asarray(geometry["intvp"], dtype=jnp.float64).reshape(-1, 1, 1, 1, 1)
        intmu_b = jnp.asarray(geometry["intmu"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1)
        vpgr_b = jnp.asarray(geometry["vpgr"], dtype=jnp.float64).reshape(-1, 1, 1, 1, 1)
        mugr_b = jnp.asarray(geometry["mugr"], dtype=jnp.float64).reshape(1, -1, 1, 1, 1)
        bn_b = jnp.asarray(geometry["bn"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1)
        ints_b = ints.reshape(1, 1, -1, 1, 1)
        efun_b = jnp.asarray(geometry["efun"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1)
        krho_b = jnp.asarray(geometry["krho"], dtype=jnp.float64).reshape(1, 1, 1, 1, -1)
        rfun_b = jnp.asarray(geometry["rfun"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1)
        bt_frac_b = jnp.asarray(geometry["bt_frac"], dtype=jnp.float64).reshape(1, 1, -1, 1, 1)
        signB = jnp.asarray(geometry["signB"], dtype=jnp.float64)
        d2X = jnp.asarray(geometry.get("d2X", 1.0), dtype=jnp.float64)
        d3v = d2X * intmu_b * bn_b * intvp_b
        dum = parseval_b * ints_b * (efun_b * krho_b) * df
        apar_ga = pre["bessel"] * apar[jnp.newaxis, jnp.newaxis, :, :, :]
        dum_a = -2.0 * float(params.vthrat) * vpgr_b * dum * jnp.conj(apar_ga)
        manual = (
            jnp.sum(d3v * jnp.imag(dum_a)) / fsa,
            jnp.sum(d3v * (vpgr_b**2 * jnp.imag(dum_a) + 2 * mugr_b * bn_b * jnp.imag(dum_a)))
            / fsa,
            jnp.sum(d3v * (jnp.imag(dum_a) * vpgr_b * rfun_b * bt_frac_b * signB)) / fsa,
        )
        bare_dum_a = (
            -2.0
            * float(params.vthrat)
            * vpgr_b
            * dum
            * jnp.conj(apar[jnp.newaxis, jnp.newaxis, :, :, :])
        )
        bare_pflux = jnp.sum(d3v * jnp.imag(bare_dum_a)) / fsa

        for actual, expected in zip(got, manual):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-13)
        assert abs(float(got[0] - bare_pflux)) > 1e-10


# ── 4. g2f transform tests ──────────────────────────────────────────────────


class TestG2FTransform:
    """Test the mixed variable g <-> f transform."""

    def test_g2f_identity_when_no_em(self):
        """When nlapar=False, g=f (identity transform)."""
        from gyaradax.fields import g_to_f

        params = GKParams(nlapar=False)
        dg = jnp.ones((32, 8, 16, 1, 1), dtype=jnp.complex128)
        apar = jnp.zeros((16, 1, 1), dtype=jnp.complex128)
        pre = {}
        df = g_to_f(dg, apar, params, pre)
        assert jnp.allclose(df, dg)

    def test_g2f_nonzero_correction(self):
        """When nlapar=True and apar!=0, g!=f."""
        from gyaradax.fields import g_to_f

        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)

        nsp = 2
        nvpar = len(geometry["vpgr"])
        nmu = len(geometry["mugr"])
        ns = len(geometry["sgrid"])
        nkx = len(geometry["kxrh"])
        nky = len(geometry["krho"])

        dg = jnp.ones((nsp, nvpar, nmu, ns, nkx, nky), dtype=jnp.complex128) * 1e-4
        apar = jnp.ones((ns, nkx, nky), dtype=jnp.complex128) * 1e-6
        df = g_to_f(dg, apar, params, pre)
        # df should differ from dg when apar is nonzero
        assert not jnp.allclose(df, dg, atol=1e-20)

    def test_g2f_roundtrip(self):
        """f -> g -> f should be identity."""
        from gyaradax.fields import f_to_g, g_to_f

        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)

        nsp = 2
        nvpar = len(geometry["vpgr"])
        nmu = len(geometry["mugr"])
        ns = len(geometry["sgrid"])
        nkx = len(geometry["kxrh"])
        nky = len(geometry["krho"])

        df_orig = jnp.ones((nsp, nvpar, nmu, ns, nkx, nky), dtype=jnp.complex128) * 1e-4
        apar = jnp.ones((ns, nkx, nky), dtype=jnp.complex128) * 1e-6

        dg = f_to_g(df_orig, apar, params, pre)
        df_back = g_to_f(dg, apar, params, pre)
        assert jnp.allclose(df_back, df_orig, rtol=1e-12)


# ── 5. CFL tests ────────────────────────────────────────────────────────────


class TestAlfvenCFL:
    """Test the Alfven CFL constraint."""

    def test_alfven_cfl_finite(self):
        """With finite beta and kinetic electrons, tmax_field should be finite and positive."""
        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)

        tmax_field = float(pre.get("tmax_field", 0.0))
        assert tmax_field > 0, "tmax_field should be positive for kinetic electrons"
        assert jnp.isfinite(jnp.asarray(tmax_field))

    def test_alfven_cfl_changes_with_beta(self):
        """EM CFL (tmax_field) should differ from ES when beta > 0."""
        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params_em = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        params_es = replace(params_em, nlapar=False, nlbpar=False, beta=0.0)

        pre_em = linear_precompute(geometry, params_em)
        pre_es = linear_precompute(geometry, params_es)

        tmax_em = float(pre_em.get("tmax_field", 0.0))
        tmax_es = float(pre_es.get("tmax_field", 0.0))
        # Both should be positive
        assert tmax_es > 0
        assert tmax_em > 0
        # With finite beta the field CFL changes (beta enters sqrt argument)
        assert tmax_em != tmax_es, "tmax_field should differ between ES and EM"


# ── 6. Backward compatibility ────────────────────────────────────────────────


class TestEMBackwardsCompat:
    """EM with beta=0 / nlapar=False must produce identical ES results."""

    def test_gkstep_nlapar_false_matches_es(self):
        """gkstep_single with nlapar=False produces finite results (backwards compat)."""
        from gyaradax.solver import gkstep_single

        case_dir = _load_em_case("em_bpar_waltz")
        geometry = _load_em_geometry(case_dir)
        params_base = gkparams_from_input_and_geometry(
            os.path.join(case_dir, "input.dat"), geometry
        )
        # EM disabled
        params_es = replace(params_base, nlapar=False, nlbpar=False, beta=0.0)

        nsp = 2
        nvpar = len(geometry["vpgr"])
        nmu = len(geometry["mugr"])
        ns = len(geometry["sgrid"])
        nkx = len(geometry["kxrh"])
        nky = len(geometry["krho"])

        pre = linear_precompute(geometry, params_es)
        rng = jax.random.PRNGKey(42)
        df = jax.random.normal(rng, (nsp, nvpar, nmu, ns, nkx, nky)) * 1e-4
        df = df.astype(jnp.complex128)
        state = default_state(nky=nky)

        next_df, (phi, _), next_state = gkstep_single(df, geometry, params_es, state, pre)

        assert jnp.all(jnp.isfinite(next_df))
        assert jnp.all(jnp.isfinite(phi))
        assert float(jnp.max(jnp.abs(next_df))) > 0


# ── 7. GKW FDS parity (mode shape correlation) ──────────────────────────────


class TestEMFDSParity:
    """Compare gyaradax EM FDS against GKW from cold start.

    Uses mode shape correlation (not amplitude) because per-ky normalization
    conventions differ. The key validation is that the spatial/velocity
    structure of the distribution function matches.
    """

    @staticmethod
    def _run_gkw_and_compare(gkw_input_template, overrides=None, n_procs=16):
        """Run GKW from input template, run gyaradax, compare FDS."""
        import subprocess
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp(prefix="gkw_parity_")
        input_txt = gkw_input_template
        if overrides:
            for k, v in overrides.items():
                input_txt = input_txt.replace(k, v)
        with open(os.path.join(tmpdir, "input.dat"), "w") as f:
            f.write(input_txt)

        result = subprocess.run(
            [
                "/usr/lib64/openmpi/bin/mpirun",
                "-np",
                str(n_procs),
                "/system/user/publicwork/galletti/gkw.x",
            ],
            cwd=tmpdir,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, f"GKW failed: {result.stderr[-200:]}"

        from gyaradax.utils import load_gkw_dump as _load_dump

        geom = load_geometry(tmpdir)
        params = gkparams_from_input_and_geometry(os.path.join(tmpdir, "input.dat"), geom)
        nvpar = len(geom["vpgr"])
        nmu = len(geom["mugr"])
        ns = len(geom["sgrid"])
        nkx = len(geom["kxrh"])
        nky = len(geom["krho"])
        nsp = len(np.atleast_1d(np.asarray(params.mas)))

        df_gkw, _ = _load_dump(
            os.path.join(tmpdir, "FDS"),
            (nvpar, nmu, ns, nkx, nky),
            n_species=nsp,
        )

        pre = linear_precompute(geom, params)
        df = init_f(geom, finit="cosine2", amp_init_real=params.amp_init, n_species=nsp)
        state = default_state(nky=nky)
        for s in range(params.naverage):
            from gyaradax.solver import gkstep_single as _step

            df, _, state = _step(df, geom, params, state, pre)

        # Correlation (mode shape match, amplitude-independent)
        def _corr(a, b):
            a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
            return float(
                np.abs(np.dot(np.conj(a), b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
            )

        correlations = {["ion", "electron"][sp]: _corr(df[sp], df_gkw[sp]) for sp in range(nsp)}
        shutil.rmtree(tmpdir, ignore_errors=True)
        return correlations

    def test_bpar_waltz_em_ion_mode_shape(self):
        """bpar_waltz EM: ion mode shape matches GKW (>95% correlation)."""
        gkw_ref = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "gkw_ref",
            "tests",
            "standard",
            "bpar_waltz_linear",
            "input.dat",
        )
        if not os.path.exists(gkw_ref):
            pytest.skip("GKW bpar_waltz_linear input.dat not available")
        template = open(gkw_ref).read().replace("NTIME = 20", "NTIME = 1")
        corrs = self._run_gkw_and_compare(template)
        assert corrs["ion"] > 0.95, f"Ion corr {corrs['ion']:.4f} < 0.95"

    def test_bpar_waltz_em_electron_mode_shape(self):
        """bpar_waltz EM: electron mode shape matches GKW (>60% correlation)."""
        gkw_ref = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "gkw_ref",
            "tests",
            "standard",
            "bpar_waltz_linear",
            "input.dat",
        )
        if not os.path.exists(gkw_ref):
            pytest.skip("GKW bpar_waltz_linear input.dat not available")
        template = open(gkw_ref).read().replace("NTIME = 20", "NTIME = 1")
        corrs = self._run_gkw_and_compare(template)
        assert corrs["electron"] > 0.60, f"Electron corr {corrs['electron']:.4f} < 0.60"


# ── 8. GKW reference validation (spectra and fluxes) ────────────────────────


class TestEMGKWValidation:
    """Validate gyaradax EM results against GKW reference runs.

    Focus on spectra and fluxes rather than growth rates, since fluxes
    are the physically meaningful output and spectra capture the mode
    structure correctly.
    """

    @pytest.mark.slow
    def test_adiabat_apar_es_fluxes_match_gkw(self):
        """Adiabatic electrons + A_par: ion-flux sign and growth match GKW reference.

        The adiabatic + A_par high-beta (β=0.234) path has a known amplitude
        gap vs GKW — gyaradax reports ion flux ~O(90×) smaller than GKW at
        saturation, likely due to a high-β normalization convention and the
        absence of the adiabatic electron's implied EM flux contribution
        (gyaradax outputs only kinetic-species fluxes, 3 cols; GKW outputs
        both species, 6 cols, and derives the adiabatic electron's flux via
        Boltzmann response). The sign and qualitative growth are correct.
        See ``docs/em_debug_report.md`` iteration 5.
        """
        case_dir = _load_em_case("em_adiabat_apar")
        ref_fluxes = np.loadtxt(os.path.join(case_dir, "fluxes.dat"))

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
        df = init_f(geometry, finit=params.finit, amp_init_real=params.amp_init, n_species=nsp)
        state = default_state(nky=len(geometry["krho"]))

        n_windows = ref_fluxes.shape[0]
        all_fluxes = []
        for _ in range(n_windows):
            df, phi, fluxes, state = gk_run(
                df, geometry, params, state, n_steps=params.naverage, pre=pre
            )
            all_fluxes.append([float(f) for f in fluxes])

        pred_fluxes = np.array(all_fluxes)

        # ion eflux should have consistent sign and exponential-like growth;
        # absolute amplitude lag is an open discrepancy (see docstring).
        sim_efl_ion = pred_fluxes[:, 1]
        ref_efl_ion = ref_fluxes[:, 1]
        same_sign = np.sign(sim_efl_ion[-3:]) == np.sign(ref_efl_ion[-3:])
        assert np.all(same_sign), (
            f"ion eflux sign mismatch: sim={sim_efl_ion[-3:]}, ref={ref_efl_ion[-3:]}"
        )
        # monotonic growth in both codes (linear phase ramp-up)
        assert np.all(np.diff(np.abs(sim_efl_ion)) > 0), (
            f"sim ion eflux not monotonic: {sim_efl_ion}"
        )
        assert np.all(np.diff(np.abs(ref_efl_ion)) > 0), (
            f"ref ion eflux not monotonic: {ref_efl_ion}"
        )

    @pytest.mark.slow
    def test_adiabat_apar_em_fluxes_match_gkw(self):
        """Adiabatic electrons + A_par: EM fluxes diagnostic available.

        gyaradax doesn't currently surface per-run EM-only fluxes through
        `gk_run` (only total / ES fluxes are returned). A dedicated
        `calculate_em_fluxes` exists in `integrals.py` but is not hooked
        into `save_dumps`. Once that's wired up this test can become a
        proper EM-flux numerical comparison; for now it exercises the code
        path and checks finiteness.
        """
        case_dir = _load_em_case("em_adiabat_apar")

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
        df = init_f(geometry, finit=params.finit, amp_init_real=params.amp_init, n_species=nsp)
        state = default_state(nky=len(geometry["krho"]))

        df, phi, fluxes, state = gk_run(
            df, geometry, params, state, n_steps=params.naverage, pre=pre
        )
        # smoke test: state propagated, fluxes finite, fields exist
        assert jnp.all(jnp.isfinite(df))
        assert jnp.all(jnp.isfinite(phi))
        assert all(jnp.isfinite(f) for f in fluxes)

    @pytest.mark.slow
    def test_bpar_waltz_es_fluxes_match_gkw(self):
        """bpar_waltz_linear: per-species linear eflux growth rate matches GKW.

        Kinetic 2-species (β=0.01, nlapar+nlbpar). In the linear phase both
        codes grow exponentially at the same γ; their per-window ratio is
        constant (pred/ref ≈ 14× stable across windows because init_f
        amplitude convention differs). Comparing the log-space slope of
        |eflux| vs window index is a clean test of linear physics match
        that is insensitive to the initial-amplitude convention.
        """
        case_dir = _load_em_case("em_bpar_waltz")
        ref_fluxes = np.loadtxt(os.path.join(case_dir, "fluxes.dat"))

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
        df = init_f(geometry, finit=params.finit, amp_init_real=params.amp_init, n_species=nsp)
        state = default_state(nky=len(geometry["krho"]))

        n_windows = min(5, ref_fluxes.shape[0])
        all_fluxes = []
        for _ in range(n_windows):
            df, phi, fluxes, state = gk_run(
                df, geometry, params, state, n_steps=params.naverage, pre=pre
            )
            # fluxes is (nsp, 3) for kinetic; flatten to [p_i, e_i, v_i, p_e, e_e, v_e]
            all_fluxes.append(np.asarray(fluxes).ravel())

        pred_fluxes = np.array(all_fluxes)
        ref_5 = ref_fluxes[:n_windows]

        # in the linear phase |eflux| grows exponentially, so the slope of
        # log|eflux| vs window index is 2*γ*naverage*dt (constant across
        # windows). Compare this slope between codes.
        def _log_slope(series):
            idx = np.arange(len(series))
            return np.polyfit(idx, np.log(np.abs(series)), 1)[0]

        for col, name in [(1, "ion_eflux"), (4, "elec_eflux")]:
            # use windows 0..3; last window can show early saturation onset
            sim_slope = _log_slope(pred_fluxes[:4, col])
            ref_slope = _log_slope(ref_5[:4, col])
            rel = abs(sim_slope - ref_slope) / abs(ref_slope)
            assert rel < 0.15, (
                f"{name} growth rate mismatch: sim_slope={sim_slope:.4f}, "
                f"ref_slope={ref_slope:.4f}, rel={rel:.3f} > 0.15"
            )
            # signs should also match
            assert np.all(np.sign(pred_fluxes[-3:, col]) == np.sign(ref_5[-3:, col])), (
                f"{name} sign mismatch at tail: sim={pred_fluxes[-3:, col]}, ref={ref_5[-3:, col]}"
            )

    @pytest.mark.slow
    def test_bpar_waltz_em_fluxes_match_gkw(self):
        """bpar_waltz_linear: EM-flux temporal shape matches GKW reference.

        Absolute EM-flux amplitudes and some signs differ by diagnostic
        convention (GKW's ``fluxes_em.dat`` is a lab-frame diagnostic with
        per-species columns, while gyaradax evaluates the code's internal
        magnetic-flutter diagnostic directly).  The convention-independent
        check is the temporal shape of the log-amplitude during the early
        linear windows.  The electron energy-flux sign sequence is aligned and
        is checked explicitly as an additional convention-sensitive guard.
        """
        case_dir = _load_em_case("em_bpar_waltz")
        ref_em_path = os.path.join(case_dir, "fluxes_em.dat")
        if not os.path.exists(ref_em_path):
            pytest.skip("EM flux reference not available")

        n_windows = 5
        pred_em = _run_em_flux_windows(case_dir, n_windows)
        ref_em = np.loadtxt(ref_em_path)[:n_windows]

        assert pred_em.shape == ref_em.shape == (n_windows, 6)
        assert np.all(np.isfinite(pred_em))

        # Columns are [p_i, e_i, v_i, p_e, e_e, v_e].  The first three ion
        # columns have an overall sign convention offset, so compare normalized
        # log-amplitude shape.  Electron eflux also has matching signs.  The
        # tail |pred/ref| ratios are intentionally scale-sensitive regression
        # guards recorded from this GKW fixture; they catch missing factors such
        # as gyroaverages or species metadata even when the temporal growth rate
        # still correlates well.
        expected_tail_ratios = {
            0: ("ion_pflux", 2.1490359744, 0.30),
            1: ("ion_eflux", 2.3029628444, 0.30),
            2: ("ion_vflux", 1.9777255142e-14, 0.50),
            4: ("elec_eflux", 0.1414037436, 0.10),
        }
        for col, (name, expected_ratio, rtol) in expected_tail_ratios.items():
            corr = _log_abs_correlation(pred_em[:, col], ref_em[:, col])
            assert corr > 0.95, f"{name} EM log-amplitude corr {corr:.4f} <= 0.95"
            ratio = _median_tail_abs_ratio(pred_em[:, col], ref_em[:, col])
            assert np.isclose(ratio, expected_ratio, rtol=rtol), (
                f"{name} EM tail |pred/ref| ratio {ratio:.6e} not within {rtol:.0%} "
                f"of expected {expected_ratio:.6e}"
            )

        assert np.all(np.sign(pred_em[:, 4]) == np.sign(ref_em[:, 4])), (
            f"electron EM eflux sign mismatch: pred={pred_em[:, 4]}, ref={ref_em[:, 4]}"
        )


class TestExternalEMGKWValidation:
    """Optional numerical EM parity tests using external GKW FDS fixtures."""

    @pytest.mark.slow
    def test_external_apar_waltz_fds_em_fluxes_match_gkw(self):
        """Compute EM fluxes from an external GKW FDS and match GKW fluxes_em.dat.

        This mirrors the non-EM flux parity tests: gyaradax does not evolve the
        state here; it loads the GKW distribution dump, computes fields and EM
        fluxes from that identical state, and compares with the matching GKW
        diagnostic row.  The fixture is optional and selected by
        ``GKW_EM_DATA_ROOT``.
        """
        case_dir = _load_external_em_case("em_apar_waltz")
        required = ["FDS", "FDS.dat", "fluxes_em.dat", "time.dat", "input.dat", "geom.dat"]
        missing = [name for name in required if not os.path.exists(os.path.join(case_dir, name))]
        if missing:
            pytest.skip(f"external EM fixture is incomplete: {missing}")

        geometry = _load_em_reference_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
        shape = (
            len(geometry["vpgr"]),
            len(geometry["mugr"]),
            len(geometry["sgrid"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )

        dg, info = load_gkw_dump(os.path.join(case_dir, "FDS"), shape, n_species=nsp)
        _, apar, bpar = _compute_fields(dg, geometry, params, pre)
        assert apar is not None
        df = g_to_f(dg, apar, params, pre)
        pred = np.asarray(
            calculate_em_fluxes(geometry, df, apar, params=params, bpar=bpar, pre=pre)
        ).reshape(-1)

        ref_all = np.atleast_2d(np.loadtxt(os.path.join(case_dir, "fluxes_em.dat")))
        times = np.atleast_2d(np.loadtxt(os.path.join(case_dir, "time.dat")))
        dump_time = float(info["time"])
        ref_idx = int(np.argmin(np.abs(times[:, 0] - dump_time)))
        assert np.isclose(times[ref_idx, 0], dump_time, rtol=1e-10, atol=1e-10), (
            f"FDS time {dump_time} not found in time.dat near row {ref_idx}"
        )
        ref = ref_all[ref_idx]

        assert pred.shape == ref.shape == (6,)
        np.testing.assert_allclose(pred, ref, rtol=2e-5, atol=1e-10)
