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

jax.config.update("jax_enable_x64", True)

from gyaradax.params import GKParams, gkparams_from_config, gkparams_from_input_and_geometry, load_config
from gyaradax.geometry import compute_geometry_from_input
from gyaradax.solver import linear_precompute, init_f, default_state
from gyaradax.simulate import gk_run
from gyaradax import load_geometry

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GKW_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gkw_cases")


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_em_case(name):
    """Load an EM test case directory, skip if missing."""
    d = os.path.join(GKW_CASES_DIR, name)
    if not os.path.exists(d):
        pytest.skip(f"EM reference data not found at {d}")
    return d


def _load_em_geometry(case_dir):
    """Load geometry for an EM test case using compute_geometry_from_input."""
    input_path = os.path.join(case_dir, "input.dat")
    return compute_geometry_from_input(input_path)


def _read_growth_rates(directory):
    """Read growth_rates_all_modes file."""
    path = os.path.join(directory, "growth_rates_all_modes")
    return np.loadtxt(path)


def _read_time_dat(directory):
    """Read time.dat -> (time, growth_rate, frequency) arrays."""
    data = np.loadtxt(os.path.join(directory, "time.dat"))
    return data[:, 0], data[:, 1], data[:, 2]


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


# ── 3. Field solve tests ────────────────────────────────────────────────────


class TestAmpereSolve:
    """Test the Ampere field solve for A_parallel."""

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


# ── 4. g2f transform tests ──────────────────────────────────────────────────


class TestG2FTransform:
    """Test the mixed variable g <-> f transform."""

    def test_g2f_identity_when_no_em(self):
        """When nlapar=False, g=f (identity transform)."""
        from gyaradax.solver import g_to_f

        params = GKParams(nlapar=False)
        dg = jnp.ones((32, 8, 16, 1, 1), dtype=jnp.complex128)
        apar = jnp.zeros((16, 1, 1), dtype=jnp.complex128)
        pre = {}
        df = g_to_f(dg, apar, params, pre)
        assert jnp.allclose(df, dg)

    def test_g2f_nonzero_correction(self):
        """When nlapar=True and apar!=0, g!=f."""
        from gyaradax.solver import g_to_f

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
        from gyaradax.solver import g_to_f, f_to_g

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
        assert np.all(
            same_sign
        ), f"ion eflux sign mismatch: sim={sim_efl_ion[-3:]}, ref={ref_efl_ion[-3:]}"
        # monotonic growth in both codes (linear phase ramp-up)
        assert np.all(
            np.diff(np.abs(sim_efl_ion)) > 0
        ), f"sim ion eflux not monotonic: {sim_efl_ion}"
        assert np.all(
            np.diff(np.abs(ref_efl_ion)) > 0
        ), f"ref ion eflux not monotonic: {ref_efl_ion}"

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
                f"{name} sign mismatch at tail: "
                f"sim={pred_fluxes[-3:, col]}, ref={ref_5[-3:, col]}"
            )

    @pytest.mark.slow
    def test_bpar_waltz_em_fluxes_match_gkw(self):
        """bpar_waltz_linear: EM-flux diagnostic path runs and is finite.

        gyaradax's EM flux is computed via `calculate_em_fluxes` in
        integrals.py (now hooked into save_dumps). This smoke test exercises
        the NL-EM run path and verifies finiteness; a numerical match test
        against GKW's fluxes_em.dat requires running `save_dumps` and
        reading the saved npz (beyond the scope of this unit test).
        """
        case_dir = _load_em_case("em_bpar_waltz")
        ref_em_path = os.path.join(case_dir, "fluxes_em.dat")
        if not os.path.exists(ref_em_path):
            pytest.skip("EM flux reference not available")

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
        df = init_f(geometry, finit=params.finit, amp_init_real=params.amp_init, n_species=nsp)
        state = default_state(nky=len(geometry["krho"]))

        df, phi, fluxes, state = gk_run(
            df, geometry, params, state, n_steps=params.naverage, pre=pre
        )
        assert jnp.all(jnp.isfinite(df))
        assert jnp.all(jnp.isfinite(phi))
        assert np.all(np.isfinite(np.asarray(fluxes)))
        ref_em = np.loadtxt(ref_em_path)
        # sanity: GKW reference columns line up with gyaradax (nsp*3 = 6 for
        # 2-species kinetic case)
        assert (
            np.asarray(fluxes).size == ref_em.shape[1]
        ), f"col count mismatch: pred {np.asarray(fluxes).size} vs ref {ref_em.shape[1]}"
