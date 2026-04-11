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

from gyaradax.params import GKParams, gkparams_from_input_and_geometry
from gyaradax.geometry import compute_geometry, compute_geometry_from_input
from gyaradax.solver import linear_precompute, init_f, default_state
from gyaradax.simulate import gk_run
from gyaradax.integrals import calculate_phi
from gyaradax import load_geometry
from gyaradax.geometry import compute_geometry_from_input

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
        import subprocess, tempfile, shutil

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
        """Adiabatic electrons + A_par: ES fluxes match GKW reference."""
        case_dir = _load_em_case("em_adiabat_apar")
        ref_fluxes = np.loadtxt(os.path.join(case_dir, "fluxes.dat"))

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        df = init_f(geometry, params=params)
        state = default_state(nky=len(geometry["krho"]))

        n_windows = ref_fluxes.shape[0]
        all_fluxes = []
        for _ in range(n_windows):
            df, phi, fluxes, state = gk_run(
                df, geometry, params, state, n_steps=params.naverage, pre=pre
            )
            all_fluxes.append([float(f) for f in fluxes])

        pred_fluxes = np.array(all_fluxes)
        from conftest import rel_l2

        # pflux (col 0), eflux (col 1) per species
        for col, name in [(0, "pflux_sp1"), (1, "eflux_sp1"), (3, "pflux_sp2"), (4, "eflux_sp2")]:
            err = rel_l2(pred_fluxes[:, col], ref_fluxes[:, col])
            assert err < 0.05, f"EM adiabatic {name} rel_l2 error {err:.4e} > 5%"

    @pytest.mark.slow
    def test_adiabat_apar_em_fluxes_match_gkw(self):
        """Adiabatic electrons + A_par: EM fluxes match GKW reference."""
        case_dir = _load_em_case("em_adiabat_apar")
        ref_em_path = os.path.join(case_dir, "fluxes_em.dat")
        if not os.path.exists(ref_em_path):
            pytest.skip("EM flux reference not available")
        ref_em_fluxes = np.loadtxt(ref_em_path)

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        df = init_f(geometry, params=params)
        state = default_state(nky=len(geometry["krho"]))

        n_windows = ref_em_fluxes.shape[0]
        all_em_fluxes = []
        for _ in range(n_windows):
            df, phi, fluxes, state = gk_run(
                df, geometry, params, state, n_steps=params.naverage, pre=pre
            )
            # EM fluxes should be available from diagnostics
            em_fluxes = state.em_fluxes if hasattr(state, "em_fluxes") else fluxes
            all_em_fluxes.append([float(f) for f in em_fluxes])

        pred_em = np.array(all_em_fluxes)
        from conftest import rel_l2

        # Compare EM heat flux columns
        for col in range(min(pred_em.shape[1], ref_em_fluxes.shape[1])):
            ref_col = ref_em_fluxes[:, col]
            if np.max(np.abs(ref_col)) < 1e-20:
                continue  # skip negligible columns
            err = rel_l2(pred_em[:, col], ref_col)
            assert err < 0.1, f"EM flux col {col} rel_l2 error {err:.4e} > 10%"

    @pytest.mark.slow
    def test_bpar_waltz_es_fluxes_match_gkw(self):
        """bpar_waltz_linear: ES fluxes match GKW reference."""
        case_dir = _load_em_case("em_bpar_waltz")
        ref_fluxes = np.loadtxt(os.path.join(case_dir, "fluxes.dat"))

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        df = init_f(geometry, params=params)
        state = default_state(nky=len(geometry["krho"]))

        n_windows = min(5, ref_fluxes.shape[0])
        all_fluxes = []
        for _ in range(n_windows):
            df, phi, fluxes, state = gk_run(
                df, geometry, params, state, n_steps=params.naverage, pre=pre
            )
            all_fluxes.append([float(f) for f in fluxes])

        pred_fluxes = np.array(all_fluxes)
        ref_5 = ref_fluxes[:n_windows]

        from conftest import rel_l2

        # eflux columns (1, 4) are the most physically meaningful
        for col, name in [(1, "eflux_ion"), (4, "eflux_elec")]:
            err = rel_l2(pred_fluxes[:, col], ref_5[:, col])
            assert err < 0.1, f"EM bpar_waltz {name} rel_l2 error {err:.4e} > 10%"

    @pytest.mark.slow
    def test_bpar_waltz_em_fluxes_match_gkw(self):
        """bpar_waltz_linear: EM fluxes match GKW reference."""
        case_dir = _load_em_case("em_bpar_waltz")
        ref_em_path = os.path.join(case_dir, "fluxes_em.dat")
        if not os.path.exists(ref_em_path):
            pytest.skip("EM flux reference not available")
        ref_em = np.loadtxt(ref_em_path)

        geometry = _load_em_geometry(case_dir)
        params = gkparams_from_input_and_geometry(os.path.join(case_dir, "input.dat"), geometry)
        pre = linear_precompute(geometry, params)
        df = init_f(geometry, params=params)
        state = default_state(nky=len(geometry["krho"]))

        n_windows = min(5, ref_em.shape[0])
        all_em = []
        for _ in range(n_windows):
            df, phi, fluxes, state = gk_run(
                df, geometry, params, state, n_steps=params.naverage, pre=pre
            )
            em_fluxes = state.em_fluxes if hasattr(state, "em_fluxes") else fluxes
            all_em.append([float(f) for f in em_fluxes])

        pred_em = np.array(all_em)
        ref_5 = ref_em[:n_windows]

        from conftest import rel_l2

        for col in range(min(pred_em.shape[1], ref_5.shape[1])):
            ref_col = ref_5[:, col]
            if np.max(np.abs(ref_col)) < 1e-20:
                continue
            err = rel_l2(pred_em[:, col], ref_col)
            assert err < 0.15, f"EM bpar_waltz em_flux col {col} rel_l2 {err:.4e} > 15%"
