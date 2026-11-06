"""gyaradax.transport: the ql-gyx / nl-gyx TORAX transport models."""

import math

import numpy as np
import pytest

pytest.importorskip("torax")

import jax
import jax.numpy as jnp
import torax
from torax._src.test_utils import core_profile_helpers, default_configs

from gyaradax.quasilinear import calibration
from gyaradax.transport import GyaradaxNLConfig, GyaradaxQLConfig, GyaradaxQLTransportModel
from gyaradax.transport import diagnostics
from gyaradax.transport.base import face_profiles, quasilinear_inputs, topology

GRID = dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32, ikxspace=5, krhomax=1.4)


def torax_config(transport):
    config = default_configs.get_default_config_dict()
    config["transport"] = {"core_transport_models": {"gyaradax": transport}}
    return torax.ToraxConfig.from_dict(config)


def transport_call(transport):
    tc = torax_config(transport)
    cp, rp, geo = core_profile_helpers.make_initial_inputs(tc)
    model = tc.transport.core_transport_models["gyaradax"].build_transport_model()
    coeffs, _ = model.call_implementation(
        rp.transport.core_transport_model_params["gyaradax"],
        rp,
        geo,
        cp,
        np.zeros_like(geo.rho_face_norm, dtype=bool),
        model.initial_state(rp),
    )
    return model, coeffs, geo, cp


def stub(flux):
    """A `_per_radius` that returns ``flux(params)`` and hands a warm start back unchanged."""

    def per_radius(self, params, geom, df_prev=None, warm=None):
        out = (*flux(params), {})
        return out if df_prev is None else (*out, df_prev)

    return per_radius


constant_flux = stub(lambda params: (jnp.asarray(1.0), jnp.asarray(0.5), jnp.asarray(0.0)))


def build(**fields):
    return GyaradaxQLConfig.from_dict({"model_name": "ql-gyx", **fields}).build_transport_model()


def test_fluxes_reach_torax_in_its_units(monkeypatch):
    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", constant_flux)
    _, coeffs, geo, cp = transport_call({"model_name": "ql-gyx", **GRID})
    qli = quasilinear_inputs(cp, geo)
    # 1 GKW gyroBohm -> chi = 2 sqrt(2) chiGB(L_ref = R) / (R/L_T); the axis has R/L_T -> 0
    gkw = 2.0 * math.sqrt(2.0) * np.asarray(qli.chiGB) * float(geo.a_minor / geo.R_major)
    np.testing.assert_allclose(
        np.asarray(coeffs.chi_face_ion)[1:], (gkw / np.asarray(qli.lref_over_lti))[1:], rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(coeffs.chi_face_el)[1:],
        (0.5 * gkw / np.asarray(qli.lref_over_lte))[1:],
        rtol=1e-6,
    )


def test_transport_run_with_cheap_flux(monkeypatch):
    def stiff(params):
        q_i = 0.5 * jnp.maximum(params.rlt - 4.0, 0.0) ** 1.5
        return q_i, q_i, jnp.asarray(0.0)

    stiff_flux = stub(stiff)

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", stiff_flux)
    tc = torax_config({"model_name": "ql-gyx", "rho_match": (0.3, 0.6, 0.85), **GRID})
    history = torax.run_simulation(tc, progress_bar=False, max_steps=3)
    t_i = np.asarray(history.simulation_output_to_xr().profiles.T_i)
    assert history.times.shape[0] == 4 and np.all(np.isfinite(t_i)) and np.all(t_i > 0)


@pytest.mark.parametrize("batching", ["vmap", "loop"])
def test_batching_matches_devices(monkeypatch, batching):
    stiff_flux = stub(lambda params: (params.rlt**1.5, params.rln, params.q))

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", stiff_flux)
    fields = {"model_name": "ql-gyx", "rho_match": (0.3, 0.5, 0.7), **GRID}
    _, ref, _, _ = transport_call(fields)
    _, out, _, _ = transport_call({**fields, "batching": batching})
    np.testing.assert_array_equal(np.asarray(out.chi_face_ion), np.asarray(ref.chi_face_ion))


def test_config_and_grid():
    tc = torax_config({"model_name": "nl-gyx", "rho_match": (0.3, 0.6)})
    assert type(
        tc.transport.core_transport_models["gyaradax"].build_transport_model()
    ).__name__ == ("GyaradaxNLTransportModel")
    model = build(grid_preset="kinetic", nky=48, vpar_max=3.5, rule="sat0", cn=2.0)
    assert (model.nkx, model.nmu, model.nky, model.vpar_max, model.cn) == (191, 12, 48, 3.5, 2.0)
    sat2 = calibration.load("auto", "sat2")
    model = build(rule="sat2", kinetic=True)
    assert model.cn == sat2["parameters"]["cn"] and model.nkx == 191
    assert dict(model.rule_options) == calibration.rule_kwargs(sat2)
    with pytest.raises(ValueError, match="no calibration"):
        build(rule="sat0")
    assert (
        build(rule="canonical", kinetic=True).cn
        == calibration.load("auto", "canonical")["parameters"]["cn"]
    )
    with pytest.warns(RuntimeWarning, match="resolution floor"):
        assert build(**{**GRID, "nkx": 43}).nkx == 85
    with pytest.raises(ValueError, match="must be odd"):
        build(nkx=192)
    with pytest.raises(NotImplementedError, match="adiabatic"):
        GyaradaxNLConfig.from_dict(
            {"model_name": "nl-gyx", "kinetic": True}
        ).build_transport_model()


def test_calibration_from_a_json_file(tmp_path):
    path = tmp_path / "mine.json"
    cal = {
        "rule": "sat2",
        "parameters": {"cn": 2.0, "flux_exponent": 0.7},
        "electrons": ["adiabatic"],
    }
    calibration.save(cal, path)
    model = build(rule="sat2", calibration=str(path), rule_options={"gate_sharpness": 50.0})
    assert model.cn == 2.0
    assert dict(model.rule_options) == {"flux_exponent": 0.7, "gate_sharpness": 50.0}
    with pytest.raises(ValueError, match="no kinetic fit"):
        build(rule="sat2", calibration=str(path), kinetic=True)


def test_diagnostics_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", constant_flux)
    diagnostics.reset_sinks()
    path = str(tmp_path / "diag.jsonl")
    transport_call(
        {"model_name": "ql-gyx", "rho_match": (0.3, 0.6), "diagnostics_path": path, **GRID}
    )
    rows = diagnostics.load_jsonl(path)
    assert {r["radius"] for r in rows if r["call"] == rows[-1]["call"]} == {0, 1}
    assert len({r["rho_face"] for r in rows}) == 2
    assert all(r["qi_gb"] == 1.0 and isinstance(r["shat"], float) for r in rows)


@pytest.mark.slow
def test_ql_linear_solve():
    _, coeffs, _, _ = transport_call(
        {"model_name": "ql-gyx", "rho_match": (0.5, 0.7), "n_steps_linear": 300, **GRID}
    )
    chi = np.asarray(coeffs.chi_face_ion)
    assert np.all(np.isfinite(chi)) and np.all(chi >= 0.0) and chi.max() > 0.0


def test_ikxspace_follows_shat():
    from gyaradax.transport.base import ikxspace_for_shat

    choices = (1, 2, 3, 4, 6, 8, 12, 16)
    for shat, want in ((0.05, 1), (0.4, 2), (0.8, 4), (1.0, 6), (2.0, 12), (5.0, 16), (-1.0, 6)):
        assert choices[int(ikxspace_for_shat(shat, choices))] == want


def test_topology_bank_matches_static_topologies():
    from gyaradax.transport.base import select_topology, topology, topology_bank

    choices = (2, 5, 9)
    bank = topology_bank(85, 32, 16, choices)
    for i, k in enumerate(choices):
        ref, got = topology(85, 32, k, 16), select_topology(bank, jnp.asarray(i))
        for key, v in ref.items():
            if key == "flipped":
                for kk, vv in v.items():
                    np.testing.assert_array_equal(np.asarray(got[key][kk]), np.asarray(vv))
            elif hasattr(v, "shape"):
                np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(v))
            else:
                assert got[key] == v


def test_radius_params_carry_collisions_and_shat_ikxspace(monkeypatch):
    seen = []

    def record(self, params, geom, df_prev=None, warm=None):
        seen.append((params, geom))
        out = (jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.0), {})
        return out if df_prev is None else (*out, df_prev)

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", record)
    fields = {"model_name": "ql-gyx", "rho_match": (0.3, 0.7), **GRID, "ikxspace_choices": (1, 2, 4, 8)}
    model, _, geo, cp = transport_call(fields)
    qli = quasilinear_inputs(cp, geo)
    seen.clear()
    model._per_radius_from_profiles(10, qli, face_profiles(cp, geo), geo)
    params, geom = seen[0]
    assert params.collisions
    np.testing.assert_allclose(float(params.coll_nref), float(cp.kinetic.n_e.face_value()[10]) / 1e19)
    np.testing.assert_allclose(float(params.coll_tref), float(cp.kinetic.T_i.face_value()[10]))
    np.testing.assert_allclose(float(params.coll_rref), float(geo.R_major))
    want = (1, 2, 4, 8)[int(np.searchsorted((1, 2, 4, 8), np.floor(6.53 * abs(float(params.shat))), "right")) - 1]
    ref = topology(GRID["nkx"], GRID["nky"], max(want, 1), GRID["ns"])
    np.testing.assert_array_equal(np.asarray(geom["mode_label"]), np.asarray(ref["mode_label"]))
    off = build(**{**fields, "collisions": False})
    seen.clear()
    off._per_radius_from_profiles(10, qli, face_profiles(cp, geo), geo)
    assert not seen[0][0].collisions


def test_loop_batching_runs_a_simulation(monkeypatch):
    def stiff(params):
        q_i = 0.5 * jnp.maximum(params.rlt - 4.0, 0.0) ** 1.5
        return q_i, q_i, jnp.asarray(0.0)

    stiff_flux = stub(stiff)

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", stiff_flux)
    tc = torax_config({"model_name": "ql-gyx", "rho_match": (0.3, 0.6, 0.85), "batching": "loop", **GRID})
    history = torax.run_simulation(tc, progress_bar=False, max_steps=2)
    assert np.all(np.isfinite(np.asarray(history.simulation_output_to_xr().profiles.T_i)))


def test_warm_start_carries_the_distribution(monkeypatch, caplog):
    calls = []

    def per_radius(self, params, geom, df_prev=None, warm=None):
        calls.append(warm)
        return jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.0), {}, df_prev + 1.0

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", per_radius)
    tc = torax_config({"model_name": "ql-gyx", "rho_match": (0.3, 0.6), "batching": "loop", **GRID})
    cp, rp, geo = core_profile_helpers.make_initial_inputs(tc)
    model = tc.transport.core_transport_models["gyaradax"].build_transport_model()
    state = model.initial_state(rp)
    assert state.df.shape == (2, 32, 8, 16, 85, 32) and np.all(np.asarray(state.ikxspace_index) == -1)
    args = (rp.transport.core_transport_model_params["gyaradax"], rp, geo, cp, np.zeros_like(geo.rho_face_norm, dtype=bool))
    _, state = model.call_implementation(*args, state)
    assert np.all(np.asarray(state.ikxspace_index) == 0) and np.allclose(np.asarray(state.df), 1.0)
    _, state = model.call_implementation(*args, state)
    assert np.allclose(np.asarray(state.df), 2.0)
    assert not build(warm_start=False, **GRID)._warm_df_shape()
    monkeypatch.setattr(
        GyaradaxQLTransportModel, "_per_radius",
        lambda self, params, geom, df_prev=None, warm=None: (*(jnp.asarray(jnp.nan),) * 3, {}, df_prev * jnp.nan),
    )
    with caplog.at_level("ERROR", logger="gyaradax.transport"):
        coeffs, _ = model.call_implementation(*args, state)
        jax.effects_barrier()
    assert np.isnan(np.asarray(coeffs.chi_face_ion)).any()
    assert sum("is not finite" in r.getMessage() for r in caplog.records) == 2
