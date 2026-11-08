"""gyaradax.transport: the ql-gyx / nl-gyx TORAX transport models."""

import json
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
from gyaradax.geometry import build_topology
from gyaradax.transport import base
from gyaradax.transport.base import ikxspace_for_shat, local_inputs, quasilinear_inputs

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


def test_config_and_grid():
    tc = torax_config({"model_name": "nl-gyx", "rho_match": (0.3, 0.6)})
    assert type(
        tc.transport.core_transport_models["gyaradax"].build_transport_model()
    ).__name__ == ("GyaradaxNLTransportModel")
    model = build(grid_preset="kinetic", nky=48, vpar_max=3.5, rule="sat0", cn=2.0)
    assert (model.nkx, model.nmu, model.nky, model.vpar_max) == (191, 12, 48, 3.5)
    assert model.channel_params == (("Qi", (("cn", 2.0),)),)
    for rule in ("sat2", "canonical"):
        cal = calibration.load("auto", rule)
        model = build(rule=rule, kinetic=True)
        want = calibration.channel_parameters(cal, "kinetic")
        assert {ch: dict(kw) for ch, kw in model.channel_params} == want and set(want) == {"Qi", "Qe", "Gamma"}
        assert dict(model.rule_options) == cal["options"] and model.nkx == 191
    with pytest.raises(ValueError, match="no calibration"):
        build(rule="sat0")
    with pytest.raises(ValueError, match="must be odd"):
        build(nkx=192)
    assert GyaradaxNLConfig.from_dict({"model_name": "nl-gyx", "kinetic": True}).build_transport_model().kinetic


def test_calibration_from_a_json_file(tmp_path):
    path = tmp_path / "mine.json"
    cal = {
        "rule": "sat2",
        "options": {"kx_width_model": "tglf"},
        "parameters": {"adiabatic": {"Qi": {"cn": 2.0, "flux_exponent": 0.7}}},
    }
    calibration.save(cal, path)
    model = build(rule="sat2", calibration=str(path), rule_options={"gate_sharpness": 50.0})
    assert model.channel_params == (("Qi", (("cn", 2.0), ("flux_exponent", 0.7))),)
    assert dict(model.rule_options) == {"kx_width_model": "tglf", "gate_sharpness": 50.0}
    with pytest.raises(ValueError, match="no kinetic fit"):
        build(rule="sat2", calibration=str(path), kinetic=True)


def test_diagnostics_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", constant_flux)
    path = str(tmp_path / "diag.jsonl")
    transport_call(
        {"model_name": "ql-gyx", "rho_match": (0.3, 0.6), "diagnostics_path": path, **GRID}
    )
    rows = [json.loads(line) for line in open(path)]
    assert {r["radius"] for r in rows if r["call"] == rows[-1]["call"]} == {0, 1}
    assert len({r["rho_face"] for r in rows}) == 2
    assert all(r["qi_gb"] == 1.0 and isinstance(r["shat"], float) and r["t"] == 0.0 and r["wall_s"] > 0 for r in rows)


@pytest.mark.slow
@pytest.mark.parametrize("kinetic", [False, True])
def test_nl_solve(kinetic):
    fields = {"model_name": "nl-gyx", "rho_match": (0.5,), "kinetic": kinetic, "n_steps_nl": 60, "nl_block": 20, "nl_tail_blocks": 2}
    _, coeffs, _, _ = transport_call({**fields, **GRID, "nvpar": 16, "nmu": 4, "ns": 8})
    for chi in (coeffs.chi_face_ion, coeffs.chi_face_el):
        assert np.all(np.isfinite(np.asarray(chi)))


@pytest.mark.slow
def test_ql_linear_solve():
    _, coeffs, _, _ = transport_call(
        {"model_name": "ql-gyx", "rho_match": (0.5, 0.7), "n_steps_linear": 300, **GRID}
    )
    chi = np.asarray(coeffs.chi_face_ion)
    assert np.all(np.isfinite(chi)) and np.all(chi >= 0.0) and chi.max() > 0.0


def test_ikxspace_follows_shat():
    choices = (1, 2, 3, 4, 6, 8, 12, 16)
    for shat, want in ((0.05, 1), (0.4, 2), (0.8, 4), (1.0, 6), (2.0, 12), (5.0, 16), (-1.0, 6)):
        assert ikxspace_for_shat(shat, choices) == want


def test_radius_params_carry_collisions_and_shat_ikxspace(monkeypatch):
    seen = []

    def record(self, params, geom, df_prev=None, warm=None):
        seen.append((params, geom))
        out = (jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.0), {})
        return out if df_prev is None else (*out, df_prev)

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", record)
    fields = {"model_name": "ql-gyx", "rho_match": (0.3, 0.7), **GRID, "ikxspace_choices": (1, 2, 4, 8)}
    model, _, geo, cp = transport_call(fields)
    loc = local_inputs(cp, geo, quasilinear_inputs(cp, geo), (0.5,))
    loc = {k: v[0] for k, v in loc.items()}
    params = model.gkparams(loc)
    i = int(np.argmin(np.abs(np.asarray(geo.rho_face_norm) - 0.5)))
    assert params.collisions
    np.testing.assert_allclose(float(params.coll_nref), float(cp.kinetic.n_e.face_value()[i]) / 1e19)
    np.testing.assert_allclose(float(params.coll_tref), float(cp.kinetic.T_i.face_value()[i]))
    np.testing.assert_allclose(float(params.coll_rref), float(geo.R_major))
    want = (1, 2, 4, 8)[int(np.searchsorted((1, 2, 4, 8), np.floor(6.53 * abs(float(params.shat))), "right")) - 1]
    assert model.ikxspace_at(float(params.shat)) == max(want, 1)
    geom = model.geometry(params, model.ikxspace_at(float(params.shat)))
    ref = build_topology(nkx=GRID["nkx"], nky=GRID["nky"], ikxspace=max(want, 1), ns=GRID["ns"])
    np.testing.assert_array_equal(np.asarray(geom["mode_label"]), np.asarray(ref["mode_label"]))
    assert not build(**{**fields, "collisions": False}).gkparams(loc).collisions


def test_runs_a_simulation(monkeypatch):
    def stiff(params):
        q_i = 0.5 * jnp.maximum(params.rlt - 4.0, 0.0) ** 1.5
        return q_i, q_i, jnp.asarray(0.0)

    stiff_flux = stub(stiff)

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", stiff_flux)
    tc = torax_config({"model_name": "ql-gyx", "rho_match": (0.3, 0.6, 0.85), **GRID})
    history = torax.run_simulation(tc, progress_bar=False, max_steps=2)
    assert np.all(np.isfinite(np.asarray(history.simulation_output_to_xr().profiles.T_i)))


def test_warm_start_carries_the_distribution(monkeypatch, caplog):
    base._compiled.cache_clear()

    calls = []

    def per_radius(self, params, geom, df_prev=None, warm=None):
        calls.append(1)
        q_i = 0.5 * jnp.maximum(params.rlt - 4.0, 0.0) ** 1.5
        return q_i, q_i, jnp.asarray(0.0), {}, df_prev + 1.0

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", per_radius)
    base._WARM.clear()
    # 0.75 and 0.875 share a face on the 4-cell grid; each tube still keeps its own state
    tc = torax_config({"model_name": "ql-gyx", "rho_match": (0.3, 0.75, 0.875), **GRID})
    history = torax.run_simulation(tc, progress_bar=False, max_steps=3)
    assert np.all(np.isfinite(np.asarray(history.simulation_output_to_xr().profiles.T_i)))
    # every call inside TORAX's loop added one to the stored distribution of each radius
    assert len(base._WARM) == 3
    counts = {float(np.asarray(df).real.flat[0]) for df, _ in base._WARM.values()}
    assert len(counts) == 1 and counts.pop() >= 3
    assert all(i == GRID["ikxspace"] for _, i in base._WARM.values())
    assert not build(warm_start=False, **GRID)._warm_df_shape()

    monkeypatch.setattr(
        GyaradaxQLTransportModel, "_per_radius",
        lambda self, params, geom, df_prev=None, warm=None: (*(jnp.asarray(jnp.nan),) * 3, {}, df_prev * jnp.nan),
    )
    base._compiled.cache_clear()
    cp, rp, geo = core_profile_helpers.make_initial_inputs(tc)
    model = tc.transport.core_transport_models["gyaradax"].build_transport_model()
    args = (rp.transport.core_transport_model_params["gyaradax"], rp, geo, cp, np.zeros_like(geo.rho_face_norm, dtype=bool))
    with caplog.at_level("ERROR", logger="gyaradax.transport"):
        coeffs, _ = model.call_implementation(*args, model.initial_state(rp))
        jax.effects_barrier()
    assert np.isnan(np.asarray(coeffs.chi_face_ion)).any()
    reported = {r.getMessage().split(" is not finite")[0] for r in caplog.records if "is not finite" in r.getMessage()}
    assert len(reported) == 2
    base._WARM.clear()


def test_diverged_solve_hands_torax_nan(monkeypatch):
    def per_radius(self, params, geom, df_prev=None, warm=None):
        return jnp.asarray(1.0), jnp.asarray(0.5), jnp.asarray(0.0), {}, df_prev * jnp.nan

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", per_radius)
    base._WARM.clear()
    _, coeffs, _, _ = transport_call({"model_name": "ql-gyx", "rho_match": (0.5,), **GRID})
    base._WARM.clear()
    assert np.isnan(np.asarray(coeffs.chi_face_ion)).any()
