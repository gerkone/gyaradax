"""gyaradax.transport: the ql-gyx / nl-gyx TORAX transport models."""

import math

import numpy as np
import pytest

pytest.importorskip("torax")

import jax.numpy as jnp
import torax
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.test_utils import default_configs

from gyaradax.quasilinear import calibration
from gyaradax.transport import GyaradaxNLConfig, GyaradaxQLConfig, GyaradaxQLTransportModel
from gyaradax.transport import diagnostics, units

GRID = dict(nvpar=32, nmu=8, ns=16, nkx=85, nky=32, ikxspace=5, krhomax=1.4)


def torax_config(transport):
    config = default_configs.get_default_config_dict()
    config["transport"] = {"core_transport_models": {"gyaradax": transport}}
    return torax.ToraxConfig.from_dict(config)


def transport_call(transport):
    tc = torax_config(transport)
    rp = build_runtime_params.RuntimeParamsProvider.from_config(tc)(t=tc.numerics.t_initial)
    geo = tc.geometry.build_provider(t=tc.numerics.t_initial)
    cp = initialization.initial_core_profiles(
        runtime_params=rp,
        geo=geo,
        source_models=tc.sources.build_models(),
        neoclassical_model=tc.neoclassical.build_model(),
    )
    model = tc.transport.core_transport_models["gyaradax"].build_transport_model()
    coeffs = model.call_implementation(
        rp.transport.core_transport_model_params["gyaradax"],
        rp,
        geo,
        cp,
        np.zeros_like(geo.rho_face_norm, dtype=bool),
    )
    return model, coeffs, geo, cp


def constant_flux(self, params, geom):
    return jnp.asarray(1.0), jnp.asarray(0.5), jnp.asarray(0.0), {}


def build(**fields):
    return GyaradaxQLConfig.from_dict(
        {"model_name": "ql-gyx", **fields}
    ).build_transport_model()


def test_fluxes_reach_torax_in_its_units(monkeypatch):
    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", constant_flux)
    _, coeffs, geo, cp = transport_call({"model_name": "ql-gyx", **GRID})
    qli = units.build_quasilinear_inputs(cp, geo)
    # 1 GKW gyroBohm -> chi = 2 sqrt(2) chiGB / (R/L_T); the axis has R/L_T -> 0
    gkw = 2.0 * math.sqrt(2.0) * np.asarray(qli.chiGB)
    np.testing.assert_allclose(
        np.asarray(coeffs.chi_face_ion)[1:], (gkw / np.asarray(qli.lref_over_lti))[1:], rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(coeffs.chi_face_el)[1:],
        (0.5 * gkw / np.asarray(qli.lref_over_lte))[1:],
        rtol=1e-6,
    )


def test_transport_run_with_cheap_flux(monkeypatch):
    def stiff_flux(self, params, geom):
        q_i = 0.5 * jnp.maximum(params.rlt - 4.0, 0.0) ** 1.5
        return q_i, q_i, jnp.asarray(0.0), {}

    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", stiff_flux)
    tc = torax_config({"model_name": "ql-gyx", "rho_match": (0.3, 0.6, 0.85), **GRID})
    history = torax.run_simulation(tc, progress_bar=False, max_steps=3)
    t_i = np.asarray(history.simulation_output_to_xr().profiles.T_i)
    assert history.times.shape[0] == 4 and np.all(np.isfinite(t_i)) and np.all(t_i > 0)


@pytest.mark.parametrize("batching", ["vmap", "loop"])
def test_batching_matches_devices(monkeypatch, batching):
    def stiff_flux(self, params, geom):
        return params.rlt**1.5, params.rln, params.q, {}

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
    with pytest.raises(ValueError, match="no kinetic fit"):
        build(rule="canonical", kinetic=True)
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
    calibration.save({"rule": "sat2", "parameters": {"cn": 2.0, "flux_exponent": 0.7}}, path)
    model = build(rule="sat2", calibration=str(path), rule_options={"gate_sharpness": 50.0})
    assert model.cn == 2.0
    assert dict(model.rule_options) == {"flux_exponent": 0.7, "gate_sharpness": 50.0}


def test_diagnostics_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(GyaradaxQLTransportModel, "_per_radius", constant_flux)
    diagnostics.reset_sinks()
    path = str(tmp_path / "diag.jsonl")
    transport_call(
        {"model_name": "ql-gyx", "rho_match": (0.3, 0.6), "diagnostics_path": path, **GRID}
    )
    rows = diagnostics.load_jsonl(path)
    assert sorted((r["call"], r["radius"]) for r in rows) == [(0, 0), (0, 1)]
    assert len({r["rho_face"] for r in rows}) == 2
    assert all(r["qi_gb"] == 1.0 and isinstance(r["shat"], float) for r in rows)


@pytest.mark.slow
def test_ql_linear_solve():
    _, coeffs, _, _ = transport_call(
        {"model_name": "ql-gyx", "rho_match": (0.5, 0.7), "n_steps_linear": 300, **GRID}
    )
    chi = np.asarray(coeffs.chi_face_ion)
    assert np.all(np.isfinite(chi)) and np.all(chi >= 0.0) and chi.max() > 0.0
