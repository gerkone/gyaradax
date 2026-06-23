from pathlib import Path

from omegaconf import OmegaConf

from gyaradax.params import gkparams_from_config, gkparams_from_runtime
from gyaradax.utils import load_runtime_params


def _write_input(tmp_path: Path, control_body: str) -> Path:
    path = tmp_path / "input.dat"
    path.write_text(
        f"&CONTROL\n{control_body}\n/\n&GRIDSIZE\nadiabatic_electrons = .true.\n/\n",
        encoding="utf-8",
    )
    return path


def test_gkw_runtime_nonlinear_absent_nl_dtim_est_defaults_adaptive_true() -> None:
    params = gkparams_from_runtime({"non_linear": True})
    assert params.adaptive_dt is True
    assert params.fac_dtim_est == 0.95
    assert params.fac_dtim_nl == 1.0


def test_gkw_runtime_nonlinear_false_nl_dtim_est_disables_adaptive() -> None:
    params = gkparams_from_runtime({"non_linear": True, "nl_dtim_est": False})
    assert params.adaptive_dt is False


def test_gkw_runtime_linear_true_nl_dtim_est_still_disables_adaptive() -> None:
    params = gkparams_from_runtime({"non_linear": False, "nl_dtim_est": True})
    assert params.adaptive_dt is False


def test_gkw_runtime_fac_dtim_est_and_fac_dtim_nl_parse_separately(tmp_path: Path) -> None:
    input_path = _write_input(
        tmp_path,
        "NON_LINEAR = .true.\nNL_DTIM_EST = .true.\nFAC_DTIM_EST = 0.77\nFAC_DTIM_NL = 0.33\n",
    )
    runtime = load_runtime_params(str(input_path))
    params = gkparams_from_runtime(runtime)

    assert params.adaptive_dt is True
    assert params.fac_dtim_est == 0.77
    assert params.fac_dtim_nl == 0.33
    # Backward-compatible alias for legacy nonlinear-CFL callers/configs.
    assert params.cfl_safety == 0.33


def test_gkw_runtime_spectral_radius_does_not_enable_adaptive() -> None:
    params = gkparams_from_runtime(
        {"non_linear": False, "nl_dtim_est": True, "spectral_radius": True}
    )
    assert params.spectral_radius is True
    assert params.adaptive_dt is False


def test_yaml_adaptive_dt_default_remains_false() -> None:
    cfg = OmegaConf.create(
        {
            "solver": {},
            "physics": {},
            "geometry": {},
            "grid": {"adiabatic_electrons": True},
        }
    )
    params = gkparams_from_config(cfg)
    assert params.adaptive_dt is False
    assert params.fac_dtim_est == 0.95
    assert params.fac_dtim_nl == 0.95


def test_yaml_adaptive_dt_true_without_safety_preserves_legacy_cfl_default() -> None:
    cfg = OmegaConf.create(
        {
            "solver": {"adaptive_dt": True, "non_linear": True},
            "physics": {},
            "geometry": {},
            "grid": {"adiabatic_electrons": True},
        }
    )
    params = gkparams_from_config(cfg)
    assert params.adaptive_dt is True
    assert params.cfl_safety == 0.95
    assert params.fac_dtim_nl == 0.95


def test_yaml_cfl_safety_keeps_legacy_nonlinear_safety_alias() -> None:
    cfg = OmegaConf.create(
        {
            "solver": {"cfl_safety": 0.42},
            "physics": {},
            "geometry": {},
            "grid": {"adiabatic_electrons": True},
        }
    )
    params = gkparams_from_config(cfg)
    assert params.cfl_safety == 0.42
    assert params.fac_dtim_nl == 0.42


def test_yaml_fac_dtim_nl_wins_over_cfl_safety_alias() -> None:
    cfg = OmegaConf.create(
        {
            "solver": {"adaptive_dt": True, "cfl_safety": 0.42, "fac_dtim_nl": 0.73},
            "physics": {},
            "geometry": {},
            "grid": {"adiabatic_electrons": True},
        }
    )
    params = gkparams_from_config(cfg)
    assert params.cfl_safety == 0.73
    assert params.fac_dtim_nl == 0.73
