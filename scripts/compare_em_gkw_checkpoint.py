#!/usr/bin/env python3
"""Compare a gyaradax run against a loaded GKW EM checkpoint case.

The case directory is expected to contain at least ``input.dat``, ``geom.dat``,
``FDS``, and the legacy diagnostic files produced by GKW (for example
``fluxes.dat`` and ``fluxes_em.dat``).  The script evolves gyaradax from the
same ``init_f`` initial condition for the requested number of small steps,
compares the evolved state with the binary GKW ``FDS`` dump, and separates
state/evolution differences from same-state diagnostic differences.

Examples:

    python scripts/compare_em_gkw_checkpoint.py \
      /local00/bioinf/volkmann/gyrokinetics/em_validation/observables_window/linear_apar_b01_window_001

    python scripts/compare_em_gkw_checkpoint.py CASE_DIR --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from _runtime_config_loader import configure_runtime_env


# Parse runtime flags before importing JAX or gyaradax modules.  CUDA device
# visibility and XLA preallocation must be configured before JAX initializes.
_runtime_parser = argparse.ArgumentParser(add_help=False)
_runtime_parser.add_argument("--device", type=int, default=-1)
_runtime_parser.add_argument("--device-list", type=str, default=None)
_runtime_parser.add_argument("--preallocate", choices=("true", "false"), default="false")
_runtime_args, _ = _runtime_parser.parse_known_args()
configure_runtime_env(
    device=_runtime_args.device,
    device_list=_runtime_args.device_list,
    preallocate=_runtime_args.preallocate,
)

from gyaradax.jax_config import enable_x64

enable_x64()

import jax.numpy as jnp

from gyaradax import load_geometry
from gyaradax.fields import _compute_fields, g_to_f
from gyaradax.integrals import calculate_em_fluxes, get_integrals
from gyaradax.params import gkparams_from_input_and_geometry
from gyaradax.precompute import linear_precompute
from gyaradax.simulate import gk_run
from gyaradax.solver import default_state, init_f
from gyaradax.utils import load_gkw_dump, parse_input_dat


def _as_flat_real_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    return arr.reshape(-1)


def _relative_l2(pred: Any, ref: Any) -> float:
    pred_arr = np.asarray(pred)
    ref_arr = np.asarray(ref)
    denom = float(np.linalg.norm(ref_arr.reshape(-1)))
    if denom == 0.0:
        return math.inf if float(np.linalg.norm(pred_arr.reshape(-1))) != 0.0 else 0.0
    return float(np.linalg.norm((pred_arr - ref_arr).reshape(-1)) / denom)


def _max_abs_diff(pred: Any, ref: Any) -> float:
    return float(np.max(np.abs(np.asarray(pred) - np.asarray(ref))))


def _state_metrics(pred: Any, ref: Any) -> dict[str, float]:
    pred_arr = np.asarray(pred)
    ref_arr = np.asarray(ref)
    return {
        "rel_l2": _relative_l2(pred_arr, ref_arr),
        "max_abs": _max_abs_diff(pred_arr, ref_arr),
        "pred_l2": float(np.linalg.norm(pred_arr.reshape(-1))),
        "ref_l2": float(np.linalg.norm(ref_arr.reshape(-1))),
        "pred_max_abs": float(np.max(np.abs(pred_arr))),
        "ref_max_abs": float(np.max(np.abs(ref_arr))),
    }


def _ascii_half_ulp(value: float) -> float:
    """Approximate half-ulp for GKW ``es13.5`` legacy diagnostic output."""
    value = abs(float(value))
    if value == 0.0 or not math.isfinite(value):
        return 5.0e-6
    return 0.5e-5 * 10.0 ** math.floor(math.log10(value))


def _column_metrics(pred: np.ndarray, ref: np.ndarray) -> list[dict[str, Any]]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    n = max(pred.size, ref.size)
    out: list[dict[str, Any]] = []
    for i in range(n):
        p = float(pred[i]) if i < pred.size else math.nan
        r = float(ref[i]) if i < ref.size else math.nan
        abs_diff = abs(p - r)
        rel_diff = abs_diff / abs(r) if r != 0.0 and math.isfinite(r) else None
        env = _ascii_half_ulp(r)
        out.append(
            {
                "col": i,
                "pred": p,
                "ref": r,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "ascii_half_ulp": env,
                "within_ascii_envelope": bool(abs_diff <= env),
            }
        )
    return out


def _load_numeric_table(path: Path) -> np.ndarray:
    """Load GKW ASCII numeric output, accepting overflowed exponents.

    GKW legacy diagnostics use a narrow ``es13.5`` format. Values with
    three-digit exponents can appear as ``1.14010+100`` rather than
    ``1.14010E+100``. Insert the missing exponent marker before NumPy parsing.
    """
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        return np.empty((0, 0), dtype=float)
    text = "\n".join(raw_lines)
    text = re.sub(r"(?<=[0-9.])([+-][0-9]{2,})(?=\s|$)", r"E\1", text)
    arr = np.asarray(np.loadtxt(StringIO(text)))
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if len(raw_lines) == 1:
            return arr.reshape(1, -1)
        return arr.reshape(-1, 1)
    return arr


def _load_reference_row(path: Path, row_index: int) -> np.ndarray | None:
    if not path.exists():
        return None
    data = _load_numeric_table(path)
    if data.size == 0:
        return None
    idx = min(max(row_index, 0), data.shape[0] - 1)
    return np.asarray(data[idx]).reshape(-1)


def _reference_row_index(case_dir: Path, dump_time: float | None) -> int:
    time_path = case_dir / "time.dat"
    if dump_time is None or not time_path.exists():
        return -1
    times = _load_numeric_table(time_path)
    if times.size == 0:
        return -1
    return int(np.argmin(np.abs(times[:, 0] - dump_time)))


def _input_control(case_dir: Path) -> dict[str, Any]:
    parsed = parse_input_dat(str(case_dir / "input.dat"))
    control = parsed.get("control", {})
    return control if isinstance(control, dict) else {}


def _default_small_steps(case_dir: Path) -> int:
    control = _input_control(case_dir)
    ntime = int(control.get("ntime", 1))
    naverage = int(control.get("naverage", 1))
    return ntime * naverage


def _normalization_disabled_by_gkw(case_dir: Path) -> bool:
    control = _input_control(case_dir)
    return bool(
        control.get("normalized") is False and control.get("normalize_per_toroidal_mode") is False
    )


def _load_case(case_dir: Path) -> tuple[dict[str, Any], Any, int, tuple[int, ...]]:
    geometry = load_geometry(str(case_dir))
    params = gkparams_from_input_and_geometry(str(case_dir / "input.dat"), geometry)
    nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])
    shape = (
        len(geometry["vpgr"]),
        len(geometry["mugr"]),
        len(geometry["sgrid"]),
        len(geometry["kxrh"]),
        len(geometry["krho"]),
    )
    return geometry, params, nsp, shape


def _diagnostics(df_mixed: Any, geometry: dict[str, Any], params: Any, pre: Any) -> dict[str, Any]:
    """Return ES/EM diagnostics using GKW mixed-variable semantics."""
    phi, apar, bpar = _compute_fields(df_mixed, geometry, params, pre)
    diag_df = df_mixed
    if params.nlapar:
        diag_df = g_to_f(df_mixed, apar, params, pre)

    _, es_fluxes = get_integrals(
        diag_df,
        geometry,
        params=params,
        pre=pre,
        adiabatic_electrons=params.adiabatic_electrons,
    )
    result: dict[str, Any] = {"es": _as_flat_real_array(es_fluxes)}

    if apar is not None:
        result["em"] = _as_flat_real_array(
            calculate_em_fluxes(geometry, diag_df, apar, params=params, bpar=None, pre=pre)
        )
    if bpar is not None:
        result["bpar"] = _as_flat_real_array(
            calculate_em_fluxes(geometry, diag_df, None, params=params, bpar=bpar, pre=pre)
        )
    return result


def compare_case(case_dir: Path, n_steps: int | None, disable_normalization: str) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    if n_steps is None:
        n_steps = _default_small_steps(case_dir)

    geometry, params0, nsp, shape = _load_case(case_dir)
    original_naverage = int(params0.naverage)
    disabled_by_gkw = _normalization_disabled_by_gkw(case_dir)
    should_disable = disable_normalization == "always" or (
        disable_normalization == "auto" and disabled_by_gkw
    )
    params = replace(params0, naverage=10**9) if should_disable else params0

    pre = linear_precompute(geometry, params)
    gkw_df, dump_info = load_gkw_dump(str(case_dir / "FDS"), shape, n_species=nsp)
    dump_time_value = dump_info.get("time")
    dump_time = float(dump_time_value) if dump_time_value is not None else None

    df0 = init_f(
        geometry,
        finit=params.finit,
        amp_init_real=params.amp_init,
        norm_eps=params.norm_eps,
        n_species=nsp,
        params=params,
    )
    state0 = default_state(nky=len(geometry["krho"]))
    evolved_df, _, _, state = gk_run(df0, geometry, params, state0, n_steps=n_steps, pre=pre)

    evolved_np = np.asarray(evolved_df)
    gkw_np = np.asarray(gkw_df)
    state_comparison: dict[str, Any] = {"overall": _state_metrics(evolved_np, gkw_np)}
    if evolved_np.ndim == 6:
        state_comparison["species"] = [
            _state_metrics(evolved_np[i], gkw_np[i]) for i in range(evolved_np.shape[0])
        ]

    diag_gkw_state = _diagnostics(gkw_df, geometry, params, pre)
    diag_evolved = _diagnostics(evolved_df, geometry, params, pre)

    ref_idx = _reference_row_index(case_dir, dump_time)
    ref_rows = {
        "es": _load_reference_row(case_dir / "fluxes.dat", ref_idx),
        "em": _load_reference_row(case_dir / "fluxes_em.dat", ref_idx),
        "bpar": _load_reference_row(case_dir / "fluxes_bpar.dat", ref_idx),
    }

    diagnostic_comparison: dict[str, Any] = {}
    for key, pred in diag_gkw_state.items():
        ref = ref_rows.get(key)
        if ref is not None:
            diagnostic_comparison[f"gkw_fds_to_ascii_{key}"] = _column_metrics(pred, ref)
    for key, pred in diag_evolved.items():
        ref = ref_rows.get(key)
        if ref is not None:
            diagnostic_comparison[f"evolved_to_ascii_{key}"] = _column_metrics(pred, ref)
    for key, pred in diag_evolved.items():
        if key in diag_gkw_state:
            diagnostic_comparison[f"evolved_to_gkw_fds_diag_{key}"] = _column_metrics(
                pred, diag_gkw_state[key]
            )

    return {
        "case_dir": str(case_dir),
        "n_steps": n_steps,
        "original_naverage": original_naverage,
        "normalization_disabled_by_gkw": disabled_by_gkw,
        "normalization_disabled_for_gyaradax": should_disable,
        "n_species": nsp,
        "dump_time": dump_time,
        "gyaradax_time": float(state.time),
        "reference_row_index": ref_idx,
        "state_comparison": state_comparison,
        "diagnostic_comparison": diagnostic_comparison,
    }


def _fmt_float(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.6e}"
    except (TypeError, ValueError):
        return str(x)


def print_text_report(result: dict[str, Any]) -> None:
    print(f"case: {result['case_dir']}")
    print(
        "steps: "
        f"{result['n_steps']} small steps; "
        f"gyaradax_time={_fmt_float(result['gyaradax_time'])}; "
        f"gkw_dump_time={_fmt_float(result['dump_time'])}"
    )
    print(
        "normalization: "
        f"gkw_disabled={result['normalization_disabled_by_gkw']} "
        f"gyaradax_disabled={result['normalization_disabled_for_gyaradax']}"
    )
    print("\nstate comparison (gyaradax evolved vs GKW binary FDS)")
    overall = result["state_comparison"]["overall"]
    print(
        "  overall: "
        f"rel_l2={overall['rel_l2']:.12e} "
        f"max_abs={overall['max_abs']:.12e} "
        f"pred_l2={overall['pred_l2']:.12e} "
        f"ref_l2={overall['ref_l2']:.12e}"
    )
    for i, metrics in enumerate(result["state_comparison"].get("species", [])):
        print(f"  species {i}: rel_l2={metrics['rel_l2']:.12e} max_abs={metrics['max_abs']:.12e}")

    print("\ndiagnostic comparisons")
    for name, rows in result["diagnostic_comparison"].items():
        max_abs = max((float(row["abs_diff"]) for row in rows), default=0.0)
        max_rel = max(
            (float(row["rel_diff"]) for row in rows if row["rel_diff"] is not None),
            default=0.0,
        )
        outside = sum(1 for row in rows if not row["within_ascii_envelope"])
        print(
            f"  {name}: max_abs={max_abs:.12e} "
            f"max_rel={max_rel:.12e} outside_ascii={outside}/{len(rows)}"
        )
        for row in rows:
            rel = _fmt_float(row["rel_diff"])
            print(
                f"    col {row['col']}: pred={row['pred']:.12e} "
                f"ref={row['ref']:.12e} abs={row['abs_diff']:.12e} "
                f"rel={rel} ascii_half_ulp={row['ascii_half_ulp']:.1e} "
                f"within={row['within_ascii_envelope']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="GKW generated case directory")
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="CUDA device index; -1 preserves existing CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--device-list",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids; overrides --device.",
    )
    parser.add_argument(
        "--preallocate",
        choices=("true", "false"),
        default="false",
        help="Set XLA_PYTHON_CLIENT_PREALLOCATE before JAX import.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Small gyaradax steps to run. Defaults to NTIME*NAVERAGE from input.dat.",
    )
    parser.add_argument(
        "--disable-linear-normalization",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Disable gyaradax linear-window normalization by setting params.naverage "
            "very large. 'auto' does this when GKW normalized=false and "
            "normalize_per_toroidal_mode=false."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    result = compare_case(args.case_dir, args.n_steps, args.disable_linear_normalization)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_text_report(result)


if __name__ == "__main__":
    main()
