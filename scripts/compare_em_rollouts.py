#!/usr/bin/env python3
"""Compare persisted gyaradax EM rollouts against matching GKW diagnostics.

The gyaradax rollouts are produced by ``scripts/generate_em_gyaradax_rollouts.py``
and live under ``<rollout-root>/<case-key>/``.  Matching GKW generated data is
resolved from ``<gkw-root>`` by translating keys such as
``observables__linear_apar_b01`` to ``observables/linear_apar_b01``.

Examples:

    python scripts/compare_em_rollouts.py --cases observables__linear_apar_b01

    python scripts/compare_em_rollouts.py \
      --rollout-root /local00/bioinf/volkmann/gyrokinetics/em_validation/gyaradax_rollouts \
      --gkw-root /local00/bioinf/volkmann/gyrokinetics/em_validation \
      --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GKW_ROOT = Path("/local00/bioinf/volkmann/gyrokinetics/em_validation")
DEFAULT_ROLLOUT_ROOT = DEFAULT_GKW_ROOT / "gyaradax_rollouts"
DATASETS = (
    ("time", "time.npy", "time.dat"),
    ("fluxes", "fluxes.npy", "fluxes.dat"),
    ("fluxes_em", "fluxes_em.npy", "fluxes_em.dat"),
    ("fluxes_bpar", "fluxes_bpar.npy", "fluxes_bpar.dat"),
)


def _load_gkw_table(path: Path) -> np.ndarray | None:
    """Load a GKW ASCII table with robust one-column and exponent handling."""
    if not path.exists():
        return None
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        return None
    text = "\n".join(raw_lines)
    # GKW es13.5 overflows three-digit exponents as e.g. 1.14010+100.
    text = re.sub(r"(?<=[0-9.])([+-][0-9]{2,})(?=\s|$)", r"E\1", text)
    arr = np.asarray(np.loadtxt(StringIO(text)))
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if len(raw_lines) == 1:
            return arr.reshape(1, -1)
        return arr.reshape(-1, 1)
    return arr


def _load_rollout_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return np.asarray(arr)


def _case_key_from_rollout_dir(path: Path) -> str:
    return path.resolve().name


def resolve_gkw_case_dir(case_key: str, gkw_root: Path) -> Path:
    """Resolve rollout key to a generated GKW case directory."""
    if "__" in case_key:
        parts = case_key.split("__")
        candidate = gkw_root.joinpath(*parts)
        if candidate.exists():
            return candidate
    if "_steps_" in case_key:
        prefix, steps = case_key.rsplit("_steps_", 1)
        candidate = gkw_root / prefix / f"steps_{steps}"
        if candidate.exists():
            return candidate
    candidates = [
        gkw_root / case_key,
        gkw_root / "observables" / case_key,
        gkw_root / "observables_window" / case_key,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return the most likely path for a useful error message.
    return candidates[0]


def _available_rollout_dirs(
    *,
    rollout_root: Path,
    cases: list[str] | None,
    rollout_dirs: list[Path] | None,
) -> list[Path]:
    dirs: list[Path] = []
    if rollout_dirs:
        dirs.extend(path.resolve() for path in rollout_dirs)
    if cases:
        dirs.extend((rollout_root / case).resolve() for case in cases)
    if not dirs:
        dirs.extend(sorted(path for path in rollout_root.iterdir() if path.is_dir()))
    return dirs


def _as_2d_common(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred2 = np.atleast_2d(pred)
    ref2 = np.atleast_2d(ref)
    if pred2.shape[0] == 1 and pred.ndim == 1:
        pred2 = pred.reshape(-1, 1)
    if ref2.shape[0] == 1 and ref.ndim == 1:
        ref2 = ref.reshape(-1, 1)
    rows = min(pred2.shape[0], ref2.shape[0])
    cols = min(pred2.shape[1], ref2.shape[1])
    return pred2[:rows, :cols], ref2[:rows, :cols]


def _time_vector(arr: np.ndarray) -> np.ndarray:
    """Return a 1D physical-time vector from a robustly loaded time table."""
    arr2 = np.asarray(arr)
    if arr2.ndim == 1:
        return arr2.astype(float)
    if arr2.size == 0:
        return np.asarray([], dtype=float)
    return arr2[:, 0].astype(float)


def _time_grid_metrics(pred_time: np.ndarray, ref_time: np.ndarray) -> dict[str, Any]:
    pred_vec = _time_vector(pred_time)
    ref_vec = _time_vector(ref_time)
    rows = min(pred_vec.shape[0], ref_vec.shape[0])
    row_diff = pred_vec[:rows] - ref_vec[:rows]
    finite = np.isfinite(row_diff)
    common_start = max(float(np.nanmin(pred_vec)), float(np.nanmin(ref_vec)))
    common_end = min(float(np.nanmax(pred_vec)), float(np.nanmax(ref_vec)))
    return {
        "pred_rows": int(pred_vec.shape[0]),
        "ref_rows": int(ref_vec.shape[0]),
        "row_pairs": int(rows),
        "row_max_abs_mismatch": float(np.nanmax(np.abs(row_diff[finite])))
        if np.any(finite)
        else None,
        "row_median_abs_mismatch": float(np.nanmedian(np.abs(row_diff[finite])))
        if np.any(finite)
        else None,
        "common_time_start": common_start,
        "common_time_end": common_end,
        "common_time_valid": bool(common_start <= common_end),
    }


def _interp_to_reference_time(
    pred: np.ndarray,
    ref: np.ndarray,
    pred_time: np.ndarray,
    ref_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Interpolate gyaradax values to GKW reference times on the common interval."""
    pred2 = np.atleast_2d(pred)
    ref2 = np.atleast_2d(ref)
    if pred2.shape[0] == 1 and pred.ndim == 1:
        pred2 = pred.reshape(-1, 1)
    if ref2.shape[0] == 1 and ref.ndim == 1:
        ref2 = ref.reshape(-1, 1)
    pred_vec = _time_vector(pred_time)
    ref_vec = _time_vector(ref_time)
    pred_rows = min(pred2.shape[0], pred_vec.shape[0])
    ref_rows = min(ref2.shape[0], ref_vec.shape[0])
    pred2 = pred2[:pred_rows]
    pred_vec = pred_vec[:pred_rows]
    ref2 = ref2[:ref_rows]
    ref_vec = ref_vec[:ref_rows]
    cols = min(pred2.shape[1], ref2.shape[1])
    pred2 = pred2[:, :cols]
    ref2 = ref2[:, :cols]
    common_start = max(float(np.nanmin(pred_vec)), float(np.nanmin(ref_vec)))
    common_end = min(float(np.nanmax(pred_vec)), float(np.nanmax(ref_vec)))
    tol = 1e-12 * max(1.0, abs(common_start), abs(common_end))
    target_mask = (
        np.isfinite(ref_vec) & (ref_vec >= common_start - tol) & (ref_vec <= common_end + tol)
    )
    target_time = ref_vec[target_mask]
    ref_common = ref2[target_mask]
    pred_common = np.empty((target_time.shape[0], cols), dtype=float)
    for col in range(cols):
        source_mask = np.isfinite(pred_vec) & np.isfinite(pred2[:, col])
        source_time = pred_vec[source_mask]
        source_values = pred2[source_mask, col]
        if target_time.shape[0] == 0:
            pred_common[:, col] = np.nan
            continue
        if source_time.shape[0] == 1:
            close = np.abs(target_time - source_time[0]) <= tol
            pred_common[:, col] = np.where(close, source_values[0], np.nan)
            continue
        if source_time.shape[0] < 2:
            pred_common[:, col] = np.nan
            continue
        order = np.argsort(source_time)
        source_time = source_time[order]
        source_values = source_values[order]
        unique_time, unique_indices = np.unique(source_time, return_index=True)
        unique_values = source_values[unique_indices]
        if unique_time.shape[0] < 2:
            pred_common[:, col] = np.nan
            continue
        pred_common[:, col] = np.interp(target_time, unique_time, unique_values)
    info = {
        "alignment": "time",
        "interpolation": "gyaradax_to_gkw_time_linear_value",
        "target": "gkw_time",
        "points_compared": int(target_time.shape[0]),
        "common_time_start": common_start,
        "common_time_end": common_end,
        "_slope_x": target_time,
    }
    return pred_common, ref_common, info


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _log_slope(
    values: np.ndarray, floor: float, x_values: np.ndarray | None = None
) -> float | None:
    amp = np.abs(values)
    if x_values is None:
        x = np.arange(values.shape[0], dtype=float)
    else:
        x = np.asarray(x_values, dtype=np.float64)
        if x.shape[0] != values.shape[0]:
            return None
    mask = np.isfinite(amp) & np.isfinite(x) & (amp > floor)
    if int(np.sum(mask)) < 3:
        return None
    idx = x[mask]
    logs = np.log(amp[mask])
    if float(np.std(logs)) == 0.0:
        return 0.0
    if float(np.std(idx)) == 0.0:
        return None
    return float(np.polyfit(idx, logs, 1)[0])


def _onset_window(
    arr: np.ndarray,
    *,
    abs_threshold: float,
    explosive_ratio: float,
    floor: float,
) -> dict[str, Any]:
    if arr.size == 0:
        return {"window": None, "reason": "empty"}
    per_row = np.max(np.abs(arr), axis=1)
    for i, value in enumerate(per_row):
        if not math.isfinite(float(value)):
            return {"window": i, "reason": "nonfinite", "value": float(value)}
        if float(value) > abs_threshold:
            return {"window": i, "reason": "abs_threshold", "value": float(value)}
        if i >= 5:
            recent = per_row[i - 5 : i + 1]
            baseline = max(float(np.median(per_row[: min(3, i)])), floor)
            monotone = bool(np.all(np.diff(recent) >= 0.0))
            if monotone and float(value) / baseline > explosive_ratio:
                return {
                    "window": i,
                    "reason": "monotone_explosive_growth",
                    "value": float(value),
                    "baseline": baseline,
                    "ratio": float(value) / baseline,
                }
    return {"window": None, "reason": "none", "value": float(per_row[-1])}


def _dataset_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    abs_threshold: float,
    explosive_ratio: float,
    rel_floor: float,
    alignment_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_pred_shape = list((pred.shape[0], 1) if pred.ndim == 1 else np.atleast_2d(pred).shape)
    original_ref_shape = list((ref.shape[0], 1) if ref.ndim == 1 else np.atleast_2d(ref).shape)
    pred, ref = _as_2d_common(pred, ref)
    compared_shape = list(pred.shape)
    diff = pred - ref
    finite = np.isfinite(pred) & np.isfinite(ref)
    denom = np.maximum(np.abs(ref), rel_floor)
    rel = np.abs(diff) / denom
    finite_rel = rel[finite]

    log_corrs: list[float] = []
    slope_ratios: list[float] = []
    slope_x = None if alignment_info is None else alignment_info.get("_slope_x")
    for col in range(pred.shape[1]):
        p_amp = np.abs(pred[:, col])
        r_amp = np.abs(ref[:, col])
        mask = np.isfinite(p_amp) & np.isfinite(r_amp) & (p_amp > rel_floor) & (r_amp > rel_floor)
        if int(np.sum(mask)) >= 3:
            corr = _safe_corr(np.log(p_amp[mask]), np.log(r_amp[mask]))
            if corr is not None:
                log_corrs.append(corr)
        p_slope = _log_slope(pred[:, col], rel_floor, slope_x)
        r_slope = _log_slope(ref[:, col], rel_floor, slope_x)
        if p_slope is not None and r_slope is not None and abs(r_slope) > 1e-14:
            slope_ratios.append(float(p_slope / r_slope))

    sign_mask = finite & (np.abs(pred) > rel_floor) & (np.abs(ref) > rel_floor)
    if int(np.sum(sign_mask)):
        sign_agreement = float(np.mean(np.sign(pred[sign_mask]) == np.sign(ref[sign_mask])))
    else:
        sign_agreement = None

    pred_onset = _onset_window(
        pred, abs_threshold=abs_threshold, explosive_ratio=explosive_ratio, floor=rel_floor
    )
    ref_onset = _onset_window(
        ref, abs_threshold=abs_threshold, explosive_ratio=explosive_ratio, floor=rel_floor
    )

    result = {
        "shape_pred": original_pred_shape,
        "shape_ref": original_ref_shape,
        "shape_compared": compared_shape,
        "truncated_to_common_shape": original_pred_shape != compared_shape
        or original_ref_shape != compared_shape,
        "rows_compared": int(pred.shape[0]),
        "cols_compared": int(pred.shape[1]),
        "max_abs_pred": float(np.nanmax(np.abs(pred))) if pred.size else 0.0,
        "max_abs_ref": float(np.nanmax(np.abs(ref))) if ref.size else 0.0,
        "max_abs_diff": float(np.nanmax(np.abs(diff))) if diff.size else 0.0,
        "max_rel_diff": float(np.nanmax(finite_rel)) if finite_rel.size else None,
        "median_rel_diff": float(np.nanmedian(finite_rel)) if finite_rel.size else None,
        "log_amplitude_corr_median": float(np.median(log_corrs)) if log_corrs else None,
        "log_amplitude_corr_min": float(np.min(log_corrs)) if log_corrs else None,
        "log_slope_ratio_median": float(np.median(slope_ratios)) if slope_ratios else None,
        "log_slope_ratio_min": float(np.min(slope_ratios)) if slope_ratios else None,
        "log_slope_ratio_max": float(np.max(slope_ratios)) if slope_ratios else None,
        "sign_agreement": sign_agreement,
        "gyaradax_onset": pred_onset,
        "gkw_onset": ref_onset,
    }
    if alignment_info is not None:
        result["alignment"] = {k: v for k, v in alignment_info.items() if not k.startswith("_")}
    return result


def _classify_case(case_key: str, dataset_results: dict[str, Any]) -> str:
    if case_key.startswith("observables_window__"):
        return "window checkpoint"
    if "__linear_" in case_key:
        any_div = any(
            result["gkw_onset"].get("window") is not None
            or result["gyaradax_onset"].get("window") is not None
            for name, result in dataset_results.items()
            if name != "time"
        )
        return (
            "full linear observables: divergent/runaway" if any_div else "full linear observables"
        )
    if "__nonlinear_" in case_key:
        any_div = any(
            result["gkw_onset"].get("window") is not None
            or result["gyaradax_onset"].get("window") is not None
            for name, result in dataset_results.items()
            if name != "time"
        )
        return (
            "nonlinear observables: divergent/runaway"
            if any_div
            else "nonlinear observables: bounded"
        )
    return "other"


def compare_rollout(
    rollout_dir: Path,
    *,
    gkw_root: Path,
    abs_threshold: float,
    explosive_ratio: float,
    rel_floor: float,
    align: str,
) -> dict[str, Any]:
    case_key = _case_key_from_rollout_dir(rollout_dir)
    gkw_dir = resolve_gkw_case_dir(case_key, gkw_root)
    if not rollout_dir.exists():
        raise FileNotFoundError(f"rollout directory not found: {rollout_dir}")
    if not gkw_dir.exists():
        raise FileNotFoundError(f"GKW directory not found for {case_key}: {gkw_dir}")

    manifest_path = rollout_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    datasets: dict[str, Any] = {}
    missing: list[str] = []
    pred_time = _load_rollout_array(rollout_dir / "time.npy")
    ref_time = _load_gkw_table(gkw_dir / "time.dat")
    time_grid = (
        _time_grid_metrics(pred_time, ref_time)
        if pred_time is not None and ref_time is not None
        else None
    )
    for name, rollout_file, gkw_file in DATASETS:
        pred = _load_rollout_array(rollout_dir / rollout_file)
        ref = _load_gkw_table(gkw_dir / gkw_file)
        if pred is None or ref is None:
            if pred is not None or ref is not None:
                missing.append(name)
            continue
        if align == "time" and name != "time" and pred_time is not None and ref_time is not None:
            pred_cmp, ref_cmp, alignment_info = _interp_to_reference_time(
                pred, ref, pred_time, ref_time
            )
            datasets[name] = _dataset_metrics(
                pred_cmp,
                ref_cmp,
                abs_threshold=abs_threshold,
                explosive_ratio=explosive_ratio,
                rel_floor=rel_floor,
                alignment_info=alignment_info,
            )
        else:
            datasets[name] = _dataset_metrics(
                pred,
                ref,
                abs_threshold=abs_threshold,
                explosive_ratio=explosive_ratio,
                rel_floor=rel_floor,
                alignment_info={"alignment": align} if name != "time" else None,
            )

    return {
        "case_key": case_key,
        "rollout_dir": str(rollout_dir),
        "gkw_dir": str(gkw_dir),
        "classification": _classify_case(case_key, datasets),
        "align": align,
        "time_grid": time_grid,
        "manifest": {
            key: manifest.get(key)
            for key in ("n_windows", "window_steps", "small_steps", "gyaradax_final_time")
            if key in manifest
        },
        "missing_partial_datasets": missing,
        "datasets": datasets,
    }


def _print_case(result: dict[str, Any]) -> None:
    print(f"\n## {result['case_key']} [{result['classification']}]")
    print(f"GKW: {result['gkw_dir']}")
    manifest = result.get("manifest", {})
    if manifest:
        print("manifest: " + ", ".join(f"{k}={v}" for k, v in manifest.items()))
    time_grid = result.get("time_grid")
    if time_grid:
        print(
            "time-grid: "
            f"common=[{time_grid['common_time_start']:.6e}, {time_grid['common_time_end']:.6e}], "
            f"row_max_mismatch={_fmt_optional(time_grid['row_max_abs_mismatch'])}, "
            f"row_median_mismatch={_fmt_optional(time_grid['row_median_abs_mismatch'])}"
        )
    for name, metrics in result["datasets"].items():
        g_on = metrics["gkw_onset"]
        p_on = metrics["gyaradax_onset"]
        align_info: dict[str, Any] = metrics.get("alignment") or {}
        extra = ""
        if align_info.get("alignment") == "time":
            extra = (
                f" interp_points={align_info.get('points_compared')}"
                f" common=[{align_info.get('common_time_start'):.6e},"
                f"{align_info.get('common_time_end'):.6e}]"
            )
        print(
            f"  {name}: rows={metrics['rows_compared']} cols={metrics['cols_compared']} "
            f"max_abs_diff={metrics['max_abs_diff']:.3e} "
            f"max_rel={_fmt_optional(metrics['max_rel_diff'])} "
            f"median_rel={_fmt_optional(metrics['median_rel_diff'])} "
            f"log_corr_med={_fmt_optional(metrics['log_amplitude_corr_median'])} "
            f"slope_ratio_med={_fmt_optional(metrics['log_slope_ratio_median'])} "
            f"sign={_fmt_optional(metrics['sign_agreement'])} "
            f"GKW_onset={g_on.get('window')}:{g_on.get('reason')} "
            f"GY_onset={p_on.get('window')}:{p_on.get('reason')}"
            f"{extra}"
        )


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=None, help="Rollout case keys to compare")
    parser.add_argument(
        "--rollout-dirs", nargs="*", type=Path, default=None, help="Explicit rollout directories"
    )
    parser.add_argument("--gkw-root", type=Path, default=DEFAULT_GKW_ROOT)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument(
        "--align",
        choices=("row", "time"),
        default="time",
        help=(
            "Comparison alignment. 'time' (default) interpolates gyaradax diagnostics "
            "onto GKW physical times over the common time interval; 'row' preserves "
            "legacy row-index comparison."
        ),
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=1e8,
        help="Magnitude threshold for contextual divergence/onset flagging",
    )
    parser.add_argument(
        "--explosive-ratio",
        type=float,
        default=1e6,
        help="Monotone growth ratio for explosive-growth onset flagging",
    )
    parser.add_argument("--rel-floor", type=float, default=1e-30)
    args = parser.parse_args()

    rollout_dirs = _available_rollout_dirs(
        rollout_root=args.rollout_root, cases=args.cases, rollout_dirs=args.rollout_dirs
    )
    results = [
        compare_rollout(
            rollout_dir,
            gkw_root=args.gkw_root,
            abs_threshold=args.abs_threshold,
            explosive_ratio=args.explosive_ratio,
            rel_floor=args.rel_floor,
            align=args.align,
        )
        for rollout_dir in rollout_dirs
    ]

    payload = {
        "gkw_root": str(args.gkw_root.resolve()),
        "rollout_root": str(args.rollout_root.resolve()),
        "abs_threshold": args.abs_threshold,
        "explosive_ratio": args.explosive_ratio,
        "rel_floor": args.rel_floor,
        "align": args.align,
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"Compared {len(results)} rollouts; align={args.align}; "
            f"abs_threshold={args.abs_threshold:.3e}; explosive_ratio={args.explosive_ratio:.3e}"
        )
        for result in results:
            _print_case(result)


if __name__ == "__main__":
    main()
