#!/usr/bin/env python3
"""Generate persistent gyaradax EM rollouts for GKW generated cases.

The script consumes existing GKW generated data directories containing
``input.dat`` and ``geom.dat``.  It runs gyaradax from the matching initial
condition over GKW diagnostic windows and writes compact NumPy outputs under an
external root for side-by-side analysis with GKW diagnostics.

Examples:

    python scripts/generate_em_gyaradax_rollouts.py \
      --case-dirs /local00/bioinf/volkmann/gyrokinetics/em_validation/observables_window/linear_apar_b01_window_001 \
      --n-windows 1 --device 0 --overwrite

    python scripts/generate_em_gyaradax_rollouts.py \
      --gkw-root /local00/bioinf/volkmann/gyrokinetics/em_validation \
      --cases nonlinear_apar_b01 nonlinear_apar_bpar_b01 \
      --n-windows 1 --device 0 --overwrite
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from _runtime_config_loader import configure_runtime_env


# Runtime flags must be applied before importing JAX or gyaradax modules.
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
from gyaradax.utils import parse_input_dat


DEFAULT_GKW_ROOT = Path("/local00/bioinf/volkmann/gyrokinetics/em_validation")
DEFAULT_OUTPUT_ROOT = DEFAULT_GKW_ROOT / "gyaradax_rollouts"


def _as_flat_array(value: Any) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def _input_control(case_dir: Path) -> dict[str, Any]:
    parsed = parse_input_dat(str(case_dir / "input.dat"))
    control = parsed.get("control", {})
    return control if isinstance(control, dict) else {}


def _normalization_disabled_by_gkw(case_dir: Path) -> bool:
    control = _input_control(case_dir)
    return bool(
        control.get("normalized") is False and control.get("normalize_per_toroidal_mode") is False
    )


def _load_gkw_table(path: Path) -> np.ndarray | None:
    """Load a GKW ASCII table with correct one-column row shape.

    ``np.atleast_2d(np.loadtxt(...))`` turns an N-row one-column file into
    ``(1, N)``.  GKW ``time.dat`` can be exactly such a file, so use the
    number of non-empty data lines to distinguish one row from one column.
    Also accept GKW ``es13.5`` overflow exponents such as ``1.14010+100``.
    """
    if not path.exists():
        return None
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        return None
    text = "\n".join(raw_lines)
    text = re.sub(r"(?<=[0-9.])([+-][0-9]{2,})(?=\s|$)", r"E\1", text)
    data = np.loadtxt(StringIO(text))
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if len(raw_lines) == 1:
            return arr.reshape(1, -1)
        return arr.reshape(-1, 1)
    return arr


def _time_rows(case_dir: Path) -> np.ndarray | None:
    data = _load_gkw_table(case_dir / "time.dat")
    if data is None or data.size == 0:
        return None
    return np.asarray(data[:, 0], dtype=np.float64)


def _available_windows(case_dir: Path, default_windows: int) -> int:
    times = _time_rows(case_dir)
    if times is not None:
        return int(times.shape[0])
    return default_windows


def _case_output_name(case_dir: Path, gkw_root: Path) -> str:
    case_dir = case_dir.resolve()
    gkw_root = gkw_root.resolve()
    try:
        rel = case_dir.relative_to(gkw_root)
        return "__".join(rel.parts)
    except ValueError:
        return case_dir.name


def _candidate_case_dirs(gkw_root: Path, case: str) -> list[Path]:
    case_path = Path(case)
    if case_path.is_absolute() or case_path.exists():
        return [case_path]
    candidates = [
        gkw_root / case,
        gkw_root / "observables" / case,
        gkw_root / "observables_window" / case,
        gkw_root / "observables_rollout_full" / case,
        gkw_root / "observables_rollout_short" / case,
    ]
    if "_steps_" in case:
        prefix, steps = case.rsplit("_steps_", 1)
        candidates.append(gkw_root / prefix / f"steps_{steps}")
    return candidates


def _resolve_case_dirs(gkw_root: Path, cases: list[str], case_dirs: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for raw_dir in case_dirs:
        path = raw_dir.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"case directory does not exist: {path}")
        resolved.append(path)
    for case in cases:
        matches = [
            p.expanduser().resolve() for p in _candidate_case_dirs(gkw_root, case) if p.exists()
        ]
        if not matches:
            tried = ", ".join(str(p) for p in _candidate_case_dirs(gkw_root, case))
            raise FileNotFoundError(f"could not resolve case {case!r}; tried: {tried}")
        resolved.append(matches[0])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _diagnostics(
    df_mixed: Any, geometry: dict[str, Any], params: Any, pre: Any
) -> dict[str, np.ndarray]:
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
    result = {"fluxes": _as_flat_array(es_fluxes)}
    if apar is not None:
        result["fluxes_em"] = _as_flat_array(
            calculate_em_fluxes(geometry, diag_df, apar, params=params, bpar=None, pre=pre)
        )
    if bpar is not None:
        result["fluxes_bpar"] = _as_flat_array(
            calculate_em_fluxes(geometry, diag_df, None, params=params, bpar=bpar, pre=pre)
        )
    return result


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def generate_rollout(
    *,
    case_dir: Path,
    output_dir: Path,
    n_windows: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory exists: {output_dir} (use --overwrite)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = load_geometry(str(case_dir))
    params0 = gkparams_from_input_and_geometry(str(case_dir / "input.dat"), geometry)
    control = _input_control(case_dir)
    window_steps = int(params0.naverage)
    total_windows = _available_windows(case_dir, int(control.get("ntime", 1)))
    if n_windows is not None:
        total_windows = min(total_windows, n_windows)
    if total_windows < 1:
        raise ValueError(f"no windows selected for {case_dir}")

    disabled_by_gkw = _normalization_disabled_by_gkw(case_dir)
    params = replace(params0, naverage=10**9) if disabled_by_gkw else params0
    pre = linear_precompute(geometry, params)
    nsp = 1 if params.adiabatic_electrons else int(jnp.asarray(params.mas).shape[0])

    df = init_f(
        geometry,
        finit=params.finit,
        amp_init_real=params.amp_init,
        norm_eps=params.norm_eps,
        n_species=nsp,
        params=params,
    )
    state = default_state(nky=len(geometry["krho"]))

    times: list[float] = []
    fluxes: list[np.ndarray] = []
    fluxes_em: list[np.ndarray] = []
    fluxes_bpar: list[np.ndarray] = []
    state_norm: list[float] = []
    max_abs_df: list[float] = []

    has_em = False
    has_bpar = False
    for _ in range(total_windows):
        df, _, _, state = gk_run(df, geometry, params, state, n_steps=window_steps, pre=pre)
        diag = _diagnostics(df, geometry, params, pre)
        times.append(float(state.time))
        fluxes.append(diag["fluxes"])
        if "fluxes_em" in diag:
            has_em = True
            fluxes_em.append(diag["fluxes_em"])
        if "fluxes_bpar" in diag:
            has_bpar = True
            fluxes_bpar.append(diag["fluxes_bpar"])
        df_np = np.asarray(df)
        state_norm.append(float(np.linalg.norm(df_np.reshape(-1))))
        max_abs_df.append(float(np.max(np.abs(df_np))))

    np.save(output_dir / "time.npy", np.asarray(times, dtype=np.float64))
    np.save(output_dir / "fluxes.npy", np.stack(fluxes))
    if has_em:
        np.save(output_dir / "fluxes_em.npy", np.stack(fluxes_em))
    if has_bpar:
        np.save(output_dir / "fluxes_bpar.npy", np.stack(fluxes_bpar))
    np.save(output_dir / "state_norm.npy", np.asarray(state_norm, dtype=np.float64))
    np.save(output_dir / "max_abs_df.npy", np.asarray(max_abs_df, dtype=np.float64))
    np.save(output_dir / "final_df.npy", np.asarray(df))

    gkw_times = _time_rows(case_dir)
    if gkw_times is not None:
        np.save(output_dir / "gkw_time.npy", gkw_times[:total_windows])

    manifest: dict[str, Any] = {
        "case_dir": str(case_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "created_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "n_windows": total_windows,
        "window_steps": window_steps,
        "small_steps": total_windows * window_steps,
        "gyaradax_final_time": float(state.time),
        "n_species": nsp,
        "adiabatic_electrons": bool(params.adiabatic_electrons),
        "nlapar": bool(params.nlapar),
        "nlbpar": bool(params.nlbpar),
        "normalization_disabled_by_gkw": disabled_by_gkw,
        "normalization_disabled_for_gyaradax": disabled_by_gkw,
        "outputs": {
            "time.npy": list(np.asarray(times).shape),
            "fluxes.npy": list(np.stack(fluxes).shape),
            "state_norm.npy": [len(state_norm)],
            "max_abs_df.npy": [len(max_abs_df)],
            "final_df.npy": list(np.asarray(df).shape),
        },
    }
    if has_em:
        manifest["outputs"]["fluxes_em.npy"] = list(np.stack(fluxes_em).shape)
    if has_bpar:
        manifest["outputs"]["fluxes_bpar.npy"] = list(np.stack(fluxes_bpar).shape)
    if gkw_times is not None:
        manifest["outputs"]["gkw_time.npy"] = [int(min(gkw_times.shape[0], total_windows))]
    _write_manifest(output_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gkw-root", type=Path, default=DEFAULT_GKW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--case-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--device-list", type=str, default=None)
    parser.add_argument("--preallocate", choices=("true", "false"), default="false")
    args = parser.parse_args()

    if not args.cases and not args.case_dirs:
        parser.error("provide at least one --cases entry or --case-dirs entry")

    case_dirs = _resolve_case_dirs(args.gkw_root, args.cases, args.case_dirs)
    summaries: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        name = _case_output_name(case_dir, args.gkw_root)
        out_dir = args.output_root / name
        manifest = generate_rollout(
            case_dir=case_dir,
            output_dir=out_dir,
            n_windows=args.n_windows,
            overwrite=args.overwrite,
        )
        summaries.append(manifest)
        print(
            f"[OK] {case_dir} -> {out_dir} "
            f"windows={manifest['n_windows']} final_time={manifest['gyaradax_final_time']:.6e}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "manifest.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {index_path}")


if __name__ == "__main__":
    main()
