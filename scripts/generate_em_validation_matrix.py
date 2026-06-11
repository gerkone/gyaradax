#!/usr/bin/env python3
"""Generate the EM validation case matrix and settings registry.

This is the single source for EM validation case names, stages, structured
settings metadata, and proposed generated-data paths.  It scans
``gkw_ref/em_validation_inputs/*.input.dat`` and can optionally materialize the
canonical rollout matrix for each base case:

    <base>_window_001.input.dat
    <base>_rollout_short.input.dat
    <base>_rollout_full.input.dat

Default mode is read-only: print an inventory and completeness summary.
Mutation is opt-in via ``--materialize`` and/or ``--write-settings``.

Examples:

    python scripts/generate_em_validation_matrix.py
    python scripts/generate_em_validation_matrix.py --materialize
    python scripts/generate_em_validation_matrix.py --write-settings
    python scripts/generate_em_validation_matrix.py --materialize --write-settings --check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("gkw_ref/em_validation_inputs")
DEFAULT_TEMPLATE_DIR = Path("gkw_ref/em_validation_templates")
DEFAULT_MATRIX = Path("gkw_ref/em_validation_matrix.json")
DEFAULT_SETTINGS_DIR = Path("gkw_ref/em_validation_cases")
DEFAULT_DATA_ROOT = Path("/local00/bioinf/volkmann/gyrokinetics/em_validation")
TEMPLATE_SUFFIX = ".input.dat"
REQUIRED_STAGES = ("window_001", "rollout_short", "rollout_full")
SOURCE_PRIORITY = ("source", "rollout_full", "rollout", "window_001", "fixed_steps")
DEFAULT_SHORT_NTIME = 10

SECTION_RE = re.compile(r"^\s*&([A-Za-z0-9_]+)\s*$")
ASSIGN_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*,?\s*$")


@dataclass(frozen=True)
class TemplateInfo:
    case: str
    base_case: str
    stage: str
    path: Path
    ntime: int | None
    naverage: int | None
    setting: str


def _strip_comment(line: str) -> str:
    return line.split("!", 1)[0].strip()


def _parse_scalar(raw: str) -> Any:
    value = raw.strip().rstrip(",").strip()
    if not value:
        return ""
    lower = value.lower()
    if lower in {".true.", "true", "t"}:
        return True
    if lower in {".false.", "false", "f"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) > 1:
            return [_parse_scalar(part) for part in parts]
    number = value.replace("D", "E").replace("d", "e")
    try:
        if re.fullmatch(r"[+-]?\d+", number):
            return int(number)
        return float(number)
    except ValueError:
        return value


def _parse_input_dat(path: Path) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue
        if line == "/":
            current = None
            continue
        match = SECTION_RE.match(line)
        if match:
            current = match.group(1).lower()
            if current == "species":
                sections.setdefault(current, []).append({})
            else:
                sections.setdefault(current, {})
            continue
        section_name = current
        if section_name is None:
            section_name = "_top"
            current = section_name
            sections.setdefault(section_name, {})
        assert section_name is not None
        assign = ASSIGN_RE.match(line)
        if not assign:
            continue
        key = assign.group(1).lower()
        value = _parse_scalar(assign.group(2))
        target: dict[str, Any]
        if section_name == "species":
            target = sections[section_name][-1]
        else:
            target = sections.setdefault(section_name, {})
        if key in target:
            old = target[key]
            if isinstance(old, list):
                old.append(value)
            else:
                target[key] = [old, value]
        else:
            target[key] = value
    return sections


def _case_name(path: Path) -> str:
    name = path.name
    return name[: -len(TEMPLATE_SUFFIX)] if name.endswith(TEMPLATE_SUFFIX) else path.stem


def _stage(case: str) -> str:
    if case.endswith("_window_001"):
        return "window_001"
    if case.endswith("_rollout_full"):
        return "rollout_full"
    if case.endswith("_rollout_short"):
        return "rollout_short"
    if "_steps_" in case:
        return "fixed_steps"
    if case.startswith(("linear_", "nonlinear_")):
        return "rollout"
    return "source"


def _base_case(case: str) -> str:
    for suffix in ("_window_001", "_rollout_full", "_rollout_short"):
        if case.endswith(suffix):
            return case[: -len(suffix)]
    if "_steps_" in case:
        return case.rsplit("_steps_", 1)[0]
    return case


def _setting(case: str) -> str:
    if case.startswith("bpar_bench_apar_beta_"):
        return "bpar_waltz_apar_beta_scan"
    if case.startswith("bpar_bench_bpar_beta_"):
        return "bpar_waltz_apar_bpar_beta_scan"
    if case.startswith("beta_bench_beta_"):
        return "cpc_apar_beta_scan"
    if case.startswith("waltz_bpar_linear"):
        return "bpar_waltz_open_parallel_boundary"
    if case.startswith("diag_lin_all_em"):
        return "diag_lin_all_em"
    if case.startswith("nonlinear_apar_bpar"):
        return "nonlinear_apar_bpar_waltz"
    if case.startswith("nonlinear_apar"):
        return "nonlinear_apar_waltz"
    if case.startswith("nonlinear_bpar_only"):
        return "nonlinear_bpar_only_waltz"
    if case.startswith("apar_bpar"):
        return "fixed_step_apar_bpar"
    if case.startswith("apar_only"):
        return "fixed_step_apar_only"
    if case.startswith("bpar_only"):
        return "fixed_step_bpar_only"
    if case.startswith("linear_apar_bpar"):
        return "linear_apar_bpar_waltz"
    if case.startswith("linear_apar"):
        return "linear_apar_waltz"
    if case.startswith("linear_bpar"):
        return "linear_bpar_waltz"
    return "uncategorized"


def _field_set(control: dict[str, Any]) -> str:
    nlapar = bool(control.get("nlapar", False))
    nlbpar = bool(control.get("nlbpar", False))
    if nlapar and nlbpar:
        return "apar_bpar"
    if nlapar:
        return "apar_only"
    if nlbpar:
        return "bpar_only"
    return "electrostatic"


def _legacy_gkw_dir(data_root: Path, case: str) -> Path:
    if "_steps_" in case:
        prefix, steps = case.rsplit("_steps_", 1)
        return data_root / prefix / f"steps_{steps}"
    if case.endswith("_window_001"):
        return data_root / "observables_window" / case
    if case.endswith("_rollout_full"):
        return data_root / "observables_rollout_full" / case
    if case.endswith("_rollout_short"):
        return data_root / "observables_rollout_short" / case
    if case.startswith(("linear_", "nonlinear_")):
        return data_root / "observables" / case
    return data_root / case


def _structured_paths(data_root: Path, regime: str, stage: str, case: str) -> dict[str, str]:
    base = data_root / regime
    return {
        "gkw": str(base / "gkw" / stage / case),
        "gyaradax": str(base / "gyaradax" / stage / case),
        "checkpoint_comparison": str(base / "comparisons" / "checkpoints" / f"{case}.json"),
        "rollout_comparison": str(base / "comparisons" / "rollouts" / f"{case}.json"),
    }


def _case_record(input_path: Path, data_root: Path) -> dict[str, Any]:
    case = _case_name(input_path)
    parsed = _parse_input_dat(input_path)
    control = parsed.get("control", {})
    gridsize = parsed.get("gridsize", {})
    mode = parsed.get("mode", {})
    geom = parsed.get("geom", {})
    spcgeneral = parsed.get("spcgeneral", {})
    rotation = parsed.get("rotation", {})
    species = parsed.get("species", [])
    diagnostic = parsed.get("diagnostic", {})

    stage = _stage(case)
    regime = "nonlinear" if bool(control.get("non_linear", False)) else "linear"
    field_set = _field_set(control)
    uprim_values = [sp.get("uprim", 0.0) for sp in species if isinstance(sp, dict)]
    vcor = rotation.get("vcor", 0.0)

    return {
        "case": case,
        "base_case": _base_case(case),
        "template": str(input_path),
        "regime": regime,
        "stage": stage,
        "setting": _setting(case),
        "field_set": field_set,
        "paths": {
            "legacy_gkw": str(_legacy_gkw_dir(data_root, case)),
            "structured": _structured_paths(data_root, regime, stage, case),
        },
        "physics": {
            "nonlinear": bool(control.get("non_linear", False)),
            "nlapar": bool(control.get("nlapar", False)),
            "nlbpar": bool(control.get("nlbpar", False)),
            "adiabatic_electrons": bool(spcgeneral.get("adiabatic_electrons", True)),
            "beta": spcgeneral.get("beta"),
            "collisions": bool(control.get("collisions", False)),
            "rotation_vcor": vcor,
            "has_rotation": bool(vcor),
            "uprim": uprim_values,
            "has_uprim": any(bool(value) for value in uprim_values),
            "cf_trap": bool(rotation.get("cf_trap", False)),
            "cf_drift": bool(rotation.get("cf_drift", False)),
        },
        "grid": {
            "nx": gridsize.get("nx"),
            "nmod": gridsize.get("nmod"),
            "n_s_grid": gridsize.get("n_s_grid"),
            "n_vpar_grid": gridsize.get("n_vpar_grid"),
            "n_mu_grid": gridsize.get("n_mu_grid"),
            "nperiod": gridsize.get("nperiod"),
            "number_of_species": gridsize.get("number_of_species"),
        },
        "mode": {
            "kthrho": mode.get("kthrho"),
            "krhomax": mode.get("krhomax"),
            "mode_box": mode.get("mode_box"),
        },
        "geometry": {"q": geom.get("q"), "shat": geom.get("shat"), "eps": geom.get("eps")},
        "time": {
            "ntime": control.get("ntime"),
            "naverage": control.get("naverage"),
            "dtim": control.get("dtim"),
            "gamatol": control.get("gamatol"),
        },
        "numerics": {
            "parallel_boundary_conditions": control.get("parallel_boundary_conditions"),
            "order_of_the_scheme": control.get("order_of_the_scheme"),
            "disp_par": control.get("disp_par"),
            "disp_vp": control.get("disp_vp"),
            "fac_dtim_est": control.get("fac_dtim_est"),
        },
        "species": species,
        "diagnostics": sorted(diagnostic.keys()) if isinstance(diagnostic, dict) else [],
    }


def _read_int(parsed: dict[str, Any], section: str, key: str) -> int | None:
    value = parsed.get(section, {})
    if not isinstance(value, dict):
        return None
    raw = value.get(key)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _template_infos(input_dir: Path) -> list[TemplateInfo]:
    infos: list[TemplateInfo] = []
    for path in sorted(input_dir.glob(f"*{TEMPLATE_SUFFIX}")):
        case = _case_name(path)
        parsed = _parse_input_dat(path)
        infos.append(
            TemplateInfo(
                case=case,
                base_case=_base_case(case),
                stage=_stage(case),
                path=path,
                ntime=_read_int(parsed, "control", "ntime"),
                naverage=_read_int(parsed, "control", "naverage"),
                setting=_setting(case),
            )
        )
    return infos


def _variant_case(base_case: str, stage: str) -> str:
    if stage == "window_001":
        return f"{base_case}_window_001"
    if stage == "rollout_short":
        return f"{base_case}_rollout_short"
    if stage == "rollout_full":
        return f"{base_case}_rollout_full"
    raise ValueError(f"unsupported stage: {stage}")


def _stage_for_matrix_base(infos: list[TemplateInfo]) -> bool:
    return {info.stage for info in infos} != {"fixed_steps"}


def _choose_source(infos: list[TemplateInfo]) -> TemplateInfo:
    by_stage = {info.stage: info for info in infos}
    for stage in SOURCE_PRIORITY:
        if stage in by_stage:
            return by_stage[stage]
    return infos[0]


def _peer_full_ntime(base_info: TemplateInfo, all_infos: list[TemplateInfo]) -> int | None:
    values = sorted(
        {
            info.ntime
            for info in all_infos
            if info.setting == base_info.setting
            and info.stage in {"rollout_full", "rollout"}
            and info.ntime is not None
        }
    )
    return values[-1] if values else None


def _target_ntime(stage: str, source: TemplateInfo, all_infos: list[TemplateInfo]) -> int:
    if stage == "window_001":
        return 1
    if stage == "rollout_short":
        full = _target_ntime("rollout_full", source, all_infos)
        return max(1, min(DEFAULT_SHORT_NTIME, full))
    if stage == "rollout_full":
        if source.stage in {"rollout_full", "rollout"} and source.ntime is not None:
            return source.ntime
        inferred = _peer_full_ntime(source, all_infos)
        if inferred is not None:
            return inferred
        if source.stage == "source" and source.ntime is not None and source.ntime > 1:
            return source.ntime
        return DEFAULT_SHORT_NTIME
    raise ValueError(f"unsupported stage: {stage}")


def _format_fortran_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    return str(value)


def _replace_assignment(text: str, key: str, value: int | str) -> str:
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s*,?\s*)$", re.IGNORECASE | re.MULTILINE
    )
    new, count = pattern.subn(rf"\g<1>{value}\g<3>", text, count=1)
    if count == 0:
        raise ValueError(f"could not find assignment for {key}")
    return new


def _replace_assignments(text: str, replacements: dict[str, Any]) -> str:
    for key, value in replacements.items():
        text = _replace_assignment(text, key, _format_fortran_value(value))
    return text


def _render_variant(
    source: TemplateInfo, target_case: str, target_stage: str, all_infos: list[TemplateInfo]
) -> str:
    text = source.path.read_text(encoding="utf-8")
    text = _replace_assignment(text, "NTIME", _target_ntime(target_stage, source, all_infos))
    header = (
        f"! Generated by scripts/generate_em_validation_matrix.py from {source.path.name}\n"
        f"! Matrix case: {target_case}; stage: {target_stage}\n"
    )
    if text.startswith("! Generated by scripts/generate_em_validation_matrix.py"):
        return text
    return header + text


def _missing_variants(input_dir: Path) -> list[tuple[str, str, TemplateInfo, Path]]:
    infos = _template_infos(input_dir)
    by_base: dict[str, list[TemplateInfo]] = {}
    existing_cases = {info.case for info in infos}
    for info in infos:
        by_base.setdefault(info.base_case, []).append(info)

    missing: list[tuple[str, str, TemplateInfo, Path]] = []
    for base_case, base_infos in sorted(by_base.items()):
        if not _stage_for_matrix_base(base_infos):
            continue
        source = _choose_source(base_infos)
        for stage in REQUIRED_STAGES:
            case = _variant_case(base_case, stage)
            if case in existing_cases:
                continue
            missing.append((case, stage, source, input_dir / f"{case}{TEMPLATE_SUFFIX}"))
    return missing


def materialize_sources_from_matrix(
    matrix_path: Path, template_dir: Path, input_dir: Path, *, overwrite: bool
) -> int:
    if not matrix_path.exists():
        return 0
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    input_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for family in matrix.get("families", []):
        template_path = template_dir / str(family["template"])
        template_text = template_path.read_text(encoding="utf-8")
        for case in family.get("cases", []):
            case_name = str(case["case"])
            out_path = input_dir / f"{case_name}{TEMPLATE_SUFFIX}"
            if out_path.exists() and not overwrite:
                continue
            text = _replace_assignments(template_text, case.get("replacements", {}))
            header = (
                f"! Generated by scripts/generate_em_validation_matrix.py from "
                f"{template_path.relative_to(template_dir.parent)}\n"
                f"! Matrix source case: {case_name}\n"
            )
            out_path.write_text(header + text, encoding="utf-8")
            count += 1
    return count


def materialize_missing(input_dir: Path, *, overwrite: bool) -> int:
    infos = _template_infos(input_dir)
    missing = _missing_variants(input_dir)
    for case, stage, source, path in missing:
        if path.exists() and not overwrite:
            raise FileExistsError(f"target exists: {path}")
        path.write_text(_render_variant(source, case, stage, infos), encoding="utf-8")
        print(f"wrote {path}")
    return len(missing)


def _groups(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = {
        "by_regime": {},
        "by_stage": {},
        "by_setting": {},
        "by_field_set": {},
        "by_base_case": {},
    }
    for record in cases:
        case = record["case"]
        for key, value in (
            ("by_regime", record["regime"]),
            ("by_stage", record["stage"]),
            ("by_setting", record["setting"]),
            ("by_field_set", record["field_set"]),
            ("by_base_case", record["base_case"]),
        ):
            grouped[key].setdefault(value, []).append(case)
    for bucket in grouped.values():
        for names in bucket.values():
            names.sort()
    return grouped


def _stage_completeness(cases: list[dict[str, Any]]) -> dict[str, Any]:
    required = list(REQUIRED_STAGES)
    by_base: dict[str, dict[str, Any]] = {}
    for record in cases:
        base = str(record["base_case"])
        item = by_base.setdefault(
            base,
            {
                "regime": record["regime"],
                "setting": record["setting"],
                "field_set": record["field_set"],
                "present_stages": [],
                "cases": [],
            },
        )
        item["present_stages"].append(record["stage"])
        item["cases"].append(record["case"])
    complete: list[str] = []
    incomplete: dict[str, Any] = {}
    fixed_step_only: dict[str, Any] = {}
    for base, item in sorted(by_base.items()):
        present = sorted(set(item["present_stages"]))
        item["present_stages"] = present
        item["cases"] = sorted(item["cases"])
        if present == ["fixed_steps"]:
            item["missing_stages"] = []
            fixed_step_only[base] = item
            continue
        missing = [stage for stage in required if stage not in present]
        item["missing_stages"] = missing
        if missing:
            incomplete[base] = item
        else:
            complete.append(base)
    return {
        "required_stages": required,
        "complete": complete,
        "incomplete": incomplete,
        "fixed_step_only": fixed_step_only,
    }


def build_registry(input_dir: Path, data_root: Path) -> dict[str, Any]:
    templates = sorted(input_dir.glob(f"*{TEMPLATE_SUFFIX}"))
    cases = [_case_record(path, data_root) for path in templates]
    names = [case["case"] for case in cases]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    return {
        "schema_version": 1,
        "generated_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "data_root": str(data_root),
        "case_count": len(cases),
        "duplicate_case_names": duplicate_names,
        "groups": _groups(cases),
        "stage_completeness": _stage_completeness(cases),
        "cases": cases,
    }


def write_registry(registry: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    groups_dir = output_dir / "groups"
    cases_dir.mkdir(exist_ok=True)
    groups_dir.mkdir(exist_ok=True)

    (output_dir / "index.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for record in registry["cases"]:
        (cases_dir / f"{record['case']}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for group_name, group in registry["groups"].items():
        (groups_dir / f"{group_name}.json").write_text(
            json.dumps(group, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output_dir / "stage_completeness.json").write_text(
        json.dumps(registry["stage_completeness"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_summary(registry: dict[str, Any], input_dir: Path) -> None:
    missing = _missing_variants(input_dir)
    print(f"cases: {registry['case_count']}")
    print(f"missing matrix variants: {len(missing)}")
    for group_name, group in registry["groups"].items():
        print(f"\n{group_name}:")
        for name, cases in sorted(group.items()):
            print(f"  {name}: {len(cases)}")
    completeness = registry["stage_completeness"]
    print("\nstage_completeness:")
    print(f"  required: {', '.join(completeness['required_stages'])}")
    print(f"  complete base cases: {len(completeness['complete'])}")
    print(f"  incomplete base cases: {len(completeness['incomplete'])}")
    print(f"  fixed-step-only bases: {len(completeness['fixed_step_only'])}")
    if missing:
        print("\nmissing:")
        for case, stage, source, path in missing:
            print(f"  {case}: stage={stage} source={source.path.name} -> {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--settings-dir", type=Path, default=DEFAULT_SETTINGS_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--materialize", action="store_true", help="create missing matrix templates"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite generated targets if present"
    )
    parser.add_argument(
        "--write-settings", action="store_true", help="write generated JSON settings"
    )
    parser.add_argument(
        "--check", action="store_true", help="fail on duplicates or incomplete matrix"
    )
    parser.add_argument(
        "--json", action="store_true", help="print full registry JSON after materialization"
    )
    args = parser.parse_args(argv)

    if args.materialize:
        source_count = materialize_sources_from_matrix(
            args.matrix, args.template_dir, args.input_dir, overwrite=args.overwrite
        )
        variant_count = materialize_missing(args.input_dir, overwrite=args.overwrite)
        print(f"materialized source templates: {source_count}")
        print(f"materialized missing variants: {variant_count}")

    registry = build_registry(args.input_dir, args.data_root)
    if args.json:
        print(json.dumps(registry, indent=2, sort_keys=True))
    else:
        print_summary(registry, args.input_dir)

    if args.write_settings:
        write_registry(registry, args.settings_dir)
        print(f"\nwrote {args.settings_dir / 'index.json'}")
        print(f"wrote {args.settings_dir / 'cases'}/*.json")
        print(f"wrote {args.settings_dir / 'groups'}/*.json")
        print(f"wrote {args.settings_dir / 'stage_completeness.json'}")

    if args.check:
        failures: list[str] = []
        if registry["duplicate_case_names"]:
            failures.append("duplicate case names: " + ", ".join(registry["duplicate_case_names"]))
        incomplete = registry["stage_completeness"]["incomplete"]
        if incomplete:
            failures.append("incomplete base cases: " + ", ".join(sorted(incomplete)))
        missing = _missing_variants(args.input_dir)
        if missing:
            failures.append(
                "missing matrix variants: " + ", ".join(case for case, _, _, _ in missing)
            )
        if failures:
            for failure in failures:
                print(failure)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
