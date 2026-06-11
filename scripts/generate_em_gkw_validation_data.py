#!/usr/bin/env python3
"""Generate GKW reference outputs for gyaradax EM validation.

This script runs a GKW executable against the small input templates in
``gkw_ref/em_validation_inputs`` and stores generated outputs under an external
root such as ``/local00/bioinf/volkmann/gyrokinetics/em_validation``.

Example:

    python scripts/generate_em_gkw_validation_data.py \
      --cases apar_only_steps_001 apar_bpar_steps_001 bpar_only_steps_001

The generated data is intentionally not written into the repository. Tests can
consume it via ``GKW_EM_DATA_ROOT=<output-root>``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_GKW_ROOT = Path("/system/user/galletti/git/gkw")
DEFAULT_GKW_EXE = DEFAULT_GKW_ROOT / "gkw_pike_stable.x"
DEFAULT_INPUT_DIR = Path("gkw_ref/em_validation_inputs")
DEFAULT_REGISTRY = Path("gkw_ref/em_validation_cases/index.json")
DEFAULT_OUTPUT_ROOT = Path("/local00/bioinf/volkmann/gyrokinetics/em_validation")
DEFAULT_MPIRUN = Path("/usr/lib64/openmpi/bin/mpirun")
DEFAULT_MPI_ARGS = ["--mca", "btl", "^openib", "--mca", "mtl", "^ofi"]

_KEEP_PATTERNS = (
    "FDS",
    "FDS.dat",
    "input.dat",
    "input.out",
    "out",
    "run.log",
    "manifest.json",
    "gkwdata.meta",
    "file_count",
    "time.dat",
    "geom.dat",
    "parallel.dat",
    "vpgr.dat",
    "intvp.dat",
    "intmu.dat",
    "sgrid",
    "kxrh",
    "krho",
    "krloc",
    "kzeta",
    "fluxes.dat",
    "fluxes_em.dat",
    "fluxes_bpar.dat",
    "fluxes_lab.dat",
    "fluxes_em_lab.dat",
    "fluxes_bpar_lab.dat",
    "growth.dat",
    "growth_rates_all_modes",
    "frequencies.dat",
    "frequencies_all_modes",
)
_KEEP_PREFIXES = (
    "apar",
    "bpar",
    "zphi",
    "zevo",
)
_KEEP_DIRS = (
    "spectrum",
    "spectra_3D",
)

_NPROCS_RE = re.compile(r"\bn_procs_(?:s|vpar|mu|sp)\s*=\s*([0-9]+)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_rev(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _parse_nprocs(input_path: Path) -> int:
    text = input_path.read_text(encoding="utf-8")
    vals = [int(m.group(1)) for m in _NPROCS_RE.finditer(text)]
    if not vals:
        return 1
    n = 1
    for val in vals:
        n *= val
    return max(n, 1)


def _case_name(input_path: Path) -> str:
    name = input_path.name
    suffix = ".input.dat"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return input_path.stem


def _case_output_dir(output_root: Path, case: str) -> Path:
    marker = "_steps_"
    if marker in case:
        prefix, steps = case.rsplit(marker, 1)
        return output_root / prefix / f"steps_{steps}"
    if "_window_001" in case:
        return output_root / "observables_window" / case
    if case.endswith("_rollout_full"):
        return output_root / "observables_rollout_full" / case
    if case.endswith("_rollout_short"):
        return output_root / "observables_rollout_short" / case
    if case.startswith(("linear_", "nonlinear_")):
        return output_root / "observables" / case
    return output_root / case


def _should_keep_output(path: Path, run_dir: Path) -> bool:
    rel = path.relative_to(run_dir)
    parts = rel.parts
    name = path.name
    if parts and parts[0] in _KEEP_DIRS:
        return True
    if name in _KEEP_PATTERNS:
        return True
    if name.startswith(_KEEP_PREFIXES):
        return True
    if name.startswith(("kxspec", "kyspec")):
        return True
    return name.endswith("_lab.dat") and name.startswith("fluxes")


def _collect_outputs(run_dir: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and _should_keep_output(p, run_dir):
            name = str(p.relative_to(run_dir))
            outputs[name] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
    return outputs


def _run_case(
    *,
    input_path: Path,
    output_root: Path,
    run_dir: Path | None,
    registry_record: dict[str, Any] | None,
    gkw_exe: Path,
    gkw_root: Path,
    mpirun: Path,
    mpi_args: list[str],
    use_mpi: str,
    overwrite: bool,
    dry_run: bool,
    env: dict[str, str],
) -> dict[str, Any]:
    case = _case_name(input_path)
    run_dir = _case_output_dir(output_root, case) if run_dir is None else run_dir
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory exists: {run_dir} (use --overwrite)")
        if not dry_run:
            shutil.rmtree(run_dir)
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, run_dir / "input.dat")

    nprocs = _parse_nprocs(input_path)
    should_mpi = use_mpi == "always" or (use_mpi == "auto" and nprocs > 1)
    if should_mpi:
        command = [str(mpirun), *mpi_args, "-np", str(nprocs), str(gkw_exe)]
    else:
        command = [str(gkw_exe)]

    manifest: dict[str, Any] = {
        "case": case,
        "input_template": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "run_dir": str(run_dir.resolve()),
        "created_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "gkw_executable": str(gkw_exe.resolve()),
        "gkw_executable_sha256": _sha256(gkw_exe) if gkw_exe.exists() else None,
        "gkw_root": str(gkw_root.resolve()),
        "gkw_git_rev": _git_rev(gkw_root),
        "nprocs": nprocs,
        "command": command,
        "dry_run": dry_run,
        "registry_record": registry_record,
    }

    if dry_run:
        return manifest

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        proc = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    manifest["returncode"] = proc.returncode
    manifest["outputs"] = _collect_outputs(run_dir)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return manifest


def _resolve_inputs(input_dir: Path, cases: list[str]) -> list[Path]:
    if cases:
        paths = []
        for case in cases:
            p = input_dir / case
            if p.suffix != ".dat":
                p = input_dir / f"{case}.input.dat"
            paths.append(p)
    else:
        paths = sorted(input_dir.glob("*.input.dat"))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing input templates: " + ", ".join(missing))
    return paths


def _load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_registry_records(
    registry: dict[str, Any],
    *,
    cases: list[str],
    regime: str | None,
    stage: str | None,
) -> list[dict[str, Any]]:
    requested = set(cases)
    records: list[dict[str, Any]] = []
    for record in registry.get("cases", []):
        if requested and record.get("case") not in requested:
            continue
        if regime is not None and record.get("regime") != regime:
            continue
        if stage is not None and record.get("stage") != stage:
            continue
        if not requested and record.get("stage") == "source":
            continue
        records.append(record)
    if requested:
        matched = {record["case"] for record in records}
        missing = sorted(requested - matched)
        if missing:
            raise FileNotFoundError("missing registry cases: " + ", ".join(missing))
    return records


def _registry_input_path(record: dict[str, Any], input_dir: Path) -> Path:
    template = Path(str(record.get("template", "")))
    if template.exists():
        return template
    return input_dir / f"{record['case']}.input.dat"


def _registry_gkw_dir(record: dict[str, Any]) -> Path:
    return Path(str(record["paths"]["structured"]["gkw"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--legacy-paths", action="store_true")
    parser.add_argument("--regime", choices=("linear", "nonlinear"), default=None)
    parser.add_argument(
        "--stage",
        choices=("fixed_steps", "window_001", "rollout", "rollout_short", "rollout_full"),
        default=None,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gkw-root", type=Path, default=DEFAULT_GKW_ROOT)
    parser.add_argument("--gkw-exe", type=Path, default=DEFAULT_GKW_EXE)
    parser.add_argument("--mpirun", type=Path, default=DEFAULT_MPIRUN)
    parser.add_argument(
        "--mpi-arg",
        action="append",
        default=None,
        help="Extra/replacement mpirun arg; repeatable. Defaults to OpenMPI fabric exclusions.",
    )
    parser.add_argument("--mpi", choices=("auto", "always", "never"), default="auto")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[],
        help="Case stems, e.g. apar_only_steps_001. Defaults to all templates.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    gkw_exe = args.gkw_exe.resolve()
    gkw_root = args.gkw_root.resolve()
    mpirun = args.mpirun.resolve()
    mpi_args = args.mpi_arg if args.mpi_arg is not None else DEFAULT_MPI_ARGS

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not gkw_exe.exists():
        raise FileNotFoundError(f"GKW executable not found: {gkw_exe}")
    if args.mpi != "never" and not mpirun.exists():
        raise FileNotFoundError(f"mpirun not found: {mpirun}")

    env = os.environ.copy()
    # Avoid the active Python virtualenv's linker/tool wrappers affecting OpenMPI-launched GKW.
    env["PATH"] = "/usr/lib64/openmpi/bin:/usr/bin:/bin:" + env.get("PATH", "")

    manifests = []
    use_registry = not args.no_registry and args.registry.exists()
    if use_registry and not args.legacy_paths:
        registry = _load_registry(args.registry)
        records = _select_registry_records(
            registry, cases=args.cases, regime=args.regime, stage=args.stage
        )
        for record in records:
            input_path = _registry_input_path(record, input_dir)
            if not input_path.exists():
                raise FileNotFoundError(f"missing input template: {input_path}")
            manifest = _run_case(
                input_path=input_path,
                output_root=output_root,
                run_dir=_registry_gkw_dir(record),
                registry_record=record,
                gkw_exe=gkw_exe,
                gkw_root=gkw_root,
                mpirun=mpirun,
                mpi_args=mpi_args,
                use_mpi=args.mpi,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                env=env,
            )
            manifests.append(manifest)
            status = "DRY" if args.dry_run else "OK"
            print(f"[{status}] {manifest['case']} -> {manifest['run_dir']}")
    else:
        if args.regime is not None or args.stage is not None:
            raise ValueError("--regime/--stage filters require registry mode")
        for input_path in _resolve_inputs(input_dir, args.cases):
            manifest = _run_case(
                input_path=input_path,
                output_root=output_root,
                run_dir=None,
                registry_record=None,
                gkw_exe=gkw_exe,
                gkw_root=gkw_root,
                mpirun=mpirun,
                mpi_args=mpi_args,
                use_mpi=args.mpi,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                env=env,
            )
            manifests.append(manifest)
            status = "DRY" if args.dry_run else "OK"
            print(f"[{status}] {manifest['case']} -> {manifest['run_dir']}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "manifest.json"
        summary_path.write_text(json.dumps(manifests, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
