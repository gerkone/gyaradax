"""Guards for canonical EM test inputs.

EM unit tests should not grow independent duplicate input.dat files.  Shared EM
physics cases belong in the EM validation template/matrix machinery under
``gkw_ref/em_validation_templates`` and ``gkw_ref/em_validation_matrix.json``.
"""

from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
EM_TEMPLATE_DIR = REPO_ROOT / "gkw_ref" / "em_validation_templates"
GKW_CASES_DIR = REPO_ROOT / "tests" / "data" / "gkw_cases"


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out: list[str] = []
    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        if char == "!" and not in_single and not in_double:
            break
        out.append(char)
    return "".join(out)


def _normalized_input(path: Path) -> str:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw).strip().lower()
        if line:
            lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(lines)


def test_no_exact_duplicate_em_input_cases() -> None:
    """Tracked EM inputs/templates should not duplicate each other exactly."""
    paths = sorted(EM_TEMPLATE_DIR.glob("*.input.dat"))
    paths += sorted(GKW_CASES_DIR.glob("*/input.dat"))

    by_content: dict[str, list[Path]] = {}
    for path in paths:
        by_content.setdefault(_normalized_input(path), []).append(path)

    duplicates = [items for items in by_content.values() if len(items) > 1]
    formatted = [
        ", ".join(str(path.relative_to(REPO_ROOT)) for path in items) for items in duplicates
    ]
    assert not duplicates, "duplicate EM input cases found: " + "; ".join(formatted)
