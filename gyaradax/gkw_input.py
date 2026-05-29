"""Neutral helpers for parsed GKW ``input.dat`` data.

This module intentionally does not replace ``parse_input_dat``.  It provides
small coercion and section-order helpers that preserve the existing parsed
schema and repeated-section ordering quirks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ParsedInput = Mapping[str, Mapping[str, Any]]

_TRUE_STRINGS = (".true.", "true", "t")
_FALSE_STRINGS = (".false.", "false", "f")


def as_float(value: Any, default: float) -> float:
    """Coerce a parsed namelist scalar to float, preserving default-on-None."""
    return float(default) if value is None else float(value)


def as_int(value: Any, default: int) -> int:
    """Coerce a parsed namelist scalar to int, preserving default-on-None."""
    return int(default) if value is None else int(value)


def as_bool(value: Any, default: bool) -> bool:
    """Coerce a parsed namelist scalar to bool using current GKW conventions.

    Parsed booleans are already real ``bool`` values.  String handling is kept
    for callers that pass raw or manually constructed parsed dictionaries.  An
    unrecognized non-bool value falls back to the supplied default, matching the
    local helper behavior previously used by ``load_runtime_params``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lv = value.strip().lower()
        if lv in _TRUE_STRINGS:
            return True
        if lv in _FALSE_STRINGS:
            return False
    return bool(default)


def species_section_keys(parsed: ParsedInput, limit: int | None = None) -> tuple[str, ...]:
    """Return repeated ``&SPECIES`` section keys in parser insertion order.

    ``parse_input_dat`` names repeated sections ``species``, ``species0``,
    ``species00``, ... by repeatedly appending ``0``.  This helper deliberately
    preserves that existing key-order quirk by iterating the parsed mapping in
    insertion order and filtering keys with the historical ``startswith`` rule.
    """
    keys = tuple(k for k in parsed if k.startswith("species"))
    return keys if limit is None else keys[:limit]


def species_blocks(parsed: ParsedInput, limit: int | None = None) -> tuple[Mapping[str, Any], ...]:
    """Return repeated ``&SPECIES`` blocks in parser insertion order."""
    return tuple(parsed[key] for key in species_section_keys(parsed, limit=limit))
