"""Bundled saturation-rule calibrations, one JSON file per fit.

A calibration holds the rule it belongs to, the fitted `parameters` (always `cn`, plus e.g. sat2's
`flux_exponent`), the rule `options` held fixed during the fit, optionally the flux-tube `grid`
it was fitted at, and the `fit` record with its `test` (and `cross_validation`) metrics.
"""

import functools
import json
import pathlib

_DIR = pathlib.Path(__file__).parent

# the calibration `auto` resolves to, per rule
DEFAULTS = {"canonical": "canonical_iter_hybrid_v1", "sat2": "sat2_v1"}


def names():
    return sorted(p.stem for p in _DIR.glob("*.json"))


@functools.lru_cache(maxsize=16)
def load(spec, rule=None):
    """Calibration from a bundled name, a JSON path, or `auto` (the rule's default); None if empty."""
    if not spec:
        return None
    if spec == "auto":
        spec = DEFAULTS.get(rule)
        if spec is None:
            return None
    path = _DIR / f"{spec}.json"
    with open(path if path.exists() else spec) as f:
        cal = json.load(f)
    if rule is not None and cal["rule"] != rule:
        raise ValueError(f"calibration {spec!r} is for rule {cal['rule']!r}, not {rule!r}")
    return cal


def rule_kwargs(cal):
    """Keyword arguments of the rule besides cn: the fitted parameters and the fixed options."""
    return {**cal.get("options", {}), **{k: v for k, v in cal["parameters"].items() if k != "cn"}}


def save(cal, path):
    with open(path, "w") as f:
        json.dump(cal, f, indent=1)
        f.write("\n")
