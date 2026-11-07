"""Saturation-rule calibrations: the bundled JSON records in `weights/`.

A record names its `rule`, the rule `options` shared by every channel, and the fitted `parameters` per electron model
(`adiabatic`, `kinetic`) and flux channel (`Qi`, `Qe`, `Gamma`): always `cn`, plus e.g. sat2's `flux_exponent`. The
QL weights are taken on the kx = 0 chain (`weights: kx_chain0`), as `point_eval.ql_at_point` evaluates them.
"""

import functools
import json
import pathlib

_DIR = pathlib.Path(__file__).parent / "weights"

# the calibration `auto` resolves to, per rule
DEFAULTS = {"canonical": "canonical_v3", "sat2": "sat2_v3"}


def names():
    return sorted(p.stem for p in _DIR.glob("*.json"))


@functools.lru_cache(maxsize=16)
def load(spec, rule=None):
    """A calibration record: a bundled name, a JSON path, or ``"auto"`` for the rule's default.

    Args:
        spec: Name, path or ``"auto"``.
        rule: Rule whose default ``"auto"`` picks.

    Returns:
        The record dict, or None when ``spec`` is empty.
    """
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


def channel_parameters(cal, electrons):
    """Fitted rule keyword arguments per flux channel of one electron model (``"adiabatic"`` or ``"kinetic"``)."""
    if electrons not in cal["parameters"]:
        raise ValueError(f"calibration of {cal['rule']!r} has no {electrons} fit")
    return {ch: dict(kw) for ch, kw in cal["parameters"][electrons].items()}


def save(cal, path):
    with open(path, "w") as f:
        json.dump(cal, f, indent=1)
        f.write("\n")
