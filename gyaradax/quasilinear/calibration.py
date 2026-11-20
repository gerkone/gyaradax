"""Saturation-rule calibrations: the bundled JSON records in `weights/`.

A record names its `rule`, the rule `options`, and the fitted `parameters` per electron model (`adiabatic`,
`kinetic`): always `cn`, plus e.g. sat2's `flux_exponent`. They set one saturated intensity shared by every flux channel
(`Qi`, `Qe`, `Gamma`), which differ only in their QL weights. The QL weights are taken on the kx = 0 chain
(`weights: kx_chain0`), as `point_eval.ql_at_point` evaluates them.
"""

import functools
import json
import pathlib

_DIR = pathlib.Path(__file__).parent / "weights"

# the calibration `auto` resolves to, per rule
DEFAULTS = {"canonical": "canonical_v4", "sat2": "sat2_v4"}


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


def parameters(cal, electrons):
    """Fitted rule keyword arguments of one electron model (``"adiabatic"`` or ``"kinetic"``), shared by all channels."""
    if electrons not in cal["parameters"]:
        raise ValueError(f"calibration of {cal['rule']!r} has no {electrons} fit")
    kw = cal["parameters"][electrons]
    if any(isinstance(v, dict) for v in kw.values()):
        raise ValueError(f"calibration of {cal['rule']!r} has parameters per flux channel; refit it with one set per electron model")
    return dict(kw)


def save(cal, path):
    with open(path, "w") as f:
        json.dump(cal, f, indent=1)
        f.write("\n")
