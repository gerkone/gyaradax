"""Host-side diagnostics of the gyaradax transport models.

Each (transport call, radius) can append one jsonl row (local parameters, fluxes before
the unit conversion, steps used, wallclock). Rows leave the jitted step through unordered
io_callbacks, since the radii are batched; the host labels them from the arrival order and
the static number of rows per call.
"""

import dataclasses
import functools
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

_SINKS: Dict[Tuple[str, int], "DiagnosticsSink"] = {}


def _to_native(value: Any) -> Any:
    arr = np.asarray(value)
    return arr.item() if arr.ndim == 0 else arr.tolist()


@dataclasses.dataclass(eq=False)
class DiagnosticsSink:
    """Host-side writer for one transport model's diagnostics."""

    diagnostics_path: str
    n_radii: int
    n_rows: int = 0
    t_begin: Optional[float] = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.diagnostics_path)), exist_ok=True)
        # a stale file would break the arrival-order labelling
        open(self.diagnostics_path, "w").close()

    @property
    def call_index(self) -> int:
        return self.n_rows // max(self.n_radii, 1)

    @property
    def radius_index(self) -> int:
        return self.n_rows % max(self.n_radii, 1)

    def mark_begin(self, _rho_idx) -> np.ndarray:
        if self.radius_index == 0:
            self.t_begin = time.perf_counter()
        return np.int32(0)

    def write(self, payload, static=None) -> None:
        now = time.perf_counter()
        row: Dict[str, Any] = {**(static or {}), **{k: _to_native(v) for k, v in payload.items()}}
        # devices deliver their rows concurrently
        with self.lock:
            row = {
                "call": self.call_index,
                "radius": self.radius_index,
                "wall_s": None if self.t_begin is None else now - self.t_begin,
                **row,
            }
            with open(self.diagnostics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            self.n_rows += 1


def get_sink(diagnostics_path: str, n_radii: int) -> Optional[DiagnosticsSink]:
    if not diagnostics_path:
        return None
    key = (diagnostics_path, n_radii)
    if key not in _SINKS:
        _SINKS[key] = DiagnosticsSink(diagnostics_path=diagnostics_path, n_radii=n_radii)
    return _SINKS[key]


def reset_sinks() -> None:
    _SINKS.clear()


def mark_begin(sink: Optional[DiagnosticsSink], rho_idx):
    """Stamp the start clock of a radius and return `rho_idx` unchanged.

    The callback's zero is added to the index so the work below depends on it; an
    optimization_barrier token gets eliminated and the stamp drifts past the work.
    """
    if sink is None:
        return rho_idx
    idx = jnp.asarray(rho_idx, dtype=jnp.int32)
    token = io_callback(sink.mark_begin, jax.ShapeDtypeStruct((), jnp.int32), idx, ordered=False)
    return idx + token


def record(sink, static, payload) -> None:
    """Emit one row; `static` is host-side, the payload traced."""
    if sink is None:
        return
    io_callback(functools.partial(sink.write, static=static), (), payload, ordered=False)


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
