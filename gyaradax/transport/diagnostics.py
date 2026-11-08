"""jsonl diagnostics of the gyaradax transport models: one row per transport call and radius."""

import json
import os
from typing import Any, Dict, List

import numpy as np

# transport calls written so far per diagnostics path
_CALLS: Dict[str, int] = {}


def _to_native(value: Any) -> Any:
    arr = np.asarray(value)
    return arr.item() if arr.ndim == 0 else arr.tolist()


def write_rows(path: str, model: str, t: float, rows: List[Dict[str, Any]]) -> None:
    """Append the rows of one transport call; the first call to a path starts the file afresh."""
    call = _CALLS.get(path, -1) + 1
    _CALLS[path] = call
    if call == 0:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w" if call == 0 else "a") as f:
        for j, row in enumerate(rows):
            native = {k: _to_native(v) for k, v in row.items()}
            f.write(json.dumps({"call": call, "radius": j, "model": model, "t": t, **native}) + "\n")


def reset() -> None:
    _CALLS.clear()


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
