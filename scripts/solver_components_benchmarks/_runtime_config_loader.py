"""Load gyaradax.runtime_config without importing the gyaradax package."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable, cast

_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "gyaradax" / "runtime_config.py"
_SPEC = spec_from_file_location("_gyaradax_runtime_config", _RUNTIME_CONFIG_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"could not load runtime config from {_RUNTIME_CONFIG_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

configure_runtime_env = cast(Callable[..., None], getattr(_MODULE, "configure_runtime_env"))
_RUNTIME_CONFIG: Any = _MODULE
