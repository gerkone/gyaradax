"""Tests for backend selection and optional CUDA imports."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from gyaradax.backends._jax import JAXOps
from gyaradax.state import GKPre


def _fail_cuda_import() -> None:
    raise AssertionError("CUDA backend should not be imported for this path")


def test_backends_package_import_does_not_require_cuda_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU/base import should not eagerly import the optional CUDA backend module."""
    monkeypatch.delitem(sys.modules, "gyaradax.backends", raising=False)
    monkeypatch.delitem(sys.modules, "gyaradax.backends._cuda", raising=False)

    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "gyaradax.backends._cuda" or name.startswith("gyaradax.backends._cuda"):
            raise ImportError("simulated missing optional CUDA module")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    module = importlib.import_module("gyaradax.backends")

    assert hasattr(module, "create_ops")
    assert "gyaradax.backends._cuda" not in sys.modules


def test_jax_backend_selection_does_not_load_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import gyaradax.backends as backends

    monkeypatch.setattr(backends, "_load_cuda_backend", _fail_cuda_import)

    ops = backends.create_ops(GKPre({}), backend="jax")

    assert isinstance(ops, JAXOps)


def test_auto_without_gpu_does_not_load_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import gyaradax.backends as backends

    monkeypatch.setattr(backends.jax, "devices", lambda: [SimpleNamespace(platform="cpu")])
    monkeypatch.setattr(backends, "_load_cuda_backend", _fail_cuda_import)

    ops = backends.create_ops(GKPre({}), backend="auto")

    assert isinstance(ops, JAXOps)


def test_auto_with_gpu_and_cuda_import_failure_falls_back_to_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gyaradax.backends as backends

    def raise_import_error() -> None:
        raise ImportError("simulated missing optional CUDA module")

    monkeypatch.setattr(backends.jax, "devices", lambda: [SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(backends, "_load_cuda_backend", raise_import_error)

    ops = backends.create_ops(GKPre({}), backend="auto")

    assert isinstance(ops, JAXOps)


def test_forced_cuda_without_gpu_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import gyaradax.backends as backends

    monkeypatch.setattr(backends.jax, "devices", lambda: [SimpleNamespace(platform="cpu")])
    monkeypatch.setattr(backends, "_load_cuda_backend", _fail_cuda_import)

    with pytest.raises(RuntimeError, match="backend='cuda' but no GPU found"):
        backends.create_ops(GKPre({}), backend="cuda")


def test_forced_cuda_with_import_failure_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gyaradax.backends as backends

    def raise_import_error() -> None:
        raise ImportError("simulated missing optional CUDA module")

    monkeypatch.setattr(backends.jax, "devices", lambda: [SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(backends, "_load_cuda_backend", raise_import_error)

    with pytest.raises(RuntimeError, match="backend='cuda' but CUDA backend could not be imported"):
        backends.create_ops(GKPre({}), backend="cuda")
