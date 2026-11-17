"""Tests for pre-JAX runtime environment configuration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_runtime_config() -> ModuleType:
    """Load runtime_config without importing the gyaradax package or JAX."""
    path = Path(__file__).resolve().parents[2] / "gyaradax" / "core" / "runtime_config.py"
    spec = importlib.util.spec_from_file_location("_gyaradax_runtime_config_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_device_order(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_DEVICE_ORDER")


def test_runtime_config_module_does_not_import_jax(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "jax", raising=False)

    module = _load_runtime_config()

    assert module is not None
    assert "jax" not in sys.modules


def test_device_sets_cuda_visible_devices(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    module.configure_runtime_env(device=3)

    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_device_sets_pci_order_unless_set(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)

    module.configure_runtime_env(device=1)
    assert module.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    module.configure_runtime_env(device=2)
    assert module.os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_device_warns_when_replacing_visible_devices(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    with pytest.warns(UserWarning, match="CUDA_VISIBLE_DEVICES=3"):
        module.configure_runtime_env(device=0)
    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_device_list_takes_precedence(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    module.configure_runtime_env(device=3, device_list="0,2")

    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == "0,2"


def test_device_minus_one_leaves_existing_cuda_visible_devices(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    module.configure_runtime_env(device=-1)

    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_preallocate_defaults_false_without_overriding_existing(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    module.configure_runtime_env()
    assert module.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    module.configure_runtime_env(preallocate=False)
    assert module.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


def test_preallocate_none_leaves_environment_untouched(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    module.configure_runtime_env(preallocate=None)

    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in module.os.environ


def test_multi_gpu_axes_append_existing_xla_flags(monkeypatch) -> None:
    module = _load_runtime_config()
    monkeypatch.setenv("XLA_FLAGS", "--existing=true")

    module.configure_runtime_env(n_gpus_sp=1, n_gpus_vp=2, n_gpus_mu=1)

    xla_flags = module.os.environ["XLA_FLAGS"]
    assert xla_flags.startswith("--existing=true ")
    assert "--xla_gpu_enable_latency_hiding_scheduler=true" in xla_flags
    assert "--xla_gpu_enable_pipelined_all_reduce=true" in xla_flags
    assert "--xla_gpu_enable_pipelined_all_gather=true" in xla_flags
    assert "--xla_gpu_enable_while_loop_double_buffering=true" in xla_flags
