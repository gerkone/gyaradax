"""Optional run telemetry and profiling, written into the run directory.

``RunTelemetry`` appends JSON lines to ``telemetry.jsonl``: one ``start`` record
(environment, devices, configuration), one ``block`` record per checkpoint block
(wall time, throughput, dt statistics, device memory in use and its peak since the
process started) and an ``end`` record. ``profile_block`` records one block with the
JAX profiler into ``profile/step_NNNNNN/`` (viewable in Perfetto / TensorBoard) and
writes ``profile_summary.json`` with the GPU time per kernel of that block. Both are
off unless the run asks for them.
"""

from __future__ import annotations

import contextlib
import datetime
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import defaultdict
from typing import Any, Iterator

import jax
import numpy as np


def _git_revision() -> str | None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        rev = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{rev}{'-dirty' if dirty else ''}"


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def device_memory(devices) -> dict[str, list[float]]:
    peak, in_use = [], []
    for d in devices:
        try:
            stats = d.memory_stats() or {}
        except Exception:
            stats = {}
        peak.append(round(stats.get("peak_bytes_in_use", 0) / 2**30, 3))
        in_use.append(round(stats.get("bytes_in_use", 0) / 2**30, 3))
    return {"process_peak_gib": peak, "in_use_gib": in_use}


class RunTelemetry:
    """Per-block run telemetry appended to ``<output_dir>/telemetry.jsonl``."""

    def __init__(self, output_dir: str, info: dict[str, Any], devices=None):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "telemetry.jsonl")
        self.devices = list(devices) if devices is not None else jax.devices()
        record = {
            "event": "start",
            "time": _now(),
            "host": socket.gethostname(),
            "argv": sys.argv,
            "gyaradax": _git_revision(),
            "jax": jax.__version__,
            "devices": [f"{d.id}: {d.device_kind}" for d in self.devices],
            "env": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("XLA_", "CUDA_VISIBLE", "GYARADAX_", "JAX_"))
            },
            **info,
        }
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def compiled(self, seconds: float) -> None:
        self._write({"event": "compile", "time": _now(), "seconds": round(seconds, 3)})

    def block(
        self,
        step: int,
        sim_time: float,
        n_steps: int,
        wall: float,
        dt_info: Any = None,
        profiled: bool = False,
    ) -> None:
        record: dict[str, Any] = {
            "event": "block",
            "profiled": profiled,
            "time": _now(),
            "step": step,
            "sim_time": sim_time,
            "n_steps": n_steps,
            "wall_s": round(wall, 4),
            "ms_per_step": round(1e3 * wall / max(n_steps, 1), 4),
            "steps_per_s": round(n_steps / wall, 3) if wall > 0 else None,
            **device_memory(self.devices),
        }
        if dt_info is not None:
            dt = np.asarray(dt_info["dt_used"]).reshape(-1)
            if dt.size:
                record["dt"] = {
                    "min": float(dt.min()),
                    "mean": float(dt.mean()),
                    "max": float(dt.max()),
                }
        self._write(record)

    def close(self, total_wall: float, steps: int, status: str = "completed") -> None:
        self._write(
            {
                "event": "end",
                "time": _now(),
                "status": status,
                "wall_s": round(total_wall, 3),
                "steps": steps,
            }
        )


def kernel_summary(trace_dir: str, n_steps: int, top: int = 40) -> dict[str, Any]:
    """GPU time per kernel from a JAX profiler trace (CUPTI), per step and in total."""
    from jax.profiler import ProfileData

    paths = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    if not paths:
        return {"kernels": [], "note": "no trace found"}
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path in paths:
        for plane in ProfileData.from_file(path).planes:
            if not plane.name.startswith("/device:GPU"):
                continue
            for line in plane.lines:
                # stream lines hold kernels and copies; the others summarise XLA ops
                if "Stream" not in line.name:
                    continue
                for ev in line.events:
                    agg[ev.name][0] += ev.duration_ns
                    agg[ev.name][1] += 1
    total = sum(v[0] for v in agg.values())
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]
    return {
        "n_steps": n_steps,
        "gpu_ms_per_step": round(total / max(n_steps, 1) / 1e6, 4),
        "kernels": [
            {
                "name": name,
                "ms_per_step": round(ns / max(n_steps, 1) / 1e6, 4),
                "share": round(ns / total, 4) if total else 0.0,
                "calls_per_step": round(count / max(n_steps, 1), 2),
            }
            for name, (ns, count) in rows
        ],
    }


@contextlib.contextmanager
def profile_block(output_dir: str, n_steps: int, step: int) -> Iterator[None]:
    """Trace the enclosed block, which starts at ``step``, into ``<output_dir>/profile/step_NNNNNN``.

    ``profile_summary.json`` gets the kernel times of this trace only.
    """
    trace_dir = os.path.join(output_dir, "profile", f"step_{step:06d}")
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.makedirs(trace_dir)
    with jax.profiler.trace(trace_dir):
        yield
    summary = {"step": step, **kernel_summary(trace_dir, n_steps)}
    with open(os.path.join(output_dir, "profile_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"profile: {trace_dir} ({summary.get('gpu_ms_per_step', 0):.3f} GPU ms/step)")
    for row in summary["kernels"][:8]:
        print(f"  {row['ms_per_step']:9.3f} ms/step {100 * row['share']:5.1f}%  {row['name'][:80]}")
