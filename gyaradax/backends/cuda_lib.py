"""Locating, building and probing the CUDA kernel library.

A build belongs to the Python environment whose CUDA runtime, cuFFT and nvJitLink it links:
``gyaradax build-cuda`` installs it under ``$PREFIX/lib/gyaradax/<key>/``, ``key`` hashing the
kernel sources and the jaxlib version. Lookup order: ``$GYARADAX_CUDA_LIB``, the environment's
build for this key, then a development build next to the sources. Standard library only:
``python cuda_lib.py probe`` prints this host's state as JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Sequence

KERNELS_DIR = Path(__file__).resolve().parent / "cuda_kernels"
LIB_NAME = "libgyaradax_cuda.so"
BUILD_INFO = "build.json"
SOURCE_SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".txt", ".cmake"}


def _jaxlib_version() -> str:
    try:
        return metadata.version("jaxlib")
    except metadata.PackageNotFoundError:
        return "none"


def build_key() -> str:
    """Hash of the kernel sources and the jaxlib version: the directory a matching build lives in."""
    h = hashlib.sha256(_jaxlib_version().encode())
    for path in sorted(KERNELS_DIR.rglob("*")):
        rel = path.relative_to(KERNELS_DIR)
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and not rel.parts[0].startswith("_"):
            h.update(str(rel).encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def env_build_dir(prefix: Optional[str] = None) -> Path:
    return Path(prefix or sys.prefix) / "lib" / "gyaradax" / build_key()


def resolve() -> Path:
    """The library to load: ``$GYARADAX_CUDA_LIB``, the environment's build, else the checkout's."""
    override = os.environ.get("GYARADAX_CUDA_LIB")
    if override:
        return Path(override)
    built = env_build_dir() / LIB_NAME
    return built if built.exists() else KERNELS_DIR / LIB_NAME


def build_info(lib: Path) -> dict[str, Any]:
    try:
        return json.loads((lib.parent / BUILD_INFO).read_text())
    except (OSError, ValueError):
        return {}


def gpu_compute_caps() -> list[int]:
    """Compute capabilities of the visible GPUs (``8.0`` -> 80), from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    caps = {int(line.strip().replace(".", "")) for line in out.splitlines() if line.strip()}
    return sorted(caps)


def probe() -> dict[str, Any]:
    """This host's CUDA library: the path picked, its build record, and whether it serves the GPUs.

    ``usable`` is false when there is no library, or when its build record lacks a GPU's
    architecture; a development build has no record and counts as usable.
    """
    lib = resolve()
    info = build_info(lib)
    caps = gpu_compute_caps()
    archs = [int(a) for a in info.get("archs", [])]
    missing = [c for c in caps if archs and c not in archs]
    exists = lib.exists()
    reason = "" if exists else "no library"
    if exists and missing:
        reason = f"built for sm {archs}, GPU is sm {missing}"
    return {
        "host": socket.gethostname(),
        "python": sys.executable,
        "key": build_key(),
        "lib": str(lib),
        "exists": exists,
        "archs": archs,
        "gpus": caps,
        "usable": exists and not missing,
        "reason": reason,
    }


def _tool(name: str) -> str:
    path = shutil.which(
        name, path=os.pathsep.join([str(Path(sys.prefix) / "bin"), os.environ.get("PATH", "")])
    )
    if path is None:
        raise SystemExit(f"error: {name} not found in {sys.prefix}/bin or on PATH")
    return path


def host_compiler() -> Optional[str]:
    """``$CXX``, else the environment's conda C++ compiler (the toolchain its CUDA libraries match)."""
    if os.environ.get("CXX"):
        return os.environ["CXX"]
    found = sorted(Path(sys.prefix, "bin").glob("*-conda-linux-gnu-c++"))
    return str(found[0]) if found else None


def build(
    archs: Sequence[int],
    lto_archs: Optional[Sequence[int]] = None,
    prefix: Optional[str] = None,
    jobs: Optional[int] = None,
    keep: bool = False,
) -> Path:
    """Build the kernels for ``archs`` with this environment's toolchain and install them for it."""
    dest = env_build_dir(prefix)
    env = dict(
        os.environ,
        PATH=os.pathsep.join([str(Path(sys.prefix) / "bin"), os.environ.get("PATH", "")]),
    )
    work = Path(tempfile.mkdtemp(prefix="gyaradax-cuda-"))
    configure = [
        _tool("cmake"),
        "-S",
        str(KERNELS_DIR),
        "-B",
        str(work),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGPU_ARCHITECTURES={';'.join(map(str, archs))}",
    ]
    cxx = host_compiler()
    if cxx:
        configure += [f"-DCMAKE_CXX_COMPILER={cxx}", f"-DCMAKE_CUDA_HOST_COMPILER={cxx}"]
    if lto_archs:
        configure.append(f"-DLTO_ARCHITECTURES={';'.join(map(str, lto_archs))}")
    try:
        subprocess.run(configure, check=True, env=env)
        subprocess.run(
            [_tool("cmake"), "--build", str(work), "-j", str(jobs or os.cpu_count() or 4)],
            check=True,
            env=env,
        )
        built = next(work.rglob(LIB_NAME))
        nvcc = subprocess.run(
            [_tool("nvcc"), "--version"], capture_output=True, text=True, env=env
        ).stdout
        dest.mkdir(parents=True, exist_ok=True)
        tmp = dest / f".{LIB_NAME}.{os.getpid()}"
        shutil.copy2(built, tmp)
        os.replace(tmp, dest / LIB_NAME)
        record = {
            "key": build_key(),
            "archs": sorted(int(a) for a in archs),
            "lto_archs": sorted(int(a) for a in lto_archs) if lto_archs else "cmake default",
            "jaxlib": _jaxlib_version(),
            "nvcc": nvcc.strip().splitlines()[-1] if nvcc.strip() else "",
            "cxx": cxx or "default",
            "sources": str(KERNELS_DIR),
            "host": socket.gethostname(),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (dest / BUILD_INFO).write_text(json.dumps(record, indent=1) + "\n")
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    return dest / LIB_NAME


if __name__ == "__main__":
    if sys.argv[1:] != ["probe"]:
        raise SystemExit("usage: python cuda_lib.py probe")
    print(json.dumps(probe()))
