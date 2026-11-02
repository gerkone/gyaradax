#!/usr/bin/env python3
"""Lightweight CUDA experiment comparison harness.

This script intentionally does not register an untested JAX FFI custom call.
Use it as the starting point for a concrete experiment: first verify that the
shared library built by CMake exists, then add a small deterministic JAX
reference and a kernel-specific call path once the kernel ABI is known.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("_build/libgyaradax_cuda_experiments.so"),
        help="Path to the experiment shared library built by CMake.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=16,
        help="Size of the deterministic JAX reference vector.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.library.exists():
        raise SystemExit(
            f"Experiment library not found: {args.library}\n"
            "Build it first with: mkdir -p _build && cd _build && "
            "cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build ."
        )

    import jax
    import jax.numpy as jnp

    x = jnp.linspace(0.0, 1.0, args.size, dtype=jnp.float64)
    alpha = jnp.asarray(2.0, dtype=x.dtype)
    reference = alpha * x
    reference.block_until_ready()

    print(f"library: {args.library}")
    print(f"jax backend: {jax.default_backend()}")
    print(f"reference shape: {reference.shape}")
    print(f"reference checksum: {float(jnp.sum(reference)):.16e}")
    print(
        "No CUDA kernel was invoked. Extend this harness with the experiment's "
        "C ABI or JAX FFI call before using it for numerical validation."
    )


if __name__ == "__main__":
    main()
