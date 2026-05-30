#!/usr/bin/env python3
"""Run all solver component benchmarks and print a summary table.

Usage:
    PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
      python solver_components_benchmarks/run_all.py --device 1
"""

import argparse
import os
import sys
from pathlib import Path

from _runtime_config_loader import configure_runtime_env

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
configure_runtime_env(device=_early.device)

from gyaradax.jax_config import enable_x64

enable_x64()

sys.path.insert(0, str(Path(__file__).parent))

CONFIG = "configs/iteration_13.yaml"

COMPONENTS = [
    ("C1 _apply_parallel", "bench_apply_parallel"),
    ("C2 _apply_vpar", "bench_apply_vpar"),
    ("C3 linear_rhs", "bench_linear_rhs"),
    ("C4 nonlinear_term_iii", "bench_nonlinear"),
    ("C5 _compute_phi", "bench_phi_solve"),
    ("C6 pack/unpack_spectrum", "bench_pack_spectrum"),
    ("C7 gkstep_single", "bench_rk4_step"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default=CONFIG)
    parser.add_argument("--mp", action="store_true", help="enable mixed precision")
    args = parser.parse_args()

    import subprocess

    for label, module_name in COMPONENTS:
        print(f"\n>>> Running {label}...")
        script_path = Path(__file__).parent / f"{module_name}.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--device",
            str(args.device),
            "--config",
            args.config,
        ]
        if args.mp:
            cmd.append("--mp")

        env = os.environ.copy()
        env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

        subprocess.run(cmd, check=True, env=env)

    print(f"\n{'='*60}")
    print("All component benchmarks complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
