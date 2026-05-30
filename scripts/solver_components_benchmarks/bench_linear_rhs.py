#!/usr/bin/env python3
"""C3: linear_rhs — full linear operator (all terms, all species).

Architecture: solver.py delegates full implementation and shape dispatch (5D/6D)
to backend. Backend (JAX/CUDA) handles Terms I, II, IV, V, VII, VIII + dissipation.
"""

import argparse
import sys
from pathlib import Path

from _runtime_config_loader import configure_runtime_env

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
configure_runtime_env(device=_early.device)

import jax
from gyaradax.jax_config import enable_x64

enable_x64()

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_setup,
    BenchTimer,
    roofline_report,
    check_accuracy,
    analyze_cost,
    BASELINES_DIR,
)


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'=' * 60}")
    print("C3: linear_rhs  (full linear operator)")
    print(f"{'=' * 60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    pre_gk = pre
    baseline = BASELINES_DIR / "linear_rhs.npz"

    from gyaradax.backends import create_ops

    results = {}
    for backend in ["jax", "cuda"]:
        print(f"\n  -- Backend: {backend.upper()}")
        try:
            ops = create_ops(pre_gk, backend=backend, mixed_precision=mixed_precision)
        except Exception as e:
            print(f"     [SKIP] {backend} not available: {e}")
            continue

        @jax.jit
        def fn(d, p):
            return ops.linear_rhs(d, p, geom, params, pre_gk)

        out = fn(df, phi)
        rel_l2 = check_accuracy(out, baseline, "output")

        print("     [XLA] Analyzing cost...")
        flops, bytes_rw = analyze_cost(fn, df, phi)

        mean_ms, std_ms = BenchTimer(lambda d=df, p=phi: fn(d, p).block_until_ready()).run()
        print(f"     timing: {mean_ms:.3f} ± {std_ms:.3f} ms")

        r = roofline_report(f"linear_rhs ({backend})", mean_ms, flops, bytes_rw)
        r["rel_l2"] = rel_l2
        results[backend] = r

    if "jax" in results and "cuda" in results:
        speedup = results["jax"]["mean_ms"] / results["cuda"]["mean_ms"]
        print(f"\n  Final Speedup (CUDA vs JAX): {speedup:.2f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
