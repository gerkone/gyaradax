#!/usr/bin/env python3
"""C4: nonlinear_term_iii — FFT Poisson bracket (ExB nonlinearity).

OPTIM.md §4.4: AI ≈ 1.85 FLOP/byte, ~9.8B FLOPs/species, ~5.3 GB R+W/species.
Benchmarks both mixed_precision=True (default) and False (full FP64).

Architecture: solver.py delegates full implementation and shape dispatch (5D/6D)
to backend. Backend (JAX/CUDA) handles pseudospectral ExB bracket via FFT.
"""
import argparse, os, sys
from pathlib import Path

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_setup,
    BenchTimer,
    roofline_report,
    check_accuracy,
    analyze_cost,
    BASELINES_DIR,
)
from gyaradax.solver import nonlinear_term_iii, GKPre


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C4: nonlinear_term_iii  (FFT Poisson bracket)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    pre_gk = GKPre(pre)
    field = df  # 5D adiabatic
    baseline = BASELINES_DIR / "nonlinear.npz"

    from gyaradax.backends import create_ops

    results = {}
    for backend in ["jax", "cuda"]:
        print(f"\n{'='*40}")
        print(f"Backend: {backend.upper()}")
        print(f"{'='*40}")

        try:
            ops = create_ops(pre_gk, field, backend=backend)
        except Exception as e:
            print(f"  [SKIP] {backend} not available: {e}")
            continue

        backend_results = {}
        for label, mp, bkey in [
            ("mixed_precision=True  (default)", True, "output_mp"),
            ("mixed_precision=False (full FP64)", False, "output_fp64"),
        ]:
            print(f"\n  -- {label}")

            @jax.jit
            def fn(f, p, m=mp):
                return ops.nonlinear_term_iii(f, p, geom, mixed_precision=m)

            out = fn(field, phi)

            rel_l2 = check_accuracy(out, baseline, bkey)
            print(f"  [XLA] Analyzing cost...")
            flops, bytes_rw = analyze_cost(fn, field, phi)

            mean_ms, std_ms = BenchTimer(lambda f=field, p=phi: fn(f, p).block_until_ready()).run()

            print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
            r = roofline_report(
                f"nonlinear ({backend}, {('mp' if mp else 'fp64')})", mean_ms, flops, bytes_rw
            )
            r["rel_l2"] = rel_l2
            backend_results[mp] = r

        results[backend] = backend_results

    if "jax" in results and "cuda" in results:
        print(f"\nSpeedups (CUDA vs JAX):")
        for mp_val in [True, False]:
            label = "MP" if mp_val else "FP64"
            t_jax = results["jax"][mp_val]["mean_ms"]
            t_cuda = results["cuda"][mp_val]["mean_ms"]
            print(f"  {label:6s}: {t_jax / t_cuda:.2f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
