#!/usr/bin/env python3
"""C4: nonlinear_term_iii — FFT Poisson bracket (ExB nonlinearity).

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


def run(config="configs/iteration_13.yaml", mixed_precision=False, test_z2z=False):
    """Benchmark nonlinear_term_iii.
    
    Args:
        test_z2z: If True, test both R2C and Z2Z FFT modes.
                 If False, use R2C only (default).
    """
    print(f"\n{'='*60}")
    print("C4: nonlinear_term_iii  (FFT Poisson bracket)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    pre_gk = GKPre(pre)
    field = df
    baseline = BASELINES_DIR / "nonlinear.npz"

    from gyaradax.backends import create_ops

    results = {}
    z2z_values = [False, True] if test_z2z else [False]
    
    for z2z in z2z_values:
        z2z_label = "Z2Z" if z2z else "R2C"
        if len(z2z_values) > 1:
            print(f"\n{'#'*60}")
            print(f"#  FFT Mode: {z2z_label} (use_z2z={z2z})")
            print(f"{'#'*60}")
        
        for backend in ["jax", "cuda"]:
            print(f"\n{'='*40}")
            print(f"Backend: {backend.upper()} ({z2z_label})")
            print(f"{'='*40}")

            backend_results = {}
            for label, mp, bkey in [
                ("mixed_precision=True  (default)", True, "output_mp"),
                ("mixed_precision=False (full FP64)", False, "output_fp64"),
            ]:
                print(f"\n  -- {label}")

                try:
                    ops = create_ops(pre_gk, backend=backend, use_z2z=z2z, mixed_precision=mp)
                except Exception as e:
                    print(f"  [SKIP] {backend} ({label}) not available: {e}")
                    continue

                @jax.jit
                def fn(f, p):
                    return ops.nonlinear_term_iii(f, p, geom)

                out = fn(field, phi)

                rel_l2 = check_accuracy(out, baseline, bkey)
                print(f"  [XLA] Analyzing cost...")
                flops, bytes_rw = analyze_cost(fn, field, phi)

                mean_ms, std_ms = BenchTimer(lambda f=field, p=phi: fn(f, p).block_until_ready()).run()

                print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
                r = roofline_report(
                    f"nonlinear ({backend}, {z2z_label}, {('mp' if mp else 'fp64')})", mean_ms, flops, bytes_rw
                )
                r["rel_l2"] = rel_l2
                backend_results[(mp, z2z)] = r

            results[(backend, z2z)] = backend_results

    if len(z2z_values) > 1:
        print(f"\n{'#'*60}")
        print(f"#  FFT Mode Comparison (Z2Z vs R2C)")
        print(f"{'#'*60}")
        for backend in ["jax", "cuda"]:
            if (backend, True) in results and (backend, False) in results:
                print(f"\n  {backend.upper()}:")
                for mp_val in [True, False]:
                    mp_label = "MP" if mp_val else "FP64"
                    t_r2c = results[(backend, False)][(mp_val, False)]["mean_ms"]
                    t_z2z = results[(backend, True)][(mp_val, True)]["mean_ms"]
                    speedup = t_r2c / t_z2z if t_z2z > 0 else float("inf")
                    print(f"    {mp_label:6s}: Z2Z vs R2C = {speedup:.2f}x (R2C: {t_r2c:.3f} ms, Z2Z: {t_z2z:.3f} ms)")

    if any(b == "jax" for (b, _) in results) and any(b == "cuda" for (b, _) in results):
        print(f"\nSpeedups (CUDA vs JAX):")
        for z2z in z2z_values:
            z2z_label = "Z2Z" if z2z else "R2C"
            for mp_val in [True, False]:
                label = f"{'Z2Z' if z2z else 'R2C'}-{('MP' if mp_val else 'FP64')}"
                t_jax = results[("jax", z2z)][(mp_val, z2z)]["mean_ms"]
                t_cuda = results[("cuda", z2z)][(mp_val, z2z)]["mean_ms"]
                print(f"  {label:10s}: {t_jax / t_cuda:.2f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true", help="Enable mixed-precision mode")
    parser.add_argument(
        "--z2z",
        action="store_true",
        help="Test both R2C and Z2Z FFT modes (default: R2C only)",
    )
    args = parser.parse_args()
    
    run(args.config, args.mp, args.z2z)
