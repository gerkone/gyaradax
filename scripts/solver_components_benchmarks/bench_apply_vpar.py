#!/usr/bin/env python3
"""C2: _apply_vpar — 5-point velocity-space stencil."""

import argparse
import os
import sys
from pathlib import Path

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_setup,
    BenchTimer,
    roofline_report,
    check_accuracy,
    analyze_cost,
    BASELINES_DIR,
)
from gyaradax.solver import GKPre
from gyaradax.backends import create_ops
import gyaradax.stencils as stencils

# Internal definition removed; using production _apply_vpar_fn instead.


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C2: _apply_vpar  (5-point vpar stencil)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df
    pre_gk = GKPre(pre)


    results = {}
    backends = []
    for b in ["jax", "cuda"]:
        try:
            ops = create_ops(pre_gk, backend=b, mixed_precision=mixed_precision)
            backends.append((b, ops))
        except Exception as e:
            print(f"  [SKIP] {b} backend not available: {e}")

    baseline = BASELINES_DIR / "apply_vpar.npz"

    # 1. Individual Stencils (D1 and D4)
    for label, coeffs, bkey in [
        ("VPAR_D1 (streaming)", stencils.VPAR_D1, "output_d1"),
        ("VPAR_D4 (dissipation)", stencils.VPAR_D4, "output_d4"),
    ]:
        print(f"\n  -- {label}")
        backend_times = {}

        for bname, ops in backends:
            from functools import partial

            # Internal function for jit/cost
            def _core(f, c, ops_in):
                return ops_in._apply_vpar(f, c)

            c_tuple = tuple(coeffs.tolist())

            # Accuracy and timing function
            run_fn = jax.jit(partial(_core, c=c_tuple, ops_in=ops))

            # Accuracy check
            out = run_fn(field)
            rel_l2 = check_accuracy(out, baseline, bkey)

            # Performance timing
            mean_ms, _ = BenchTimer(lambda: run_fn(field).block_until_ready()).run()
            backend_times[bname] = mean_ms

            # Reporting
            print(f"     [{bname.upper():4s}] {mean_ms:7.3f} ms  (rel_l2={rel_l2:.2e})")

            if bname == "cuda":
                # Only report roofline for CUDA/FFI
                flops, bytes_rw = analyze_cost(run_fn, field)
                roofline_report(f"_apply_vpar ({label[:4]}, {bname})", mean_ms, flops, bytes_rw)

        if "jax" in backend_times and "cuda" in backend_times:
            print(f"     Speedup: {backend_times['jax']/backend_times['cuda']:.2f}x")

    # 2. Dual Fused Stencil
    print("\n  -- VPAR_D1+D4 Dual Fusion")
    c1, c4 = tuple(stencils.VPAR_D1), tuple(stencils.VPAR_D4)
    dual_times = {}

    for bname, ops in backends:
        from functools import partial

        def _core_dual(f, ops_in):
            return ops_in._apply_vpar_dual(f, c1, c4)

        run_fn_dual = jax.jit(partial(_core_dual, ops_in=ops))

        out_dual = run_fn_dual(field)
        # Check accuracy for both outputs
        if isinstance(out_dual, tuple):
            rel_l2_d1 = check_accuracy(out_dual[0], baseline, "output_d1")
            rel_l2_d4 = check_accuracy(out_dual[1], baseline, "output_d4")
            l2_str = f"d1={rel_l2_d1:.1e}, d4={rel_l2_d4:.1e}"
        else:
            l2_str = "N/A"

        mean_ms, _ = BenchTimer(lambda: run_fn_dual(field)[0].block_until_ready()).run()
        dual_times[bname] = mean_ms
        print(f"     [{bname.upper():4s}] {mean_ms:7.3f} ms  (rel_l2: {l2_str})")

        if bname == "cuda":
            flops, bytes_rw = analyze_cost(run_fn_dual, field)
            roofline_report(f"_apply_vpar_dual ({bname})", mean_ms, flops, bytes_rw)

    if "jax" in dual_times and "cuda" in dual_times:
        print(f"     Speedup: {dual_times['jax']/dual_times['cuda']:.2f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
