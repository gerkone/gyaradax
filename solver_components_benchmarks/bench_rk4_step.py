#!/usr/bin/env python3
"""C7: gkstep_single — full RK4 time step (linear and nonlinear).

Benchmarks CUDA vs JAX ops backends with proper warmup, full
synchronisation, statistical reporting, and roofline analysis.

Architecture: solver.py delegates linear_rhs and nonlinear_term_iii to backend.
Backend handles all shape dispatch (5D adiabatic / 6D kinetic electrons).
"""

import argparse
import os
import sys
import functools
from pathlib import Path
from dataclasses import replace

# ---------------------------------------------------------------------------
# Early device selection (before any JAX import)
# ---------------------------------------------------------------------------
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    load_setup,
    BenchTimer,
    roofline_report,
    check_accuracy,
    analyze_cost,
    BASELINES_DIR,
)
from gyaradax.solver import gkstep_single, default_state, GKPre  # noqa: E402
from gyaradax.backends import create_ops  # noqa: E402

# ---------------------------------------------------------------------------
# Number of warmup iterations before timing
# ---------------------------------------------------------------------------
N_WARMUP = 3


# ---------------------------------------------------------------------------
# Core benchmark helper — eliminates duplication between phases
# ---------------------------------------------------------------------------
def _bench_phase(
    phase_name: str,
    *,
    df,
    geom,
    params,
    state,
    pre_gk,
    baseline_path: Path,
    baseline_key_df: str,
    baseline_key_phi: str,
    backend_forced: str | None,
    test_z2z: bool = False,
    mixed_precision: bool = True,
):
    """Run one benchmark phase (linear or nonlinear) across backends.
    
    Args:
        test_z2z: If True, test both R2C and Z2Z FFT modes (nonlinear only).
                  If False, use R2C only (linear or default).
        mixed_precision: Use mixed precision (FP32 FFTs) for nonlinear bracket.
    """

    print(f"\n[PHASE] {phase_name}")
    
    z2z_values = [False, True] if test_z2z else [False]
    
    for z2z in z2z_values:
        z2z_label = "Z2Z" if z2z else "R2C"
        if len(z2z_values) > 1:
            print(f"\n  {'#'*50}")
            print(f"  #  FFT Mode: {z2z_label} (use_z2z={z2z})")
            print(f"  {'#'*50}")

        for bname in ["jax", "cuda"]:
            if backend_forced and bname != backend_forced:
                continue

            print(f"\n  -- Backend: {bname.upper()} ({z2z_label})")

            # --- create backend ops -------------------------------------------
            try:
                ops = create_ops(pre_gk, backend=bname, use_z2z=z2z, mixed_precision=mixed_precision)
            except Exception as e:
                print(f"     [SKIP] {bname} not available: {e}")
                continue

            # --- build jitted function ----------------------------------------
            @jax.jit
            def fn(d, s):
                return gkstep_single(d, geom, params, s, pre_gk, ops=ops)

            # --- warmup -------------------------------------------------------
            for _ in range(N_WARMUP):
                jax.block_until_ready(fn(df, state))

            # --- accuracy check -----------------------------------------------
            out_df, (out_phi, _), _ = fn(df, state)
            jax.block_until_ready((out_df, out_phi))
            check_accuracy(out_df, baseline_path, baseline_key_df)
            check_accuracy(out_phi, baseline_path, baseline_key_phi)

            # --- timing -------------------------------------------------------
            timer = BenchTimer(lambda: jax.block_until_ready(fn(df, state)))
            mean_ms, std_ms = timer.run()
            print(f"     timing: {mean_ms:.3f} ± {std_ms:.3f} ms")

            # --- cost / roofline ----------------------------------------------
            print("     [XLA] Analyzing cost...")
            flops, bytes_rw = analyze_cost(fn, df, state)
            roofline_report(f"{phase_name} ({bname}, {z2z_label})", mean_ms, flops, bytes_rw)
    
    # Print comparison if both modes were tested
    if len(z2z_values) > 1 and not backend_forced:
        print(f"\n  {'='*50}")
        print(f"  FFT Mode Comparison (Z2Z vs R2C)")
        print(f"  {'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(
    config: str = "configs/iteration_13.yaml",
    mixed_precision: bool = False,
    backend_forced: str | None = None,
    nonlinear_z2z: bool = False,
):
    """Benchmark RK4 step.
    
    Args:
        nonlinear_z2z: If True, test nonlinear phase with both R2C/Z2Z modes.
                      If False, use R2C only for nonlinear (linear never uses Z2Z).
    """
    print(f"\n{'=' * 60}")
    print(f"C7: gkstep_single  (Full RK4 Step)")
    print(f"{'=' * 60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    state = default_state(nky=df.shape[-1])
    pre_gk = GKPre(pre)

    baseline = BASELINES_DIR / "rk4_step.npz"
    if not baseline.exists():
        sys.exit(
            f"[ERROR] Baseline file not found: {baseline}\n"
            f"        Run the baseline generator first."
        )

    shared = dict(
        df=df,
        geom=geom,
        state=state,
        pre_gk=pre_gk,
        baseline_path=baseline,
        backend_forced=backend_forced,
        mixed_precision=mixed_precision,
    )

    # Phase 1 — Linear RK4 (no Z2Z, linear doesn't use it)
    _bench_phase(
        "Linear RK4 Step",
        params=replace(params, non_linear=False),
        baseline_key_df="out_df_linear",
        baseline_key_phi="out_phi_linear",
        test_z2z=False,
        **shared,
    )

    # Phase 2 — Nonlinear RK4 (optionally test Z2Z vs R2C)
    _bench_phase(
        "Nonlinear RK4 Step",
        params=replace(params, non_linear=True),
        baseline_key_df="out_df_nonlinear",
        baseline_key_phi="out_phi_nonlinear",
        test_z2z=nonlinear_z2z,
        **shared,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gkstep_single: CUDA vs JAX backends")
    parser.add_argument(
        "--device", type=int, default=1, help="CUDA device index (already applied at import)"
    )
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true", help="Enable mixed-precision mode")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["jax", "cuda"],
        help="Run only this backend (default: both)",
    )
    parser.add_argument(
        "--z2z",
        action="store_true",
        help="Test nonlinear phase with both R2C/Z2Z modes (default: R2C only)",
    )
    args = parser.parse_args()
    
    run(args.config, args.mp, args.backend, args.z2z)
