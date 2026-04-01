#!/usr/bin/env python3
"""C8: gkstep_scan — N-step RK4 via jax.lax.scan.

Wraps gkstep_single in a jax.lax.scan loop so that XLA sees the full
N-step computation graph and can apply cross-step kernel fusion,
buffer reuse, and scheduling optimisations that a single-step
benchmark cannot capture.

Reports:
  - Total wall time for N steps (single fused HLO program).
  - Amortised per-step time  (total / N).
  - Single-step reference time (for comparison / fusion speedup).
  - Roofline metrics for the fused program.

Usage:
  python bench_rk4_scan.py --nsteps 50 --backend cuda
  python bench_rk4_scan.py --nsteps 100 --config configs/big.yaml --mp
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

import jax                                       # noqa: E402
import jax.lax                                   # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                          # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import (                             # noqa: E402
    load_setup,
    BenchTimer,
    roofline_report,
    check_accuracy,
    analyze_cost,
    BASELINES_DIR,
)
from gyaradax.solver import gkstep_single, default_state, GKPre  # noqa: E402
from gyaradax.backends import create_ops                          # noqa: E402

# ---------------------------------------------------------------------------
N_WARMUP = 3          # warmup calls before timing
DEFAULT_NSTEPS = 50   # default scan length
# ---------------------------------------------------------------------------


def _build_scan_fn(geom, params, pre_gk, ops):
    """Return a jitted function that runs N RK4 steps under lax.scan.

    The scan carries (df, state) and discards per-step diagnostics to
    keep memory bounded.  XLA compiles the entire N-step loop into one
    HLO program, enabling cross-step fusion.
    """

    def body(carry, _unused):
        """Single scan iteration — one RK4 step."""
        df, state = carry
        new_df, (new_phi, aux), new_state = gkstep_single(
            df, geom, params, state, pre_gk, ops=ops,
        )
        # Carry forward only what the next step needs.
        # Store a scalar diagnostic (e.g. phi norm) so we can sanity-check
        # without materialising the full phi array at every step.
        phi_norm = jnp.linalg.norm(new_phi.ravel())
        return (new_df, new_state), phi_norm

    @functools.partial(jax.jit, static_argnames=("n",))
    def scan_fn(df, state, n):
        (final_df, final_state), phi_norms = jax.lax.scan(
            body,
            init=(df, state),
            xs=None,          # no per-step input
            length=n,
        )
        return final_df, final_state, phi_norms

    return scan_fn


def _build_single_fn(geom, params, pre_gk, ops):
    """Single-step jitted reference (for fusion speedup comparison)."""

    @functools.partial(jax.jit, static_argnames=("ops",))
    def fn(df, state, ops):
        return gkstep_single(df, geom, params, state, pre_gk, ops=ops)

    return fn


def _bench_scan_phase(
    phase_name: str,
    *,
    nsteps: int,
    df,
    geom,
    params,
    state,
    pre_gk,
    baseline_path: Path,
    baseline_key_df: str,
    baseline_key_phi: str,
    backend_forced: str | None,
):
    """Benchmark one phase (linear / nonlinear) with scan and single-step."""

    print(f"\n{'─' * 60}")
    print(f"  {phase_name}   (N = {nsteps} steps)")
    print(f"{'─' * 60}")

    for bname in ["jax", "cuda"]:
        if backend_forced and bname != backend_forced:
            continue

        print(f"\n  ── Backend: {bname.upper()}")

        # --- backend ops --------------------------------------------------
        try:
            ops = create_ops(pre_gk, df, backend=bname)
        except Exception as e:
            print(f"     [SKIP] {bname} not available: {e}")
            continue

        # ==================================================================
        # A) Single-step reference
        # ==================================================================
        print(f"\n     [A] Single-step reference")

        single_fn = _build_single_fn(geom, params, pre_gk, ops)

        for _ in range(N_WARMUP):
            jax.block_until_ready(single_fn(df, state, ops))

        # Accuracy on single step
        out_df, (out_phi, _), _ = single_fn(df, state, ops)
        jax.block_until_ready((out_df, out_phi))
        check_accuracy(out_df,  baseline_path, baseline_key_df)
        check_accuracy(out_phi, baseline_path, baseline_key_phi)

        timer_single = BenchTimer(
            lambda: jax.block_until_ready(single_fn(df, state, ops))
        )
        single_mean_ms, single_std_ms = timer_single.run()
        print(f"         single-step : {single_mean_ms:.3f} ± {single_std_ms:.3f} ms")

        # ==================================================================
        # B) Scan-fused N-step benchmark
        # ==================================================================
        print(f"\n     [B] Scan-fused  ({nsteps} steps)")

        scan_fn = _build_scan_fn(geom, params, pre_gk, ops)

        # Warmup — the first call triggers XLA compilation of the full
        # scan program which can be substantially slower than single-step
        # compilation.
        print(f"         compiling scan HLO ({nsteps} steps)...")
        for _ in range(N_WARMUP):
            jax.block_until_ready(scan_fn(df, state, nsteps))

        # Accuracy: run one scan step and compare to single-step baseline.
        # A 1-step scan must produce the same result as the single-step fn.
        scan_1_df, _, _ = scan_fn(df, state, 1)
        jax.block_until_ready(scan_1_df)
        check_accuracy(scan_1_df, baseline_path, baseline_key_df)

        # Check that phi norms are finite after N steps (divergence guard).
        final_df, _, phi_norms = scan_fn(df, state, nsteps)
        jax.block_until_ready((final_df, phi_norms))

        n_finite = int(jnp.isfinite(phi_norms).sum())
        if n_finite < nsteps:
            print(f"         [WARN] {nsteps - n_finite}/{nsteps} steps "
                  f"produced non-finite phi — solution may be diverging")
        else:
            print(f"         phi norms: all {nsteps} steps finite ✓")

        # Timing
        timer_scan = BenchTimer(
            lambda: jax.block_until_ready(scan_fn(df, state, nsteps))
        )
        scan_mean_ms, scan_std_ms = timer_scan.run()
        amort_ms = scan_mean_ms / nsteps
        amort_std = scan_std_ms / nsteps

        print(f"         total time  : {scan_mean_ms:.3f} ± {scan_std_ms:.3f} ms")
        print(f"         per-step    : {amort_ms:.3f} ± {amort_std:.3f} ms")

        # ==================================================================
        # C) Fusion analysis
        # ==================================================================
        naive_total_ms = single_mean_ms * nsteps
        speedup = naive_total_ms / scan_mean_ms if scan_mean_ms > 0 else float("inf")
        overhead_pct = (1.0 - speedup) * 100  # negative = fusion helped

        print(f"\n     [C] Fusion analysis")
        print(f"         naive  N×single : {naive_total_ms:.3f} ms")
        print(f"         scan   fused    : {scan_mean_ms:.3f} ms")
        print(f"         fusion speedup  : {speedup:.2f}×")
        if speedup > 1.0:
            print(f"         XLA saved       : {naive_total_ms - scan_mean_ms:.3f} ms "
                  f"({(1 - 1/speedup)*100:.1f}% reduction)")
        else:
            print(f"         overhead        : {-overhead_pct:.1f}% "
                  f"(scan adds scheduling cost — expected for small N)")

        # ==================================================================
        # D) Roofline for the fused program
        # ==================================================================
        print(f"         [XLA] Analyzing fused cost...")
        try:
            flops, bytes_rw = analyze_cost(scan_fn, df, state, nsteps)
            # Report per-step roofline by dividing aggregate by N
            roofline_report(
                f"{phase_name} scan/{nsteps} ({bname})",
                amort_ms,
                flops / nsteps,
                bytes_rw / nsteps,
            )
        except Exception as e:
            print(f"         [SKIP] cost analysis failed: {e}")
            # Fallback: report raw timing only
            roofline_report(f"{phase_name} scan/{nsteps} ({bname})",
                            amort_ms, 0, 0)


# ---------------------------------------------------------------------------
# Sweep over multiple N values to show how fusion scales
# ---------------------------------------------------------------------------
def _sweep_nsteps(
    phase_name: str,
    *,
    nsteps_list: list[int],
    df,
    geom,
    params,
    state,
    pre_gk,
    backend_forced: str | None,
):
    """Run the scan benchmark at several N values and print a summary table."""

    print(f"\n{'═' * 60}")
    print(f"  SWEEP: {phase_name}")
    print(f"{'═' * 60}")

    for bname in ["jax", "cuda"]:
        if backend_forced and bname != backend_forced:
            continue

        try:
            ops = create_ops(pre_gk, df, backend=bname)
        except Exception:
            continue

        print(f"\n  Backend: {bname.upper()}")

        # Single-step baseline
        single_fn = _build_single_fn(geom, params, pre_gk, ops)
        for _ in range(N_WARMUP):
            jax.block_until_ready(single_fn(df, state, ops))
        timer = BenchTimer(
            lambda: jax.block_until_ready(single_fn(df, state, ops))
        )
        single_ms, _ = timer.run()

        # Table header
        print(f"\n  {'N':>6}  {'total_ms':>10}  {'per_step_ms':>12}  "
              f"{'naive_ms':>10}  {'speedup':>8}  {'finite':>6}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}")

        scan_fn = _build_scan_fn(geom, params, pre_gk, ops)

        for n in nsteps_list:
            # Warmup (recompiles if n changes because it's static)
            for _ in range(N_WARMUP):
                jax.block_until_ready(scan_fn(df, state, n))

            # Check finite
            _, _, phi_norms = scan_fn(df, state, n)
            jax.block_until_ready(phi_norms)
            n_fin = int(jnp.isfinite(phi_norms).sum())

            # Time
            timer = BenchTimer(
                lambda n=n: jax.block_until_ready(scan_fn(df, state, n))
            )
            total_ms, _ = timer.run()
            per_step = total_ms / n
            naive = single_ms * n
            spdup = naive / total_ms if total_ms > 0 else float("inf")

            print(f"  {n:>6}  {total_ms:>10.3f}  {per_step:>12.3f}  "
                  f"{naive:>10.3f}  {spdup:>7.2f}×  {n_fin:>4}/{n}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(
    config: str = "configs/iteration_13.yaml",
    mixed_precision: bool = False,
    backend_forced: str | None = None,
    nsteps: int = DEFAULT_NSTEPS,
    sweep: bool = False,
):
    print(f"\n{'=' * 60}")
    print(f"C8: gkstep_scan  (Fused N-Step RK4 Benchmark)")
    print(f"{'=' * 60}")
    print(f"    config  : {config}")
    print(f"    mixed-p : {mixed_precision}")
    print(f"    nsteps  : {nsteps}")
    print(f"    sweep   : {sweep}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    state = default_state(nky=df.shape[-1])
    pre_gk = GKPre(pre)

    baseline = BASELINES_DIR / "rk4_step.npz"
    if not baseline.exists():
        sys.exit(
            f"[ERROR] Baseline file not found: {baseline}\n"
            f"        Run the baseline generator first."
        )

    phases = [
        ("Linear RK4",    replace(params, non_linear=False),
         "out_df_linear",    "out_phi_linear"),
        ("Nonlinear RK4", replace(params, non_linear=True),
         "out_df_nonlinear", "out_phi_nonlinear"),
    ]

    for phase_name, p_var, key_df, key_phi in phases:

        # --- Detailed benchmark at the requested N -----------------------
        _bench_scan_phase(
            phase_name,
            nsteps=nsteps,
            df=df,
            geom=geom,
            params=p_var,
            state=state,
            pre_gk=pre_gk,
            baseline_path=baseline,
            baseline_key_df=key_df,
            baseline_key_phi=key_phi,
            backend_forced=backend_forced,
        )

        # --- Optional sweep across multiple N values ---------------------
        if sweep:
            _sweep_nsteps(
                phase_name,
                nsteps_list=[1, 5, 10, 25, 50, 100, 200],
                df=df,
                geom=geom,
                params=p_var,
                state=state,
                pre_gk=pre_gk,
                backend_forced=backend_forced,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark N-step fused RK4 via jax.lax.scan: CUDA vs JAX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=1,
                        help="CUDA device index")
    parser.add_argument("--config", type=str,
                        default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true",
                        help="Enable mixed-precision mode")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["jax", "cuda"],
                        help="Run only this backend (default: both)")
    parser.add_argument("--nsteps", type=int, default=DEFAULT_NSTEPS,
                        help="Number of RK4 steps in the scan loop")
    parser.add_argument("--sweep", action="store_true",
                        help="Also run a sweep over multiple N values")
    args = parser.parse_args()
    run(args.config, args.mp, args.backend, args.nsteps, args.sweep)