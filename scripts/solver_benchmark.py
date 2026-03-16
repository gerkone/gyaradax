#!/usr/bin/env python3
"""
Benchmark and verification tool for the Gyaradax JAX solver.
Used to track performance gains and numerical correctness during optimization.
"""
# Bootstrap JAX environment BEFORE any JAX imports
from gyaradax.bootstrap import init_jax

init_jax()

import time
import argparse
import numpy as np

from gyaradax.simulate import _setup_simulation, _init_condition
from gyaradax.solver import gksolve, linear_precompute


def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-30) -> float:
    """Calculate relative L2 error."""
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark Gyaradax JAX solver")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of steps for timing"
    )
    parser.add_argument(
        "--reference", type=str, help="Path to .npz reference to verify against"
    )
    parser.add_argument(
        "--save-reference", type=str, help="Path to save final state as reference"
    )
    parser.add_argument(
        "--resume-k-file", type=str, help="Path to GKW K-file to resume from"
    )
    parser.add_argument(
        "--time-average-flux",
        action="store_true",
        help="Calculate time-averaged heat flux (last 80 big steps)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed runtime info"
    )
    parser.add_argument("--device", type=int, default=None, help="GPU device ID to use")

    args = parser.parse_args()

    # 2. Setup (Logic now entirely on the top level via bootstrap and standard imports)
    print(f"[*] Loading configuration: {args.config}")
    params, geometry, _, _, _ = _setup_simulation(args.config, None, False, {})

    # Handle K-file resumption if specified
    if args.resume_k_file:
        print(f"[*] Resuming from K-file: {args.resume_k_file}")
        df, state = _init_condition(
            None, args.resume_k_file, geometry, params, args.verbose
        )
    else:
        df, state = _init_condition(None, None, geometry, params, args.verbose)

    # Determine grid resolution
    res = df.shape
    num_elements = np.prod(res)
    size_mb = (num_elements * 16) / (1024 * 1024)  # complex128 is 16 bytes

    print(f"[*] Grid resolution: {res} (vpar, mu, s, kx, ky)")
    print(f"[*] State size: {num_elements:,} elements ({size_mb:.2f} MB)")

    # 3. Precompute
    print("[*] Running precompute...")
    t0 = time.time()
    pre = linear_precompute(geometry, params)
    # block on one result to ensure compilation completes
    for k, v in pre.items():
        if hasattr(v, "block_until_ready"):
            v.block_until_ready()
            break
    t_pre = time.time() - t0
    print(f"[*] Precompute complete in {t_pre:.3f}s")

    # 4. Solvers Benchmarks
    print(f"[*] Running solver for {args.steps} steps...")

    # If reference is provided, we might want to be careful with total steps.
    # We do a 1-step warmup using the INITIAL state, but discard the result
    # for the benchmark run to ensure we start from step 0 (or resume step).

    print("[*] Performing warmup/compilation (1 step)...")
    _ = gksolve(df, geometry, params, state, n_steps=1, pre=pre)

    # Actual Benchmark loop (using JAX internal scan)
    # We run EXACTLY args.steps from the INITIAL state.
    t_run_start = time.time()
    df_next, (phi, fluxes), state_next = gksolve(
        df, geometry, params, state, n_steps=args.steps, pre=pre
    )

    if hasattr(df_next, "block_until_ready"):
        df_next.block_until_ready()
    t_total = time.time() - t_run_start

    m_steps_s = args.steps / t_total
    print(f"[*] Solver loop complete in {t_total:.3f}s ({m_steps_s:.2f} steps/s)")

    # 5. Verification
    if args.reference:
        print(f"[*] Verifying against reference: {args.reference}")
        ref = np.load(args.reference)
        ref_df = ref["df"]

        err = rel_l2(np.array(df_next), ref_df)
        print(f"[*] Final state relative L2 difference: {err:.4e}")

        if err < 1e-10:
            print("[SUCCESS] Result significantly matches baseline.")
        else:
            print("[WARNING] Result deviates from baseline!")

    if args.save_reference:
        print(f"[*] Saving final state to {args.save_reference}")
        np.savez(
            args.save_reference,
            df=np.array(df_next),
            time=np.array(state_next.time),
            step=np.array(state_next.step),
        )


if __name__ == "__main__":
    run_benchmark()
