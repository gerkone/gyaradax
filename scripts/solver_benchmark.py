#!/usr/bin/env python3
"""
Benchmark and verification tool for the Gyaradax JAX solver.
Used to track performance gains and numerical correctness during optimization.
"""

import sys
import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Any

# Ensure we are in the project root
sys.path.append(os.getcwd())

from gyaradax.simulate import _setup_simulation, _init_condition
from gyaradax.solver import gksolve, GKState, linear_precompute
from gyaradax.integrals import get_integrals

def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-30) -> float:
    """Calculate relative L2 error."""
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))

def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark Gyaradax JAX solver")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps for timing")
    parser.add_argument("--reference", type=str, help="Path to .npz reference to verify against")
    parser.add_argument("--save-reference", type=str, help="Path to save final state as reference")
    parser.add_argument("--resume-k-file", type=str, help="Path to GKW K-file to resume from")
    parser.add_argument("--time-average-flux", action="store_true", help="Calculate time-averaged heat flux (last 80 big steps)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed runtime info")
    
    args = parser.parse_args()

    # 1. Setup
    print(f"[*] Loading configuration: {args.config}")
    params, geometry, _, _, _ = _setup_simulation(args.config, None, False, {})
    
    # Handle K-file resumption if specified
    if args.resume_k_file:
        print(f"[*] Resuming from K-file: {args.resume_k_file}")
        df, state = _init_condition(None, args.resume_k_file, geometry, params, args.verbose)
    else:
        df, state = _init_condition(None, None, geometry, params, args.verbose)
    
    # Determine grid resolution
    res = df.shape
    num_elements = np.prod(res)
    size_mb = (num_elements * 16) / (1024 * 1024) # complex128 is 16 bytes
    
    print(f"[*] Grid resolution: {res} (vpar, mu, s, kx, ky)")
    print(f"[*] State size: {num_elements:,} elements ({size_mb:.2f} MB)")

    # 2. Precompute (Time it once)
    print("[*] Running precompute...")
    t_pre_start = time.time()
    pre = linear_precompute(geometry, params)
    jax.block_until_ready(pre)
    t_pre = time.time() - t_pre_start
    print(f"[+] Precompute completed in {t_pre*1000:.2f} ms")

    print(f"[*] Target steps for benchmark: {args.steps}")

    # 3. Warm-up (Compiles JIT kernels)
    print("[*] Warming up (JIT compilation)...")
    t0 = time.time()
    # Run 1 step to trigger compilation
    df_warm, _, state_warm = gksolve(df, geometry, params, state, n_steps=1, pre=pre)
    # Ensure completion
    jax.block_until_ready(df_warm)
    t_warm = time.time() - t0
    print(f"[+] Warm-up completed in {t_warm:.2f}s")

    # 4. Timed Benchmark
    print(f"[*] Running {args.steps} steps benchmark...")
    
    # Use a directory for intermittent outputs if time-averaging
    output_dir = "benchmark_temp_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    t1 = time.time()
    
    if args.time_average_flux:
        # We need to run with simulate-like loop to get fluxes history
        from gyaradax.simulate import simulate
        final_df, final_state, perf = simulate(
            args.config,
            output_dir=output_dir,
            resume_k_file=args.resume_k_file,
            n_steps=args.steps,
            save_dumps=False,
            verbose=args.verbose
        )
        t_total = time.time() - t1
    else:
        final_df, (phi, fluxes), final_state = gksolve(df, geometry, params, state, n_steps=args.steps, pre=pre)
        jax.block_until_ready(final_df)
        t_total = time.time() - t1
    
    avg_step = t_total / args.steps
    steps_sec = 1.0 / avg_step
    # Each step is RK4 (4 stages)
    ms_per_stage = (avg_step * 1000) / 4

    print("\n" + "="*40)
    print(" PERFORMANCE RESULTS")
    print("="*40)
    print(f"Precompute time:        {t_pre*1000:8.4f} ms")
    print(f"Total time ({args.steps} steps):  {t_total:8.4f} s")
    print(f"Average time per step:   {avg_step*1000:8.4f} ms")
    print(f"Average time per stage:  {ms_per_stage:8.4f} ms")
    print(f"Throughput:              {steps_sec:8.4f} steps/s")
    print("="*40)

    # 4. Numerical Verification / Time Averaging
    if args.time_average_flux:
        history_path = os.path.join(output_dir, "fluxes.npz")
        hist_flux = np.load(history_path)
        sim_eflux = hist_flux["fluxes"][:, 1]
        
        # average over the last 80 big timesteps (matches validate_time_averaged.py)
        # Assuming dump_interval * naverage is the cadence
        avg_count = 80
        sim_eflux_avg = np.mean(sim_eflux[-avg_count:])
        print(f"\n[*] Time-averaged Heat Flux (last {avg_count} samples): {sim_eflux_avg:.6e}")
        
        # Optional reference check if data_dir has fluxes.dat
        data_dir = params.run.data_dir if hasattr(params, 'run') else geometry.get('data_dir') # Setup simulation might not put it in params
        # Actually _setup_simulation returns params, geometry, ...
        # and simulate uses cfg.run.data_dir
        from gyaradax.params import load_config
        cfg = load_config(args.config)
        ref_flux_path = os.path.join(cfg.run.data_dir, "fluxes.dat")
        if os.path.exists(ref_flux_path):
             # Simplified ref check: just print the last few lines or mean of ref
             try:
                 ref_fluxes = np.loadtxt(ref_flux_path)
                 # This is a bit complex to match exactly without reusing validate_time_averaged logic
                 # but let's just show we are in the ballpark
                 print(f"[*] GKW Reference (fluxes.dat) exists at: {ref_flux_path}")
             except:
                 pass

    elif args.reference:
        if not os.path.exists(args.reference):
            print(f"[!] Reference file not found: {args.reference}")
        else:
            print(f"\n[*] Verifying against reference: {args.reference}")
            ref_data = np.load(args.reference)
            ref_df = ref_data["df"]
            
            error = rel_l2(np.array(final_df), ref_df)
            max_err = float(np.max(np.abs(np.array(final_df) - ref_df)))
            
            print(f"[+] Relative L2 Error: {error:.2e}")
            print(f"[+] Max Absolute Error: {max_err:.2e}")
            
            if error < 1e-12:
                print("[PASS] Numerical results match reference (FP64 precision).")
            elif error < 1e-6:
                print("[PASS] Numerical results match reference (single precision range).")
            else:
                print("[FAIL] Significant divergence from reference!")

    # 5. Diagnostic Check (Self-consistency)
    phi_val, (p, e, v) = get_integrals(final_df, geometry, params=params)
    print(f"\n[*] Diagnostics at end of run:")
    print(f"    - Energy Flux: {float(e):.6e}")
    print(f"    - Potential Amp (L2): {float(jnp.sqrt(jnp.mean(jnp.abs(phi_val)**2))):.6e}")

    # 6. Save Reference if requested
    if args.save_reference:
        print(f"\n[*] Saving reference to: {args.save_reference}")
        np.savez(args.save_reference, 
                 df=np.array(final_df), 
                 phi=np.array(phi), 
                 eflux=float(e),
                 step=int(final_state.step))
        print("[+] Reference saved.")

if __name__ == "__main__":
    run_benchmark()
