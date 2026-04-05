#!/usr/bin/env python3
"""
Benchmark script for the paper's performance comparison section.

Measures gyaradax throughput, memory, and (estimated) FLOP utilization,
and parses GKW perform.dat / perfloop.dat / slurm files for comparison.

Usage:
    python scripts/paper_benchmark.py --config configs/iteration_13a.yaml --device 0
    python scripts/paper_benchmark.py --config configs/iteration_13a.yaml --device 0 --kinetic-config configs/v3_kiteration_991_half_rlt.yaml
"""

import os
import re
import sys
import time
import json
import argparse
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gyaradax import load_geometry, GKParams, gk_init, gksolve
from gyaradax.solver import (
    linear_precompute,
    _compute_phi,
)
from gyaradax.backends import create_ops
from gyaradax.params import gkparams_from_config, load_config


# ---------------------------------------------------------------------------
# GKW reference parsing
# ---------------------------------------------------------------------------


def parse_gkw_perform(path):
    """Parse GKW perform.dat into a dict of {label: (n_calls, total_sec, pct)}."""
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: label (variable width)  n_calls  total_sec  pct
            match = re.match(r"^(.+?)\s{2,}(\d+)\s+([\d.E+\-]+)\s+([\d.]+)$", line)
            if match:
                label = match.group(1).strip()
                n_calls = int(match.group(2))
                total_sec = float(match.group(3))
                pct = float(match.group(4))
                results[label] = {"n_calls": n_calls, "total_sec": total_sec, "pct": pct}
    return results


def parse_gkw_perfloop(path):
    """Parse GKW perfloop.dat — one wall-time per large step."""
    times = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                times.append(float(parts[1]))
    return np.array(times)


def parse_slurm_info(path):
    """Extract basic info from a slurm output file."""
    info = {}
    with open(path) as f:
        text = f.read()
    m = re.search(r"numtasks=(\d+)", text)
    if m:
        info["mpi_tasks"] = int(m.group(1))
    m = re.search(r"numnodes=(\d+)", text)
    if m:
        info["nodes"] = int(m.group(1))
    m = re.search(r"Running on master node:\s+(\S+)", text)
    if m:
        info["node"] = m.group(1)
    return info


def collect_gkw_timing(data_dir):
    """Collect all available GKW timing info from a data directory."""
    result = {}

    perform_path = os.path.join(data_dir, "perform.dat")
    if os.path.exists(perform_path):
        result["perform"] = parse_gkw_perform(perform_path)

    perfloop_path = os.path.join(data_dir, "perfloop.dat")
    if os.path.exists(perfloop_path):
        loop_times = parse_gkw_perfloop(perfloop_path)
        if len(loop_times) > 0:
            result["perfloop"] = {
                "n_steps": len(loop_times),
                "mean_s": float(np.mean(loop_times)),
                "std_s": float(np.std(loop_times)),
                "total_s": float(np.sum(loop_times)),
            }

    slurm_files = [f for f in os.listdir(data_dir) if f.startswith("slurm-")]
    if slurm_files:
        result["slurm"] = parse_slurm_info(os.path.join(data_dir, slurm_files[0]))

    return result


# ---------------------------------------------------------------------------
# gyaradax benchmarking
# ---------------------------------------------------------------------------


def bench_fn(fn, n_warmup=2, n_iters=10, label=""):
    """Benchmark a JAX function. Returns mean_ms, std_ms."""
    for _ in range(n_warmup):
        result = fn()
        jax.block_until_ready(result)

    times = []
    for _ in range(n_iters):
        t0 = time.time()
        result = fn()
        jax.block_until_ready(result)
        times.append((time.time() - t0) * 1000)

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    if label:
        print(f"  {label:35s}: {mean_ms:8.2f} +/- {std_ms:5.2f} ms")
    return mean_ms, std_ms


def get_device_info():
    """Get JAX device info."""
    dev = jax.devices()[0]
    info = {
        "platform": dev.platform,
        "device_kind": dev.device_kind,
    }
    if hasattr(dev, "memory_stats"):
        try:
            stats = dev.memory_stats()
            if stats:
                info["peak_bytes_in_use"] = stats.get("peak_bytes_in_use", None)
                info["bytes_limit"] = stats.get("bytes_limit", None)
        except Exception:
            pass
    return info


def get_memory_usage():
    """Get current JAX memory usage."""
    dev = jax.devices()[0]
    try:
        stats = dev.memory_stats()
        if stats:
            return {
                "peak_mb": stats.get("peak_bytes_in_use", 0) / 1e6,
                "current_mb": stats.get("bytes_in_use", 0) / 1e6,
                "limit_mb": stats.get("bytes_limit", 0) / 1e6,
            }
    except Exception:
        pass
    return {}


def estimate_flops_per_step(grid_shape, non_linear=True, n_species=1):
    """
    Rough FLOP estimate per RK4 step.

    Counts dominant operations: FFTs (nonlinear), stencil applications (linear),
    phi solve, and Bessel/Maxwellian evaluations.
    """
    if len(grid_shape) == 6:
        nsp, nvp, nmu, ns, nkx, nky = grid_shape
    else:
        nvp, nmu, ns, nkx, nky = grid_shape
        nsp = n_species

    N = nsp * nvp * nmu * ns * nkx * nky  # total grid points

    # per RHS evaluation (called 4x per RK4 step):
    # - linear terms: ~30 flops/point (drifts, streaming, mirror, drives, dissipation)
    # - phi solve: O(nvp * nmu * ns * nkx * nky) reductions + divisions
    # - nonlinear: 4 FFTs per s-slice * O(5 * nkx*nky*log(nkx*nky)) each
    linear_flops = 30 * N
    phi_flops = 10 * ns * nkx * nky  # reduction + division
    if non_linear:
        fft_size = nkx * nky
        nl_flops = ns * 4 * 5 * fft_size * max(1, np.log2(fft_size))  # per species
        nl_flops *= nsp
    else:
        nl_flops = 0

    rhs_flops = linear_flops + phi_flops + nl_flops
    step_flops = 4 * rhs_flops  # RK4 = 4 RHS evals
    return int(step_flops)


def benchmark_gyaradax(config_path, n_steps=120, n_blocks=5, device=None):
    """Run full gyaradax benchmark. Returns a results dict."""
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    cfg = load_config(config_path)
    params = gkparams_from_config(cfg)

    # force nonlinear for fair comparison
    params = GKParams(
        **{
            **{k: getattr(params, k) for k in params.__dataclass_fields__},
            "non_linear": True,
        }
    )

    # geometry
    if hasattr(cfg.run, "data_dir") and os.path.exists(cfg.run.data_dir):
        geometry = load_geometry(cfg.run.data_dir)
    else:
        from gyaradax.simulate import _geometry_from_config

        geometry = _geometry_from_config(cfg)

    df, geometry, state = gk_init(geometry, params)
    grid_shape = df.shape
    mode = "kinetic" if not params.adiabatic_electrons else "adiabatic"

    print(f"config: {config_path}")
    print(f"mode: {mode}, grid: {grid_shape}, dt: {params.dt}")
    print(f"device: {jax.devices()[0]}")

    results = {
        "config": config_path,
        "mode": mode,
        "grid_shape": list(grid_shape),
        "dt": params.dt,
        "device": get_device_info(),
    }

    # precompute
    t0 = time.time()
    pre = linear_precompute(geometry, params)
    jax.block_until_ready(jax.tree.leaves(pre))
    t_pre = time.time() - t0
    print(f"precompute: {t_pre:.3f}s")
    results["precompute_s"] = t_pre

    # memory after precompute
    mem = get_memory_usage()
    if mem:
        print(f"memory after precompute: {mem['current_mb']:.0f} MB / {mem['limit_mb']:.0f} MB")
        results["memory_after_precompute_mb"] = mem["current_mb"]

    # component benchmarks
    print("\ncomponent benchmarks:")
    ops = create_ops(pre, backend=params.backend, use_z2z=params.use_z2z)

    phi = _compute_phi(df, geometry, params, pre)

    phi_ms, phi_std = bench_fn(
        lambda: _compute_phi(df, geometry, params, pre),
        label="phi solve",
    )
    results["phi_ms"] = phi_ms

    lin_ms, lin_std = bench_fn(
        lambda: ops.linear_rhs(df, phi, geometry, params, pre),
        label="linear rhs",
    )
    results["linear_rhs_ms"] = lin_ms

    if params.non_linear:
        nl_ms, nl_std = bench_fn(
            lambda: ops.nonlinear_term_iii(df, phi, geometry, mixed_precision=params.mixed_precision),
            label="nonlinear rhs (term iii)",
        )
        results["nonlinear_rhs_ms"] = nl_ms

    # full solver: warmup
    print(f"\nfull solver: {n_steps} steps/block, {n_blocks} blocks")
    print("warmup (compilation)...")
    t0 = time.time()
    df_w, _, state_w = gksolve(df, geometry, params, state, n_steps=n_steps, pre=pre)
    jax.block_until_ready(df_w)
    t_warmup = time.time() - t0
    print(f"compilation: {t_warmup:.2f}s")
    results["compilation_s"] = t_warmup

    # memory after compilation
    mem = get_memory_usage()
    if mem:
        print(
            f"memory after compilation: {mem['peak_mb']:.0f} MB peak / {mem['limit_mb']:.0f} MB limit"
        )
        results["peak_memory_mb"] = mem["peak_mb"]
        results["memory_limit_mb"] = mem["limit_mb"]

    # timed blocks
    df_cur, state_cur = df_w, state_w
    block_times = []
    for i in range(n_blocks):
        t0 = time.time()
        df_cur, (phi_out, fluxes), state_cur = gksolve(
            df_cur, geometry, params, state_cur, n_steps=n_steps, pre=pre
        )
        jax.block_until_ready(df_cur)
        dt_block = time.time() - t0
        block_times.append(dt_block)
        sps = n_steps / dt_block
        ms_per_step = dt_block * 1000 / n_steps
        print(
            f"  block {i+1}/{n_blocks}: {dt_block:.3f}s ({sps:.1f} steps/s, {ms_per_step:.2f} ms/step)"
        )

    times = np.array(block_times)
    mean_sps = n_steps / np.mean(times)
    std_sps = n_steps * np.std(times) / np.mean(times) ** 2
    ms_per_step = np.mean(times) * 1000 / n_steps

    print(f"\n{'='*60}")
    print(f"  total: {n_steps * n_blocks} steps in {np.sum(times):.3f}s")
    print(f"  throughput: {mean_sps:.2f} +/- {std_sps:.2f} steps/s")
    print(f"  latency: {ms_per_step:.2f} ms/step")

    results["n_steps_total"] = n_steps * n_blocks
    results["total_time_s"] = float(np.sum(times))
    results["steps_per_sec"] = float(mean_sps)
    results["steps_per_sec_std"] = float(std_sps)
    results["ms_per_step"] = float(ms_per_step)
    results["block_times_s"] = block_times

    # FLOP estimate
    est_flops = estimate_flops_per_step(grid_shape, non_linear=params.non_linear)
    achieved_flops = est_flops * mean_sps
    results["est_flops_per_step"] = est_flops
    results["est_gflops"] = achieved_flops / 1e9
    print(f"  est. FLOP/step: {est_flops:.2e}")
    print(f"  est. throughput: {achieved_flops/1e9:.2f} GFLOP/s")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# comparison summary
# ---------------------------------------------------------------------------


def print_gkw_summary(data_dir, label="GKW"):
    """Print GKW timing summary from a data directory."""
    gkw = collect_gkw_timing(data_dir)
    if not gkw:
        print(f"  {label}: no timing data found in {data_dir}")
        return gkw

    print(f"\n{label} ({data_dir}):")
    if "slurm" in gkw:
        s = gkw["slurm"]
        print(f"  hardware: {s.get('mpi_tasks', '?')} MPI tasks, {s.get('nodes', '?')} node(s)")

    if "perform" in gkw:
        perf = gkw["perform"]
        if "rk4" in perf:
            rk4 = perf["rk4"]
            n_steps = rk4["n_calls"]
            total_s = rk4["total_sec"]
            sps = n_steps / total_s
            ms_step = total_s * 1000 / n_steps
            print(f"  rk4: {n_steps} steps in {total_s:.1f}s")
            print(f"  throughput: {sps:.2f} steps/s ({ms_step:.2f} ms/step)")

        # component breakdown
        print("  breakdown:")
        for key in [
            "Non linear terms: FFT, No MPI",
            "Linear terms matmul, No MPI",
            "Calc Fields",
            "Copy fdis vector to tmp",
        ]:
            if key in perf:
                p = perf[key]
                print(f"    {key:40s}: {p['total_sec']:.1f}s ({p['pct']:.1f}%)")

    if "perfloop" in gkw:
        pl = gkw["perfloop"]
        # perfloop measures wall-time per "large step" (naverage RK4 steps)
        print(f"  per-block wall time: {pl['mean_s']*1000:.1f} +/- {pl['std_s']*1000:.1f} ms")

    return gkw


def main():
    parser = argparse.ArgumentParser(description="Paper benchmark: gyaradax vs GKW")
    parser.add_argument("--config", type=str, required=True, help="gyaradax config yaml")
    parser.add_argument("--kinetic-config", type=str, default=None, help="optional kinetic config")
    parser.add_argument(
        "--gkw-dir",
        type=str,
        action="append",
        default=[],
        help="additional GKW data dirs to parse timing from (repeatable)",
    )
    parser.add_argument("--steps", type=int, default=120, help="steps per block")
    parser.add_argument("--blocks", type=int, default=5, help="number of timed blocks")
    parser.add_argument("--device", type=int, default=None, help="GPU device index")
    parser.add_argument("--output", type=str, default=None, help="save results JSON")
    parser.add_argument(
        "--gkw-only", action="store_true", help="only print GKW timing, skip gyaradax"
    )
    args = parser.parse_args()

    all_results = {}

    # --- GKW reference timing ---
    cfg = load_config(args.config)
    if hasattr(cfg.run, "data_dir") and os.path.exists(cfg.run.data_dir):
        gkw_adiabatic = print_gkw_summary(cfg.run.data_dir, label="GKW (adiabatic)")
        all_results["gkw_adiabatic"] = gkw_adiabatic

    if args.kinetic_config:
        kcfg = load_config(args.kinetic_config)
        if hasattr(kcfg.run, "data_dir") and os.path.exists(kcfg.run.data_dir):
            gkw_kinetic = print_gkw_summary(kcfg.run.data_dir, label="GKW (kinetic)")
            all_results["gkw_kinetic"] = gkw_kinetic

    for i, gkw_dir in enumerate(args.gkw_dir):
        if os.path.exists(gkw_dir):
            label = f"GKW ({os.path.basename(gkw_dir.rstrip('/'))})"
            gkw_extra = print_gkw_summary(gkw_dir, label=label)
            all_results[f"gkw_extra_{i}"] = gkw_extra

    if args.gkw_only:
        return

    # --- gyaradax benchmark ---
    print(f"\n{'='*60}")
    print("gyaradax (adiabatic)")
    print(f"{'='*60}")
    results_adiabatic = benchmark_gyaradax(
        args.config, n_steps=args.steps, n_blocks=args.blocks, device=args.device
    )
    all_results["gyaradax_adiabatic"] = results_adiabatic

    if args.kinetic_config:
        print(f"\n{'='*60}")
        print("gyaradax (kinetic)")
        print(f"{'='*60}")
        results_kinetic = benchmark_gyaradax(
            args.kinetic_config, n_steps=args.steps, n_blocks=args.blocks, device=args.device
        )
        all_results["gyaradax_kinetic"] = results_kinetic

    # --- comparison table ---
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'':30s} {'gyaradax':>15s} {'GKW':>15s} {'speedup':>10s}")
    print("-" * 72)

    for mode_key, gkw_key in [
        ("gyaradax_adiabatic", "gkw_adiabatic"),
        ("gyaradax_kinetic", "gkw_kinetic"),
    ]:
        if mode_key not in all_results:
            continue
        gr = all_results[mode_key]
        label = gr["mode"]

        gkw_sps = None
        if gkw_key in all_results and "perform" in all_results[gkw_key]:
            perf = all_results[gkw_key]["perform"]
            if "rk4" in perf:
                rk4 = perf["rk4"]
                gkw_sps = rk4["n_calls"] / rk4["total_sec"]

        gkw_str = f"{gkw_sps:.2f}" if gkw_sps else "N/A"
        speedup = f"{gr['steps_per_sec'] / gkw_sps:.1f}x" if gkw_sps else "N/A"
        print(f"  {label:28s} {gr['steps_per_sec']:>12.2f} s/s {gkw_str:>12s} s/s {speedup:>10s}")

    # save
    if args.output:
        # convert numpy types for JSON
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=_convert)
        print(f"\nresults saved to {args.output}")


if __name__ == "__main__":
    main()
