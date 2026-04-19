"""benchmark multi-GPU scaling of gyaradax on an adiabatic NL run.

Runs the same config across a matrix of device meshes, measures wall
time per block after a JIT warmup block, and emits a markdown table
with the speedup numbers + a log-log scaling plot.

Usage:
  python scripts/benchmark_sharding.py \
      --config configs/iteration_13.yaml \
      --meshes 1 2x1 4x1 2x2 4x2 \
      --device-ids 0,2,3,4,5,6,7 \
      --n-steps 2000 \
      --output docs/multi_gpu_scaling.png
"""

import argparse
import os
import subprocess
import sys
import time


def _run_one(mesh_str, device_ids, config, n_steps):
    """Fork a subprocess because each run needs its own CUDA_VISIBLE_DEVICES."""
    if mesh_str == "1":
        vp, mu = 1, 1
        dev_sub = device_ids[:1]
    else:
        vp, mu = (int(x) for x in mesh_str.split("x"))
        dev_sub = device_ids[: vp * mu]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in dev_sub)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
    env["PYTHONUNBUFFERED"] = "1"

    code = f"""
import time, jax, numpy as np
from gyaradax import sharding, load_config
from gyaradax.params import gkparams_from_config
from gyaradax.geometry import compute_geometry
from gyaradax.solver import linear_precompute
from gyaradax.simulate import gk_init, gk_run

cfg = load_config('{config}')
params = gkparams_from_config(cfg, non_linear=True, adaptive_dt=False,
    dt=0.005, n_gpus_vp={vp}, n_gpus_mu={mu})
geom = compute_geometry(q=params.q, shat=params.shat, eps=params.eps,
    ns=cfg.grid.ns, nkx=cfg.grid.nkx, nky=cfg.grid.nky,
    nvpar=cfg.grid.nvpar, nmu=cfg.grid.nmu, vpar_max=cfg.grid.vpar_max,
    nperiod=cfg.grid.nperiod, krhomax=cfg.grid.krhomax,
    ikxspace=cfg.grid.ikxspace, adiabatic_electrons=True,
    geom_type='circ', signB=params.signB)
pre = linear_precompute(geom, params)
df, geom, state = gk_init(geom, params, n_species=1)
mesh = sharding.build_mesh(params)
if mesh is not None:
    grid = sharding.grid_shape_from(params, geom)
    df = sharding.shard_df(df, mesh, grid)
    pre = sharding.shard_pre(pre, mesh, grid)

# warmup (pays JIT once)
df, phi, flx, state = gk_run(df, geom, params, state, n_steps=100, pre=pre)
jax.block_until_ready(df)

t0 = time.time()
df, phi, flx, state = gk_run(df, geom, params, state, n_steps={n_steps}, pre=pre)
jax.block_until_ready(df)
dt = time.time() - t0
print(f'ELAPSED={{dt:.3f}}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        print(f"[mesh {mesh_str}] FAILED:", result.stderr.strip().splitlines()[-5:])
        return None
    for line in result.stdout.splitlines():
        if line.startswith("ELAPSED="):
            return float(line.split("=", 1)[1])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/iteration_13.yaml")
    p.add_argument("--meshes", nargs="+", default=["1", "2x1", "4x1", "2x2", "4x2"])
    p.add_argument("--device-ids", default="0,2,3,4,5,6,7",
                   help="comma-separated physical device ids to use")
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--output", default="docs/multi_gpu_scaling.png")
    args = p.parse_args()

    device_ids = [int(x) for x in args.device_ids.split(",")]
    results = {}
    for mesh_str in args.meshes:
        print(f"[mesh {mesh_str}] running…", flush=True)
        t0 = time.time()
        elapsed = _run_one(mesh_str, device_ids, args.config, args.n_steps)
        print(f"[mesh {mesh_str}] done in {time.time()-t0:.1f}s wall; elapsed={elapsed}")
        results[mesh_str] = elapsed

    baseline = results.get("1")
    print()
    print(f"| mesh (vp×mu) | GPUs | t / {args.n_steps} steps (s) | speedup |")
    print("|--------------|------|------|---------|")
    for mesh_str in args.meshes:
        t = results[mesh_str]
        if mesh_str == "1":
            vp, mu = 1, 1
        else:
            vp, mu = (int(x) for x in mesh_str.split("x"))
        n = vp * mu
        if t is None:
            print(f"| {vp}×{mu} | {n} | FAIL | — |")
        elif baseline is None:
            print(f"| {vp}×{mu} | {n} | {t:.2f} | — |")
        else:
            print(f"| {vp}×{mu} | {n} | {t:.2f} | {baseline/t:.2f}× |")

    if baseline is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs, ys = [], []
        for mesh_str in args.meshes:
            t = results[mesh_str]
            if t is None:
                continue
            if mesh_str == "1":
                n = 1
            else:
                vp, mu = (int(x) for x in mesh_str.split("x"))
                n = vp * mu
            xs.append(n)
            ys.append(baseline / t)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(xs, ys, "o-", label="measured")
        ax.plot([xs[0], xs[-1]], [xs[0], xs[-1]], "k--", alpha=0.4, label="ideal")
        ax.set_xlabel("# GPUs")
        ax.set_ylabel("speedup vs 1 GPU")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        ax.set_title(f"gyaradax adiabatic NL ({args.config})")
        fig.tight_layout()
        fig.savefig(args.output, dpi=120)
        print(f"\nplot: {args.output}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
