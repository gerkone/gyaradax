#!/usr/bin/env python3
"""Generate HLO dumps for linear RK4 integration."""

import os
import sys
import argparse
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))


def setup_test_data(pre, jax, jnp, dtype):
    """Create test data matching solver state.

    df:  [nv, nmu, ns, nkx, nky]  — vmapped over axis 2 (s)
    phi: [ns, nkx, nky]           — vmapped over axis 0 (s); phi_s is [nkx, nky]
    """
    nv, nmu, ns, nkx, nky = pre["bessel"].shape

    key = jax.random.PRNGKey(42)
    df  = jax.random.normal(key, (nv, nmu, ns, nkx, nky), dtype=dtype)
    phi = jax.random.normal(key, (ns, nkx, nky),           dtype=dtype)

    return df, phi


def run_linear_rhs(ops, df, phi, geom, params, pre):
    """Run linear RHS operator via JAXOps.linear_rhs."""
    return ops.linear_rhs(df, phi, geom, params, pre)


def run_linear_rk4_step(ops, df, geom, params, pre, dt):
    """Run a single linear RK4 integration step.

    Computes df_new = df + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    where k_i = RHS(df_i) and RHS computes phi then linear terms.
    """
    from gyaradax.solver import _compute_phi

    def _rhs(df_i):
        phi_i = _compute_phi(df_i, geom, params, pre)
        return ops.linear_rhs(df_i, phi_i, geom, params, pre)

    k1 = _rhs(df)
    k2 = _rhs(df + 0.5 * dt * k1)
    k3 = _rhs(df + 0.5 * dt * k2)
    k4 = _rhs(df + dt * k3)

    dt6 = dt / 6.0
    dt3 = dt / 3.0
    return df + dt6 * k1 + dt3 * k2 + dt3 * k3 + dt6 * k4


def main():
    parser = argparse.ArgumentParser(description="Generate HLO dumps for linear RK4.")
    parser.add_argument(
        "--full-dump", 
        action="store_true", 
        help="Enable full XLA dump to ./xla_hlo_linear (generates many files)"
    )
    args = parser.parse_args()

    if args.full_dump:
        os.environ["XLA_FLAGS"] = "--xla_dump_to=./xla_hlo_linear"
        os.makedirs("./xla_hlo_linear", exist_ok=True)
    
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Generating linear RK4 HLO dumps...")

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)

    from gyaradax.backends._jax import JAXOps
    from gyaradax.types import GKPre
    from gyaradax.params import load_config, gkparams_from_config
    from gyaradax.solver import linear_precompute
    from gyaradax.geometry import compute_geometry

    cfg = load_config(str(root / "configs" / "iteration_13.yaml"))
    geom_cfg = cfg["geometry"]
    grid_cfg = cfg["grid"]
    
    geom = compute_geometry(
        q=float(geom_cfg["q"]),
        shat=float(geom_cfg["shat"]),
        eps=float(geom_cfg["eps"]),
        ns=int(grid_cfg["ns"]),
        nkx=int(grid_cfg["nkx"]),
        nky=int(grid_cfg["nky"]),
        nvpar=int(grid_cfg["nvpar"]),
        nmu=int(grid_cfg["nmu"]),
        vpar_max=float(grid_cfg.get("vpar_max", 3.0)),
        nperiod=int(grid_cfg.get("nperiod", 1)),
        kxmax=float(geom_cfg.get("kxmax", 0.0)),
        krhomax=float(grid_cfg.get("krhomax", 1.4)),
        ikxspace=int(grid_cfg.get("ikxspace", 5)),
        signB=float(geom_cfg.get("signB", 1.0)),
        Rref=float(geom_cfg.get("Rref", 100.0)),
        geom_type=str(geom_cfg.get("geometry_model", "circ")),
    )
    
    from dataclasses import replace
    
    params = gkparams_from_config(cfg)
    params = replace(params, non_linear=False)
    
    pre = linear_precompute(geom, params)
    pre_gk = GKPre(pre)
    
    df, phi = setup_test_data(pre, jax, jnp, jnp.complex128)
    
    ops = JAXOps(pre_gk, use_z2z=False)
    
    dt = 0.01
    
    lowered_rhs64 = jax.jit(run_linear_rhs).lower(
        ops, df, phi, geom, params, pre
    )
    hlo_text_rhs64 = lowered_rhs64.as_text()
    with open("linear_rhs_only.hlo.txt", "w") as f:
        f.write(hlo_text_rhs64)
    print(f"  RHS:  linear_rhs_only.hlo.txt ({len(hlo_text_rhs64)} chars)")
    
    lowered_rk464 = jax.jit(run_linear_rk4_step).lower(
        ops, df, geom, params, pre, dt
    )
    hlo_text_rk464 = lowered_rk464.as_text()
    with open("linear_rk4_fp64.hlo.txt", "w") as f:
        f.write(hlo_text_rk464)
    print(f"  FP64: linear_rk4_fp64.hlo.txt ({len(hlo_text_rk464)} chars)")
    
    df_f32 = df.astype(jnp.complex64)
    phi_f32 = phi.astype(jnp.complex64)
    
    lowered_rk432 = jax.jit(run_linear_rk4_step).lower(
        ops, df_f32, geom, params, pre, dt
    )
    hlo_text_rk432 = lowered_rk432.as_text()
    with open("linear_rk4_fp32.hlo.txt", "w") as f:
        f.write(hlo_text_rk432)
    print(f"  FP32: linear_rk4_fp32.hlo.txt ({len(hlo_text_rk432)} chars)")
    
    if args.full_dump:
        print(f"  XLA:  ./xla_hlo_linear/")
    print("Done.")


if __name__ == "__main__":
    main()
