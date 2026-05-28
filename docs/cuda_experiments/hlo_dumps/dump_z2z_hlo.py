#!/usr/bin/env python3
"""Generate HLO dumps for Z2Z fp64 and fp32 variants."""

import os
import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))


def setup_test_data(pre, jax, jnp, dtype):
    """Create test data matching solver state.

    df:  [nv, nmu, ns, nkx, nky]  — vmapped over axis 2 (s)
    phi: [ns, nkx, nky]           — vmapped over axis 0 (s); phi_s is [nkx, nky]
    """
    nv, nmu, ns, nkx, nky = pre["bessel"].shape

    key = jax.random.PRNGKey(42)
    df = jax.random.normal(key, (nv, nmu, ns, nkx, nky), dtype=dtype)
    phi = jax.random.normal(key, (ns, nkx, nky), dtype=dtype)

    return df, phi


def run_z2z_nonlinear(ops, df, phi, geom):
    """Run Z2Z nonlinear term via JAXOps.nonlinear_term_iii."""
    return ops.nonlinear_term_iii(
        df,
        phi,
        geom,
        efun_sign=1.0,
        fft_prefactor=1.0 + 0.0j,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate HLO dumps for Z2Z.")
    parser.add_argument(
        "--full-dump",
        action="store_true",
        help="Enable full XLA dump to ./xla_hlo (generates many files)",
    )
    args = parser.parse_args()

    # Set environment variables BEFORE importing JAX and other modules that use it
    if args.full_dump:
        os.environ["XLA_FLAGS"] = "--xla_dump_to=./xla_hlo"
        os.makedirs("./xla_hlo", exist_ok=True)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Generating Z2Z HLO dumps...")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from gyaradax.backends._jax import JAXOps
    from gyaradax.state import GKPre
    from gyaradax.params import load_config, gkparams_from_config
    from gyaradax.solver import linear_precompute
    from gyaradax.geometry import compute_geometry

    cfg = load_config(str(repo_root / "configs" / "iteration_13.yaml"))
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

    params = gkparams_from_config(cfg)
    pre = linear_precompute(geom, params)
    pre_gk = GKPre(pre)

    df, phi = setup_test_data(pre, jax, jnp, jnp.complex128)

    ops_z2z_fp64 = JAXOps(pre_gk, use_z2z=True, mixed_precision=False)
    ops_z2z_fp32 = JAXOps(pre_gk, use_z2z=True, mixed_precision=True)

    # FP64 HLO
    lowered64 = jax.jit(run_z2z_nonlinear).lower(ops_z2z_fp64, df, phi, geom)
    hlo_text64 = lowered64.as_text()
    with open("z2z_fp64.hlo.txt", "w") as f:
        f.write(hlo_text64)
    print(f"  FP64: z2z_fp64.hlo.txt ({len(hlo_text64)} chars)")

    # FP32 HLO
    lowered32 = jax.jit(run_z2z_nonlinear).lower(ops_z2z_fp32, df, phi, geom)
    hlo_text32 = lowered32.as_text()
    with open("z2z_fp32.hlo.txt", "w") as f:
        f.write(hlo_text32)
    print(f"  FP32: z2z_fp32.hlo.txt ({len(hlo_text32)} chars)")

    if args.full_dump:
        print("  XLA: ./xla_hlo/")
    print("Done.")


if __name__ == "__main__":
    main()
