#!/usr/bin/env python3
"""Generate baseline npz files for each solver component.

Run once from the repo root:
    PYTHONPATH=. JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
      python solver_components_benchmarks/generate_baselines.py --device 1

Writes solver_components_benchmarks/baselines/<component>.npz.
Each file contains the exact inputs and expected output for that component
so that bench_*.py files can verify numerical correctness against them.
"""
import argparse
import os
import sys
from pathlib import Path

# parse --device before JAX import
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_p.add_argument("--config", type=str, default="configs/iteration_13.yaml")
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import load_setup, BASELINES_DIR

BASELINES_DIR.mkdir(exist_ok=True)


def save(name: str, **arrays):
    path = BASELINES_DIR / f"{name}.npz"
    np.savez(path, **{k: np.array(v) for k, v in arrays.items()})
    print(f"  saved {path.name}  ({', '.join(arrays)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    args = parser.parse_args()

    print(f"Device: {jax.devices()[0]}")
    print(f"Config: {args.config}\n")

    print("Loading setup...")
    df, phi, geom, params, pre = load_setup(args.config)

    # for adiabatic: df is (nv, nmu, ns, nkx, nky)
    # extract a single-species 5D field for C1/C2/C4
    field5d = df  # already 5D for adiabatic

    # ── C1: _apply_parallel ────────────────────────────────────────────────
    print("\nC1: _apply_parallel")
    from gyaradax.solver import _compute_linear_rhs

    # replicate closure logic from solver.py:758-768
    s_shift   = pre["s_shift"]
    kx_shift  = pre["kx_shift"]
    valid_shift = pre["valid_shift"]

    from gyaradax.backends import create_ops
    ops = create_ops(pre, field5d, backend="jax")

    @jax.jit
    def _apply_parallel(field, coeffs):
        return ops._apply_parallel(field, coeffs)


    out_c1 = _apply_parallel(field5d, pre["s_total_upar"])
    save("apply_parallel",
         field=field5d, coeffs=pre["s_total_upar"],
         output=out_c1)

    # ── C2: _apply_vpar ───────────────────────────────────────────────────
    print("\nC2: _apply_vpar")
    from gyaradax import stencils

    @jax.jit
    def _apply_vpar(field, coeffs):
        return ops._apply_vpar(field, coeffs)


    out_c2_d1 = _apply_vpar(field5d, stencils.VPAR_D1)
    out_c2_d4 = _apply_vpar(field5d, stencils.VPAR_D4)
    save("apply_vpar",
         field=field5d,
         coeffs_d1=stencils.VPAR_D1, output_d1=out_c2_d1,
         coeffs_d4=stencils.VPAR_D4, output_d4=out_c2_d4)

    # ── C3: _linear_rhs_core (via _compute_linear_rhs) ───────────────────
    print("\nC3: _compute_linear_rhs")

    @jax.jit
    def _lin_rhs():
        return _compute_linear_rhs(df, phi, geom, params, pre, ops)


    out_c3 = _lin_rhs()
    save("linear_rhs",
         df=df, phi=phi,
         output=out_c3)

    # ── C4: nonlinear_term_iii ────────────────────────────────────────────
    print("\nC4: nonlinear_term_iii")
    from gyaradax.solver import nonlinear_term_iii

    mp = params.mixed_precision

    @jax.jit
    def _nl_mp():
        return ops.nonlinear_term_iii(field5d, phi, geom, mixed_precision=True)

    @jax.jit
    def _nl_fp64():
        return ops.nonlinear_term_iii(field5d, phi, geom, mixed_precision=False)


    out_c4_mp   = _nl_mp()
    out_c4_fp64 = _nl_fp64()
    save("nonlinear",
         field=field5d, phi=phi,
         output_mp=out_c4_mp,
         output_fp64=out_c4_fp64)

    # ── C5: _compute_phi ─────────────────────────────────────────────────
    print("\nC5: _compute_phi")
    from gyaradax.solver import _compute_phi

    @jax.jit
    def _phi():
        return _compute_phi(df, geom, params, pre)

    out_c5 = _phi()
    save("phi_solve",
         df=df,
         output=out_c5)

    # ── C6: pack/unpack_half_spectrum ────────────────────────────────────
    print("\nC6: pack/unpack_half_spectrum")
    from gyaradax.solver import pack_half_spectrum, unpack_half_spectrum

    jind   = pre["nl_jind"]
    mrad   = int(pre["nl_mrad"])
    mphi   = int(pre["nl_mphi"])
    mphiw3 = int(pre["nl_mphiw3"])
    nkx, nky = field5d.shape[-2], field5d.shape[-1]

    # use a (nv, nmu, nkx, nky) spectral slice as input (one s-slice of field)
    spec_in = field5d[:, :, 0, :, :]  # (nv, nmu, nkx, nky)

    @jax.jit
    def _pack():
        return pack_half_spectrum(spec_in, jind, mrad, mphiw3)

    @jax.jit
    def _unpack():
        packed = pack_half_spectrum(spec_in, jind, mrad, mphiw3)
        return unpack_half_spectrum(packed, jind, nky)

    out_packed   = _pack()
    out_unpacked = _unpack()
    save("pack_spectrum",
         spec_in=spec_in, jind=jind,
         output_packed=out_packed,
         output_unpacked=out_unpacked)

    # ── C7: gkstep_single (full RK4 step) ───────────────────────────────────
    print("\nC7: gkstep_single")
    from gyaradax.solver import gkstep_single, default_state, GKPre
    from dataclasses import replace

    state = default_state(nky=df.shape[-1])
    pre_gk = GKPre(pre)

    @jax.jit
    def _rk4_linear(d, s):
        return gkstep_single(d, geom, replace(params, non_linear=False), s, pre_gk, ops=ops)

    @jax.jit
    def _rk4_nonlinear(d, s):
        return gkstep_single(d, geom, replace(params, non_linear=True), s, pre_gk, ops=ops)


    out_df_lin, (out_phi_lin, _), _ = _rk4_linear(df, state)
    out_df_nl,  (out_phi_nl,  _), _ = _rk4_nonlinear(df, state)
    save("rk4_step",
         df=df,
         out_df_linear=out_df_lin,    out_phi_linear=out_phi_lin,
         out_df_nonlinear=out_df_nl,  out_phi_nonlinear=out_phi_nl)

    print("\nAll baselines written to", BASELINES_DIR)


if __name__ == "__main__":
    main()
