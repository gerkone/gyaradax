#!/usr/bin/env python3
import argparse, os, sys, ctypes
from pathlib import Path

# Provide standard bench_apply_vpar script arguments
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import ffi

sys.path.insert(0, str(Path(__file__).parent))
from common import load_setup, BenchTimer, roofline_report, check_accuracy, analyze_cost, BASELINES_DIR
import gyaradax.stencils as stencils

# ffi is already imported

lib_path = Path(__file__).parent.parent / "cuda_augmentations" / "liblto_bracket.so"
if not lib_path.exists():
    print(f"  [ERROR] {lib_path} not found. Please compile it first.")
    sys.exit(1)

_lib = ctypes.cdll.LoadLibrary(str(lib_path))
ffi.register_ffi_target("apply_vpar_stencil_ffi", ffi.pycapsule(_lib.apply_vpar_stencil_ffi), platform="CUDA")
ffi.register_ffi_target("apply_vpar_dual_stencil_ffi", ffi.pycapsule(_lib.apply_vpar_dual_stencil_ffi), platform="CUDA")

def _apply_vpar_cuda(field, coeffs):
    nv = field.shape[0]
    inner_size = 1
    for d in field.shape[1:]:
        inner_size *= d
    
    return ffi.ffi_call(
        "apply_vpar_stencil_ffi",
        jax.ShapeDtypeStruct(field.shape, field.dtype)
    )(field,
      c0=float(coeffs[0]), c1=float(coeffs[1]), 
      c2=float(coeffs[2]), c3=float(coeffs[3]), c4=float(coeffs[4]),
      nv=np.int32(nv), inner_size=np.int32(inner_size))

def _apply_vpar_dual_cuda(field, coeffs_d1, coeffs_d4):
    nv = field.shape[0]
    inner_size = 1
    for d in field.shape[1:]:
        inner_size *= d
        
    return ffi.ffi_call(
        "apply_vpar_dual_stencil_ffi",
        (jax.ShapeDtypeStruct(field.shape, field.dtype),
         jax.ShapeDtypeStruct(field.shape, field.dtype))
    )(field,
      c0_d1=float(coeffs_d1[0]), c1_d1=float(coeffs_d1[1]), c2_d1=float(coeffs_d1[2]), c3_d1=float(coeffs_d1[3]), c4_d1=float(coeffs_d1[4]),
      c0_d4=float(coeffs_d4[0]), c1_d4=float(coeffs_d4[1]), c2_d4=float(coeffs_d4[2]), c3_d4=float(coeffs_d4[3]), c4_d4=float(coeffs_d4[4]),
      nv=np.int32(nv), inner_size=np.int32(inner_size))

def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C2-CUDA: _apply_vpar  (5-point vpar stencil)")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)
    field = df

    baseline = BASELINES_DIR / "apply_vpar.npz"

    for label, coeffs, bkey in [
        ("VPAR_D1 (CUDA window)", stencils.VPAR_D1, "output_d1"),
        ("VPAR_D4 (CUDA window)", stencils.VPAR_D4, "output_d4"),
    ]:
        print(f"\n  -- {label}")
        coeffs_tuple = tuple(float(x) for x in coeffs)
        out = _apply_vpar_cuda(field, coeffs_tuple)
        rel_l2 = check_accuracy(out, baseline, bkey)
        
        @jax.jit
        def run_bench(f): return _apply_vpar_cuda(f, coeffs_tuple)

        # Cost: 1 read array, 1 write array per call.
        flo_op = field.size * 10
        byte_rw = field.size * 16 * 2

        mean_ms, std_ms = BenchTimer(lambda f=field: run_bench(f).block_until_ready()).run()
        print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
        roofline_report(f"_apply_vpar ({label[:6]})", mean_ms, flo_op, byte_rw)

    # Benchmark V1 (Dual fused stencil)
    print(f"\n  -- VPAR_D1+D4 Fused (CUDA Dual window)")
    c_d1 = tuple(float(x) for x in stencils.VPAR_D1)
    c_d4 = tuple(float(x) for x in stencils.VPAR_D4)
    out_d1, out_d4 = _apply_vpar_dual_cuda(field, c_d1, c_d4)
    
    # Check accuracy
    rel_l2_d1 = check_accuracy(out_d1, baseline, "output_d1")
    rel_l2_d4 = check_accuracy(out_d4, baseline, "output_d4")
    print(f"  accuracy [OK] rel_l2(D1)={rel_l2_d1:.3e}, rel_l2(D4)={rel_l2_d4:.3e}")

    @jax.jit
    def run_bench_dual(f): return _apply_vpar_dual_cuda(f, c_d1, c_d4)

    flo_op_dual = field.size * 20
    byte_rw_dual = field.size * 16 * 3 # 1 read, 2 writes

    mean_ms, std_ms = BenchTimer(lambda f=field: run_bench_dual(f)[0].block_until_ready()).run()
    print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms (For BOTH combined)")
    roofline_report(f"_apply_vpar_dual", mean_ms, flo_op_dual, byte_rw_dual)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
