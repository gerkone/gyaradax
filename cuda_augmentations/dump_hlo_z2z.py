#!/usr/bin/env python3
"""Generate HLO dumps for JAX Z2Z fp32 and fp64 to analyze performance gap."""
import argparse
import os
import sys
from pathlib import Path

root = Path(__file__).parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--config", type=str, default=str(root / "configs" / "iteration_13.yaml"))
parser.add_argument("--output-dir", type=str, default=str(root / "hlo_dumps"))
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "solver_components_benchmarks"))
from common import load_setup
from gyaradax.backends._jax import JAXOps
from gyaradax.backends import create_ops
from gyaradax.types import GKPre

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print(f"HLO Dump Generator: JAX Z2Z fp32 vs fp64")
print(f"{'='*80}")
print(f"  Output dir: {output_dir}")
print(f"  Config: {args.config}")

df, phi, geom, params, pre = load_setup(args.config)
pre_gk = GKPre(pre)

jax_z2z = JAXOps(pre_gk, use_z2z=True)

mrad, mphi = pre["nl_mrad"], pre["nl_mphi"]
nkx, nky = df.shape[-2], df.shape[-1]

print(f"  df shape: {df.shape}")
print(f"  phi shape: {phi.shape}")
print(f"  Grid: mrad={mrad}, mphi={mphi}, nkx={nkx}, nky={nky}")

jax_z2z_fp64 = JAXOps(pre, use_z2z=True, mixed_precision=False)
jax_z2z_fp32 = JAXOps(pre, use_z2z=True, mixed_precision=True)

def run_z2z_fp64(d, p):
    return jax_z2z_fp64.nonlinear_term_iii(
        d, p, geom, efun_sign=1.0, fft_prefactor=1.0 + 0.0j
    )

def run_z2z_fp32(d, p):
    return jax_z2z_fp32.nonlinear_term_iii(
        d, p, geom, efun_sign=1.0, fft_prefactor=1.0 + 0.0j
    )

fp64_dump_dir = output_dir / "xla_hlo_fp64"
fp64_dump_dir.mkdir(parents=True, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_to={fp64_dump_dir}"

print("\nCompiling fp64 version...")
compiled_fp64 = jax.jit(run_z2z_fp64).lower(df, phi).compile()
hlo_text_fp64 = compiled_fp64.as_text()

fp64_text_path = output_dir / "z2z_fp64.hlo.txt"
with open(fp64_text_path, "w") as f:
    f.write(hlo_text_fp64)
print(f"  Written: {fp64_text_path}")
print(f"  XLA HLO dump dir: {fp64_dump_dir}")

fp32_dump_dir = output_dir / "xla_hlo_fp32"
fp32_dump_dir.mkdir(parents=True, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_to={fp32_dump_dir}"
jax.clear_caches()

print("\nCompiling fp32 version...")
compiled_fp32 = jax.jit(run_z2z_fp32).lower(df, phi).compile()
hlo_text_fp32 = compiled_fp32.as_text()

fp32_text_path = output_dir / "z2z_fp32.hlo.txt"
with open(fp32_text_path, "w") as f:
    f.write(hlo_text_fp32)
print(f"  Written: {fp32_text_path}")
print(f"  XLA HLO dump dir: {fp32_dump_dir}")

print("\n" + "="*80)
print("HLO Dump Summary")
print("="*80)

import os
for fname in ["z2z_fp64.hlo.txt", "z2z_fp32.hlo.txt"]:
    fpath = output_dir / fname
    size = os.path.getsize(fpath)
    print(f"  {fname:20s}: {size:>10,} bytes")

print("\nAnalyzing key differences...")

def count_ops(hlo_text, dtype_name):
    lines = hlo_text.split("\n")
    op_counts = {}
    for line in lines:
        if "fft" in line.lower() or "convert" in line.lower() or "multiply" in line.lower():
            for op in ["fft", "convert", "multiply", "add", "subtract", "negate"]:
                if op in line.lower():
                    op_counts[op] = op_counts.get(op, 0) + 1
    return op_counts

fp64_ops = count_ops(hlo_text_fp64, "fp64")
fp32_ops = count_ops(hlo_text_fp32, "fp32")

print(f"\n{'Operation':15s} | {'fp64 count':12s} | {'fp32 count':12s}")
print("-" * 45)
all_ops = set(fp64_ops.keys()) | set(fp32_ops.keys())
for op in sorted(all_ops):
    print(f"{op:15s} | {fp64_ops.get(op, 0):12d} | {fp32_ops.get(op, 0):12d}")

print("\nHLO dumps generated successfully!")
print(f"Use tools like `hlo-opt` or compare the text files to analyze differences.")
