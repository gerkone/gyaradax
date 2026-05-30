#!/usr/bin/env python3
"""C6: pack_half_spectrum / unpack_half_spectrum — FFT index permutation."""

import argparse
import sys
from pathlib import Path

from _runtime_config_loader import configure_runtime_env

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
configure_runtime_env(device=_early.device)

import jax
from gyaradax.jax_config import enable_x64

enable_x64()

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_setup,
    BenchTimer,
    roofline_report,
    check_accuracy,
    analyze_cost,
    BASELINES_DIR,
)
from gyaradax.solver import pack_half_spectrum, unpack_half_spectrum


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'=' * 60}")
    print("C6: pack_half_spectrum / unpack_half_spectrum")
    print(f"{'=' * 60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)

    jind = pre["nl_jind"]
    mrad = int(pre["nl_mrad"])
    int(pre["nl_mphi"])
    mphiw3 = int(pre["nl_mphiw3"])
    nky = df.shape[-1]

    # (nv, nmu, nkx, nky) — one s-slice
    spec_in = df[:, :, 0, :, :]

    @jax.jit
    def fn_pack(s):
        return pack_half_spectrum(s, jind, mrad, mphiw3)

    @jax.jit
    def fn_unpack(p):
        return unpack_half_spectrum(p, jind, nky)

    baseline = BASELINES_DIR / "pack_spectrum.npz"

    print("\n  -- pack_half_spectrum")
    out_pack = fn_pack(spec_in)
    check_accuracy(out_pack, baseline, "output_packed")

    print("  [XLA] Analyzing cost...")
    flops_p, bytes_p = analyze_cost(fn_pack, spec_in)

    mean_ms, std_ms = BenchTimer(lambda s=spec_in: fn_pack(s).block_until_ready()).run()
    print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
    roofline_report("pack_half_spectrum", mean_ms, flops_p, bytes_p)

    print("\n  -- unpack_half_spectrum")
    out_unpack = fn_unpack(out_pack)
    check_accuracy(out_unpack, baseline, "output_unpacked")

    print("  [XLA] Analyzing cost...")
    flops_u, bytes_u = analyze_cost(fn_unpack, out_pack)

    mean_ms2, std_ms2 = BenchTimer(lambda p=out_pack: fn_unpack(p).block_until_ready()).run()
    print(f"  timing: {mean_ms2:.3f} ± {std_ms2:.3f} ms")
    roofline_report("unpack_half_spectrum", mean_ms2, flops_u, bytes_u)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
