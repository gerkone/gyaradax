#!/usr/bin/env python3
"""C6: pack_half_spectrum / unpack_half_spectrum — FFT index permutation.

OPTIM.md §4.6: Pure memory movement, 0 FLOPs.
Per pack: input 11.1 MB, output 27.1 MB (+ 27.1 MB zero-init).
Per unpack: input 27.1 MB, output 11.1 MB.
"""
import argparse, os, sys
from pathlib import Path

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--device", type=int, default=1)
_early, _ = _p.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_early.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from common import load_setup, BenchTimer, roofline_report, check_accuracy, BASELINES_DIR

# (nv=32, nmu=8, nkx=85, nky=32) complex128 = 11.1 MB  in
# (nv=32, nmu=8, mrad=135, mphiw3=49) complex128 = 27.1 MB out
BYTES_PACK   = (11.1 + 27.1 + 27.1) * 1e6  # read + write + zero-init
BYTES_UNPACK = (27.1 + 11.1) * 1e6          # read + write


def run(config="configs/iteration_13.yaml", mixed_precision=False):
    print(f"\n{'='*60}")
    print("C6: pack_half_spectrum / unpack_half_spectrum")
    print(f"{'='*60}")

    df, phi, geom, params, pre = load_setup(config, mixed_precision)

    from gyaradax.solver import pack_half_spectrum, unpack_half_spectrum

    jind   = pre["nl_jind"]
    mrad   = int(pre["nl_mrad"])
    mphi   = int(pre["nl_mphi"])
    mphiw3 = int(pre["nl_mphiw3"])
    nky    = df.shape[-1]

    # (nv, nmu, nkx, nky) — one s-slice
    spec_in = df[:, :, 0, :, :]

    fn_pack   = jax.jit(lambda: pack_half_spectrum(spec_in, jind, mrad, mphiw3))
    fn_unpack = jax.jit(lambda: unpack_half_spectrum(fn_pack(), jind, nky))

    baseline = BASELINES_DIR / "pack_spectrum.npz"

    print("\n  -- pack_half_spectrum")
    out_pack = fn_pack()
    check_accuracy(out_pack, baseline, "output_packed")
    mean_ms, std_ms = BenchTimer(fn_pack).run()
    print(f"  timing: {mean_ms:.3f} ± {std_ms:.3f} ms")
    roofline_report("pack_half_spectrum", mean_ms, 0, BYTES_PACK)

    print("\n  -- unpack_half_spectrum")
    out_unpack = fn_unpack()
    check_accuracy(out_unpack, baseline, "output_unpacked")
    mean_ms2, std_ms2 = BenchTimer(fn_unpack).run()
    print(f"  timing: {mean_ms2:.3f} ± {std_ms2:.3f} ms")
    roofline_report("unpack_half_spectrum", mean_ms2, 0, BYTES_UNPACK)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--config", type=str, default="configs/iteration_13.yaml")
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    run(args.config, args.mp)
