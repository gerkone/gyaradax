"""Shared utilities for solver component benchmarks.

Usage in each bench_*.py:
    from common import load_setup, BenchTimer, roofline_report, check_accuracy
"""
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── must be imported after CUDA_VISIBLE_DEVICES is set by the caller ────────
import jax
import jax.numpy as jnp

# ── GPU Hardware Specs ────────────────────────────────────────────────────────

def get_gpu_specs():
    """Detect current GPU and return (BW_GBS, FP64_TFLOPS, kind, name)."""
    try:
        kind = jax.devices()[0].device_kind
    except Exception:
        kind = "Unknown"

    specs = None
    # GPU specs database: (Memory Bandwidth in TB/s, FP64 TFLOPS, Display Name)
    # Note: Insertion order matters here. We check for more specific strings 
    # (like "H100 PCIe") before general ones (like "H100").
    # Reference Links:
    # - NVIDIA Blackwell Datasheet: https://resources.nvidia.com/en-us-blackwell-architecture
    # - NVIDIA Hopper Datasheet: https://resources.nvidia.com/en-us-tensor-core
    # - NVIDIA Ampere Datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    gpu_database = {
        "B300": {"bw_tbs": 8.0, "fp64_tflops": 37.0, "name": "Blackwell B300"},
        "B200": {"bw_tbs": 8.0, "fp64_tflops": 37.0, "name": "Blackwell B200"},
        "B100": {"bw_tbs": 8.0, "fp64_tflops": 30.0, "name": "Blackwell B100"},
        "H200 NVL": {"bw_tbs": 4.8, "fp64_tflops": 30.0, "name": "Hopper H200 NVL"},
        "H200": {"bw_tbs": 4.8, "fp64_tflops": 34.0, "name": "Hopper H200 SXM"},
        "H100 PCIe": {"bw_tbs": 2.0, "fp64_tflops": 25.6, "name": "Hopper H100 PCIe"},
        "H100": {"bw_tbs": 3.35, "fp64_tflops": 34.0, "name": "Hopper H100 SXM"},
        "A100-PCIE-40GB": {"bw_tbs": 1.55, "fp64_tflops": 9.7, "name": "Ampere A100 PCIe 40GB"},
        "A100-SXM4-40GB": {"bw_tbs": 1.55, "fp64_tflops": 9.7, "name": "Ampere A100 SXM4 40GB"},
        "A100": {"bw_tbs": 2.0, "fp64_tflops": 9.7, "name": "Ampere A100 80GB"},
        "V100-PCIe": {"bw_tbs": 0.9, "fp64_tflops": 7.0, "name": "Volta V100 PCIe"},
        "V100": {"bw_tbs": 0.9, "fp64_tflops": 7.8, "name": "Volta V100 SXM"}
    }

    # Match the device kind to our database
    for key, data in gpu_database.items():
        if key in kind:
            specs = data
            break
    if specs is None:
        print(f"  [WARN] Unknown GPU kind '{kind}' — roofline figures will be meaningless.")
        return 0.0, 0.0, kind, kind

    bw_gbs = specs["bw_tbs"] * 1024
    return bw_gbs, specs["fp64_tflops"], kind, specs["name"]

# Global defaults (can be overridden if needed)
DEFAULT_BW_GBS, DEFAULT_FP64_TFLOPS, DEVICE_KIND, DEVICE_MODEL = get_gpu_specs()
BASELINES_DIR = Path(__file__).parent / "baselines"


# ── Setup ────────────────────────────────────────────────────────────────────

def load_setup(config_path: str = "configs/iteration_13.yaml", mixed_precision: bool = False):
    """Load real solver state from config + K-file checkpoint.

    Returns
    -------
    df   : jnp.ndarray  (nv, nmu, ns, nkx, nky) complex128, adiabatic 5D slice
    phi  : jnp.ndarray  (ns, nkx, nky) complex128
    geom : dict
    params : GKParams
    pre  : dict  (linear_precompute output)
    """
    from gyaradax import load_config, load_geometry
    from gyaradax.params import gkparams_from_config
    from gyaradax.simulate import _geometry_from_config
    from gyaradax.solver import linear_precompute, _compute_phi, GKState
    from gyaradax.utils import load_gkw_k_dump, read_gkw_dump_time, K_files

    cfg = load_config(config_path)
    params = gkparams_from_config(cfg, mixed_precision=mixed_precision)

    data_dir = getattr(cfg.run, "data_dir", None)
    if data_dir and os.path.exists(os.path.join(data_dir, "geom.dat")):
        geom = load_geometry(data_dir)
    else:
        geom = _geometry_from_config(cfg)

    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])

    # load K-file checkpoint
    k_path = None
    if data_dir:
        ks = K_files(data_dir)
        if ks:
            k_path = os.path.join(data_dir, ks[0])

    if k_path is not None:
        shape = tuple(len(geom[k]) for k in ("intvp", "intmu", "ints", "kxrh", "krho"))
        df = load_gkw_k_dump(k_path, shape, n_species=n_species)
        t_start = read_gkw_dump_time(k_path + ".dat") if os.path.exists(k_path + ".dat") else 0.0
    else:
        from gyaradax.simulate import gk_init
        df, _ = gk_init(geom, params, n_species=n_species)
        t_start = 0.0

    pre = linear_precompute(geom, params)
    phi = _compute_phi(df, geom, params, pre)
    
    print(f"  device   : {DEVICE_KIND} (detected as {DEVICE_MODEL})")
    print(f"  peak BW  : {DEFAULT_BW_GBS:.0f} GB/s")
    print(f"  peak FP64: {DEFAULT_FP64_TFLOPS:.1f} TFLOP/s")

    # for adiabatic: df is 5D (nv, nmu, ns, nkx, nky)
    # for kinetic:   df is 6D (nsp, nv, nmu, ns, nkx, nky); squeeze species=0 when needed
    print(f"  df shape : {df.shape}  dtype: {df.dtype}")
    print(f"  phi shape: {phi.shape}  dtype: {phi.dtype}")
    return df, phi, geom, params, pre


# ── Timing ───────────────────────────────────────────────────────────────────

class BenchTimer:
    """Warm-up then time a JIT-compiled function, returning mean ± std ms."""

    def __init__(self, fn: Callable, n_warmup: int = 3, n_trials: int = 15):
        self.fn = fn
        self.n_warmup = n_warmup
        self.n_trials = n_trials

    def run(self) -> tuple[float, float]:
        for _ in range(self.n_warmup):
            jax.block_until_ready(self.fn())
        times = []
        for _ in range(self.n_trials):
            t0 = time.perf_counter()
            jax.block_until_ready(self.fn())
            times.append((time.perf_counter() - t0) * 1e3)
        arr = np.array(times)
        return float(arr.mean()), float(arr.std())


# ── Reporting ─────────────────────────────────────────────────────────────────

def roofline_report(
    label: str,
    mean_ms: float,
    flops: float,
    bytes_rw: float,
    bw_gbs: float = DEFAULT_BW_GBS,
    fp64_tflops: float = DEFAULT_FP64_TFLOPS,
) -> dict:
    """Print roofline stats and return a result dict.

    Parameters
    ----------
    flops    : total FLOPs for one call (0 for pure-memory kernels)
    bytes_rw : total bytes read + written for one call
    bw_gbs   : peak memory bandwidth of the target device (GB/s)
    """
    achieved_gbs = bytes_rw / 1e9 / (mean_ms / 1e3)
    pct_bw = 100.0 * achieved_gbs / bw_gbs if bw_gbs > 0 else float("nan")
    if flops > 0:
        ai_achieved = flops / bytes_rw
        if bw_gbs > 0 and fp64_tflops > 0:
            roofline_tflops = min(flops / bytes_rw * bw_gbs / 1024, fp64_tflops)
            pct_roof = 100.0 * (flops / (mean_ms / 1e3)) / (roofline_tflops * 1e12)
            roof_str = f"{pct_roof:.0f}% roofline"
        else:
            roof_str = "roofline N/A"
        print(
            f"  {label:30s}  {mean_ms:7.3f} ms  "
            f"{achieved_gbs:6.1f} GB/s ({pct_bw:.0f}% BW)  "
            f"AI={ai_achieved:.3f}  {roof_str}"
        )
    else:
        print(
            f"  {label:30s}  {mean_ms:7.3f} ms  "
            f"{achieved_gbs:6.1f} GB/s ({pct_bw:.0f}% BW)  "
            f"AI=0 (pure memory)"
        )
    return {
        "label": label,
        "mean_ms": mean_ms,
        "achieved_gbs": achieved_gbs,
        "pct_bw": pct_bw,
    }


def check_accuracy(out: jnp.ndarray, baseline_path: str, key: str) -> float:
    """Compare output against saved baseline; print and return rel_l2."""
    path = Path(baseline_path)
    if not path.exists():
        print(f"  [SKIP] baseline not found: {path} — run generate_baselines.py first")
        return float("nan")
    data = np.load(path)
    ref = jnp.array(data[key])
    rel_l2 = float(jnp.linalg.norm(out - ref) / jnp.linalg.norm(ref))
    status = "OK" if rel_l2 < 1e-10 else "FAIL"
    print(f"  accuracy [{status}] rel_l2 = {rel_l2:.3e}  (vs {path.name}:{key})")
    return rel_l2
