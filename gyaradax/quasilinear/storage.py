"""Storage options for campaign data: dtype casting, keep levels, npz saving.

Configured from the run/campaign YAML `storage:` section:
  dtype: f64 (default) | f32 | bf16 | f16
  compress: false (default) | true          -> np.savez_compressed
  keep: full (default) | ql | gamma         -> which harvest fields to store
"""

import numpy as np

# fields never downcast: tiny, and they feed fits/axes directly
KEEP_F64 = ("gamma", "growth_rate", "krho", "kxrh", "ds", "dt", "X", "Y", "F", "gamma_max")
# keep levels: 'ql' drops the raw complex phi field; 'gamma' keeps only stability info
KEEP_DROP = {"full": (), "ql": ("phi",), "gamma": None}
GAMMA_ONLY_FIELDS = ("gamma", "growth_rate", "n_steps")


def storage_dtype(name):
    """Resolve a storage dtype name (f64/f32/bf16/f16) to a numpy dtype."""
    if name in ("f64", None, ""):
        return np.dtype(np.float64)
    if name == "f32":
        return np.dtype(np.float32)
    if name == "f16":
        return np.dtype(np.float16)
    if name == "bf16":
        import ml_dtypes

        return np.dtype(ml_dtypes.bfloat16)
    raise ValueError(f"unknown storage dtype: {name!r} (use f64/f32/bf16/f16)")


def apply_storage(arrays, dtype="f64", keep="full", keep_f64=KEEP_F64):
    """Filter fields per the keep level, then cast floats to the storage dtype."""
    if keep not in KEEP_DROP:
        raise ValueError(f"unknown storage keep level: {keep!r} (use full/ql/gamma)")
    if KEEP_DROP[keep] is None:
        arrays = {k: v for k, v in arrays.items() if k in GAMMA_ONLY_FIELDS}
    else:
        arrays = {k: v for k, v in arrays.items() if k not in KEEP_DROP[keep]}
    dt = storage_dtype(dtype)
    out = {}
    for k, v in arrays.items():
        a = np.asarray(v)
        if k in keep_f64 or dt == np.float64:
            out[k] = a
        elif np.issubdtype(a.dtype, np.complexfloating):
            out[k] = a.astype(np.complex64)
        elif np.issubdtype(a.dtype, np.floating):
            out[k] = a.astype(dt)
        else:
            out[k] = a
    return out


def save_arrays(path, arrays, compress=False):
    (np.savez_compressed if compress else np.savez)(path, **arrays)


def resolve_storage(*configs):
    """Merge `storage:` sections (later configs win) into full option dict."""
    opts = {"dtype": "f64", "compress": False, "keep": "full"}
    for cfg in configs:
        opts.update((cfg or {}).get("storage", {}) or {})
    opts["compress"] = bool(opts["compress"])
    return opts
