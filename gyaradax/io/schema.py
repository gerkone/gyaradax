"""The trajectory metadata as a tree, and the trajectory contract shared with neugk_jax.

A verbatim copy of the tree and contract parts of ``neugk_jax.dataset.schema``, which reads the same
files; ``tests/unit/test_dataset.py`` checks the two agree when neugk_jax is installed.
"""

from __future__ import annotations

import numpy as np

FLUX_LABELS = ("pflux", "eflux", "vflux")
# keys left out of metadata_light (the per-element df and phi moments)
LIGHT_DROP_KEYS = (
    "df_min",
    "df_max",
    "df_var",
    "df_mean",
    "df_std",
    "phi_min",
    "phi_max",
    "phi_var",
)

# grouped layout: the keys arranged by meaning; readers see the flat layout through flatten_metadata
SCHEMA_VERSION = 3
_GROUPS = {
    "run": "simulation_code source_config stable outcome quantization hdisp_rerun_of appended_from hand_label",
    "params": "species n_species ion_species_index electron_species_index ion_temp_grad electron_temp_grad "
              "density_grad temp_ratio ion_temp electron_temp rlt rln tmp q s_hat eps coll_freq",
    "layout": "resolution df_shape phi_shape axis_order channel_layout real_space_convention phi_layout "
              "phi_convention",
    "frames": "timesteps solver_steps phic_source phic_source_frames phi_source quant_phi_err quant_phi_df_err "
              "quant_flux_err first_frame clipped_t_start thinned_from thinning_spacing",
    "fluxes": "flux flux_labels kyspec kxspec growth fluxspec",
    "flags": "sus_underresolved sus_reason broken broken_note frames_dropped frames_dropped_count banded_from_t",
}
_PREFIXED = {
    ("trace", "trace_"): "timesteps solver_steps flux growth nyq_zonal nyq_nonzonal",
    ("linear", "files", "linear_"): "bin poten_bin shape poten_shape layout poten_layout space k_bin k_bin_shape",
    ("linear", "provenance", "linear_"): "code source schema criterion dt fds gate t",
    ("linear", "convergence", "linear_"): "time settled stop stopped_early t_end_requested stop_note gamma_drift",
    ("linear", "grid", "linear_"): "ky kxrh kthnorm ky_convention",
    ("linear", "spectrum", "linear_"): "gamma omega gamma_std omega_std gamma_max peak_ky gamma_max_lowky low_ky",
    ("linear", "flux", "linear_"): "flux_labels flux flux_ky flux_kxy weight_ky weight_ky_allkx flux_ky_chain0 "
                                   "phi2_ky phi2_kxy phi2_ky_chain0",
    ("linear", "history", "linear_hist_"): "t gamma omega weight_t weight_ky",
}
_STATS = {"df": "mean var std min max", "phi": "mean var std min max", "phic": "mean var std min max",
          "flux": "mean var std min max"}


def _paths() -> dict:
    paths = {"geometry": ("geometry",), "ds": ("geometry", "ds"), "df_stats_agg_axes": ("stats", "df", "agg_axes")}
    for group, keys in _GROUPS.items():
        paths.update({k: (group, k) for k in keys.split()})
    for (*where, prefix), keys in _PREFIXED.items():
        paths.update({prefix + k: (*where, k) for k in keys.split()})
    paths["linear_rms"] = ("linear", "rms", "rms")
    paths.update({f"linear_rms_{k}": ("linear", "rms", k) for k in ("ky", "cky", "kxky", "ckxky", "count")})
    for field, keys in _STATS.items():
        paths.update({f"{field}_{k}": ("stats", field, k) for k in keys.split()})
    return paths


TREE_PATHS = _paths()


def unplaced_keys(flat: dict) -> list:
    return sorted(set(flat) - set(TREE_PATHS))


def nest_metadata(flat: dict) -> dict:
    """Grouped tree of flat metadata; keys without a place in :data:`TREE_PATHS` go to ``extra``."""
    if flat.get("schema") == SCHEMA_VERSION:
        return flat
    tree = {"schema": SCHEMA_VERSION}
    extra = {k: flat[k] for k in unplaced_keys(flat)}
    if extra:
        tree["extra"] = extra
    if "geometry" in flat:
        if "ds" in flat["geometry"]:
            raise KeyError("geometry already holds ds")
        # a copy: ds joins the geometry group of the tree, not the caller's dict
        tree["geometry"] = dict(flat["geometry"])
    for key, value in flat.items():
        if key == "geometry" or key in extra:
            continue
        *parents, leaf = TREE_PATHS[key]
        node = tree
        for p in parents:
            node = node.setdefault(p, {})
        if leaf in node:
            raise KeyError(f"{key} collides at {'.'.join(TREE_PATHS[key])}")
        node[leaf] = value
    return tree


def flatten_metadata(meta: dict) -> dict:
    """Flat view of grouped metadata (flat layouts are returned unchanged)."""
    if meta.get("schema") != SCHEMA_VERSION:
        return meta
    flat = {}
    for key, path in TREE_PATHS.items():
        node = meta
        for p in path:
            if not isinstance(node, dict) or p not in node:
                break
            node = node[p]
        else:
            flat[key] = node
    if "geometry" in flat:
        flat["geometry"] = {k: v for k, v in flat["geometry"].items() if k != "ds"}
    flat.update(meta.get("extra", {}))
    return flat


# trajectory contract shared by the gyaradax writer and neugk's preprocess and loader; gyaradax vendors this
# section verbatim. Shapes use the symbols of _dims; "S2" is the re/im axis and "*" any extent.
_F, _I, _B, _S, _D, _N = "float", "int", "bool", "str", "dict", "none"
REQUIRED, KINETIC, ADIABATIC, OPTIONAL = "required", "kinetic", "adiabatic", "optional"
METADATA_SPEC = {
    "timesteps": (REQUIRED, _F, ("T",)),
    "flux": (REQUIRED, _F, ("T", "S", 3)),
    "flux_labels": (REQUIRED, "str_seq", (3,)),
    "resolution": (REQUIRED, "int_seq", (5,)),
    "df_shape": (REQUIRED, "int_seq", (7,)),
    "n_species": (REQUIRED, _I, ()),
    "species": (REQUIRED, "str_seq", ("S",)),
    "ion_species_index": (REQUIRED, _I, ()),
    "electron_species_index": (REQUIRED, _I, ()),
    "rlt": (REQUIRED, _F, ("S",)),
    "rln": (REQUIRED, _F, ("S",)),
    "tmp": (REQUIRED, _F, ("S",)),
    "ion_temp_grad": (REQUIRED, _F, (1,)),
    "density_grad": (REQUIRED, _F, (1,)),
    "electron_temp_grad": (REQUIRED, _F, (1,)),
    "temp_ratio": (REQUIRED, _F, (1,)),
    "eps": (REQUIRED, _F, (1,)),
    "coll_freq": (REQUIRED, _F, (1,)),
    "q": (REQUIRED, _F, (1,)),
    "s_hat": (REQUIRED, _F, (1,)),
    "ds": (REQUIRED, _F, ()),
    "geometry": (REQUIRED, _D, None),
    "kyspec": (REQUIRED, _F, ("T", "nky")),
    "phi_mean": (REQUIRED, _F, ("nkx", "ns", "nky")),
    "phi_std": (REQUIRED, _F, ("nkx", "ns", "nky")),
    "phic_source": (REQUIRED, _S, ()),
    **{f"phic_{k}": (REQUIRED, _F, ("S2", "nkx", "ns", "nky")) for k in ("mean", "var", "std", "min", "max")},
    # full metadata only (LIGHT_DROP_KEYS); df moments keep size 1 on the aggregated axes
    **{f"df_{k}": (REQUIRED, _F, ("S2", "S", "*", "*", "*", "*", "*")) for k in ("mean", "var", "std", "min", "max")},
    **{f"phi_{k}": (REQUIRED, _F, ("nkx", "ns", "nky")) for k in ("min", "max", "var")},
    "kxspec": (KINETIC, _F, ("T", "nkx")),
    "growth": (OPTIONAL, _F, ("T", "nky")),
    "solver_steps": (OPTIONAL, _I, ("T",)),
    # written by the GKW conversion only; the gyaradax writer leaves them out
    "fluxspec": (OPTIONAL, _F, ("T", "nky")),
    **{f"flux_{k}": (OPTIONAL, _F, ()) for k in ("mean", "var", "std", "min", "max")},
    "simulation_code": (OPTIONAL, _S, ()),
    "source_config": (OPTIONAL, _S, ()),
    "quantization": (OPTIONAL, _S, ()),
    "axis_order": (OPTIONAL, _S, ()),
    "channel_layout": (OPTIONAL, _S, ()),
    "real_space_convention": (OPTIONAL, _S, ()),
    "phi_shape": (OPTIONAL, "int_seq", (3,)),
    "phi_layout": (OPTIONAL, _S, ()),
    "phi_convention": (OPTIONAL, _S, ()),
    "phi_source": (OPTIONAL, _S, ()),
    "phic_source_frames": (OPTIONAL, _S, ("T",)),
    "df_stats_agg_axes": (OPTIONAL, "int_seq", ("*",)),
    "ion_temp": (OPTIONAL, _F, (1,)),
    "electron_temp": (OPTIONAL, _F, (1,)),
    "quant_phi_err": (OPTIONAL, _F, ("T",)),
    "quant_phi_df_err": (OPTIONAL, _F, ("T",)),
    "quant_flux_err": (OPTIONAL, _F, ("T", "S", 3)),
    "first_frame": (OPTIONAL, _I, ()),
    "clipped_t_start": (OPTIONAL, "float_or_none", ()),
    "thinned_from": (OPTIONAL, "int_or_none", ("T",)),
    "thinning_spacing": (OPTIONAL, _F, ()),
    "appended_from": (OPTIONAL, _I, ("*",)),
    "hdisp_rerun_of": (OPTIONAL, _S, ()),
    # legacy: written by earlier gyaradax campaigns, no longer by the writer
    "outcome": (OPTIONAL, _D, None),
    "stable": (OPTIONAL, _B, ()),
    "hand_label": (OPTIONAL, _S, ()),
    **{f"trace_{k}": (OPTIONAL, "float_or_none", ("Ttr", *rest)) for k, rest in
       (("timesteps", ()), ("flux", ("*",)), ("growth", ("nky",)), ("nyq_zonal", ()), ("nyq_nonzonal", ()))},
    "trace_solver_steps": (OPTIONAL, _I, ("Ttr",)),
    "sus_underresolved": (OPTIONAL, _B, ()),
    "sus_reason": (OPTIONAL, _S, ()),
    "broken": (OPTIONAL, _S, ()),
    "broken_note": (OPTIONAL, _S, ()),
    "frames_dropped": (OPTIONAL, _S, ()),
    "frames_dropped_count": (OPTIONAL, _I, ()),
    "banded_from_t": (OPTIONAL, _F, ()),
}
# linear eigenmode: every key optional, the files and their shape keys come as a set
LINEAR_SPEC = {
    "linear_bin": (_S, ()), "linear_shape": ("int_seq", (7,)), "linear_space": (_S, ()), "linear_layout": (_S, ()),
    "linear_poten_bin": (_S, ()), "linear_poten_shape": ("int_seq", (4,)), "linear_poten_layout": (_S, ()),
    "linear_k_bin": (_S, ()), "linear_k_bin_shape": ("int_seq", ("*",)),
    "linear_code": (_S, ()), "linear_source": (_S, ()), "linear_schema": (_I, ()), "linear_criterion": (_S, ()),
    "linear_dt": (_F, ()), "linear_fds": (_S, ()), "linear_gate": (_S, ()), "linear_t": (_F, ()),
    "linear_time": (_F, ()), "linear_settled": (_B, ()), "linear_stop": (_S, ()), "linear_stopped_early": (_B, ()),
    "linear_t_end_requested": (_F, ()), "linear_stop_note": (_S, ()), "linear_gamma_drift": (_F, ()),
    "linear_ky": (_F, ("nky",)), "linear_kxrh": (_F, ("nkx",)), "linear_kthnorm": (_F, ()), "linear_ky_convention": (_S, ()),
    "linear_gamma": (_F, ("nky",)), "linear_omega": (_F, ("nky",)), "linear_gamma_std": (_F, ("nky",)),
    "linear_omega_std": (_F, ("nky",)), "linear_gamma_max": (_F, ()), "linear_peak_ky": (_F, ()),
    "linear_gamma_max_lowky": (_F, ()), "linear_low_ky": (_F, ()),
    "linear_flux_labels": (_S, (3,)), "linear_flux": (_F, ("S", 3)), "linear_flux_ky": (_F, ("S", 3, "nky")),
    "linear_flux_kxy": (_F, ("S", 3, "nkx", "nky")), "linear_weight_ky": (_F, ("S", 3, "nky")),
    "linear_weight_ky_allkx": (_F, ("S", 3, "nky")), "linear_flux_ky_chain0": (_F, ("S", 3, "nky")),
    "linear_phi2_ky": (_F, ("nky",)), "linear_phi2_kxy": (_F, ("nkx", "nky")), "linear_phi2_ky_chain0": (_F, ("nky",)),
    "linear_rms": (_F, ()), "linear_rms_ky": (_F, ("nky",)), "linear_rms_cky": (_F, ("S2", "nky")),
    "linear_rms_kxky": (_F, ("nkx", "nky")), "linear_rms_ckxky": (_F, ("S2", "nkx", "nky")), "linear_rms_count": (_I, ()),
    "linear_hist_t": (_F, ("Th",)), "linear_hist_gamma": (_F, ("Th", "nky")), "linear_hist_omega": (_F, ("Th", "nky")),
    "linear_hist_weight_t": (_F, ("Tw",)), "linear_hist_weight_ky": (_F, ("Tw", "S", 3, "nky")),
}
GEOMETRY_REQUIRED = {"kxrh": ("nkx",), "krho": ("nky",), "intvp": ("nvpar",), "intmu": ("nmu",), "ints": ("ns",),
                     "bn": ("ns",), "little_g": ("ns", 3)}
RESUME_FIELDS = {"df": "complex", "time": "float", "step": "int", "accumulated_norm_factor": "float",
                 "window_start_amp": "float", "last_growth_rate": "float", "frame": "int"}


def resume_frame(timesteps, time: float) -> int:
    ts = np.asarray(timesteps, dtype=np.float64)
    return int(np.searchsorted(ts, time + 1e-6 * max(1.0, abs(time)), side="right")) - 1


def _dims(meta: dict) -> dict:
    res = tuple(int(r) for r in np.atleast_1d(meta.get("resolution", ())))
    dims = {"T": len(np.atleast_1d(meta.get("timesteps", []))), "S": int(meta.get("n_species", 0)), "S2": 2}
    dims.update(zip(("nvpar", "nmu", "ns", "nkx", "nky"), res))
    for sym, key in (("Ttr", "trace_timesteps"), ("Th", "linear_hist_t"), ("Tw", "linear_hist_weight_t")):
        if meta.get(key) is not None:
            dims[sym] = len(np.atleast_1d(meta[key]))
    return dims


def _check_value(key: str, value, kind: str, shape, dims: dict) -> list[str]:
    if kind == _D:
        return [] if isinstance(value, dict) else [f"{key}: not a dict"]
    if value is None:
        return [] if kind.endswith("_or_none") else [f"{key}: None"]
    if kind in ("int_seq", "str_seq"):
        if not isinstance(value, (tuple, list, np.ndarray)):
            return [f"{key}: {type(value).__name__}, not a sequence"]
        if kind == "str_seq" and not all(isinstance(v, (str, np.str_)) for v in value):
            return [f"{key}: not all strings"]
        if kind == "int_seq" and not all(isinstance(v, (int, np.integer)) for v in value):
            return [f"{key}: not all ints"]
        arr_shape = (len(value),)
    else:
        arr = np.asarray(value)
        base = kind.removesuffix("_or_none")
        want = {_F: "f", _I: "iu", _B: "b", _S: "U"}[base]
        if arr.dtype.kind not in want and not (base == _F and arr.dtype.kind in "iu"):
            return [f"{key}: dtype {arr.dtype}, expected {base}"]
        arr_shape = arr.shape
    if shape is None:
        return []
    if len(arr_shape) != len(shape):
        return [f"{key}: shape {arr_shape}, expected {shape}"]
    for got, sym in zip(arr_shape, shape):
        if sym == "*":
            continue
        want_n = dims.get(sym) if isinstance(sym, str) else sym
        if want_n is not None and got != want_n:
            return [f"{key}: shape {arr_shape}, expected {shape} with {dims}"]
    return []


def check_metadata(meta: dict, light: bool = False) -> list[str]:
    """Violations of the trajectory contract in one metadata dict (flat or grouped)."""
    meta = flatten_metadata(meta)
    dims = _dims(meta)
    kinetic = int(meta.get("n_species", 0)) > 1
    out = []
    for key, (need, kind, shape) in METADATA_SPEC.items():
        required = need == REQUIRED or (need == KINETIC and kinetic) or (need == ADIABATIC and not kinetic)
        if light and key in LIGHT_DROP_KEYS:
            if key in meta:
                out.append(f"{key}: in metadata_light")
            continue
        if key not in meta:
            if required:
                out.append(f"{key}: missing")
            continue
        out += _check_value(key, meta[key], kind, shape, dims)
    for key, (kind, shape) in LINEAR_SPEC.items():
        if key in meta:
            out += _check_value(key, meta[key], kind, shape, dims)
    if "linear_bin" in meta and meta.get("linear_space") != "spectral":
        out.append(f"linear_space: {meta.get('linear_space')!r}, expected 'spectral'")
    if "linear_k_bin" in meta:
        out.append("linear_k_bin: legacy, the eigenmode is linear.bin")
    unknown = sorted(set(meta) - set(METADATA_SPEC) - set(LINEAR_SPEC))
    out += [f"{k}: not in the contract" for k in unknown]
    if "flux_labels" in meta and tuple(meta["flux_labels"]) != FLUX_LABELS:
        out.append(f"flux_labels: {tuple(meta['flux_labels'])}, expected {FLUX_LABELS}")
    if "df_shape" in meta and "resolution" in meta:
        want = (2, dims["S"], *(int(r) for r in meta["resolution"]))
        if tuple(int(x) for x in meta["df_shape"]) != want:
            out.append(f"df_shape: {tuple(meta['df_shape'])}, expected {want}")
    geom = meta.get("geometry") if isinstance(meta.get("geometry"), dict) else {}
    for key, shape in GEOMETRY_REQUIRED.items():
        if key not in geom:
            out.append(f"geometry.{key}: missing")
        else:
            out += _check_value(f"geometry.{key}", geom[key], _F, shape, dims)
    return out


def check_trajectory(path: str, require_checkpoint: bool = False) -> list[str]:
    """Violations of the trajectory contract in a trajectory directory: metadata, frame files, restart files.

    ``require_checkpoint`` also asks for ``checkpoint/step_*.npz``, which the stored dataset keeps and a fresh
    writer output does not have.
    """
    import glob
    import os
    import pickle
    import re

    out = []
    metas = {}
    for name in ("metadata.pkl", "metadata_light.pkl"):
        p = os.path.join(path, name)
        if not os.path.exists(p):
            out.append(f"{name}: missing")
            continue
        with open(p, "rb") as f:
            raw = pickle.load(f)
        if raw.get("schema") != SCHEMA_VERSION:
            out.append(f"{name}: not the grouped layout")
        if raw.get("extra"):
            out.append(f"{name}: keys without a place: {sorted(raw['extra'])}")
        metas[name] = flatten_metadata(raw)
        out += [f"{name}: {v}" for v in check_metadata(raw, light=name == "metadata_light.pkl")]
    if len(metas) == 2:
        full, light = metas["metadata.pkl"], metas["metadata_light.pkl"]
        diff = sorted(set(full) - set(LIGHT_DROP_KEYS) ^ set(light))
        if diff:
            out.append(f"metadata_light: keys differ from metadata minus LIGHT_DROP_KEYS: {diff}")
    meta = metas.get("metadata_light.pkl") or metas.get("metadata.pkl") or {}
    t = len(np.atleast_1d(meta.get("timesteps", [])))
    data = os.path.join(path, "data")
    files = os.listdir(data) if os.path.isdir(data) else []
    for kind in ("timestep", "poten", "potenc"):
        idx = sorted(int(m.group(1)) for f in files if (m := re.match(rf"{kind}_(\d+)\.", f)))
        if idx != list(range(t)):
            out.append(f"data/{kind}: frames {len(idx)} (first {idx[:1]}, last {idx[-1:]}), expected 0..{t - 1}")
    for name, key, n in (("linear.bin", "linear_shape", 7), ("linear_poten.bin", "linear_poten_shape", 4)):
        p = os.path.join(data, name)
        if os.path.exists(p) and meta.get(key) is not None:
            if os.path.getsize(p) != 4 * int(np.prod(np.asarray(meta[key], np.int64))):
                out.append(f"data/{name}: size does not match {key}")
    if os.path.exists(os.path.join(data, "linear.bin")) != os.path.exists(os.path.join(data, "linear_poten.bin")):
        out.append("data: linear.bin and linear_poten.bin come as a pair")
    if os.path.exists(os.path.join(data, "linear_k.bin")):
        out.append("data/linear_k.bin: legacy, the eigenmode is linear.bin")
    if not os.path.exists(os.path.join(path, "config.yaml")):
        out.append("config.yaml: missing")
    state = os.path.join(path, "resume", "state.npz")
    if not os.path.exists(state) or not os.path.exists(os.path.join(path, "resume", "config.yaml")):
        out.append("resume/{state.npz, config.yaml}: missing")
    else:
        with np.load(state) as z:
            for key, kind in RESUME_FIELDS.items():
                if key not in z.files:
                    out.append(f"resume/state.npz: {key} missing")
                elif kind == "complex" and z[key].dtype.kind != "c":
                    out.append(f"resume/state.npz: {key} dtype {z[key].dtype}")
            if "frame" in z.files and "time" in z.files and t:
                want = resume_frame(meta["timesteps"], float(z["time"]))
                if int(z["frame"]) != want:
                    out.append(f"resume/state.npz: frame {int(z['frame'])}, expected {want} (time {float(z['time']):.4g})")
    if require_checkpoint and not glob.glob(os.path.join(path, "checkpoint", "step_*.npz")):
        out.append("checkpoint/step_*.npz: missing")
    return out
