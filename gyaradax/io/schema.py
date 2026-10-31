"""The trajectory metadata as a tree: the flat keys grouped by meaning.

A verbatim copy of the tree part of ``neugk_jax.dataset.schema``, which reads the same files;
``tests/unit/test_dataset.py`` checks the two agree when neugk_jax is installed.
"""

from __future__ import annotations

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
