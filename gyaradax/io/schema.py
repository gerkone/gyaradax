"""The trajectory metadata tree, as in ``neugk_jax.dataset.schema`` (which also holds the trajectory contract)."""

FLUX_LABELS = ("pflux", "eflux", "vflux")
# keys left out of metadata_light (the per-element df and phi moments)
LIGHT_DROP_KEYS = ("df_min", "df_max", "df_var", "df_mean", "df_std", "phi_min", "phi_max", "phi_var")

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
    ("linear", "rms", "linear_"): "rms",
    ("linear", "rms", "linear_rms_"): "ky cky kxky ckxky count",
    **{("stats", f, f"{f}_"): "mean var std min max" for f in ("df", "phi", "phic", "flux")},
}
TREE_PATHS = {"geometry": ("geometry",), "ds": ("geometry", "ds"), "df_stats_agg_axes": ("stats", "df", "agg_axes")}
TREE_PATHS.update({k: (g, k) for g, keys in _GROUPS.items() for k in keys.split()})
TREE_PATHS.update({p + k: tuple(w) + (k,) for (*w, p), keys in _PREFIXED.items() for k in keys.split()})

_MISSING = object()


def nest_metadata(flat: dict) -> dict:
    """Grouped tree of flat metadata; keys without a place in :data:`TREE_PATHS` go to ``extra``."""
    if flat.get("schema") == SCHEMA_VERSION:
        return flat
    tree = {"schema": SCHEMA_VERSION}
    extra = {k: v for k, v in flat.items() if k not in TREE_PATHS}
    if extra:
        tree["extra"] = extra
    if "geometry" in flat:
        # a copy: ds joins the geometry group of the tree, not the caller's dict
        tree["geometry"] = dict(flat["geometry"])
    for key, value in flat.items():
        if key == "geometry" or key in extra:
            continue
        *parents, leaf = TREE_PATHS[key]
        node = tree
        for p in parents:
            node = node.setdefault(p, {})
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
            node = node.get(p, _MISSING) if isinstance(node, dict) else _MISSING
        if node is not _MISSING:
            flat[key] = node
    if "geometry" in flat:
        flat["geometry"] = {k: v for k, v in flat["geometry"].items() if k != "ds"}
    return {**flat, **meta.get("extra", {})}
