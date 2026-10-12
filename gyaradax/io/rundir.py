"""Run-directory output: diagnostic histories, restart snapshots, run metadata."""

import os
from typing import Any, Dict, Tuple, cast

import jax.numpy as jnp
import numpy as np

from gyaradax.diagnostics.diag import phi_spectra
from gyaradax.diagnostics.fluxes import _compute_em_fluxes_arr


def save_dumps(
    output_dir: str,
    df: jnp.ndarray,
    phi: jnp.ndarray,
    fluxes: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    state: Any,
    geometry: Dict[str, jnp.ndarray],
    save_dumps: bool = True,
    params: Any = None,
    pre: Any = None,
    dt_info: Any = None,
    block_start_step: int = 0,
    block_start_time: float = 0.0,
    snapshot_f32: bool = False,
):
    """Handle simulation output. Saves heavy 5D distribution snapshots if requested
    and appends diagnostic history to persistent files.

    When ``params`` and ``pre`` are supplied and ``params.nlapar`` is set, also
    writes ``fluxes_em.npz`` with the magnetic-flutter EM fluxes computed from
    the final (f, A_par, B_par). When ``dt_info`` is supplied (dict with per-step
    ``dt_used``/``dt_nl``/``dt_lin`` arrays and scalar ``dt_input``), also
    appends a per-step ``dt_history.npz`` matching GKW's debug4 columns:
    step, time, dt_used, dt_nl, dt_lin, dt_input.
    """
    os.makedirs(output_dir, exist_ok=True)

    ky_spec, kx_spec = phi_spectra(phi, geometry)

    fluxes_arr = np.asarray(fluxes)
    if fluxes_arr.ndim == 0 or (fluxes_arr.ndim == 1 and fluxes_arr.shape[0] != 3):
        fluxes_arr = np.array([fluxes[0], fluxes[1], fluxes[2]])

    # em fluxes shape matches ES: (3,) for 5D df, (nsp, 3) for 6D, including
    # the EM vflux (flutter momentum flux). A_par + B_par contributions are
    # summed into one triple; GKW writes them as separate fluxes_em/fluxes_bpar.
    em_fluxes_arr = _compute_em_fluxes_arr(df, geometry, params, pre, fluxes_arr.shape)

    diags = {
        "fluxes": fluxes_arr,
        "fluxes_em": em_fluxes_arr,
        "kx_spec": np.array(kx_spec),
        "ky_spec": np.array(ky_spec),
        "time": np.array(state.time),
        "growth": np.array(state.last_growth_rate),
        "step": np.array(state.step),
    }

    def _append_to_npz(filename, new_data):
        path = os.path.join(output_dir, filename)
        current_step = int(state.step)
        if os.path.exists(path):
            try:
                with np.load(path) as data:
                    # truncate at current_step so resume doesn't double-append
                    if "step" in data.files:
                        mask = data["step"] < current_step
                        updated = {
                            k: np.append(data[k][mask], [new_data[k]], axis=0) for k in data.files
                        }
                    else:
                        updated = {k: np.append(data[k], [new_data[k]], axis=0) for k in data.files}
            except (IOError, ValueError):
                updated = {k: np.array([v]) for k, v in new_data.items()}
        else:
            updated = {k: np.array([v]) for k, v in new_data.items()}
        np.savez(path, **updated)

    # include step and time in every diagnostic file for self-description
    common = {"step": diags["step"], "time": diags["time"]}

    _append_to_npz("fluxes.npz", {"fluxes": diags["fluxes"], **common})
    _append_to_npz("kyspec.npz", {"ky_spec": diags["ky_spec"], **common})
    _append_to_npz("kxspec.npz", {"kx_spec": diags["kx_spec"], **common})
    _append_to_npz("growth.npz", {"growth": diags["growth"], **common})
    # EM flux kept in a separate file so existing consumers of fluxes.npz are
    # unaffected; with nlapar off the array is zeros.
    _append_to_npz("fluxes_em.npz", {"fluxes_em": diags["fluxes_em"], **common})

    # dt_history: per-step record of the adaptive CFL controller, mirroring
    # GKW's debug4 dt_history.dat for diagnosing dt ramp-up on resume.
    last_dt = None
    if dt_info is not None:
        dt_used = np.asarray(dt_info["dt_used"]).reshape(-1)
        dt_nl = np.asarray(dt_info["dt_nl"]).reshape(-1)
        dt_lin = np.asarray(dt_info["dt_lin"]).reshape(-1)
        dt_input = float(np.asarray(dt_info["dt_input"]))
        n = int(dt_used.shape[0])
        if n > 0:
            # cumulative time within the block; step is the absolute step at
            # the END of each substep.
            step_arr = (
                np.asarray(block_start_step, dtype=np.int64) + 1 + np.arange(n, dtype=np.int64)
            )
            time_arr = float(block_start_time) + np.cumsum(dt_used)
            dt_input_arr = np.full((n,), dt_input, dtype=np.float64)
            _append_dt_path = os.path.join(output_dir, "dt_history.npz")
            new_data: dict[str, Any] = {
                "step": step_arr,
                "time": time_arr,
                "dt_used": dt_used.astype(np.float64),
                "dt_nl": dt_nl.astype(np.float64),
                "dt_lin": dt_lin.astype(np.float64),
                "dt_input": dt_input_arr,
            }
            if os.path.exists(_append_dt_path):
                try:
                    with np.load(_append_dt_path) as data:
                        if "step" in data.files:
                            mask = data["step"] < int(step_arr[0])
                            updated = {
                                k: np.concatenate([data[k][mask], new_data[k]]) for k in new_data
                            }
                        else:
                            updated = {k: np.concatenate([data[k], new_data[k]]) for k in new_data}
                except (IOError, ValueError):
                    updated = new_data
            else:
                updated = new_data
            np.savez(_append_dt_path, **cast(Any, updated))
            last_dt = float(dt_used[-1])

    if save_dumps:
        ckpt_name = f"step_{int(state.step):06d}.npz"
        path = os.path.join(output_dir, ckpt_name)
        # halve on-disk size: 5D df/phi to complex64, diagnostics stay f64
        df_out = np.asarray(df)
        phi_out = np.asarray(phi)
        if snapshot_f32:
            df_out = df_out.astype(np.complex64)
            phi_out = phi_out.astype(np.complex64)
        checkpoint: dict[str, Any] = {
            "df": df_out,
            "phi": phi_out,
            "fluxes": fluxes_arr,
            "time": np.array(state.time),
            "step": np.array(state.step),
            "accumulated_norm_factor": np.array(state.accumulated_norm_factor),
            "window_start_amp": np.array(state.window_start_amp),
            "last_growth_rate": np.array(state.last_growth_rate),
            "kx_spec": diags["kx_spec"],
            "ky_spec": diags["ky_spec"],
        }
        # record last dt so resume can warm-start the CFL controller from the
        # saturated state instead of params.dt.
        if last_dt is not None:
            checkpoint["dt_last"] = np.array(last_dt, dtype=np.float64)
        np.savez(path, **cast(Any, checkpoint))


def save_run_metadata(output_dir: str, config: Any, geometry: Dict[str, Any]) -> None:
    """Write config.yaml and geometry.pkl alongside a run's dumps.

    Downstream preprocessing/inference otherwise reconstructs a trajectory from
    GKW's ``input.dat`` (physical parameters) and ``geom.dat`` (geometry), which
    gyaradax runs do not produce. This records the equivalent information: the
    run configuration as ``config.yaml`` and the full geometry dict as
    ``geometry.pkl`` (jax arrays converted to numpy, mirroring how the dataset
    metadata stores ``geometry``).
    """
    import pickle

    from omegaconf import OmegaConf

    os.makedirs(output_dir, exist_ok=True)
    if config is not None:
        cfg = config if OmegaConf.is_config(config) else OmegaConf.create(config)
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

    np_geom: Dict[str, Any] = {}
    for key, value in geometry.items():
        try:
            np_geom[key] = np.asarray(value)
        except Exception:
            np_geom[key] = value  # keep non-array metadata (labels, classes) as-is
    with open(os.path.join(output_dir, "geometry.pkl"), "wb") as handle:
        pickle.dump(np_geom, handle)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a .npz checkpoint into a dictionary of arrays."""
    with np.load(path) as data:
        return {k: jnp.array(v) for k, v in data.items()}


def print_params(params, grid_shape=None):
    """Pretty-print GKParams and optional grid shape."""
    d = vars(params)

    solver_keys = [
        "dt",
        "naverage",
        "non_linear",
        "adaptive_dt",
        "cfl_safety",
        "finit",
        "adiabatic_electrons",
        "mixed_precision",
    ]
    dissipation_keys = ["disp_par", "disp_vp", "disp_x", "disp_y", "idisp"]
    species_keys = ["rlt", "rln", "mas", "tmp", "de", "signz", "vthrat"]
    geometry_keys = ["shat", "q", "eps", "kthnorm", "Rref", "d2X", "signB"]
    grid_keys = ["dvp", "sgr_dist", "kxmax", "kymax", "dgrid", "tgrid"]

    def _fmt(v):
        if hasattr(v, "shape") and v.shape:
            return "[" + ", ".join(f"{float(x):.6g}" for x in np.asarray(v).flat) + "]"
        elif isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _section(title, keys):
        print(f"  {title}:")
        for k in keys:
            if k in d:
                print(f"    {k:<24s} {_fmt(d[k])}")

    _section("solver", solver_keys)
    _section("dissipation", dissipation_keys)
    _section("species", species_keys)
    _section("geometry", geometry_keys)
    _section("grid", grid_keys)

    if grid_shape is not None:
        labels = ["nvpar", "nmu", "ns", "nkx", "nky"]
        if len(grid_shape) == 6:
            labels = ["nsp"] + labels
        dims = ", ".join(f"{name}={s}" for name, s in zip(labels, grid_shape))
        print(f"  grid shape: {dims}")

    print("=" * 88)
