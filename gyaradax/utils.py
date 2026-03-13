import os
import re
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Any


def read_gkw_dump_time(dat_path: str) -> float:
    """Read simulation time from a GKW .dat file."""
    if not os.path.exists(dat_path):
        return 0.0
    with open(dat_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"TIME\s*=\s*([0-9eE+\-.]+)", text)
    if m is None:
        return 0.0
    return float(m.group(1))


def load_gkw_dump(
    file_path: str, resolution: Tuple[int, ...]
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Load a GKW distribution function and associated metadata (.dat).
    Returns (df, info_dict).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GKW dump not found: {file_path}")

    # 1. Load distribution function
    with open(file_path, "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)
    nvpar, nmu, ns, nkx, nky = resolution
    knth = np.reshape(ff, (2, nvpar, nmu, ns, nkx, nky), order="F")
    df = jnp.array(knth[0] + 1j * knth[1], dtype=jnp.complex128)

    # 2. Load side info
    info = {"path": file_path, "time": 0.0}
    dat_path = file_path + ".dat"
    if os.path.exists(dat_path):
        info["time"] = read_gkw_dump_time(dat_path)

    return df, info


def load_gkw_k_dump(file_path: str, resolution: Tuple[int, ...]) -> jnp.ndarray:
    """Legacy wrapper for backward compatibility."""
    df, _ = load_gkw_dump(file_path, resolution)
    return df


def K_files(directory):
    """List distribution function files in a directory."""
    files = os.listdir(directory)
    digit_files = sorted(
        [file for file in files if file.isdigit()], key=lambda x: int(x)
    )
    k_files = sorted(
        [file for file in files if file.startswith("K") and not file.endswith(".dat")]
    )
    return k_files + digit_files


def poten_files(directory):
    """List potential field files in a directory."""
    files = os.listdir(directory)
    poten_files = sorted([file for file in files if file.startswith("Poten")])
    timestep_slices = [int(f.replace("Poten", "")) for f in poten_files]
    return poten_files, np.array(timestep_slices) - 1


def save_dumps(
    output_dir: str,
    df: jnp.ndarray,
    phi: jnp.ndarray,
    fluxes: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    state: Any,
    geometry: Dict[str, jnp.ndarray],
    save_dumps: bool = True,
):
    """
    Handle simulation output. Saves heavy 5D distribution snapshots if requested
    and appends diagnostic history to persistent files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Append diagnostics to history
    # Compute spectra
    phi_sq = jnp.abs(phi) ** 2
    kx_spec = jnp.sum(phi_sq, axis=(0, 2))
    ky_spec = jnp.sum(phi_sq, axis=(0, 1))

    diags = {
        "fluxes": np.array([fluxes[0], fluxes[1], fluxes[2]]),
        "kx_spec": np.array(kx_spec),
        "ky_spec": np.array(ky_spec),
        "time": np.array(state.time),
        "growth": np.array(state.last_growth_rate),
        "step": np.array(state.step),
    }

    # Internal helper to append data to an npz file
    def _append_to_npz(filename, new_data):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            with np.load(path) as data:
                # Load existing data and append new step
                updated = {
                    k: np.append(data[k], [new_data[k]], axis=0) for k in data.files
                }
        else:
            # Create new file with first entry
            updated = {k: np.array([v]) for k, v in new_data.items()}
        np.savez(path, **updated)

    # Note: We group these to avoid too many small files
    # but the user requested "fluxes.npz, kyspec.npz, kxspec.npz, growth.npz"
    _append_to_npz("fluxes.npz", {"fluxes": diags["fluxes"]})
    _append_to_npz("kyspec.npz", {"ky_spec": diags["ky_spec"]})
    _append_to_npz("kxspec.npz", {"kx_spec": diags["kx_spec"]})
    _append_to_npz(
        "growth.npz",
        {"growth": diags["growth"], "time": diags["time"], "step": diags["step"]},
    )

    # 2. Save heavy snapshot if flag is set
    if save_dumps:
        ckpt_name = f"step_{int(state.step):06d}.npz"
        path = os.path.join(output_dir, ckpt_name)
        checkpoint = {
            "df": np.array(df),
            "phi": np.array(phi),
            "pflux": np.array(fluxes[0]),
            "eflux": np.array(fluxes[1]),
            "vflux": np.array(fluxes[2]),
            "time": np.array(state.time),
            "step": np.array(state.step),
            "accumulated_norm_factor": np.array(state.accumulated_norm_factor),
            "window_start_amp": np.array(state.window_start_amp),
            "last_growth_rate": np.array(state.last_growth_rate),
            "kx_spec": diags["kx_spec"],
            "ky_spec": diags["ky_spec"],
        }
        np.savez(path, **checkpoint)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a .npz checkpoint into a dictionary of arrays."""
    with np.load(path) as data:
        return {k: jnp.array(v) for k, v in data.items()}
