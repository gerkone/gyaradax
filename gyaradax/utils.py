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


def save_checkpoint(
    path: str,
    df: jnp.ndarray,
    phi: jnp.ndarray,
    fluxes: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    state: Any,
    geometry: Dict[str, jnp.ndarray],
    **extra_data,
):
    """
    Save a full simulation snapshot to a JAX-friendly .npz file.

    Target variables include distribution functions, potential, fluxes,
    and diagnostic metadata.
    """
    # ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # compute spectra for diagnostic convenience
    # phi is [ns, nkx, nky]
    phi_sq = jnp.abs(phi) ** 2
    # kx spectrum: sum over s and ky
    kx_spec = jnp.sum(phi_sq, axis=(0, 2))
    # ky spectrum: sum over s and kx
    ky_spec = jnp.sum(phi_sq, axis=(0, 1))

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
        "kx_spec": np.array(kx_spec),
        "ky_spec": np.array(ky_spec),
    }

    # add any extra diagnostics provided by the user
    for k, v in extra_data.items():
        checkpoint[k] = np.array(v)

    np.savez(path, **checkpoint)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a .npz checkpoint into a dictionary of arrays."""
    with np.load(path) as data:
        return {k: jnp.array(v) for k, v in data.items()}
