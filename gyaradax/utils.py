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
    file_path: str, resolution: Tuple[int, ...], n_species: int = 1
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Load a GKW distribution function and associated metadata (.dat).

    Args:
        file_path: Path to the binary dump file.
        resolution: Grid shape (nvpar, nmu, ns, nkx, nky).
        n_species: Number of kinetic species stored in the dump.
            For adiabatic electrons (default), n_species=1.
            For kinetic electrons, n_species=2 (ions + electrons).

    Returns:
        (df, info_dict) where df has shape:
            (nvpar, nmu, ns, nkx, nky) when n_species=1
            (n_species, nvpar, nmu, ns, nkx, nky) when n_species>1
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GKW dump not found: {file_path}")

    # 1. Load distribution function
    with open(file_path, "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)
    nvpar, nmu, ns, nkx, nky = resolution

    if n_species == 1:
        knth = np.reshape(ff, (2, nvpar, nmu, ns, nkx, nky), order="F")
        df = jnp.array(knth[0] + 1j * knth[1], dtype=jnp.complex128)
    else:
        # GKW stores species as the outermost (slowest) Fortran index.
        # Binary layout: (2_re_im, nvpar, nmu, ns, nkx, nky, nspecies) Fortran order.
        knth = np.reshape(ff, (2, nvpar, nmu, ns, nkx, nky, n_species), order="F")
        # Combine real/imag and move species to leading axis
        df_np = knth[0] + 1j * knth[1]  # (nvpar, nmu, ns, nkx, nky, nspecies)
        df = jnp.array(
            np.moveaxis(df_np, -1, 0), dtype=jnp.complex128
        )  # (nspecies, nvpar, nmu, ns, nkx, nky)

    # 2. Load side info
    info = {"path": file_path, "time": 0.0}
    dat_path = file_path + ".dat"
    if os.path.exists(dat_path):
        info["time"] = read_gkw_dump_time(dat_path)

    return df, info


def load_gkw_k_dump(
    file_path: str, resolution: Tuple[int, ...], n_species: int = 1
) -> jnp.ndarray:
    """Legacy wrapper for backward compatibility."""
    df, _ = load_gkw_dump(file_path, resolution, n_species=n_species)
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

    # compute spectra with proper ds and parseval weighting
    ds = float(jnp.asarray(geometry["ints"])[0])
    parseval = jnp.asarray(geometry["parseval"])
    phi_sq = jnp.abs(phi) ** 2
    weighted = ds * phi_sq * parseval[None, None, :]
    kx_spec = jnp.sum(weighted, axis=(0, 2))
    ky_spec = jnp.sum(weighted, axis=(0, 1))

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
        current_step = int(state.step)
        if os.path.exists(path):
            try:
                with np.load(path) as data:
                    # Use 'step' to truncate entries strictly before the current one.
                    # This prevents overlapping history when resuming simulations.
                    if "step" in data.files:
                        mask = data["step"] < current_step
                        updated = {
                            k: np.append(data[k][mask], [new_data[k]], axis=0)
                            for k in data.files
                        }
                    else:
                        # Fallback for legacy files without 'step'
                        updated = {
                            k: np.append(data[k], [new_data[k]], axis=0)
                            for k in data.files
                        }
            except (IOError, ValueError):
                # If file is corrupted or incompatible, start fresh
                updated = {k: np.array([v]) for k, v in new_data.items()}
        else:
            # Create new file with first entry
            updated = {k: np.array([v]) for k, v in new_data.items()}
        np.savez(path, **updated)

    # Note: We group these to avoid too many small files
    # but the user requested "fluxes.npz, kyspec.npz, kxspec.npz, growth.npz"
    # We now include step and time in every file for self-description and safe appending.
    common = {"step": diags["step"], "time": diags["time"]}

    _append_to_npz("fluxes.npz", {"fluxes": diags["fluxes"], **common})
    _append_to_npz("kyspec.npz", {"ky_spec": diags["ky_spec"], **common})
    _append_to_npz("kxspec.npz", {"kx_spec": diags["kx_spec"], **common})
    _append_to_npz("growth.npz", {"growth": diags["growth"], **common})

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
