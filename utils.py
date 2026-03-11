import os
import numpy as np
import jax.numpy as jnp
from typing import Tuple, List

def load_gkw_k_dump(file_path: str, resolution: Tuple[int, ...]) -> jnp.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GKW dump not found: {file_path}")
    with open(file_path, "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)
    
    nvpar, nmu, ns, nkx, nky = resolution
    
    # Direct reshape into Project Mandate order with interleaved complex
    knth = np.reshape(ff, (2, nvpar, nmu, ns, nkx, nky), order="F")
    df = knth[0] + 1j * knth[1]
    
    return jnp.array(df, dtype=jnp.complex128)

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
