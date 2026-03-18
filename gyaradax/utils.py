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


def read_gkw_dump_dtim(dat_path: str) -> float:
    """Read timestep DTIM from a GKW .dat file."""
    if not os.path.exists(dat_path):
        return 0.0
    with open(dat_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"DTIM\s*=\s*([0-9eE+\-.]+)", text)
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


def load_gkw_k_dump(file_path: str, resolution: Tuple[int, ...], n_species: int = 1) -> jnp.ndarray:
    """Legacy wrapper for backward compatibility."""
    df, _ = load_gkw_dump(file_path, resolution, n_species=n_species)
    return df


def K_files(directory):
    """List distribution function files in a directory."""
    files = os.listdir(directory)
    digit_files = sorted([file for file in files if file.isdigit()], key=lambda x: int(x))
    k_files = sorted([file for file in files if file.startswith("K") and not file.endswith(".dat")])
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

    # Spectra use field-line-averaged |phi|^2, matching GKW conventions:
    #   ky_spec: per-mode spectral density = ds * sum_{s,kx} |phi|^2
    #   kx_spec: total per kx = ds * sum_{s,ky} parseval_ky * |phi|^2
    #     where parseval_ky = [1, 2, 2, ...] (one-sided Parseval for real fields)
    ds = float(jnp.asarray(geometry["ints"])[0])
    nky = phi.shape[-1]
    parseval_ky = jnp.array([1.0] + [2.0] * (nky - 1))
    phi_sq = jnp.abs(phi) ** 2
    ky_spec = jnp.sum(ds * phi_sq, axis=(0, 1))
    kx_spec = jnp.sum(ds * phi_sq * parseval_ky[None, None, :], axis=(0, 2))

    # fluxes: tuple of 3 scalars (adiabatic) or (nsp, 3) array (kinetic)
    fluxes_arr = np.asarray(fluxes)
    if fluxes_arr.ndim == 0 or (fluxes_arr.ndim == 1 and fluxes_arr.shape[0] != 3):
        # tuple of scalars -> (3,)
        fluxes_arr = np.array([fluxes[0], fluxes[1], fluxes[2]])

    diags = {
        "fluxes": fluxes_arr,
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
                            k: np.append(data[k][mask], [new_data[k]], axis=0) for k in data.files
                        }
                    else:
                        # Fallback for legacy files without 'step'
                        updated = {k: np.append(data[k], [new_data[k]], axis=0) for k in data.files}
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
            "fluxes": fluxes_arr,
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


def print_params(params, grid_shape=None):
    """Pretty-print GKParams and optional grid shape."""
    d = vars(params)

    # group fields
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


# --- file-loading utilities (moved from geometry.py) ---


def is_number(string):
    pattern = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
    return bool(re.fullmatch(pattern, string.strip()))


def _strip_inline_comment(line: str) -> str:
    """Strip Fortran '!' comments while respecting quoted strings."""
    out = []
    quote = None
    for ch in line:
        if quote is None and ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if quote is not None and ch == quote:
            quote = None
            out.append(ch)
            continue
        if quote is None and ch == "!":
            break
        out.append(ch)
    return "".join(out).strip()


def _split_top_level_commas(text: str):
    """Split comma-separated assignments, ignoring commas inside quotes."""
    chunks = []
    buf = []
    quote = None
    for ch in text:
        if quote is None and ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if quote is not None and ch == quote:
            quote = None
            buf.append(ch)
            continue
        if quote is None and ch == ",":
            chunk = "".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            buf = []
            continue
        buf.append(ch)
    chunk = "".join(buf).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _parse_namelist_value(value: str):
    """Parse Fortran-namelist-like scalar values into Python scalars."""
    v = value.strip().rstrip(",")
    if not v:
        return ""

    if (v.startswith("'") and v.endswith("'")) or (
        v.startswith('"') and v.endswith('"')
    ):
        return v[1:-1]

    lv = v.lower()
    if lv in (".true.", "true", "t"):
        return True
    if lv in (".false.", "false", "f"):
        return False

    # handle fortran double-exponent notation 1.0d+00
    num = v.replace("D", "e").replace("d", "e")
    if re.fullmatch(r"[+-]?\d+", num):
        try:
            return int(num)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", num):
        try:
            return float(num)
        except ValueError:
            pass
    return v


def load_geom_dat_file(file_path):
    """Load geometric parameters from a .dat file."""
    data = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    key = None
    values = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 1 and not is_number(parts[0]):
            try:
                if len(values) == 0:
                    values.extend(map(float, parts))
                    data[key] = values[0]
                    key = None
                    values = []
                    continue
                else:
                    raise ValueError
            except Exception:
                if key is not None:
                    data[key] = np.array(values, dtype=np.float64)
                key = parts[0]
                values = []
        else:
            values.extend(map(float, parts))

    if key is not None:
        data[key] = np.array(values, dtype=np.float64)

    return data


def parse_input_dat(file_path):
    """Parse GKW input.dat configuration file."""
    parsed_data: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(file_path):
        return parsed_data

    current_section = None
    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = _strip_inline_comment(raw_line)
            if not line:
                continue

            if line.startswith("&"):
                section = line[1:].strip().lower()
                while section in parsed_data:
                    section = f"{section}0"
                parsed_data[section] = {}
                current_section = section
                continue

            if line.startswith("/"):
                current_section = None
                continue

            if current_section is None:
                continue

            for assignment in _split_top_level_commas(line):
                if "=" not in assignment:
                    continue
                key, value = assignment.split("=", 1)
                key = key.strip().lower()
                parsed_data[current_section][key] = _parse_namelist_value(value)

    return parsed_data


def load_runtime_params(input_dat_path: str) -> Dict[str, Any]:
    """
    Load runtime controls for solver parity from `input.dat`.

    Returned keys are typed scalars and can be fed into GKParams creation.
    """
    inp = parse_input_dat(input_dat_path)
    control = inp.get("control", {})

    def _flt(name, default):
        val = control.get(name, default)
        return float(val) if val is not None else float(default)

    def _int(name, default):
        val = control.get(name, default)
        return int(val) if val is not None else int(default)

    def _bool(name, default):
        val = control.get(name, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            lv = val.strip().lower()
            if lv in (".true.", "true", "t"):
                return True
            if lv in (".false.", "false", "f"):
                return False
        return bool(default)

    method_val = control.get("method", "EXP")
    method = str(method_val).strip().strip("'").strip('"').upper()

    finit = inp.get("spcgeneral", {}).get("finit", "cosine2")
    if not finit:
        finit = inp.get("components", {}).get("finit", "cosine2")

    # adiabatic_electrons can appear in gridsize or spcgeneral depending on GKW version
    ae_val = inp.get("gridsize", {}).get("adiabatic_electrons")
    if ae_val is None:
        ae_val = inp.get("spcgeneral", {}).get("adiabatic_electrons", True)
    adiabatic_electrons = bool(ae_val)

    return {
        "dtim": _flt("dtim", 0.01),
        "naverage": _int("naverage", 40),
        "disp_par": _flt("disp_par", 1.0),
        "disp_vp": _flt("disp_vp", 0.2),
        "disp_x": _flt("disp_x", 0.1),
        "disp_y": _flt("disp_y", 0.1),
        "non_linear": _bool("non_linear", False),
        "nlapar": _bool("nlapar", False),
        "method": method,
        "meth": _int("meth", 0),
        "finit": finit,
        "adiabatic_electrons": adiabatic_electrons,
    }


def load_scalars(directory: str) -> Dict[str, Any]:
    """
    Extract only the scalar configuration and physics parameters from GKW files.

    This is a lightweight alternative to load_geometry, returning only the
    scalars needed for GKParams and YAML configuration.
    """
    geom = load_geom_dat_file(os.path.join(directory, "geom.dat"))
    input_data = parse_input_dat(os.path.join(directory, "input.dat"))

    # 1. extract runtime/solver params
    runtime = load_runtime_params(os.path.join(directory, "input.dat"))

    # 2. extract geometry scalars
    def _scalar(key, default=0.0):
        return float(np.asarray(geom.get(key, default)).item())

    scalars = {
        "shat": _scalar("shat", 0.0),
        "q": _scalar("q", 1.0),
        "eps": _scalar("eps", 0.0),
        "kthnorm": _scalar("kthnorm", 1.0),
        "Rref": abs(_scalar("Rref", 1.0)),
        "d2X": 1.0,
        "signB": 1.0,
    }

    # 3. extract species info (all kinetic species)
    num_sp = input_data.get("gridsize", {}).get("number_of_species", 1)
    species_keys = [k for k in input_data.keys() if k.startswith("species")][:num_sp]
    if species_keys:
        sp_mas = np.array([float(input_data[k].get("mass", 1.0)) for k in species_keys])
        sp_tmp = np.array([float(input_data[k].get("temp", 1.0)) for k in species_keys])
        sp_de = np.array([float(input_data[k].get("dens", 1.0)) for k in species_keys])
        sp_signz = np.array([float(input_data[k].get("z", 1.0)) for k in species_keys])
        sp_rlt = np.array([float(input_data[k].get("rlt", 0.0)) for k in species_keys])
        sp_rln = np.array([float(input_data[k].get("rln", 0.0)) for k in species_keys])
        sp_vthrat = np.sqrt(sp_tmp / sp_mas)

        def _maybe_scalar(arr):
            return float(arr[0]) if len(arr) == 1 else arr

        scalars.update(
            {
                "mas": _maybe_scalar(sp_mas),
                "tmp": _maybe_scalar(sp_tmp),
                "de": _maybe_scalar(sp_de),
                "signz": _maybe_scalar(sp_signz),
                "rlt": _maybe_scalar(sp_rlt),
                "rln": _maybe_scalar(sp_rln),
                "vthrat": _maybe_scalar(sp_vthrat),
            }
        )
    else:
        scalars.update(
            {
                "mas": 1.0,
                "tmp": 1.0,
                "de": 1.0,
                "signz": 1.0,
                "rlt": 1.0,
                "rln": 1.0,
                "vthrat": 1.0,
            }
        )

    # 4. extract grid/scaling info
    kxrh = np.loadtxt(os.path.join(directory, "kxrh"))
    if kxrh.ndim > 1:
        kxrh = kxrh[0]
    scalars["kxmax"] = float(np.max(np.abs(kxrh)))

    krho = np.loadtxt(os.path.join(directory, "krho"))
    if krho.ndim > 1:
        krho = krho.T[0]
    scalars["kymax"] = float(np.max(np.abs(krho / scalars["kthnorm"])))

    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))
    if vpgr.ndim > 1:
        vpgr = vpgr[0]
    scalars["dvp"] = float(np.mean(np.diff(vpgr))) if len(vpgr) > 1 else 1.0

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    scalars["sgr_dist"] = float(np.abs(sgrid[1] - sgrid[0])) if len(sgrid) > 1 else 1.0

    scalars["dgrid"] = 1.0
    if os.path.exists(os.path.join(directory, "dgrid.dat")):
        dg = np.loadtxt(os.path.join(directory, "dgrid.dat"))
        scalars["dgrid"] = float(np.asarray(dg).reshape(-1)[0])
    elif "dgrid" in geom:
        scalars["dgrid"] = float(np.asarray(geom["dgrid"]).reshape(-1)[0])

    scalars["tgrid"] = 1.0
    if os.path.exists(os.path.join(directory, "tgrid.dat")):
        tg = np.loadtxt(os.path.join(directory, "tgrid.dat"))
        scalars["tgrid"] = float(np.asarray(tg).reshape(-1)[0])
    elif "tgrid" in geom:
        scalars["tgrid"] = float(np.asarray(geom["tgrid"]).reshape(-1)[0])

    # merge with runtime
    scalars.update(runtime)
    return scalars


def load_geometry(directory):
    """Load geometry and physics parameters into JAX arrays."""
    from gyaradax.geometry import (
        _build_mode_connectivity,
        _build_pos_par_grid_classes,
        _build_parallel_shift_maps,
    )

    geom = load_geom_dat_file(os.path.join(directory, "geom.dat"))
    input_data = parse_input_dat(os.path.join(directory, "input.dat"))

    geometry = {}

    # scalar geometry controls
    if "kthnorm" in geom:
        geometry["kthnorm"] = jnp.array(
            float(np.asarray(geom["kthnorm"]).reshape(-1)[0]), dtype=jnp.float64
        )
    if "shat" in geom:
        geometry["shat"] = jnp.array(
            float(np.asarray(geom["shat"]).reshape(-1)[0]), dtype=jnp.float64
        )
    if "q" in geom:
        geometry["q"] = jnp.array(
            float(np.asarray(geom["q"]).reshape(-1)[0]), dtype=jnp.float64
        )
    if "eps" in geom:
        geometry["eps"] = jnp.array(
            float(np.asarray(geom["eps"]).reshape(-1)[0]), dtype=jnp.float64
        )

    # grids
    kxrh = np.loadtxt(os.path.join(directory, "kxrh"))
    if kxrh.ndim > 1:
        kxrh = kxrh[0]
    geometry["kxrh"] = jnp.array(kxrh, dtype=jnp.float64)

    krho = np.loadtxt(os.path.join(directory, "krho"))
    if krho.ndim > 1:
        krho = krho.T[0]
    kthnorm = (
        float(np.asarray(geom["kthnorm"]).reshape(-1)[0]) if "kthnorm" in geom else 1.0
    )
    geometry["krho"] = jnp.array(krho / kthnorm, dtype=jnp.float64)

    geometry["parseval"] = jnp.array(
        [1.0] + [float(len(geometry["krho"]))] * (len(geometry["krho"]) - 1),
        dtype=jnp.float64,
    )

    # velocity space
    intvp = np.loadtxt(os.path.join(directory, "intvp.dat"))
    if intvp.ndim > 1:
        intvp = intvp[0]
    geometry["intvp"] = jnp.array(intvp, dtype=jnp.float64)

    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))
    if vpgr.ndim > 1:
        vpgr = vpgr[0]
    geometry["vpgr"] = jnp.array(vpgr, dtype=jnp.float64)
    geometry["vpgr_rms"] = jnp.array(
        float(np.sqrt(np.mean(vpgr**2))), dtype=jnp.float64
    )
    if len(vpgr) > 1:
        geometry["dvp"] = jnp.array(float(np.mean(np.diff(vpgr))), dtype=jnp.float64)
    else:
        geometry["dvp"] = jnp.array(1.0, dtype=jnp.float64)

    if os.path.exists(os.path.join(directory, "intmu.dat")):
        intmu = np.loadtxt(os.path.join(directory, "intmu.dat"))
        if intmu.ndim == 2:
            intmu = intmu[:, 0]
        geometry["intmu"] = jnp.array(intmu, dtype=jnp.float64)

    if os.path.exists(os.path.join(directory, "vperp.dat")):
        vperp = np.loadtxt(os.path.join(directory, "vperp.dat"))
        if vperp.ndim == 2:
            vperp = vperp[:, 0]
        geometry["mugr"] = jnp.array(vperp**2 / 2.0, dtype=jnp.float64)
        geometry["mugr_rms"] = jnp.array(
            float(np.sqrt(np.mean((vperp**2 / 2.0) ** 2))),
            dtype=jnp.float64,
        )

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    ints = np.concatenate([np.array([0.0]), np.diff(sgrid)])
    ints[0] = ints[1]
    geometry["ints"] = jnp.array(ints, dtype=jnp.float64)
    geometry["sgrid"] = jnp.array(sgrid, dtype=jnp.float64)
    if len(sgrid) > 1:
        geometry["sgr_dist"] = jnp.array(
            float(np.abs(sgrid[1] - sgrid[0])), dtype=jnp.float64
        )
    else:
        geometry["sgr_dist"] = jnp.array(1.0, dtype=jnp.float64)

    # physics constants defaults
    geometry["Rref"] = jnp.array(jnp.abs(geom["Rref"]), dtype=jnp.float64)
    geometry["signz"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["tmp"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["mas"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["de"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["vthrat"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["rlt"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["rln"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["d2X"] = jnp.array(1.0, dtype=jnp.float64)
    geometry["signB"] = jnp.array(1.0, dtype=jnp.float64)

    # load species info
    num_sp = input_data.get("gridsize", {}).get("number_of_species", 1)
    species_keys = [k for k in input_data.keys() if k.startswith("species")][:num_sp]
    if species_keys:
        mas, tmp, de, signz, rlt, rln = [], [], [], [], [], []
        for k in species_keys:
            sp = input_data[k]
            mas.append(sp.get("mass", 1.0))
            tmp.append(sp.get("temp", 1.0))
            de.append(sp.get("dens", 1.0))
            signz.append(sp.get("z", 1.0))
            rlt.append(sp.get("rlt", 0.0))
            rln.append(sp.get("rln", 0.0))

        geometry["mas"] = jnp.array(mas, dtype=jnp.float64)
        geometry["tmp"] = jnp.array(tmp, dtype=jnp.float64)
        geometry["de"] = jnp.array(de, dtype=jnp.float64)
        geometry["signz"] = jnp.array(signz, dtype=jnp.float64)
        geometry["rlt"] = jnp.array(rlt, dtype=jnp.float64)
        geometry["rln"] = jnp.array(rln, dtype=jnp.float64)
        geometry["vthrat"] = jnp.sqrt(geometry["tmp"] / geometry["mas"])

    # geometry arrays
    geometry["bn"] = jnp.array(geom["bn"], dtype=jnp.float64)
    geometry["ffun"] = jnp.array(geom["F"], dtype=jnp.float64)
    if "G" in geom:
        geometry["gfun"] = jnp.array(geom["G"], dtype=jnp.float64)
    geometry["bt_frac"] = jnp.array(geom["Bt_frac"], dtype=jnp.float64)
    geometry["rfun"] = jnp.array(geom["R"], dtype=jnp.float64)
    geometry["little_g"] = jnp.array(
        np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1),
        dtype=jnp.float64,
    )

    # drift functions
    if "D_eps" in geom:
        geometry["dfun"] = jnp.array(
            np.stack([geom["D_eps"], geom["D_zeta"], geom["D_s"]], -1),
            dtype=jnp.float64,
        )
    if "H_eps" in geom:
        geometry["hfun"] = jnp.array(
            np.stack([geom["H_eps"], geom["H_zeta"], geom["H_s"]], -1),
            dtype=jnp.float64,
        )
    if "I_eps" in geom:
        geometry["ifun"] = jnp.array(
            np.stack([geom["I_eps"], geom["I_zeta"], geom["I_s"]], -1),
            dtype=jnp.float64,
        )

    # ExB function
    if "E_eps_zeta" in geom:
        geometry["efun"] = jnp.array(-geom["E_eps_zeta"], dtype=jnp.float64)

    # spectral connectivity metadata for open-parallel boundary stencils
    mode_label_path = os.path.join(directory, "mode_label")
    if os.path.exists(mode_label_path):
        mode_label = np.loadtxt(mode_label_path)
        mode_label_kxky, ixplus, ixminus, ixzero, iyzero = _build_mode_connectivity(
            mode_label, kxrh, np.asarray(geometry["krho"])
        )
        pos_classes = _build_pos_par_grid_classes(ixplus, ixminus, len(sgrid))
        s_shift, kx_shift, valid_shift = _build_parallel_shift_maps(
            ixplus, ixminus, iyzero, len(sgrid), max_shift=4
        )

        geometry["mode_label"] = jnp.array(mode_label_kxky, dtype=jnp.int32)
        geometry["ixplus"] = jnp.array(ixplus, dtype=jnp.int32)
        geometry["ixminus"] = jnp.array(ixminus, dtype=jnp.int32)
        geometry["ixzero"] = jnp.array(ixzero, dtype=jnp.int32)
        geometry["iyzero"] = jnp.array(iyzero, dtype=jnp.int32)
        geometry["pos_par_grid_class"] = jnp.array(pos_classes, dtype=jnp.int8)
        geometry["s_shift"] = jnp.array(s_shift, dtype=jnp.int32)
        geometry["kx_shift"] = jnp.array(kx_shift, dtype=jnp.int32)
        geometry["valid_shift"] = jnp.array(valid_shift, dtype=jnp.bool_)

    geometry["kxmax"] = jnp.array(float(np.max(np.abs(kxrh))), dtype=jnp.float64)
    geometry["kymax"] = jnp.array(
        float(np.max(np.abs(np.asarray(geometry["krho"])))), dtype=jnp.float64
    )

    return geometry
