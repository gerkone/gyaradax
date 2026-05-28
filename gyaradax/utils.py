import os
import re
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Any, cast


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
        # GKW Fortran layout: species is the outermost (slowest) index;
        # (2_re_im, nvpar, nmu, ns, nkx, nky, nspecies) -> move species to leading axis.
        knth = np.reshape(ff, (2, nvpar, nmu, ns, nkx, nky, n_species), order="F")
        df_np = knth[0] + 1j * knth[1]
        df = jnp.array(np.moveaxis(df_np, -1, 0), dtype=jnp.complex128)

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


def _compute_em_fluxes_arr(df, geometry, params, pre, fluxes_shape):
    """Compute magnetic-flutter EM fluxes as a numpy array shaped like the ES
    fluxes array. Returns zeros when nlapar is off, params/pre missing, or
    apar/bpar are unavailable.
    """
    zero = np.zeros(fluxes_shape, dtype=np.float64)
    if params is None or pre is None or not getattr(params, "nlapar", False):
        return zero
    # solver imports here to avoid a top-level cycle
    from gyaradax.solver import _compute_fields, g_to_f
    from gyaradax.integrals import calculate_em_fluxes

    _, apar, bpar = _compute_fields(df, geometry, params, pre)
    if apar is None:
        return zero
    df_f = g_to_f(df, apar, params, pre)
    res = calculate_em_fluxes(geometry, df_f, apar, params=params, bpar=bpar, pre=pre)
    if len(fluxes_shape) == 1:  # (3,)
        em_p, em_e, em_v = (np.asarray(r, dtype=np.float64) for r in res)
        out = np.zeros(fluxes_shape, dtype=np.float64)
        out[0] = float(em_p)
        out[1] = float(em_e)
        out[2] = float(em_v)
    else:  # 6D path returns (nsp, 3)
        arr = np.asarray(res, dtype=np.float64)
        out = np.zeros(fluxes_shape, dtype=np.float64)
        out[..., :3] = arr
    return out


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

    # spectra use field-line-averaged |phi|^2 (GKW convention):
    #   ky_spec = ds * sum_{s,kx} |phi|^2
    #   kx_spec = ds * sum_{s,ky} parseval_ky * |phi|^2,  parseval_ky = [1, 2, 2, ...]
    ds = float(jnp.asarray(geometry["ints"])[0])
    nky = phi.shape[-1]
    parseval_ky = jnp.array([1.0] + [2.0] * (nky - 1))
    phi_sq = jnp.abs(phi) ** 2
    ky_spec = jnp.sum(ds * phi_sq, axis=(0, 1))
    kx_spec = jnp.sum(ds * phi_sq * parseval_ky[None, None, :], axis=(0, 2))

    fluxes_arr = np.asarray(fluxes)
    if fluxes_arr.ndim == 0 or (fluxes_arr.ndim == 1 and fluxes_arr.shape[0] != 3):
        fluxes_arr = np.array([fluxes[0], fluxes[1], fluxes[2]])

    # em fluxes shape matches ES: (3,) for 5D df, (nsp, 3) for 6D. vflux slot is
    # zero -- EM momentum flux is not tracked by calculate_em_fluxes.
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
        checkpoint: dict[str, Any] = {
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
        # record last dt so resume can warm-start the CFL controller from the
        # saturated state instead of params.dt.
        if last_dt is not None:
            checkpoint["dt_last"] = np.array(last_dt, dtype=np.float64)
        np.savez(path, **cast(Any, checkpoint))


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

    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
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


def load_geom_dat_file(file_path: str) -> Dict[str, Any]:
    """Load geometric parameters from a .dat file."""
    data: dict[str, Any] = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    key: str | None = None
    values: list[float] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 1 and not is_number(parts[0]):
            try:
                if len(values) == 0:
                    values.extend(map(float, parts))
                    if key is not None:
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

    # collisions namelist parsing
    coll = inp.get("collisions", {})

    def _coll_bool(name, default):
        val = coll.get(name, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            lv = val.strip().lower()
            if lv in (".true.", "true", "t"):
                return True
            if lv in (".false.", "false", "f"):
                return False
        return bool(default)

    collisions_on = _bool("collisions", False)

    return {
        "dtim": _flt("dtim", 0.01),
        "naverage": _int("naverage", 40),
        "disp_par": _flt("disp_par", 1.0),
        "disp_vp": _flt("disp_vp", 0.2),
        "disp_x": _flt("disp_x", 0.0),
        "disp_y": _flt("disp_y", 0.0),
        "non_linear": _bool("non_linear", False),
        "nlapar": _bool("nlapar", False),
        "nlbpar": _bool("nlbpar", False),
        "beta": float(
            inp.get("spcgeneral", {}).get(
                "beta",
                inp.get("spcgeneral", {}).get("beta_ref", 0.0),
            )
        ),
        "method": method,
        "meth": _int("meth", 0),
        "finit": finit,
        "adiabatic_electrons": adiabatic_electrons,
        "amp_init": float(
            inp.get("spcgeneral", {}).get(
                "amp_init",
                inp.get("components", {}).get("amp_init", 1.0e-3),
            )
        ),
        "collisions": collisions_on,
        "coll_pitch_angle": _coll_bool("pitch_angle", True),
        "coll_en_scatter": _coll_bool("en_scatter", True),
        "coll_friction": _coll_bool("friction_coll", True),
        "coll_freq": float(coll.get("coll_freq", 0.0)),
        "coll_freq_override": _coll_bool("freq_override", True),
        "coll_mass_conserve": _coll_bool("mass_conserve", True),
        "coll_mom_conservation": _coll_bool("mom_conservation", False),
        "coll_ene_conservation": _coll_bool("ene_conservation", False),
        "coll_rref": float(coll.get("rref", 1.0)),
        "coll_tref": float(coll.get("tref", 1.0)),
        "coll_nref": float(coll.get("nref", 1.0)),
    }


def load_scalars(directory: str) -> Dict[str, Any]:
    """Extract scalar config and physics parameters from GKW files.

    Lightweight alternative to load_geometry: returns only the scalars
    needed for GKParams and YAML configuration.
    """
    geom = load_geom_dat_file(os.path.join(directory, "geom.dat"))
    input_data = parse_input_dat(os.path.join(directory, "input.dat"))
    runtime = load_runtime_params(os.path.join(directory, "input.dat"))

    def _scalar(key, default=0.0):
        return float(np.asarray(geom.get(key, default)).item())

    scalars: dict[str, Any] = {
        "shat": _scalar("shat", 0.0),
        "q": _scalar("q", 1.0),
        "eps": _scalar("eps", 0.0),
        "kthnorm": _scalar("kthnorm", 1.0),
        "Rref": abs(_scalar("Rref", 1.0)),
        "d2X": 1.0,
        "signB": 1.0,
    }

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

    scalars.update(runtime)
    return scalars


def load_geometry(directory):
    """Load geometry and physics parameters into JAX arrays."""
    from gyaradax.geometry.loaded import load_loaded_geometry

    return load_loaded_geometry(directory)


def pack_half_spectrum(
    spec_kxky: jnp.ndarray, jind: jnp.ndarray, mrad: int, mphiw3: int
) -> jnp.ndarray:
    out_shape = spec_kxky.shape[:-2] + (mrad, mphiw3)
    out = jnp.zeros(out_shape, dtype=jnp.complex128)
    nky = spec_kxky.shape[-1]
    return out.at[..., jind, :nky].set(spec_kxky)


def unpack_half_spectrum(spec_half: jnp.ndarray, jind: jnp.ndarray, nky: int) -> jnp.ndarray:
    return spec_half[..., jind, :nky]
