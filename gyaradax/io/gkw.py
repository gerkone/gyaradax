"""GKW I/O: dump readers, input.dat / geom.dat parsing, GKW run directory -> YAML."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np

ParsedInput = Mapping[str, Mapping[str, Any]]

_TRUE_STRINGS = (".true.", "true", "t")
_FALSE_STRINGS = (".false.", "false", "f")


def as_float(value: Any, default: float) -> float:
    """Coerce a parsed namelist scalar to float, preserving default-on-None."""
    return float(default) if value is None else float(value)


def as_int(value: Any, default: int) -> int:
    """Coerce a parsed namelist scalar to int, preserving default-on-None."""
    return int(default) if value is None else int(value)


def as_bool(value: Any, default: bool) -> bool:
    """Coerce a parsed namelist scalar to bool using current GKW conventions.

    Parsed booleans are already real ``bool`` values.  String handling is kept
    for callers that pass raw or manually constructed parsed dictionaries.  An
    unrecognized non-bool value falls back to the supplied default, matching the
    local helper behavior previously used by ``load_runtime_params``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lv = value.strip().lower()
        if lv in _TRUE_STRINGS:
            return True
        if lv in _FALSE_STRINGS:
            return False
    return bool(default)


def species_section_keys(parsed: ParsedInput, limit: int | None = None) -> tuple[str, ...]:
    """Return repeated ``&SPECIES`` section keys in parser insertion order.

    ``parse_input_dat`` names repeated sections ``species``, ``species0``,
    ``species00``, ... by repeatedly appending ``0``.  This helper deliberately
    preserves that existing key-order quirk by iterating the parsed mapping in
    insertion order and filtering keys with the historical ``startswith`` rule.
    """
    keys = tuple(k for k in parsed if k.startswith("species"))
    return keys if limit is None else keys[:limit]


def species_blocks(parsed: ParsedInput, limit: int | None = None) -> tuple[Mapping[str, Any], ...]:
    """Return repeated ``&SPECIES`` blocks in parser insertion order."""
    return tuple(parsed[key] for key in species_section_keys(parsed, limit=limit))


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


def resolve_betaprime(inp: Dict[str, Any]) -> float:
    """Resolve GKW's veta_prime from a parsed input.dat dict.

    Mirrors the flux-tube fill in components.f90 (~1995-2016):
      betaprime_type='ref'  -> betaprime_ref (input, default 0)
      betaprime_type='sp'   -> veta * sum_species(-de*tmp*(rlt+rln))
                               with veta = beta; the sum runs over ALL
                               species blocks (incl. adiabatic ones).
      betaprime_type='eq'/'file' -> not supported here.
    """
    spc = inp.get("spcgeneral", {})
    bp_type = str(spc.get("betaprime_type", "ref")).strip().strip("'").strip('"').lower()
    bp_ref = as_float(spc.get("betaprime_ref", 0.0), 0.0)
    if bp_type == "ref":
        return bp_ref
    if bp_type == "sp":
        beta = as_float(spc.get("beta", spc.get("beta_ref", 0.0)), 0.0)
        species = species_blocks(inp)
        dum = 0.0
        for sp in species:
            de = as_float(sp.get("dens", 1.0), 1.0)
            tmp = as_float(sp.get("temp", 1.0), 1.0)
            rlt = as_float(sp.get("rlt", 0.0), 0.0)
            rln = as_float(sp.get("rln", 0.0), 0.0)
            dum -= de * tmp * (rlt + rln)
        return beta * dum
    raise NotImplementedError(
        f"betaprime_type='{bp_type}' is not supported (only 'ref' and 'sp')"
    )


def load_runtime_params(input_dat_path: str) -> Dict[str, Any]:
    """
    Load runtime controls for solver parity from `input.dat`.

    Returned keys are typed scalars and can be fed into GKParams creation.
    """
    inp = parse_input_dat(input_dat_path)
    control = inp.get("control", {})

    def _flt(name, default):
        return as_float(control.get(name, default), default)

    def _int(name, default):
        return as_int(control.get(name, default), default)

    def _bool(name, default):
        return as_bool(control.get(name, default), default)

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
        return as_bool(coll.get(name, default), default)

    collisions_on = _bool("collisions", False)

    return {
        "dtim": _flt("dtim", 0.01),
        "naverage": _int("naverage", 40),
        "nl_dtim_est": _bool("nl_dtim_est", True),
        "fac_dtim_est": _flt("fac_dtim_est", 0.95),
        "fac_dtim_nl": _flt("fac_dtim_nl", 1.0),
        "spectral_radius": _bool("spectral_radius", True),
        "disp_par": _flt("disp_par", 1.0),
        "disp_vp": _flt("disp_vp", 0.2),
        "disp_x": _flt("disp_x", 0.0),
        "disp_y": _flt("disp_y", 0.0),
        # gkw reads idisp from &linear_term_switches
        "idisp": as_int(inp.get("linear_term_switches", {}).get("idisp", control.get("idisp", 2)), 2),
        "ltrapping_arakawa": _bool("ltrapping_arakawa", False),
        "non_linear": _bool("non_linear", False),
        "nlapar": _bool("nlapar", False),
        "nlbpar": _bool("nlbpar", False),
        "beta": float(
            inp.get("spcgeneral", {}).get(
                "beta",
                inp.get("spcgeneral", {}).get("beta_ref", 0.0),
            )
        ),
        "betaprime": resolve_betaprime(inp),
        "drift_gradp_type": str(
            inp.get("spcgeneral", {}).get("drift_gradp_type", "full_drift")
        ).strip().strip("'").strip('"').lower(),
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
        # GKW defaults (collisionop.f90:150-151) are .false. for both; a stock
        # GKW input omitting these keys must not be reinterpreted (worst case:
        # freq_override=True with coll_freq=0.0 silently zeroes the operator).
        "coll_freq_override": _coll_bool("freq_override", False),
        "coll_mass_conserve": _coll_bool("mass_conserve", False),
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
        "signB": float(input_data.get("geom", {}).get("signb", 1.0)),
        "signJ": float(input_data.get("geom", {}).get("signj", 1.0)),
    }

    num_sp = as_int(input_data.get("gridsize", {}).get("number_of_species", 1), 1)
    species = species_blocks(input_data, limit=num_sp)
    if species:
        sp_mas = np.array([float(sp.get("mass", 1.0)) for sp in species])
        sp_tmp = np.array([float(sp.get("temp", 1.0)) for sp in species])
        sp_de = np.array([float(sp.get("dens", 1.0)) for sp in species])
        sp_signz = np.array([float(sp.get("z", 1.0)) for sp in species])
        sp_rlt = np.array([float(sp.get("rlt", 0.0)) for sp in species])
        sp_rln = np.array([float(sp.get("rln", 0.0)) for sp in species])
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


def _to_native(v):
    """Convert numpy scalars/arrays to plain Python types for OmegaConf."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def gkw_to_yaml(gkw_dir, output_yaml):
    """Convert a GKW run directory into a complete Gyaradax YAML configuration."""
    from omegaconf import OmegaConf

    if not os.path.isdir(gkw_dir):
        raise FileNotFoundError(f"GKW run directory {gkw_dir} not found")

    # 1. extract all scalars using the library helper
    scalars = {k: _to_native(v) for k, v in load_scalars(gkw_dir).items()}

    # 2. extract grid resolution for metadata
    input_dat_path = os.path.join(gkw_dir, "input.dat")
    raw_input = parse_input_dat(input_dat_path)
    gridsize = raw_input.get("gridsize", {})

    # 3. build structured config matching GKParams.from_config expectations
    config = {
        "run": {
            "name": os.path.basename(os.path.normpath(gkw_dir)),
            "data_dir": os.path.abspath(gkw_dir),
        },
        "solver": {
            "dt": scalars.get("dtim", 0.01),
            "naverage": scalars.get("naverage", 40),
            "n_steps": scalars.get("ntime", 400),
            "dump_interval": scalars.get("ndump_ts", 3),
            "disp_par": scalars.get("disp_par", 1.0),
            "disp_vp": scalars.get("disp_vp", 0.2),
            "disp_x": scalars.get("disp_x", 0.1),
            "disp_y": scalars.get("disp_y", 0.1),
            "idisp": scalars.get("idisp", 2),
            "ltrapping_arakawa": scalars.get("ltrapping_arakawa", False),
            "non_linear": scalars.get("non_linear", False),
            "adaptive_dt": not scalars.get("adiabatic_electrons", True),
            "finit": scalars.get("finit", "cosine2"),
        },
        "physics": {
            "rlt": scalars.get("rlt", 0.0),
            "rln": scalars.get("rln", 0.0),
            "mas": scalars.get("mas", 1.0),
            "tmp": scalars.get("tmp", 1.0),
            "de": scalars.get("de", 1.0),
            "signz": scalars.get("signz", 1.0),
            "vthrat": scalars.get("vthrat", 1.0),
            "dgrid": scalars.get("dgrid", 1.0),
            "tgrid": scalars.get("tgrid", 1.0),
        },
        "geometry": {
            "shat": scalars.get("shat", 0.0),
            "q": scalars.get("q", 1.0),
            "eps": scalars.get("eps", 0.0),
            "kthnorm": scalars.get("kthnorm", 1.0),
            "Rref": scalars.get("Rref", 1.0),
            "d2X": scalars.get("d2X", 1.0),
            "signB": scalars.get("signB", 1.0),
            "signJ": scalars.get("signJ", 1.0),
            "dvp": scalars.get("dvp", 1.0),
            "sgr_dist": scalars.get("sgr_dist", 1.0),
            "kxmax": scalars.get("kxmax", 1.0),
            "kymax": scalars.get("kymax", 1.0),
        },
        "grid": {
            "nvpar": int(gridsize.get("n_vpar_grid", 0)),
            "nmu": int(gridsize.get("n_mu_grid", 0)),
            "ns": int(gridsize.get("n_s_grid", 0)),
            "nkx": int(gridsize.get("nx", 0)),
            "nky": int(gridsize.get("nmod", 0)),
            "adiabatic_electrons": scalars.get("adiabatic_electrons", True),
        },
    }

    # the analytic geometry of the run, so the config does not depend on the gkw files
    from gyaradax.geometry.geom import geometry_spec_from_input_dat

    spec = geometry_spec_from_input_dat(input_dat_path)
    config["geometry"].update(
        {"geometry_model": spec.model, "kxmax": spec.kxmax, "signB": spec.signB, "signJ": spec.signJ,
         **{k: float(v) for k, v in spec.model_params.items()}}
    )
    config["grid"].update(
        {"krhomax": spec.krhomax, "ikxspace": spec.ikxspace, "nperiod": spec.nperiod, "vpar_max": spec.vpar_max}
    )

    conf = OmegaConf.create(config)
    OmegaConf.save(conf, output_yaml)
    print(f"successfully converted {gkw_dir} to {output_yaml}")


def load_geometry(directory):
    """Load geometry and physics parameters into JAX arrays."""
    from gyaradax.geometry.loaded import load_loaded_geometry

    return load_loaded_geometry(directory)
