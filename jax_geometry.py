import jax
import jax.numpy as jnp
import numpy as np
import os
import re

# Ensure fp64
jax.config.update("jax_enable_x64", True)

def is_number(string):
    pattern = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
    return bool(re.fullmatch(pattern, string.strip()))

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
    parsed_data = {}
    if not os.path.exists(file_path):
        return parsed_data
        
    with open(file_path, "r") as file:
        content = file.read()
    
    sections = re.split(r"&\w+", content)
    section_headers = re.findall(r"&(\w+)", content)
    
    sections = [
        section.strip()
        for section in sections
        if len(section) and section[0] != "!" and section.strip()
    ]
    
    for header, section in zip(section_headers, sections):
        section_dict = {}
        params = re.findall(r"(\w+)\s*=\s*([-\d\.e\w\.]+)", section)
        for param, value in params:
            try:
                if "." in value or "e" in value:
                    section_dict[param] = float(value)
                else:
                    section_dict[param] = int(value)
            except ValueError:
                section_dict[param] = value.strip()
        while header in parsed_data:
            header = f"{header}0"
        parsed_data[header] = section_dict

    return parsed_data

def load_geometry(directory):
    """Load geometry and physics parameters into JAX arrays."""
    geom = load_geom_dat_file(os.path.join(directory, "geom.dat"))
    input_data = parse_input_dat(os.path.join(directory, "input.dat"))

    geometry = {}
    
    # Grids
    kxrh = np.loadtxt(os.path.join(directory, "kxrh"))
    if kxrh.ndim > 1: kxrh = kxrh[0]
    geometry["kxrh"] = jnp.array(kxrh, dtype=jnp.float64)
    
    krho = np.loadtxt(os.path.join(directory, "krho"))
    if krho.ndim > 1: krho = krho.T[0]
    geometry["krho"] = jnp.array(krho / geom["kthnorm"], dtype=jnp.float64)
    
    geometry["parseval"] = jnp.array([1.0] + [float(len(geometry["krho"]))] * (len(geometry["krho"]) - 1), dtype=jnp.float64)

    # Velocity space
    intvp = np.loadtxt(os.path.join(directory, "intvp.dat"))
    if intvp.ndim > 1: intvp = intvp[0]
    geometry["intvp"] = jnp.array(intvp, dtype=jnp.float64)

    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))
    if vpgr.ndim > 1: vpgr = vpgr[0]
    geometry["vpgr"] = jnp.array(vpgr, dtype=jnp.float64)

    if os.path.exists(os.path.join(directory, "intmu.dat")):
        intmu = np.loadtxt(os.path.join(directory, "intmu.dat"))
        if intmu.ndim == 2: intmu = intmu[:, 0]
        geometry["intmu"] = jnp.array(intmu, dtype=jnp.float64)
    
    if os.path.exists(os.path.join(directory, "vperp.dat")):
        vperp = np.loadtxt(os.path.join(directory, "vperp.dat"))
        if vperp.ndim == 2: vperp = vperp[:, 0]
        geometry["mugr"] = jnp.array(vperp**2 / 2.0, dtype=jnp.float64)

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    ints = np.concatenate([np.array([0.0]), np.diff(sgrid)])
    ints[0] = ints[1]
    geometry["ints"] = jnp.array(ints, dtype=jnp.float64)

    # Physics Constants Defaults
    geometry["signz"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["tmp"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["mas"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["de"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["vthrat"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["rlt"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["rln"] = jnp.array([1.0], dtype=jnp.float64)
    geometry["d2X"] = jnp.array(1.0, dtype=jnp.float64)
    geometry["signB"] = jnp.array(1.0, dtype=jnp.float64)

    # Load species info
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

    # Geometry Arrays
    geometry["bn"] = jnp.array(geom["bn"], dtype=jnp.float64)
    geometry["bt_frac"] = jnp.array(geom["Bt_frac"], dtype=jnp.float64)
    geometry["rfun"] = jnp.array(geom["R"], dtype=jnp.float64)
    geometry["little_g"] = jnp.array(np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1), dtype=jnp.float64)
    
    # Drift functions (dfun components)
    if "D_eps" in geom:
        geometry["dfun"] = jnp.array(np.stack([geom["D_eps"], geom["D_zeta"], geom["D_s"]], -1), dtype=jnp.float64)
    
    # ExB function (efun)
    if "E_eps_zeta" in geom:
        geometry["efun"] = jnp.array(-geom["E_eps_zeta"], dtype=jnp.float64) 
    
    return geometry
