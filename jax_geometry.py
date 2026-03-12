import jax
import jax.numpy as jnp
import numpy as np
import os
import re

# Ensure fp64
jax.config.update("jax_enable_x64", True)


def _build_mode_connectivity(mode_label, kxrh, krho):
    """
    Build spectral parallel-boundary connectivity from mode labels.

    Returns:
      mode_label_kxky: int32[nkx, nky]
      ixplus: int32[nkx, nky], -1 means open boundary (no connection)
      ixminus: int32[nkx, nky], -1 means open boundary (no connection)
      ixzero: int32 scalar, index of kx=0 mode
      iyzero: int32 scalar, index of ky=0 mode
    """
    mode_label = np.asarray(mode_label, dtype=np.int32)
    nkx = int(kxrh.shape[0])
    nky = int(krho.shape[0])

    if mode_label.shape == (nkx, nky):
        mode_label_kxky = mode_label
    elif mode_label.shape == (nky, nkx):
        mode_label_kxky = mode_label.T
    else:
        raise ValueError(
            f"mode_label shape {mode_label.shape} incompatible with nkx/nky=({nkx},{nky})"
        )

    ixzero = int(np.argmin(np.abs(kxrh)))
    iyzero = int(np.argmin(np.abs(krho)))

    ixplus = -np.ones((nkx, nky), dtype=np.int32)
    ixminus = -np.ones((nkx, nky), dtype=np.int32)

    for iy in range(nky):
        # ky=0 mode is always periodic in spectral mode_box runs.
        if iy == iyzero:
            ix = np.arange(nkx, dtype=np.int32)
            ixplus[:, iy] = ix
            ixminus[:, iy] = ix
            continue

        labels = mode_label_kxky[:, iy]
        for lbl in np.unique(labels):
            chain = np.where(labels == lbl)[0].astype(np.int32)
            if chain.size <= 1:
                continue
            chain = np.sort(chain)
            ixplus[chain[:-1], iy] = chain[1:]
            ixminus[chain[1:], iy] = chain[:-1]

    return mode_label_kxky, ixplus, ixminus, ixzero, iyzero


def _build_pos_par_grid_classes(ixplus, ixminus, ns):
    """
    Build pos_par_grid class values (-2,-1,0,1,2) for open parallel boundaries.
    Shape: [ns, nkx, nky]
    """
    pos = np.zeros((ns,) + ixplus.shape, dtype=np.int8)
    left_open = ixminus < 0
    right_open = ixplus < 0

    if ns >= 1:
        pos[0, left_open] = -2
        pos[ns - 1, right_open] = 2
    if ns >= 2:
        pos[1, left_open] = -1
        pos[ns - 2, right_open] = 1

    return pos


def _build_parallel_shift_maps(ixplus, ixminus, iyzero, ns, max_shift=4):
    """
    Precompute parallel shift connectivity maps for s-stencil application.

    Returns arrays with shape [2*max_shift+1, ns, nkx, nky]:
      s_shift   : target s-index
      kx_shift  : target kx-index
      valid     : whether shifted point is in-grid (open boundary aware)
    """
    nkx, nky = ixplus.shape
    nshifts = 2 * max_shift + 1

    s_shift = np.zeros((nshifts, ns, nkx, nky), dtype=np.int32)
    kx_shift = np.zeros((nshifts, ns, nkx, nky), dtype=np.int32)
    valid = np.zeros((nshifts, ns, nkx, nky), dtype=np.bool_)

    for shift_idx, delta_s in enumerate(range(-max_shift, max_shift + 1)):
        for s in range(ns):
            for kx in range(nkx):
                for ky in range(nky):
                    tgt_s = s + delta_s
                    tgt_kx = kx
                    ok = True

                    if tgt_s < 0:
                        if ky == iyzero:
                            tgt_s += ns
                        else:
                            kx_conn = ixminus[kx, ky]
                            if kx_conn >= 0:
                                tgt_kx = kx_conn
                                tgt_s += ns
                            else:
                                ok = False
                    elif tgt_s >= ns:
                        if ky == iyzero:
                            tgt_s -= ns
                        else:
                            kx_conn = ixplus[kx, ky]
                            if kx_conn >= 0:
                                tgt_kx = kx_conn
                                tgt_s -= ns
                            else:
                                ok = False

                    if ok and 0 <= tgt_s < ns:
                        s_shift[shift_idx, s, kx, ky] = tgt_s
                        kx_shift[shift_idx, s, kx, ky] = tgt_kx
                        valid[shift_idx, s, kx, ky] = True

    return s_shift, kx_shift, valid

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
    if kxrh.ndim > 1:
        kxrh = kxrh[0]
    geometry["kxrh"] = jnp.array(kxrh, dtype=jnp.float64)
    
    krho = np.loadtxt(os.path.join(directory, "krho"))
    if krho.ndim > 1:
        krho = krho.T[0]
    geometry["krho"] = jnp.array(krho / geom["kthnorm"], dtype=jnp.float64)
    
    geometry["parseval"] = jnp.array([1.0] + [float(len(geometry["krho"]))] * (len(geometry["krho"]) - 1), dtype=jnp.float64)

    # Velocity space
    intvp = np.loadtxt(os.path.join(directory, "intvp.dat"))
    if intvp.ndim > 1: intvp = intvp[0]
    geometry["intvp"] = jnp.array(intvp, dtype=jnp.float64)

    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))
    if vpgr.ndim > 1: vpgr = vpgr[0]
    geometry["vpgr"] = jnp.array(vpgr, dtype=jnp.float64)
    geometry["vpgr_rms"] = jnp.array(float(np.sqrt(np.mean(vpgr**2))), dtype=jnp.float64)
    if len(vpgr) > 1:
        geometry["dvp"] = jnp.array(float(np.mean(np.diff(vpgr))), dtype=jnp.float64)
    else:
        geometry["dvp"] = jnp.array(1.0, dtype=jnp.float64)

    if os.path.exists(os.path.join(directory, "intmu.dat")):
        intmu = np.loadtxt(os.path.join(directory, "intmu.dat"))
        if intmu.ndim == 2: intmu = intmu[:, 0]
        geometry["intmu"] = jnp.array(intmu, dtype=jnp.float64)
    
    if os.path.exists(os.path.join(directory, "vperp.dat")):
        vperp = np.loadtxt(os.path.join(directory, "vperp.dat"))
        if vperp.ndim == 2: vperp = vperp[:, 0]
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
        geometry["sgr_dist"] = jnp.array(float(np.abs(sgrid[1] - sgrid[0])), dtype=jnp.float64)
    else:
        geometry["sgr_dist"] = jnp.array(1.0, dtype=jnp.float64)

    # Physics Constants Defaults
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
    geometry["ffun"] = jnp.array(geom["F"], dtype=jnp.float64)
    if "G" in geom:
        geometry["gfun"] = jnp.array(geom["G"], dtype=jnp.float64)
    geometry["bt_frac"] = jnp.array(geom["Bt_frac"], dtype=jnp.float64)
    geometry["rfun"] = jnp.array(geom["R"], dtype=jnp.float64)
    geometry["little_g"] = jnp.array(np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1), dtype=jnp.float64)
    
    # Drift functions (dfun components)
    if "D_eps" in geom:
        geometry["dfun"] = jnp.array(np.stack([geom["D_eps"], geom["D_zeta"], geom["D_s"]], -1), dtype=jnp.float64)
    if "H_eps" in geom:
        geometry["hfun"] = jnp.array(np.stack([geom["H_eps"], geom["H_zeta"], geom["H_s"]], -1), dtype=jnp.float64)
    if "I_eps" in geom:
        geometry["ifun"] = jnp.array(np.stack([geom["I_eps"], geom["I_zeta"], geom["I_s"]], -1), dtype=jnp.float64)
    
    # ExB function (efun)
    if "E_eps_zeta" in geom:
        geometry["efun"] = jnp.array(-geom["E_eps_zeta"], dtype=jnp.float64) 

    # Spectral connectivity metadata for open-parallel boundary stencils.
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
    geometry["kymax"] = jnp.array(float(np.max(np.abs(np.asarray(geometry["krho"])))), dtype=jnp.float64)
    
    return geometry
