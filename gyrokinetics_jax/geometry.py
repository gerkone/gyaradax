import jax.numpy as jnp
import numpy as np
import os
import re


def is_number(string):
    pattern = r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
    return bool(re.fullmatch(pattern, string.strip()))


def load_geom(file_path):
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
    parsed_data = {}
    with open(file_path, "r") as file:
        content = file.read()
    sections = re.findall(r"&(\w+)(.*?)/", content, re.DOTALL)
    species_count = 0
    for header, section in sections:
        header = header.upper()
        if header == "SPECIES":
            header = f"SPECIES_{species_count}"
            species_count += 1
        section_dict = {}
        params = re.findall(r"(\w+)\s*=\s*([-\d\.e\w']+)", section)
        for param, value in params:
            value = value.strip("'")
            if is_number(value):
                section_dict[param] = (
                    float(value)
                    if ("e" in value.lower() or "." in value)
                    else int(value)
                )
            else:
                section_dict[param] = value
        parsed_data[header] = section_dict
    return parsed_data


def load_geometry(directory):
    geometry = {}
    input_data = parse_input_dat(os.path.join(directory, "input.dat"))
    geom_dat = load_geom(os.path.join(directory, "geom.dat"))

    # Use float64 for all JAX arrays
    f64 = jnp.float64

    ions = input_data.get("SPECIES_0", {})
    geometry["rlt"] = jnp.array(ions.get("rlt", 6.0), dtype=f64)
    geometry["rln"] = jnp.array(ions.get("rln", 2.0), dtype=f64)

    geom_nml = input_data.get("GEOM", {})
    geometry["shat"] = jnp.array(geom_nml.get("shat", 0.0), dtype=f64)
    geometry["q"] = jnp.array(geom_nml.get("q", 1.0), dtype=f64)
    geometry["eps"] = jnp.array(geom_nml.get("eps", 1.0), dtype=f64)

    geometry["tmp"] = jnp.array(ions.get("temp", 1.0), dtype=f64)
    geometry["mas"] = jnp.array(ions.get("mass", 1.0), dtype=f64)
    geometry["de"] = jnp.array(ions.get("dens", 1.0), dtype=f64)
    geometry["signz"] = jnp.array(ions.get("z", 1.0), dtype=f64)

    ctrl = input_data.get("CONTROL", {})
    geometry["disp_x"] = jnp.array(ctrl.get("disp_x", 0.1), dtype=f64)
    geometry["disp_y"] = jnp.array(ctrl.get("disp_y", 0.1), dtype=f64)
    geometry["disp_par"] = jnp.array(ctrl.get("disp_par", 1.0), dtype=f64)
    geometry["dtim"] = jnp.array(ctrl.get("dtim", 0.01), dtype=f64)
    geometry["nonlin_norm_fac"] = jnp.array(
        ctrl.get("nonlin_norm_fac", 1e-5), dtype=f64
    )

    geometry["vthrat"] = jnp.sqrt(geometry["tmp"] / geometry["mas"])
    Bref = geom_dat.get("Bref", 1.0)
    geometry["signB"] = jnp.array(jnp.sign(Bref), dtype=f64)
    geometry["d2X"] = jnp.array(1.0, dtype=f64)

    geometry["kxrh"] = jnp.array(
        np.loadtxt(os.path.join(directory, "kxrh"))[0], dtype=f64
    )
    geometry["krho"] = jnp.array(
        np.loadtxt(os.path.join(directory, "krho")).T[0] / geom_dat["kthnorm"],
        dtype=f64,
    )

    geometry["intvp"] = jnp.array(
        np.loadtxt(os.path.join(directory, "intvp.dat"))[0], dtype=f64
    )
    vpgr_vals = np.loadtxt(os.path.join(directory, "vpgr.dat"))[0]
    geometry["vpgr"] = jnp.array(vpgr_vals, dtype=f64)
    geometry["dvp"] = jnp.array(vpgr_vals[1] - vpgr_vals[0], dtype=f64)

    # GKW mugr is vperp^2 / 2
    nmu = 8
    mugr_arr = np.zeros(nmu)
    intmu_arr = np.zeros(nmu)
    mumax = 4.5
    dvperp = np.sqrt(2.0 * mumax) / nmu
    for j in range(1, nmu + 1):
        vperp = (j - 0.5) * dvperp
        mugr_arr[j - 1] = vperp**2 / 2.0
        intmu_arr[j - 1] = np.pi * (
            (vperp + 0.5 * dvperp) ** 2 - (vperp - 0.5 * dvperp) ** 2
        )
    geometry["intmu"] = jnp.array(intmu_arr, dtype=f64)
    geometry["mugr"] = jnp.array(mugr_arr, dtype=f64)

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    geometry["s_grid"] = jnp.array(sgrid, dtype=f64)
    ds_arr = np.diff(sgrid)
    ds_arr = np.concatenate([[ds_arr[0]], ds_arr])
    geometry["ds"] = jnp.array(ds_arr, dtype=f64)

    Jacobian = geom_dat.get("Jacobian", 1.0)
    geometry["ints"] = jnp.array(ds_arr * Jacobian, dtype=f64)

    geometry["bn"] = jnp.array(geom_dat["bn"], dtype=f64)
    geometry["bt_frac"] = jnp.array(geom_dat["Bt_frac"], dtype=f64)
    geometry["rfun"] = jnp.array(geom_dat["R"], dtype=f64)
    geometry["efun"] = jnp.array(-geom_dat["E_eps_zeta"], dtype=f64)
    geometry["ffun"] = jnp.array(geom_dat["F"], dtype=f64)
    geometry["gfun"] = jnp.array(geom_dat["G"], dtype=f64)
    geometry["little_g"] = jnp.array(
        np.stack(
            [geom_dat["g_zeta_zeta"], geom_dat["g_eps_zeta"], geom_dat["g_eps_eps"]], -1
        ),
        dtype=f64,
    )

    geometry["dfun_x"] = jnp.array(geom_dat["D_eps"], dtype=f64)
    geometry["dfun_y"] = jnp.array(geom_dat["D_zeta"], dtype=f64)

    geometry["nkx"] = jnp.array(len(geometry["kxrh"]), dtype=f64)
    geometry["ikxspace"] = jnp.array(
        input_data.get("MODE", {}).get("ikxspace", 1), dtype=f64
    )

    ny = len(geometry["krho"])
    geometry["parseval"] = jnp.array([1.0] + [2.0] * (ny - 1), dtype=f64)

    return geometry
