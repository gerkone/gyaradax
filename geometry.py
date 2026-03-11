import jax.numpy as jnp
import numpy as np
import os
import re


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

    # parse lines
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

    # final commit
    if key is not None:
        data[key] = np.array(values, dtype=np.float64)

    return data


def parse_input_dat(file_path):
    """Parse GKW input.dat configuration file."""
    parsed_data = {}
    with open(file_path, "r") as file:
        content = file.read()
    # split sections
    sections = re.split(r"&\w+", content)
    section_headers = re.findall(r"&(\w+)", content)
    # clean comments
    sections = [
        section.strip()
        for section in sections
        if len(section) and section[0] != "!" and section.strip()
    ]
    # iterate over sections
    for header, section in zip(section_headers, sections):
        section_dict = {}
        params = re.findall(r"(\w+)\s*=\s*([-\d\.e\w]+)", section)
        for param, value in params:
            try:
                section_dict[param] = (
                    float(value) if "e" in value or "." in value else int(value)
                )
            except ValueError:
                section_dict[param] = value.strip()
        while header in parsed_data:
            header = f"{header}0"
        parsed_data[header] = section_dict

    return parsed_data


def load_geometry(directory, dtype=torch.float64):
    # load geom dat file and input data
    geom = load_geom_dat_file(os.path.join(directory, "geom.dat"))
    input_data = parse_input_dat(os.path.join(directory, "input.dat"))

    geometry = {}
    # charge sign
    geometry["signz"] = torch.tensor(1.0, dtype=dtype)
    # thermal velocity ratio
    geometry["vthrat"] = torch.tensor(1.0, dtype=dtype)
    # species temperature
    geometry["tmp"] = torch.tensor(1.0, dtype=dtype)
    # species mass
    geometry["mas"] = torch.tensor(1.0, dtype=dtype)
    # metric factor
    geometry["d2X"] = torch.tensor(1.0, dtype=dtype)
    # magnetic field sign
    geometry["signB"] = torch.tensor(1.0, dtype=dtype)

    # load physics switches and beta
    control = input_data.get("control", {})

    def parse_gkw_bool(val):
        if isinstance(val, str):
            val = val.lower().strip()
            if val == ".true.":
                return 1.0
            if val == ".false.":
                return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    # parallel vector potential switch
    geometry["nlapar"] = torch.tensor(
        parse_gkw_bool(control.get("nlapar", 0.0)), dtype=dtype
    )
    # parallel magnetic field switch
    geometry["nlbpar"] = torch.tensor(
        parse_gkw_bool(control.get("nlbpar", 0.0)), dtype=dtype
    )

    # beta is often in 'parameters' or 'control'
    parameters = input_data.get("parameters", {})
    # plasma beta
    geometry["beta"] = torch.tensor(float(parameters.get("beta", 0.0)), dtype=dtype)

    # gather active species
    num_sp = 1
    for sec in input_data.values():
        if "number_of_species" in sec:
            num_sp = int(sec["number_of_species"])
            break
    species_keys = [k for k in input_data.keys() if k.startswith("species")][:num_sp]
    if species_keys:
        mas, tmp, de, signz = [], [], [], []
        for k in species_keys:
            sp = input_data[k]
            mas.append(sp.get("mass", 1.0))
            tmp.append(sp.get("temp", 1.0))
            de.append(sp.get("dens", 1.0))
            signz.append(sp.get("z", 1.0))

        geometry["mas"] = torch.tensor(mas, dtype=dtype)
        geometry["tmp"] = torch.tensor(tmp, dtype=dtype)
        geometry["de"] = torch.tensor(de, dtype=dtype)
        geometry["signz"] = torch.tensor(signz, dtype=dtype)

        # compute vthrat = sqrt(T_s / m_s)
        vthrat = [np.sqrt(t / m) for t, m in zip(tmp, mas)]
        geometry["vthrat"] = torch.tensor(vthrat, dtype=dtype)
        # if multiple species are found, electrons are kinetic, so not adiabatic
        geometry["adiabatic"] = torch.tensor(
            0.0 if len(species_keys) > 1 else 1.0, dtype=dtype
        )
    else:
        geometry["mas"] = torch.tensor([1.0], dtype=dtype)
        geometry["tmp"] = torch.tensor([1.0], dtype=dtype)
        geometry["de"] = torch.tensor([1.0], dtype=dtype)
        geometry["signz"] = torch.tensor([1.0], dtype=dtype)
        geometry["vthrat"] = torch.tensor([1.0], dtype=dtype)
        geometry["adiabatic"] = torch.tensor(1.0, dtype=dtype)

    kxrh = np.loadtxt(os.path.join(directory, "kxrh"))[0]
    krho = np.loadtxt(os.path.join(directory, "krho")).T[0] / geom["kthnorm"]
    # radial wavevectors
    geometry["kxrh"] = torch.tensor(kxrh, dtype=dtype)
    # binormal wavevectors
    geometry["krho"] = torch.tensor(krho, dtype=dtype)
    # spectral correction factor
    geometry["parseval"] = torch.tensor(
        [1.0] + [float(len(krho))] * (len(krho) - 1), dtype=dtype
    )

    # mugr and intmu
    if os.path.exists(os.path.join(directory, "intmu.dat")):
        intmu = np.loadtxt(os.path.join(directory, "intmu.dat"))
        if intmu.ndim == 2:
            intmu = intmu[:, 0]
        # magnetic moment integrals
        geometry["intmu"] = torch.tensor(intmu, dtype=dtype)
    else:
        mugr = np.zeros(8 + 1)
        intmu = np.zeros(8 + 1)
        mumax = 4.5
        dvperp = np.sqrt(2.0 * mumax) / 8
        for j in range(8 + 1):
            vperp = (j - 0.5) * dvperp
            mugr[j] = vperp**2 / 2.0
            intmu[j] = abs(
                np.pi * ((vperp + 0.5 * dvperp) ** 2 - (vperp - 0.5 * dvperp) ** 2)
            )
        geometry["intmu"] = torch.tensor(intmu[1:], dtype=dtype)

    if os.path.exists(os.path.join(directory, "vperp.dat")):
        vperp = np.loadtxt(os.path.join(directory, "vperp.dat"))
        if vperp.ndim == 2:
            vperp = vperp[:, 0]
        # magnetic moment grid
        geometry["mugr"] = torch.tensor(vperp**2 / 2.0, dtype=dtype)
    else:
        mugr = np.zeros(8 + 1)
        dvperp = np.sqrt(2.0 * 4.5) / 8
        for j in range(8 + 1):
            vperp = (j - 0.5) * dvperp
            mugr[j] = vperp**2 / 2.0
        geometry["mugr"] = torch.tensor(mugr[1:], dtype=dtype)

    intvp = np.loadtxt(os.path.join(directory, "intvp.dat"))[0]
    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))[0]
    # parallel velocity integrals
    geometry["intvp"] = torch.tensor(intvp, dtype=dtype)
    # parallel velocity grid
    geometry["vpgr"] = torch.tensor(vpgr, dtype=dtype)

    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    ints = np.concatenate([np.array([0.0]), np.diff(sgrid)])
    ints[0] = ints[1]  # CHECK
    # parallel coordinate integrals
    geometry["ints"] = torch.tensor(ints, dtype=dtype)

    # drift function
    geometry["efun"] = torch.tensor(-geom["E_eps_zeta"], dtype=dtype)
    # metric tensor components
    geometry["little_g"] = torch.tensor(
        np.stack([geom["g_zeta_zeta"], geom["g_eps_zeta"], geom["g_eps_eps"]], -1),
        dtype=dtype,
    )

    # magnetic field strength
    geometry["bn"] = torch.tensor(geom["bn"], dtype=dtype)
    # toroidal field fraction
    geometry["bt_frac"] = torch.tensor(geom["Bt_frac"], dtype=dtype)
    # major radius function
    geometry["rfun"] = torch.tensor(geom["R"], dtype=dtype)

    # if multiple species are present, adiabatic should be 0.0
    if len(geometry.get("de", [1.0])) > 1:
        geometry["adiabatic"] = torch.tensor(0.0, dtype=dtype)
    else:
        geometry["adiabatic"] = torch.tensor(
            np.squeeze(geom.get("adiabatic", 1.0)), dtype=dtype
        )

    # ensure species-specific fields are updated in the returned geometry
    for k in ["mas", "tmp", "de", "signz", "vthrat"]:
        if k in geom:
            geometry[k] = torch.tensor(geom[k], dtype=dtype)
        elif k not in geometry:
            geometry[k] = torch.tensor(1.0, dtype=dtype)

    return geometry

