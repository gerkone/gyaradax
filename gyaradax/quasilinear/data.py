"""Loaders for GKW-format linear/nonlinear sim directories.

Same file layout as gyaradax outputs (gyaradax mirrors GKW). Returns plain
numpy / jax arrays plus a config dict from input.dat. The set under
/restricteddata/ukaea/gyrokinetics/raw is the calibration target: pairs of
`iteration_N_Lin` (linear) and `iteration_N` (nonlinear).

For one linear dir:
  krho               : (nky, nx)  ky values broadcast over the extended-x grid
  kxrh               : (nky, nx)
  sgrid              : (ns,)
  growth.dat         : (nt, nky)  γ trace
  parallel.dat       : (ns*nx*nky, 15)  flattened [s, phi_re, phi_im, ...] (Fortran)
  eflux_spectra.dat  : (nt, nky)  per-ky energy flux trace
  geom.dat           : sectioned text, includes g_zeta_zeta, g_eps_zeta, g_eps_eps
  input.dat          : namelist
"""

import os
import re
from collections import defaultdict
from io import StringIO

import jax.numpy as jnp
import numpy as np


def parse_input_dat(path):
    """Parse a GKW namelist input.dat.

    Repeated sections (notably `&species`) are collected into a list under
    `<section>_list`; the bare `<section>` key holds the first occurrence
    for backward compatibility. New code should prefer `species_list` to
    handle ion + electron explicitly.
    """
    sections = {}
    section_counts = defaultdict(int)
    current = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("&"):
                name = line[1:].lower()
                section_counts[name] += 1
                current = {}
                if section_counts[name] == 1:
                    sections[name] = current
                sections.setdefault(f"{name}_list", []).append(current)
                continue
            if "=" in line and current is not None:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                val = val.split("!")[0].strip().strip(",").strip()
                try:
                    if "." in val or "e" in val.lower():
                        current[key] = float(val)
                    else:
                        current[key] = int(val)
                except ValueError:
                    current[key] = val.strip("'").strip('"')
    return sections


def _fix_fortran_floats(line):
    # "1.234-100" -> "1.234E-100" for malformed Fortran output
    return re.sub(r"(?<=\d)([+-]\d{2,3})", r"E\1", line)


def _robust_loadtxt(path):
    try:
        return np.loadtxt(path)
    except Exception:
        with open(path) as f:
            lines = [_fix_fortran_floats(line) for line in f]
        return np.loadtxt(StringIO("".join(lines)))


def _load_little_g(geom_dat_path, ns):
    """Parse geom.dat for g_zeta_zeta, g_eps_zeta, g_eps_eps blocks.

    geom.dat is sectioned: a label line (e.g. `g_eps_eps`) followed by
    free-form float values until ns are accumulated, then the next label.
    Returns (3, ns) stacked as [g_zeta_zeta, g_eps_zeta, g_eps_eps].
    """
    with open(geom_dat_path) as f:
        lines = f.readlines()

    def _is_label(s):
        s = s.strip()
        if not s:
            return False
        try:
            [float(tok) for tok in s.split()]
            return False
        except ValueError:
            return True

    blocks = {}
    i = 0
    while i < len(lines):
        if _is_label(lines[i]):
            name = lines[i].strip()
            j = i + 1
            vals = []
            while j < len(lines) and len(vals) < ns and not _is_label(lines[j]):
                try:
                    vals.extend(float(t) for t in lines[j].split())
                except ValueError:
                    break
                j += 1
            blocks[name] = np.asarray(vals[:ns], dtype=np.float64)
            i = j
        else:
            i += 1

    missing = [
        k
        for k in ("g_zeta_zeta", "g_eps_zeta", "g_eps_eps")
        if k not in blocks or blocks[k].size < ns
    ]
    if missing:
        raise ValueError(f"{geom_dat_path}: missing or short blocks: {missing}")
    return np.stack([blocks["g_zeta_zeta"], blocks["g_eps_zeta"], blocks["g_eps_eps"]], axis=0)


def load_linear_outputs(sim_dir):
    """Load a GKW-format linear sim directory (uses parallel.dat / eflux_spectra).

    Returns dict with jax arrays: phi2 (ns,nkx,nky), growth_rate (nky,),
    krho (nky,), kxrh (nkx,), little_g (3,ns), flux_weights (nky,), ds,
    config (parsed input.dat), meta.
    """
    config = parse_input_dat(os.path.join(sim_dir, "input.dat"))

    krho_raw = np.loadtxt(os.path.join(sim_dir, "krho"))
    kxrh_raw = np.loadtxt(os.path.join(sim_dir, "kxrh"))
    sgrid = np.loadtxt(os.path.join(sim_dir, "sgrid"))

    if krho_raw.ndim == 1:
        krho_vals = krho_raw.astype(np.float64)
        nky = krho_vals.shape[0]
        nx = 1
    else:
        # rows = ky values broadcast across the extended-x grid; first column is the ky list
        krho_vals = krho_raw[:, 0].astype(np.float64)
        nky, nx = krho_raw.shape

    if kxrh_raw.ndim == 1:
        kxrh_vals = kxrh_raw.astype(np.float64)
    else:
        # one kx vector per row; row 0 is constant across ky in standard GKW output
        kxrh_vals = kxrh_raw[0, :].astype(np.float64)
    ns = sgrid.shape[0]

    growth = _robust_loadtxt(os.path.join(sim_dir, "growth.dat"))
    gamma = growth[-1, :] if growth.ndim > 1 else growth[None, :][-1, :]
    if gamma.shape[0] != nky:
        if gamma.shape[0] == nky + 1:
            gamma = gamma[1:]
        else:
            raise ValueError(f"growth.dat last row has {gamma.shape[0]} cols, expected nky={nky}")

    parallel = _robust_loadtxt(os.path.join(sim_dir, "parallel.dat"))
    phi_re = parallel[:, 1]
    phi_im = parallel[:, 2]
    expected = ns * nx * nky
    if phi_re.shape[0] != expected:
        raise ValueError(
            f"parallel.dat has {phi_re.shape[0]} rows, expected ns*nx*nky = {expected}"
        )
    # fortran order: s varies fastest, then x, then ky
    phi2 = (phi_re**2 + phi_im**2).reshape((ns, nx, nky), order="F")
    little_g = _load_little_g(os.path.join(sim_dir, "geom.dat"), ns)

    spectra_path = os.path.join(sim_dir, "eflux_spectra.dat")
    if os.path.exists(spectra_path):
        fluxes = _robust_loadtxt(spectra_path)
        flux_weights = fluxes[-1, :] if fluxes.ndim == 2 else fluxes
        if flux_weights.shape[0] != nky:
            flux_weights = np.ones(nky)
    else:
        flux_weights = np.ones(nky)

    ds = float(sgrid[1] - sgrid[0]) if ns > 1 else 1.0

    return {
        "phi2": jnp.asarray(phi2),
        "growth_rate": jnp.asarray(gamma),
        "krho": jnp.asarray(krho_vals),
        "kxrh": jnp.asarray(kxrh_vals),
        "little_g": jnp.asarray(little_g),
        "flux_weights": jnp.asarray(flux_weights),
        "ds": ds,
        "config": config,
        "meta": {"ns": ns, "nkx": nx, "nky": nky, "sim_dir": sim_dir},
    }


def load_nonlinear_target(nl_dir, channel="eflux", n_average=240):
    """Time-averaged nonlinear flux from the tail of fluxes.dat.

    channel: "pflux" (col 0), "eflux" (col 1), or "vflux" (col 2).
    n_average: number of trailing rows of fluxes.dat to average (≈ last
    80 K-dumps × 3 rows/dump for the UKAEA dataset).
    """
    cols = {"pflux": 0, "eflux": 1, "vflux": 2}
    if channel not in cols:
        raise ValueError(f"unknown channel {channel}")
    fluxes = _robust_loadtxt(os.path.join(nl_dir, "fluxes.dat"))
    if fluxes.ndim == 1:
        fluxes = fluxes[None, :]
    return float(np.mean(fluxes[-n_average:, cols[channel]]))


def pair_sims(raw_root, suffix="_Lin"):
    """Return sorted list of (linear_dir, nonlinear_dir) pairs.

    Matching: `iteration_N_Lin` ↔ `iteration_N`.
    """
    entries = os.listdir(raw_root)
    linear_map = {}
    nonlinear_set = set()
    for d in entries:
        full = os.path.join(raw_root, d)
        if not os.path.isdir(full):
            continue
        if d.endswith(suffix):
            linear_map[d[: -len(suffix)]] = d
        else:
            nonlinear_set.add(d)

    pairs = []
    for base in sorted(nonlinear_set):
        if base in linear_map:
            pairs.append((os.path.join(raw_root, linear_map[base]), os.path.join(raw_root, base)))

    def sort_key(p):
        digits = "".join(c for c in os.path.basename(p[1]) if c.isdigit())
        return int(digits) if digits else 0

    return sorted(pairs, key=sort_key)


def _ion_species(config):
    """Pick the ion species: first &species block with mass >= 0.1, else the first."""
    species_list = config.get("species_list", [])
    if not species_list:
        return config.get("species", {})
    for sp in species_list:
        if float(sp.get("mass", 0.0)) >= 0.1:
            return sp
    return species_list[0]


def _electron_species(config):
    """Pick the electron species (mass < 0.1 or signz < 0) if present."""
    for sp in config.get("species_list", []):
        if float(sp.get("mass", 0.0)) < 0.1 or float(sp.get("signz", 1.0)) < 0:
            return sp
    return {}


def gradient_labels(config):
    """Ion (rlt, rln) plus (shat, q) from input.dat for plots."""
    ion = _ion_species(config)
    return {
        "rlt": ion.get("rlt"),
        "rln": ion.get("rln"),
        "shat": config.get("geom", {}).get("shat"),
        "q": config.get("geom", {}).get("q"),
    }


# itg drives dominate; electron entries are often locked to ion ones in this dataset
FEATURE_NAMES = (
    "rlt_i",
    "rln_i",
    "rlt_e",
    "rln_e",
    "shat",
    "q",
    "eps",
    "beta",
)


def physics_features(config):
    """Extract a fixed-schema feature vector from parsed input.dat. Missing keys → 0.0."""
    ion = _ion_species(config)
    el = _electron_species(config)
    gm = config.get("geom", {})
    md = config.get("mode", {})
    spc = config.get("spcgeneral", {})
    out = {
        "rlt_i": ion.get("rlt", 0.0),
        "rln_i": ion.get("rln", 0.0),
        "rlt_e": el.get("rlt", 0.0),
        "rln_e": el.get("rln", 0.0),
        "shat": gm.get("shat", 0.0),
        "q": gm.get("q", 0.0),
        "eps": gm.get("eps", 0.0),
        "beta": spc.get("beta_ref", spc.get("beta", md.get("beta", gm.get("beta", 0.0)))),
    }
    return np.asarray([float(out[k]) for k in FEATURE_NAMES], dtype=np.float64)


def is_unstable(
    sim_dir,
    n_tail=50,
    mean_threshold=-0.09243312561447203,
    std_threshold=0.31697104772709916,
):
    """Heuristic stability filter ported from quasilinear_torch.py.

    A sim is unstable if growth.dat tail-averaged γ exceeds the mean threshold
    OR the spread across ky exceeds the std threshold. Thresholds are
    dataset-specific (UKAEA set); use `growth_rate_max` for a physical cutoff.
    """
    g = _robust_loadtxt(os.path.join(sim_dir, "growth.dat"))
    if g.ndim == 1:
        g = g[None, :]
    g_mean_ky = np.mean(g[-n_tail:, :], axis=0)
    return not ((g_mean_ky < mean_threshold).all() and g_mean_ky.std() < std_threshold)


def growth_rate_max(sim_dir):
    """Max_ky γ from the last row of growth.dat. Physical unstable cutoff: γ > 0."""
    g = _robust_loadtxt(os.path.join(sim_dir, "growth.dat"))
    return float(g.max()) if g.ndim == 1 else float(g[-1].max())
