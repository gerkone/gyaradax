"""Loaded/reference GKW geometry model.

This module contains the file-backed geometry construction used by
``gyaradax.utils.load_geometry``.  It intentionally reads GKW reference/output
files directly and does not route through analytic ``compute_geometry``.
"""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from gyaradax.geometry.topology import (
    _build_mode_connectivity,
    _build_parallel_shift_maps,
    _build_pos_par_grid_classes,
)
from gyaradax.utils import load_geom_dat_file, parse_input_dat


class LoadedGKWGeometryModel:
    """Construct geometry dictionaries from GKW reference/output files."""

    name = "gkw-loaded"

    def load(self, directory: str) -> dict[str, Any]:
        """Load geometry and physics parameters into JAX arrays."""
        geom = load_geom_dat_file(os.path.join(directory, "geom.dat"))
        input_data = parse_input_dat(os.path.join(directory, "input.dat"))

        geometry: dict[str, Any] = {}

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

        kxrh = np.atleast_1d(np.loadtxt(os.path.join(directory, "kxrh")))
        if kxrh.ndim > 1:
            kxrh = kxrh[0]
        geometry["kxrh"] = jnp.array(kxrh, dtype=jnp.float64)

        krho = np.atleast_1d(np.loadtxt(os.path.join(directory, "krho")))
        if krho.ndim > 1:
            krho = krho.T[0]
        kthnorm = float(np.asarray(geom["kthnorm"]).reshape(-1)[0]) if "kthnorm" in geom else 1.0
        geometry["krho"] = jnp.array(krho / kthnorm, dtype=jnp.float64)

        # parseval correction: 1 for ky=0, 2 for ky>0 (one-sided spectrum)
        krho_vals = jnp.asarray(geometry["krho"], dtype=jnp.float64)
        geometry["parseval"] = jnp.where(jnp.abs(krho_vals) < 1e-10, 1.0, 2.0)

        intvp = np.loadtxt(os.path.join(directory, "intvp.dat"))
        if intvp.ndim > 1:
            intvp = intvp[0]
        geometry["intvp"] = jnp.array(intvp, dtype=jnp.float64)

        vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))
        if vpgr.ndim > 1:
            vpgr = vpgr[0]
        geometry["vpgr"] = jnp.array(vpgr, dtype=jnp.float64)
        geometry["vpgr_rms"] = jnp.array(float(np.sqrt(np.mean(vpgr**2))), dtype=jnp.float64)
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
            geometry["sgr_dist"] = jnp.array(float(np.abs(sgrid[1] - sgrid[0])), dtype=jnp.float64)
        else:
            geometry["sgr_dist"] = jnp.array(1.0, dtype=jnp.float64)

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

        if "E_eps_zeta" in geom:
            geometry["efun"] = jnp.array(-geom["E_eps_zeta"], dtype=jnp.float64)

        # spectral connectivity metadata for open-parallel boundary stencils
        mode_label_path = os.path.join(directory, "mode_label")
        if os.path.exists(mode_label_path):
            mode_label = np.atleast_1d(np.loadtxt(mode_label_path))
            mode_label_kxky, ixplus, ixminus, ixzero, iyzero, iyzero_bc = _build_mode_connectivity(
                mode_label, kxrh, np.asarray(geometry["krho"])
            )
            pos_classes = _build_pos_par_grid_classes(ixplus, ixminus, len(sgrid))
            s_shift, kx_shift, valid_shift = _build_parallel_shift_maps(
                ixplus, ixminus, iyzero_bc, len(sgrid), max_shift=4
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


_LOADED_GKW_GEOMETRY_MODEL = LoadedGKWGeometryModel()


def load_loaded_geometry(directory: str) -> dict[str, Any]:
    """Load a GKW reference/output geometry directory without analytic recomputation."""
    return _LOADED_GKW_GEOMETRY_MODEL.load(directory)
