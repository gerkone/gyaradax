"""Writes a campaign's configs from a base config and parameter ranges: scrambled Sobol points, then reserves.

Kinetic runs get ikxspace = floor(6.53 |shat|) (at least 1); adiabatic runs get the ikxspace whose radial box equals
that of the base eps under the base ikxspace, so dx and dy match the stored adiabatic set at any eps. With TREF set,
each run gets a collisions section with physical references (rref m, nref 1e19 m^-3, tref keV log-uniform).
Writes CAMPAIGN/configs/NAME.yaml, CAMPAIGN/design.csv and CAMPAIGN/order.txt (main runs first, reserves after a
comment line). Edit the block of constants for each campaign.

usage: PYTHONPATH=<gyaradax checkout> make_configs.py CAMPAIGN
"""

import copy
import csv
import math
import os
import sys

import numpy as np
import yaml
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "ex1"  # run prefix: ex1_000, ex1_001, ...
BASE = f"{HERE}/configs/kinetic.yaml"  # or configs/adiabatic.yaml
N_MAIN, N_RESERVE, SEED = 24, 8, 20261123
RANGES = {
    "rlt_i": (2.0, 10.0),
    "rlt_e": (2.0, 10.0),
    "rln": (0.5, 5.0),
    "te_ti": (0.7, 1.5),
    "q": (1.2, 4.0),
    "shat": (0.5, 3.0),
    "eps": (0.1, 0.19),
}
TREF = None  # e.g. (3.0, 30.0) keV for collisional runs
RREF, NREF = 6.2, 10.0


def box_length(cfg):
    from omegaconf import OmegaConf

    from gyaradax.geometry.geom import compute_geometry_from_config

    kx = np.unique(
        np.round(
            np.asarray(compute_geometry_from_config(OmegaConf.create(cfg))["kxrh"]).ravel(), 10
        )
    )
    return 2 * np.pi / np.min(np.diff(kx))


def config_for(base, name, p, tref):
    cfg = copy.deepcopy(base)
    cfg["run"] = {"name": name}
    kinetic = not cfg["grid"].get("adiabatic_electrons", True)
    ph, ge = cfg["physics"], cfg["geometry"]
    if kinetic:
        ph.update(rlt=[p["rlt_i"], p["rlt_e"]], rln=[p["rln"], p["rln"]], tmp=[1.0, p["te_ti"]])
    else:
        ph.update(rlt=p["rlt_i"], rln=p["rln"])
    for key in ("kthnorm", "kxmax", "kymax"):
        ge.pop(key, None)
    ge.update(q=p["q"], shat=p["shat"])
    if kinetic:
        ge["eps"] = p["eps"]
        cfg["grid"]["ikxspace"] = max(1, math.floor(6.53 * abs(p["shat"])))
    else:
        target = box_length(cfg)
        ge["eps"] = p["eps"]
        cfg["grid"]["ikxspace"] = int(
            max(1, round(base["grid"]["ikxspace"] * target / box_length(cfg)))
        )
    if tref is not None:
        cfg["collisions"] = {
            "enabled": True,
            "pitch_angle": True,
            "en_scatter": True,
            "friction": True,
            "freq_override": False,
            "rref": RREF,
            "nref": NREF,
            "tref": float(tref),
        }
    return cfg


def main():
    camp = os.path.abspath(sys.argv[1])
    os.makedirs(f"{camp}/configs", exist_ok=True)
    base = yaml.safe_load(open(BASE))
    keys = list(RANGES)
    u = qmc.Sobol(d=len(keys) + 1, scramble=True, seed=SEED).random(N_MAIN + N_RESERVE)
    names = []
    with open(f"{camp}/design.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "kind", *keys, "tref", "ikxspace"])
        for i, ui in enumerate(u):
            p = {
                k: round(float(RANGES[k][0] + ui[j] * (RANGES[k][1] - RANGES[k][0])), 5)
                for j, k in enumerate(keys)
            }
            tref = (
                None
                if TREF is None
                else float(np.exp(np.log(TREF[0]) + ui[-1] * np.log(TREF[1] / TREF[0])))
            )
            name = f"{NAME}_{i:03d}"
            cfg = config_for(base, name, p, tref)
            yaml.safe_dump(cfg, open(f"{camp}/configs/{name}.yaml", "w"), sort_keys=False)
            w.writerow(
                [
                    name,
                    "main" if i < N_MAIN else "reserve",
                    *(p[k] for k in keys),
                    tref or "",
                    cfg["grid"]["ikxspace"],
                ]
            )
            names.append(name)
    with open(f"{camp}/order.txt", "w") as fh:
        fh.write(
            "\n".join(names[:N_MAIN]) + "\n# reserves: move a line above this one to queue it\n"
        )
        fh.write("\n".join(f"# {n}" for n in names[N_MAIN:]) + "\n")
    print(f"{len(names)} configs ({N_MAIN} main, {N_RESERVE} reserves) in {camp}/configs")


if __name__ == "__main__":
    main()
