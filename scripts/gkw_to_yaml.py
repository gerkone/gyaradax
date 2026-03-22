import os
import argparse
from omegaconf import OmegaConf
from gyaradax.utils import parse_input_dat, load_scalars


def gkw_to_yaml(gkw_dir, output_yaml):
    """Convert a GKW run directory into a complete Gyaradax YAML configuration."""
    if not os.path.exists(gkw_dir):
        print(f"error: directory {gkw_dir} not found.")
        return

    # 1. extract all scalars using the library helper
    scalars = load_scalars(gkw_dir)

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
            "idisp": scalars.get("meth", 2),
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

    conf = OmegaConf.create(config)
    OmegaConf.save(conf, output_yaml)
    print(f"successfully converted {gkw_dir} to {output_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert GKW run directory to a complete YAML config."
    )
    parser.add_argument("gkw_dir", type=str, help="path to GKW run directory")
    parser.add_argument("output", type=str, help="path to output YAML file")
    args = parser.parse_args()

    gkw_to_yaml(args.gkw_dir, args.output)
