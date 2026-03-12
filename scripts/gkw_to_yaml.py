import os
import argparse
from omegaconf import OmegaConf
from gyaradax.geometry import parse_input_dat, load_runtime_params


def gkw_to_yaml(gkw_dir, output_yaml):
    """convert GKW input.dat and basic geometry info to a YAML configuration."""
    input_dat_path = os.path.join(gkw_dir, "input.dat")
    if not os.path.exists(input_dat_path):
        print(f"error: {input_dat_path} not found.")
        return

    # 1. extract solver parameters
    runtime = load_runtime_params(input_dat_path)

    # 2. extract grid resolution
    raw_input = parse_input_dat(input_dat_path)
    gridsize = raw_input.get("gridsize", {})

    # 3. build structured config
    config = {
        "run": {
            "name": os.path.basename(os.path.normpath(gkw_dir)),
            "data_dir": os.path.abspath(gkw_dir),
        },
        "solver": {
            "dt": runtime["dtim"],
            "naverage": runtime["naverage"],
            "disp_par": runtime["disp_par"],
            "disp_vp": runtime["disp_vp"],
            "disp_x": runtime["disp_x"],
            "disp_y": runtime["disp_y"],
            "non_linear": runtime["non_linear"],
            "enable_term_iii": True,  # default for ES setup
        },
        "grid": {
            "nvpar": int(gridsize.get("n_vpar_grid", 0)),
            "nmu": int(gridsize.get("n_mu_grid", 0)),
            "ns": int(gridsize.get("n_s_grid", 0)),
            "nkx": int(gridsize.get("nx", 0)),
            "nky": int(gridsize.get("nmod", 0)),
        },
    }

    conf = OmegaConf.create(config)
    OmegaConf.save(conf, output_yaml)
    print(f"successfully converted {input_dat_path} to {output_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert GKW input.dat to YAML config."
    )
    parser.add_argument(
        "gkw_dir", type=str, help="path to GKW run directory containing input.dat"
    )
    parser.add_argument("output", type=str, help="path to output YAML file")
    args = parser.parse_args()

    gkw_to_yaml(args.gkw_dir, args.output)
