import os
from gyaradax import load_config, gkparams_from_config, GKParams
from scripts.gkw_to_yaml import gkw_to_yaml


def test_gkw_to_yaml_conversion(nonlin_dir, tmp_path):
    output_yaml = os.path.join(tmp_path, "config.yaml")

    # 1. run conversion script
    gkw_to_yaml(nonlin_dir, output_yaml)
    assert os.path.exists(output_yaml)

    # 2. load with gyaradax
    cfg = load_config(output_yaml)
    assert hasattr(cfg, "solver")
    assert hasattr(cfg, "grid")
    assert cfg.solver.non_linear is True

    # 3. convert to GKParams
    params = gkparams_from_config(cfg)
    assert isinstance(params, GKParams)
    assert params.non_linear is True
    assert params.dt == 0.01

    # 4. grid checks
    assert cfg.grid.ns == 16
    assert cfg.grid.nkx == 85
