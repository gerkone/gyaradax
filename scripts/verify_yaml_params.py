import jax
from gyaradax import load_config, gkparams_from_config, GKParams

# ensure fp64
jax.config.update("jax_enable_x64", True)

def test_yaml_to_params():
    cfg = load_config("test_config.yaml")
    params = gkparams_from_config(cfg)
    
    print(f"Loaded params: {params}")
    
    # check some values
    assert params.non_linear is True
    assert params.dt == 0.01
    assert params.rlt > 10.0
    assert params.shat > 3.0
    assert params.q > 4.0
    assert params.kxmax > 8.0
    
    print("Verification successful: GKParams created from YAML correctly.")

if __name__ == "__main__":
    test_yaml_to_params()
