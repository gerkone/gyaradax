"""bundled gyaradax-QL Cn calibration weights + loaders.

mirrors fusion_surrogates' qlknn/models: calibrated weights ship as package
data and are resolved by name through `registry`, with a default. the torax
gyaradax-ql plugin loads the default head automatically.
"""

import pickle

from . import registry


def load_weights_from_name(name):
    """load a bundled Cn weights dict by registry name.

    returns the full payload (scalar + parametric + polynomial heads + fit
    metadata), so both the basic-ql scalar and the cn-version head are available.
    """
    path = registry.MODELS.get(name)
    if path is None:
        raise ValueError(
            f"Cn model '{name}' not in registry {list(registry.MODELS)}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_default_weights():
    """load the default bundled Cn weights payload."""
    return load_weights_from_name(registry.DEFAULT_CN_NAME)
