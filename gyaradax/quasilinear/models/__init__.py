"""bundled gyaradax-QL Cn calibration weights + loaders.

mirrors fusion_surrogates' qlknn/models: calibrated weights ship as package
data and are resolved by name through `registry`, with a default. the torax
gyaradax-ql plugin loads the default head automatically.
"""

import functools
import pickle

from . import registry


def load_weights_from_name(name):
    """Load a bundled Cn payload (scalar Cn, its grid and fit metadata) by registry name."""
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


@functools.lru_cache(maxsize=8)
def load_cn_payload(spec):
    """resolve a payload from '' (None) / 'auto' (default) / registry name / pickle path.

    unknown paths warn and return None so callers can fall back to the
    basic-ql scalar Cn.
    """
    if not spec:
        return None
    if spec == "auto":
        return load_default_weights()
    if spec in registry.MODELS:
        return load_weights_from_name(spec)
    try:
        with open(spec, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        import warnings

        warnings.warn(
            f"cn payload '{spec}' is neither a bundled model name nor an "
            "existing file; falling back to the basic-ql scalar Cn.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def select_cn_head(payload):
    """The scalar Cn of a payload (a dict with 'scalar', or a bare number); None when absent."""
    if isinstance(payload, dict):
        payload = payload.get("scalar")
    return None if payload is None else float(payload)
