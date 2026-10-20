"""registry of bundled gyaradax-QL Cn calibration weights.

mirrors fusion_surrogates' qlknn/models/registry: weights ship as package
data and are resolved by name, with a default. each entry points at a pickle
holding the canonical rule's scalar Cn, the grid it was fitted on and the fit metadata.
"""

import pathlib

DEFAULT_CN_NAME = "iter_hybrid_v1"

# name -> bundled pickle path
MODELS = {
    "iter_hybrid_v1": str(
        pathlib.Path(__file__).parent / "cn_iter_hybrid_v1.pkl"
    ),
}
