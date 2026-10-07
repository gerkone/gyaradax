#!/usr/bin/env python3
"""Backwards-compatible shim: ``gyaradax convert GKW_DIR OUTPUT.yaml``.

The converter lives in :func:`gyaradax.utils.gkw_to_yaml`; this keeps
``python -m scripts.gkw_to_yaml GKW_DIR OUTPUT.yaml`` working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gyaradax.cli import main  # noqa: E402
from gyaradax.utils import gkw_to_yaml  # noqa: E402, F401

if __name__ == "__main__":
    raise SystemExit(main(["convert", *sys.argv[1:]]))
