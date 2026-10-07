#!/usr/bin/env python3
"""Backwards-compatible shim for the ``gyaradax`` CLI.

The implementation now lives in :mod:`gyaradax.cli` so it ships with the
package and is reachable from the ``gyaradax`` console script. Prefer::

    gyaradax run configs/adiabatic_a.yaml
    gyaradax run configs/nl_em_apar.yaml --n-gpus 4

This script keeps the old ``python scripts/run.py CONFIG ...`` invocation
working, and still accepts the retired ``--kinetic`` flag (adiabatic vs
kinetic is now detected from ``grid.adiabatic_electrons`` in the config).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gyaradax.cli import main  # noqa: E402


def _translate(argv):
    """Map the legacy flag set onto the subcommand CLI."""
    argv = list(argv)
    if "--kinetic" in argv:
        argv.remove("--kinetic")
        print("note: --kinetic is no longer needed; the config decides.", file=sys.stderr)
    if not argv or argv[0] not in {"run", "bench", "info", "convert"}:
        argv.insert(0, "run")
    return argv


if __name__ == "__main__":
    raise SystemExit(main(_translate(sys.argv[1:])))
