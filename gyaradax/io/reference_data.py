"""GKW reference runs for the test suite, fetched on demand from the Hugging Face Hub.

The dataset holds one directory per case, ``<kind>/<name>/``, with ``kind`` one of
``nonlinear``, ``linear`` and ``kinetic``. Each directory is a GKW run directory in
GKW's own format (``input.dat``, ``geom.dat``, the diagnostics the tests read and a
few pairs of consecutive distribution dumps) plus the equivalent gyaradax
``config.yaml``. ``manifest.json`` lists the cases, their dump pairs and the size and
SHA-256 of every file. ``long_horizon/`` holds long reference runs as configs and
JSON diagnostics, outside the test suite.

The files live under :func:`data_root`, ``$GYARADAX_TEST_DATA`` or
``~/.cache/gyaradax/test_data``. Nothing is downloaded unless asked for, with
:func:`fetch`, ``case_dir(..., download=True)`` or
``python -m gyaradax.io.reference_data``.
"""

import argparse
import json
import os

REPO_ID = "gerkone/gyaradax-tests"
ENV_ROOT = "GYARADAX_TEST_DATA"
ENV_DOWNLOAD = "GYARADAX_DOWNLOAD_TEST_DATA"
MANIFEST = "manifest.json"


def data_root(root=None):
    if root:
        return os.path.abspath(os.path.expanduser(root))
    env = os.environ.get(ENV_ROOT)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".cache", "gyaradax", "test_data")


def download_requested():
    return os.environ.get(ENV_DOWNLOAD, "").lower() in ("1", "true", "yes")


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def fetch(kind=None, name=None, root=None, revision=None, diagnostics_only=False):
    """Download the whole dataset, one kind, or one case into the data root.

    Args:
        kind: ``"nonlinear"``, ``"linear"``, ``"kinetic"`` or ``"long_horizon"``; None for every kind.
        name: Case name within ``kind``; None for every case of it.
        root: Target directory; :func:`data_root` when None.
        revision: Dataset revision (branch, tag or commit); the latest when None.
        diagnostics_only: Only the JSON diagnostics, configs and inputs, no dumps.

    Returns:
        The data root.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "downloading the GKW reference data needs huggingface_hub: pip install -e '.[test]'"
        ) from exc
    if name is not None and kind is None:
        raise ValueError("a case name needs its kind")
    target = data_root(root)
    prefix = "" if kind is None else f"{kind}/" if name is None else f"{kind}/{name}/"
    patterns = [f"{prefix}*.json", f"{prefix}*.yaml", f"{prefix}*input*.dat", f"{prefix}*.md"] if diagnostics_only else [f"{prefix}*"]
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=target,
        allow_patterns=[MANIFEST, *patterns],
    )
    return target


def _complete(path, files):
    return all(
        os.path.isfile(os.path.join(path, f)) and os.path.getsize(os.path.join(path, f)) == meta["size"]
        for f, meta in files.items()
    )


def case_dir(manifest, kind, name, root=None, download=False):
    """Directory of one reference case, downloading it first if asked.

    Args:
        manifest: Manifest dict (:func:`load_manifest`) that lists the case.
        kind: ``"nonlinear"``, ``"linear"`` or ``"kinetic"``.
        name: Case name.
        root: Data root; :func:`data_root` when None.
        download: Fetch the case when it is missing or incomplete.

    Returns:
        The case directory, or None when it is not available.
    """
    files = manifest["cases"][kind][name]["files"]
    path = os.path.join(data_root(root), kind, name)
    if _complete(path, files):
        return path
    if download:
        fetch(kind, name, root=root)
        if _complete(path, files):
            return path
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"download the GKW reference runs from {REPO_ID}")
    parser.add_argument("--kind", choices=["nonlinear", "linear", "kinetic", "long_horizon"], help="only this kind")
    parser.add_argument("--name", help="only this case of --kind")
    parser.add_argument("--root", help=f"target directory (default ${ENV_ROOT} or ~/.cache/gyaradax/test_data)")
    parser.add_argument("--revision", help="dataset revision")
    parser.add_argument("--diagnostics-only", action="store_true", help="only the JSON diagnostics and configs")
    args = parser.parse_args(argv)
    print(fetch(args.kind, args.name, root=args.root, revision=args.revision, diagnostics_only=args.diagnostics_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
