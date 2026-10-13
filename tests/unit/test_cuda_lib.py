"""CUDA kernel library per Python environment: build key, lookup order and the host probe."""

import json
import shutil
import subprocess
import sys

from gyaradax.backends import cuda_lib


def test_build_key_follows_kernel_sources_and_the_env_build_wins(tmp_path, monkeypatch):
    kernels = tmp_path / "cuda_kernels"
    shutil.copytree(cuda_lib.KERNELS_DIR, kernels, ignore=shutil.ignore_patterns("_*", "*.so"))
    monkeypatch.setattr(cuda_lib, "KERNELS_DIR", kernels)
    monkeypatch.setattr(cuda_lib.sys, "prefix", str(tmp_path / "env"))
    monkeypatch.delenv("GYARADAX_CUDA_LIB", raising=False)
    key = cuda_lib.build_key()
    (kernels / "README.md").write_text("docs do not count\n")
    assert cuda_lib.build_key() == key
    (kernels / "kernels" / "extra.cu").write_text("// a changed kernel\n")
    assert cuda_lib.build_key() != key

    assert cuda_lib.resolve() == kernels / cuda_lib.LIB_NAME
    built = cuda_lib.env_build_dir()
    assert built == tmp_path / "env" / "lib" / "gyaradax" / cuda_lib.build_key()
    built.mkdir(parents=True)
    (built / cuda_lib.LIB_NAME).write_bytes(b"")
    (built / cuda_lib.BUILD_INFO).write_text(json.dumps({"archs": [80, 103]}))
    assert cuda_lib.resolve() == built / cuda_lib.LIB_NAME
    monkeypatch.setenv("GYARADAX_CUDA_LIB", str(tmp_path / "pinned.so"))
    assert cuda_lib.resolve() == tmp_path / "pinned.so"


def test_probe_checks_the_gpu_architecture(tmp_path, monkeypatch):
    lib = tmp_path / cuda_lib.LIB_NAME
    monkeypatch.setenv("GYARADAX_CUDA_LIB", str(lib))
    monkeypatch.setattr(cuda_lib, "gpu_compute_caps", lambda: [80])
    assert cuda_lib.probe()["reason"] == "no library"
    lib.write_bytes(b"")
    assert cuda_lib.probe()["usable"]
    (tmp_path / cuda_lib.BUILD_INFO).write_text(json.dumps({"archs": [80, 103]}))
    assert cuda_lib.probe()["usable"]
    monkeypatch.setattr(cuda_lib, "gpu_compute_caps", lambda: [90])
    probe = cuda_lib.probe()
    assert not probe["usable"] and probe["reason"] == "built for sm [80, 103], GPU is sm [90]"


def test_probe_runs_from_its_path_without_jax():
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{str(cuda_lib.__file__)!r}, 'probe']\n"
        f"runpy.run_path({str(cuda_lib.__file__)!r}, run_name='__main__')\n"
        "print(sorted(k for k in sys.modules if k.split('.')[0] in ('jax', 'jaxlib', 'numpy', 'gyaradax')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    report, imported = out.strip().splitlines()
    assert json.loads(report)["key"] == cuda_lib.build_key() and imported == "[]"
