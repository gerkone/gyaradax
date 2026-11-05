#!/bin/bash
# idempotent setup of a fresh AMD Developer Cloud ROCm container; run inside the container
set -euo pipefail
ROCM_VER="${ROCM_VER:-7.14}"
REPO="${GYX_REPO:-/root/gyaradax}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git ninja-build "amdrocm-core-dev${ROCM_VER}" "amdrocm-runtime-dev${ROCM_VER}" "amdrocm-fft-dev${ROCM_VER}"
[ -e "$REPO" ] || git clone -q -b "${GYX_BRANCH:-feat/hip}" https://github.com/gerkone/gyaradax.git "$REPO"
python3 -m pip install -q cmake pytest huggingface_hub
python3 -m pip install -q -e "$REPO"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())" 2>&1 | tail -1
cd "$REPO/gyaradax/backends/hip_kernels" && [ -f CMakeLists.txt ] && mkdir -p _build && cd _build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build . -j"$(nproc)" && cmake --install . >/dev/null
