#!/bin/bash
# Merge the NVIDIA HPC SDK's split CUDA trees (nvcc under cuda/<ver>, cuFFT under
# math_libs/<ver>/targets/<arch>) into one root for find_package(CUDAToolkit).
#
# Usage: scripts/make_cuda_root.sh [DEST] [CUDA_VER] [ARCH]   (./cuda_root 12.6 sbsa-linux)
#
# targets/ is deliberately not linked: nvcc searches it before include/, and the
# SDK's copy has no cufft.h, which breaks the cuFFT LTO callback compiles.

set -euo pipefail

DEST=${1:-$PWD/cuda_root}
VER=${2:-12.6}
ARCH=${3:-sbsa-linux}
SDK=${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_$(uname -m)/24.11}

CU=$SDK/cuda/$VER
ML=$SDK/math_libs/$VER/targets/$ARCH

for d in "$CU" "$ML"; do
    [ -d "$d" ] || { echo "error: missing $d" >&2; exit 1; }
done

rm -rf "$DEST"
mkdir -p "$DEST"/{bin,include,lib64,nvvm}

# toolkit proper: nvcc, bin2c, headers, cudart, nvJitLink, libdevice
for sub in bin include lib64 nvvm; do
    for f in "$CU/$sub"/*; do ln -sfn "$f" "$DEST/$sub/"; done
done

# math libraries: cuFFT and friends
for f in "$ML/lib"/*;     do ln -sfn "$f" "$DEST/lib64/"; done
for f in "$ML/include"/*; do ln -sfn "$f" "$DEST/include/"; done

echo "merged CUDA root: $DEST"
"$DEST/bin/nvcc" --version | tail -2
echo "cufft.h: $(readlink -f "$DEST/include/cufft.h")"
