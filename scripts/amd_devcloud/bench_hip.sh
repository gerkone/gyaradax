#!/bin/bash
# docs throughput table on the two benchmark grids, inside the container; extra args go to gyaradax bench
cd "${GYX_REPO:-/root/gyaradax}" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for c in configs/adiabatic/nonlinear/itg_high_shear.yaml configs/kinetic/nonlinear/benchmark.yaml; do
    echo "### $c $*"
    gyaradax bench "$c" --steps 30 --blocks 5 "$@" 2>&1 | grep -v -E 'rocm_pcie|rocm_executor'
done
