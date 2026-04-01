#!/bin/bash
# Run all three kinetic cases from scratch, sequentially on GPU 7
# Launch: nohup bash scripts/run_kinetic_sweep.sh > logs/kinetic_sweep.log 2>&1 &

set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.
cd /system/user/galletti/git/gyrokinetics-jax-old

echo "=== Kinetic sweep (from scratch) started at $(date) ==="

echo ""
echo "=== [1/3] half_rlt ==="
python -u scripts/run.py configs/kinetic.yaml --kinetic --device=7 --from-scratch --block-size=300 2>&1

echo ""
echo "=== [2/3] ntsks128 ==="
python -u scripts/run.py configs/kinetic_ntsks128.yaml --kinetic --device=7 --from-scratch --block-size=300 2>&1

echo ""
echo "=== [3/3] double_rlt ==="
python -u scripts/run.py configs/kinetic_double_rlt.yaml --kinetic --device=7 --from-scratch --block-size=300 2>&1

echo ""
echo "=== Kinetic sweep finished at $(date) ==="
