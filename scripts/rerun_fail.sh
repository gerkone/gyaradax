#!/bin/bash
# Rerun iteration_13 (good) and iteration_222 (fail) in parallel on GPU 6 and 7
# Launch: bash scripts/rerun_fail.sh

set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.
cd /system/user/galletti/git/gyrokinetics-jax-old
mkdir -p logs

echo "=== iteration_13 on GPU 6, iteration_222 on GPU 7 ==="

python -u scripts/run.py configs/iteration_13.yaml --device=6 --from-scratch \
  --output-dir scan/validation_outputs_iteration_13_rerun \
  > logs/rerun_13.log 2>&1 &
PID1=$!

python -u scripts/run.py configs/sweep/iteration_222.yaml --device=7 --from-scratch \
  --output-dir scan/validation_outputs_iteration_222_FAIL \
  > logs/rerun_222.log 2>&1 &
PID2=$!

echo "PID iteration_13: $PID1 (GPU 6) -> logs/rerun_13.log"
echo "PID iteration_222: $PID2 (GPU 7) -> logs/rerun_222.log"
echo "waiting..."
wait $PID1 $PID2
echo "=== both done $(date) ==="
