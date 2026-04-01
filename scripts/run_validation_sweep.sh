#!/bin/bash
# Run 30 new validation trajectories sequentially
# Launch: nohup bash scripts/run_validation_sweep.sh > logs/validation_sweep_new.log 2>&1 &

set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.
cd /system/user/galletti/git/gyrokinetics-jax-old

echo "=== Validation sweep started at $(date) ==="


echo ""
echo "=== [1/30] iteration_104 ==="
python -u scripts/run.py configs/sweep/iteration_104.yaml --device=6 --output-dir scan/validation_outputs_iteration_104 2>&1

echo ""
echo "=== [2/30] iteration_107 ==="
python -u scripts/run.py configs/sweep/iteration_107.yaml --device=6 --output-dir scan/validation_outputs_iteration_107 2>&1

echo ""
echo "=== [3/30] iteration_108 ==="
python -u scripts/run.py configs/sweep/iteration_108.yaml --device=6 --output-dir scan/validation_outputs_iteration_108 2>&1

echo ""
echo "=== [4/30] iteration_112 ==="
python -u scripts/run.py configs/sweep/iteration_112.yaml --device=6 --output-dir scan/validation_outputs_iteration_112 2>&1

echo ""
echo "=== [5/30] iteration_116 ==="
python -u scripts/run.py configs/sweep/iteration_116.yaml --device=6 --output-dir scan/validation_outputs_iteration_116 2>&1

echo ""
echo "=== [6/30] iteration_120 ==="
python -u scripts/run.py configs/sweep/iteration_120.yaml --device=6 --output-dir scan/validation_outputs_iteration_120 2>&1

echo ""
echo "=== [7/30] iteration_126 ==="
python -u scripts/run.py configs/sweep/iteration_126.yaml --device=6 --output-dir scan/validation_outputs_iteration_126 2>&1

echo ""
echo "=== [8/30] iteration_129 ==="
python -u scripts/run.py configs/sweep/iteration_129.yaml --device=6 --output-dir scan/validation_outputs_iteration_129 2>&1

echo ""
echo "=== [9/30] iteration_156 ==="
python -u scripts/run.py configs/sweep/iteration_156.yaml --device=6 --output-dir scan/validation_outputs_iteration_156 2>&1

echo ""
echo "=== [10/30] iteration_164 ==="
python -u scripts/run.py configs/sweep/iteration_164.yaml --device=6 --output-dir scan/validation_outputs_iteration_164 2>&1

echo ""
echo "=== [11/30] iteration_183 ==="
python -u scripts/run.py configs/sweep/iteration_183.yaml --device=6 --output-dir scan/validation_outputs_iteration_183 2>&1

echo ""
echo "=== [12/30] iteration_198 ==="
python -u scripts/run.py configs/sweep/iteration_198.yaml --device=6 --output-dir scan/validation_outputs_iteration_198 2>&1

echo ""
echo "=== [13/30] iteration_207 ==="
python -u scripts/run.py configs/sweep/iteration_207.yaml --device=6 --output-dir scan/validation_outputs_iteration_207 2>&1

echo ""
echo "=== [14/30] iteration_210 ==="
python -u scripts/run.py configs/sweep/iteration_210.yaml --device=6 --output-dir scan/validation_outputs_iteration_210 2>&1

echo ""
echo "=== [15/30] iteration_212 ==="
python -u scripts/run.py configs/sweep/iteration_212.yaml --device=6 --output-dir scan/validation_outputs_iteration_212 2>&1

echo ""
echo "=== [16/30] iteration_222 ==="
python -u scripts/run.py configs/sweep/iteration_222.yaml --device=6 --output-dir scan/validation_outputs_iteration_222 2>&1

echo ""
echo "=== [17/30] iteration_245 ==="
python -u scripts/run.py configs/sweep/iteration_245.yaml --device=6 --output-dir scan/validation_outputs_iteration_245 2>&1

echo ""
echo "=== [18/30] iteration_257 ==="
python -u scripts/run.py configs/sweep/iteration_257.yaml --device=6 --output-dir scan/validation_outputs_iteration_257 2>&1

echo ""
echo "=== [19/30] iteration_260 ==="
python -u scripts/run.py configs/sweep/iteration_260.yaml --device=6 --output-dir scan/validation_outputs_iteration_260 2>&1

echo ""
echo "=== [20/30] iteration_263 ==="
python -u scripts/run.py configs/sweep/iteration_263.yaml --device=6 --output-dir scan/validation_outputs_iteration_263 2>&1

echo ""
echo "=== [21/30] iteration_270 ==="
python -u scripts/run.py configs/sweep/iteration_270.yaml --device=6 --output-dir scan/validation_outputs_iteration_270 2>&1

echo ""
echo "=== [22/30] iteration_281 ==="
python -u scripts/run.py configs/sweep/iteration_281.yaml --device=6 --output-dir scan/validation_outputs_iteration_281 2>&1

echo ""
echo "=== [23/30] iteration_284 ==="
python -u scripts/run.py configs/sweep/iteration_284.yaml --device=6 --output-dir scan/validation_outputs_iteration_284 2>&1

echo ""
echo "=== [24/30] iteration_295 ==="
python -u scripts/run.py configs/sweep/iteration_295.yaml --device=6 --output-dir scan/validation_outputs_iteration_295 2>&1

echo ""
echo "=== [25/30] iteration_298 ==="
python -u scripts/run.py configs/sweep/iteration_298.yaml --device=6 --output-dir scan/validation_outputs_iteration_298 2>&1

echo ""
echo "=== [26/30] iteration_51 ==="
python -u scripts/run.py configs/sweep/iteration_51.yaml --device=6 --output-dir scan/validation_outputs_iteration_51 2>&1

echo ""
echo "=== [27/30] iteration_52 ==="
python -u scripts/run.py configs/sweep/iteration_52.yaml --device=6 --output-dir scan/validation_outputs_iteration_52 2>&1

echo ""
echo "=== [28/30] iteration_60 ==="
python -u scripts/run.py configs/sweep/iteration_60.yaml --device=6 --output-dir scan/validation_outputs_iteration_60 2>&1

echo ""
echo "=== [29/30] iteration_74 ==="
python -u scripts/run.py configs/sweep/iteration_74.yaml --device=6 --output-dir scan/validation_outputs_iteration_74 2>&1

echo ""
echo "=== [30/30] iteration_84 ==="
python -u scripts/run.py configs/sweep/iteration_84.yaml --device=6 --output-dir scan/validation_outputs_iteration_84 2>&1

echo ""
echo "=== Validation sweep finished at $(date) ==="
