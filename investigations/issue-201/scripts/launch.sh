#!/bin/bash
# usage: launch.sh GPU CONFIG OUTDIR; one nonlinear run on one gpu with the skin-gate code on PYTHONPATH
# env: GYX_CODE (gyaradax checkout with the patch applied), PY (python of the gyaradax env)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.90} PYTHONUNBUFFERED=1
export PYTHONPATH=${GYX_CODE:?set GYX_CODE}
echo "start $2 gpu $1 pid $$ $(date '+%F %T')"
${PY:-python} -m gyaradax.cli run $2 --output-dir $3
echo "end $2 rc $? $(date '+%F %T')"
