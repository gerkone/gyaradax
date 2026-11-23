#!/bin/bash
# one GPU of one host: claims the next unfinished config of CAMPAIGN/order.txt, runs it with gyaradax datagen, repeats; usage: worker.sh CAMPAIGN GPU
C=$(realpath "$1"); G=$2; H=$(hostname -s)
source "$C/campaign.env"
mkdir -p "$C/claims" "$C/runs" "$C/logs"
export CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false
log() { echo "$(date '+%F %T') $H:$G $*" >> "$C/queue.log"; }
code() { cat "$C/runs/$1/datagen_exit" 2>/dev/null; }
final() { case "$(code "$1")" in 0|2|6) return 0 ;; esac; return 1; }
stopped() { [ -f "$C/STOP" ] || [ -f "$C/STOP.$H" ] || [ -f "$C/STOP.$H.$G" ]; }
until [ "$(nvidia-smi -i "$G" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt "${BUSY_MB:-2000}" ]; do
  stopped && exit 0; sleep 60
done
log "worker up"
while ! stopped; do
  job=""
  # order.txt is re-read before every claim, so appended reserves are picked up
  for n in $(grep -v '^#' "$C/order.txt"); do
    final "$n" && continue
    mkdir "$C/claims/$n" 2>/dev/null && { job=$n; break; }
  done
  [ -z "$job" ] && break
  log "start $job"
  (cd "$REPO" && PYTHONPATH="$REPO" exec "$PY" -u -m gyaradax datagen "$C/configs/$job.yaml" --out "$C/runs" --gpus "$G" \
    ${DATASET:+--dataset "$DATASET"} --args "$FLAGS" >> "$C/logs/$job.log" 2>&1 < /dev/null) &
  echo "$H $G $! $(date '+%F %T')" > "$C/claims/$job/who"
  wait $!
  rc=$(code "$job"); log "end $job exit ${rc:-none}"
  if [ "$rc" = 75 ]; then
    rm -rf "$C/claims/$job"
    # a stop usually comes with a STOP file; give the shared filesystem time to show it
    sleep 120
  elif ! final "$job"; then
    echo "${rc:-none}" > "$C/claims/$job/failed"
  fi
done
log "worker done"
