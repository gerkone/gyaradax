#!/bin/bash
# usage: stop_at.sh RUNDIR PID STEP; kills PID once a snapshot at or past STEP has been written
while kill -0 $2 2>/dev/null; do
  f=$(ls $1/step_*.npz 2>/dev/null | tail -1)
  s=$(basename "$f" .npz | sed 's/step_0*//')
  if [ -n "$s" ] && [ "$s" -ge $3 ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 10 ]; then kill $2; echo "stopped $1 at step $s $(date '+%F %T')"; exit 0; fi
  sleep 20
done
echo "pid $2 gone before step $3 $(date '+%F %T')"
