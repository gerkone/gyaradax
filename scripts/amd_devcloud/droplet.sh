#!/bin/bash
# local driver: GYX_DROPLET=<ip> droplet.sh <setup|sync|shell|run CMD|pull>
set -euo pipefail
: "${GYX_DROPLET:?set GYX_DROPLET to the droplet ip}"
KEY="${GYX_DROPLET_KEY:-$HOME/.ssh/amd}"
CONTAINER="${GYX_CONTAINER:-rocm}"
RESULTS="${GYX_RESULTS:-/system/user/publicwork/galletti/amd_devcloud/results}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=30 "root@$GYX_DROPLET")
sync_tree() {
    (cd "$ROOT" && git ls-files -co --exclude-standard -z | tar --null -T - -cf -) |
        "${SSH[@]}" "docker exec -i $CONTAINER bash -c 'mkdir -p /root/gyaradax && tar -xf - -C /root/gyaradax'"
}
case "${1:-}" in
    setup) "${SSH[@]}" "docker exec -i $CONTAINER bash -l -s" < "$HERE/setup_container.sh"; sync_tree ;;
    sync) sync_tree ;;
    shell) "${SSH[@]}" -t "docker exec -it $CONTAINER bash -l" ;;
    run) shift; "${SSH[@]}" "docker exec -i $CONTAINER bash -lc $(printf %q "$*")" ;;
    pull) mkdir -p "$RESULTS"; "${SSH[@]}" "docker exec -i $CONTAINER tar -cf - -C /root bench" | tar -xf - -C "$RESULTS"; echo "$RESULTS" ;;
    *) echo "usage: droplet.sh <setup|sync|shell|run CMD|pull>" >&2; exit 2 ;;
esac
