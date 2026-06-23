#!/usr/bin/env bash
# Run gyaradax generation/comparison for registry-based EM validation data.
#
# Defaults to the linear validation matrix and structured registry paths:
#   /local00/.../em_validation/<regime>/gkw/<stage>/<case>
#   /local00/.../em_validation/<regime>/gyaradax/<stage>/<case>
#   /local00/.../em_validation/<regime>/comparisons/rollouts/<case>.json
#
# Override REGIME=nonlinear or STAGES="rollout_short rollout_full" as needed.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/system/apps/userenv/volkmann/gyaradax_env/bin/python}
GKW_ROOT=${GKW_ROOT:-/local00/bioinf/volkmann/gyrokinetics/em_validation}
LOG_ROOT=${LOG_ROOT:-${GKW_ROOT}/gyaradax_benchmark_comparison_logs}
REGIME=${REGIME:-linear}
STAGES=${STAGES:-"window_001 rollout_short rollout_full"}
DEVICE=${DEVICE:-0}
PREALLOCATE=${PREALLOCATE:-false}
OVERWRITE_FLAG=${OVERWRITE_FLAG:---overwrite}

mkdir -p "${LOG_ROOT}/gyaradax" "${LOG_ROOT}/compare"

run_gyaradax_stage() {
	local stage=$1
	local name="${REGIME}_${stage}"
	local log_file="${LOG_ROOT}/gyaradax/${name}.log"
	echo "[$(date --iso-8601=seconds)] START gyaradax ${name}" | tee -a "${LOG_ROOT}/status.log"
	"${PYTHON_BIN}" -u scripts/generate_em_gyaradax_rollouts.py \
		--regime "${REGIME}" \
		--stage "${stage}" \
		--device "${DEVICE}" \
		--preallocate "${PREALLOCATE}" \
		${OVERWRITE_FLAG} \
		>"${log_file}" 2>&1
	echo "[$(date --iso-8601=seconds)] DONE gyaradax ${name}" | tee -a "${LOG_ROOT}/status.log"
}

compare_stage() {
	local stage=$1
	local name="${REGIME}_${stage}"
	local json_file="${LOG_ROOT}/compare/${name}.json"
	local log_file="${LOG_ROOT}/compare/${name}.log"
	echo "[$(date --iso-8601=seconds)] START compare ${name}" | tee -a "${LOG_ROOT}/status.log"
	"${PYTHON_BIN}" -u scripts/compare_em_rollouts.py \
		--regime "${REGIME}" \
		--stage "${stage}" \
		--align time \
		--write-case-json \
		--json \
		>"${json_file}" 2>"${log_file}"
	echo "[$(date --iso-8601=seconds)] DONE compare ${name}" | tee -a "${LOG_ROOT}/status.log"
}

main() {
	cd "$(git rev-parse --show-toplevel)"
	: >"${LOG_ROOT}/status.log"
	echo "[$(date --iso-8601=seconds)] gyaradax EM comparison started" | tee -a "${LOG_ROOT}/status.log"
	echo "REGIME=${REGIME} STAGES=${STAGES} DEVICE=${DEVICE} PREALLOCATE=${PREALLOCATE}" | tee -a "${LOG_ROOT}/status.log"

	for stage in ${STAGES}; do
		run_gyaradax_stage "${stage}"
		compare_stage "${stage}"
	done

	echo "[$(date --iso-8601=seconds)] gyaradax EM comparison complete" | tee -a "${LOG_ROOT}/status.log"
}

main "$@"
