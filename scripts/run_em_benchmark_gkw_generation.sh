#!/usr/bin/env bash
# Generate EM validation GKW data unattended using the registry matrix.
#
# By default this generates the linear validation matrix stage-by-stage into the
# structured registry paths under /local00/.../em_validation/<regime>/gkw/<stage>/.
# Override REGIME=nonlinear or STAGES="window_001 rollout_short rollout_full" as needed.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/system/apps/userenv/volkmann/gyaradax_env/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/local00/bioinf/volkmann/gyrokinetics/em_validation}
LOG_ROOT=${LOG_ROOT:-${OUTPUT_ROOT}/benchmark_generation_logs}
GENERATOR=${GENERATOR:-scripts/generate_em_gkw_validation_data.py}
REGIME=${REGIME:-linear}
STAGES=${STAGES:-"window_001 rollout_short rollout_full"}
OVERWRITE_FLAG=${OVERWRITE_FLAG:---overwrite}
MPI_MODE=${MPI_MODE:-auto}

mkdir -p "${LOG_ROOT}"

run_stage() {
	local stage=$1
	local name="${REGIME}_${stage}"
	local log_file="${LOG_ROOT}/${name}.log"
	local status_file="${LOG_ROOT}/${name}.status"

	echo "[$(date --iso-8601=seconds)] START ${name}" | tee "${status_file}"
	"${PYTHON_BIN}" -u "${GENERATOR}" \
		${OVERWRITE_FLAG} \
		--mpi "${MPI_MODE}" \
		--regime "${REGIME}" \
		--stage "${stage}" \
		>"${log_file}" 2>&1
	echo "[$(date --iso-8601=seconds)] DONE ${name}" | tee -a "${status_file}"
	echo "log=${log_file}" | tee -a "${status_file}"
}

main() {
	cd "$(git rev-parse --show-toplevel)"
	echo "[$(date --iso-8601=seconds)] EM GKW generation started"
	echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
	echo "LOG_ROOT=${LOG_ROOT}"
	echo "PYTHON_BIN=${PYTHON_BIN}"
	echo "REGIME=${REGIME}"
	echo "STAGES=${STAGES}"
	echo "OVERWRITE_FLAG=${OVERWRITE_FLAG}"
	echo "MPI_MODE=${MPI_MODE}"

	for stage in ${STAGES}; do
		run_stage "${stage}"
	done

	echo "[$(date --iso-8601=seconds)] EM GKW generation complete"
}

main "$@"
