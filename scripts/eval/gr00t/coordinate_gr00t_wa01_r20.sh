#!/usr/bin/env bash
# Continue the active WA-01 r20 campaign from prefix/main rollout to formal stages.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_SCRIPT="${ROOT}/scripts/eval/gr00t/eval_gr00t_wa01.sh"
PILOT_SUITE="${PILOT_SUITE:-configs/eval/stages/wa01_gr00t_r20_pilot_v1.yaml}"
PREFIX_ROOT="${PREFIX_ROOT:-outputs/stage_prefixes/wa01_gr00t_r20_pilot_v1}"
FORMAL_SUITE="${FORMAL_SUITE:-${PREFIX_ROOT}/wa01_gr00t_r20_formal_v1.yaml}"
COORDINATOR_LOG="${COORDINATOR_LOG:-outputs/eval/logs/wa01_gr00t_r20_coordinator.log}"
CHECKPOINTS=(100000 070000 080000 090000)
GPUS=(0 1 2 3)
PORTS=(57100 57101 57102 57103)

cd "${ROOT}"
mkdir -p "$(dirname "${COORDINATOR_LOG}")"

wait_for_session() {
    local session="$1"
    while tmux has-session -t "${session}" 2>/dev/null; do
        sleep 60
    done
}

echo "Waiting for shared WA-01 expert prefixes"
wait_for_session wa01-gr00t-prefix-r20
"${EVAL_SCRIPT}" prefix-summary
"${ROOT}/.venv-robocasa/bin/python" scripts/eval/freeze_stage_suite.py \
    --stage-suite "${PILOT_SUITE}" \
    --prefix-root "${PREFIX_ROOT}" \
    --output "${FORMAL_SUITE}" \
    --stage-suite-version wa01_gr00t_stage_r20_formal_v1

echo "Waiting for all four WA-01 main rollouts"
for step in "${CHECKPOINTS[@]}"; do
    wait_for_session "wa01-gr00t-${step}-r20"
    summary="outputs/eval/wa01-gr00t-${step}-50seeds-r20/summary.json"
    [[ -f "${summary}" ]] || {
        echo "Missing main rollout summary: ${summary}" >&2
        exit 1
    }
    "${ROOT}/.venv-robocasa/bin/python" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p["publishable"] and p["recorded"] == 50' \
        "${summary}"
done

echo "Starting formal WA-01 stage evaluations"
for index in "${!CHECKPOINTS[@]}"; do
    step="${CHECKPOINTS[$index]}"
    gpu="${GPUS[$index]}"
    port="${PORTS[$index]}"
    session="wa01-gr00t-${step}-stage-r20"
    log="outputs/eval/logs/wa01_gr00t_${step}_stage_r20.log"
    if ! tmux has-session -t "${session}" 2>/dev/null; then
        tmux new-session -d -s "${session}" \
            "cd '${ROOT}' && exec env CKPT_STEP='${step}' GPU='${gpu}' RENDER_GPU='${gpu}' PORT='${port}' STAGE_SUITE='${FORMAL_SUITE}' PREFIX_ROOT='${PREFIX_ROOT}' STAGE_RUN_ID='wa01-gr00t-${step}-stage-50seeds-r20-formal-v1' ./scripts/eval/gr00t/eval_gr00t_wa01.sh full-stage >> '${log}' 2>&1"
    fi
done

for step in "${CHECKPOINTS[@]}"; do
    wait_for_session "wa01-gr00t-${step}-stage-r20"
done
echo "WA-01 GR00T r20 campaign complete"
