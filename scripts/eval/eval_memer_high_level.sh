#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {wa01|ts01|wr03} CHECKPOINT_PATH [smoke|holdout20] [extra eval args...]" >&2
  exit 2
fi

TASK="${1,,}"
CHECKPOINT_PATH="$2"
MODE="${3:-smoke}"
if [[ $# -ge 3 ]]; then
  shift 3
else
  shift 2
fi
EXTRA_ARGS=("$@")

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${ROOT}/ckpts/Qwen3-VL-4B-Instruct}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
MEMER_QWEN_ENV="${MEMER_QWEN_ENV:-memer-qwen}"
MEMER_PYTHON="${MEMER_PYTHON:-}"
if [[ -z "${MEMER_PYTHON}" && -x "${ROOT}/policy/memer/.venv/bin/python" ]]; then
  # The project venv carries both the Qwen inference stack and the LeRobot
  # dataset reader.  The training-only conda env may not include LeRobot.
  MEMER_PYTHON="${ROOT}/policy/memer/.venv/bin/python"
fi
MEMER_HF_HOME="${MEMER_HF_HOME:-/tmp/memer_eval_hf}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
# Checkpoints produced by the current qwen-vl-finetune loader saw the leading
# MemER instruction under the assistant role. Override this for future models
# trained after that loader contract is changed.
SYSTEM_ROLE="${SYSTEM_ROLE:-assistant}"

case "${TASK}" in
  wa01)
    DATASET_PATH="${ROOT}/lerobot_data/memer/wa01_200seeds_v061_subtasks_v2"
    CAMERA_LAYOUT="${ROOT}/configs/memer/wa01_rgb_camera_layout.json"
    KEYFRAME_RULES="${ROOT}/configs/memer/wa01_keyframe_rules.json"
    HOLDOUT_EPISODES="2,3,5,6,7,8,9,10,11,13,14,16,20,23,30,36,42,45,52,59"
    INSTRUCTION_ARGS=(
      --high-level-instruction
      "Start at the coffee machine with two identical prepared coffees. Add one sugar cube to either coffee, carry both coffees to the dining counter, and pick up the coffee cup you sugared."
    )
    ;;
  ts01)
    DATASET_PATH="${ROOT}/lerobot_data/memer/ts01_200seeds_v061_subtasks_v2"
    CAMERA_LAYOUT="${ROOT}/configs/memer/ts01_rgb_camera_layout.json"
    KEYFRAME_RULES="${ROOT}/configs/memer/ts01_keyframe_rules.json"
    HOLDOUT_EPISODES="0,1,2,5,6,7,8,9,10,16,17,19,20,21,23,24,28,32,42,43"
    # TS01 keeps each episode's K=1/2/3 instruction from meta/tasks.parquet.
    INSTRUCTION_ARGS=()
    ;;
  wr03)
    DATASET_PATH="${ROOT}/lerobot_data/memer/wr03_200seeds_v061_subtasks_v2"
    CAMERA_LAYOUT="${ROOT}/configs/memer/wr03_rgb_camera_layout.json"
    KEYFRAME_RULES="${ROOT}/configs/memer/wr03_keyframe_rules.json"
    HOLDOUT_EPISODES="2,3,4,5,6,7,10,11,12,13,14,15,18,19,20,21,22,23,26,27"
    INSTRUCTION_ARGS=(
      --high-level-instruction
      "Place the can into one of the two cabinets. Close both cabinet doors, remember where the can was stored, then retrieve it."
    )
    ;;
  *)
    echo "Unsupported task: ${TASK}; expected wa01, ts01, or wr03." >&2
    exit 2
    ;;
esac

CHECKPOINT_PATH="$(realpath "${CHECKPOINT_PATH}")"
BASE_MODEL_PATH="$(realpath "${BASE_MODEL_PATH}")"
if [[ -z "${PROCESSOR_PATH}" ]]; then
  if [[ -f "${CHECKPOINT_PATH}/preprocessor_config.json" ]]; then
    PROCESSOR_PATH="${CHECKPOINT_PATH}"
  else
    PROCESSOR_PATH="${BASE_MODEL_PATH}"
  fi
fi
PROCESSOR_PATH="$(realpath "${PROCESSOR_PATH}")"

test -f "${DATASET_PATH}/meta/info.json"
test -f "${CAMERA_LAYOUT}"
test -f "${KEYFRAME_RULES}"
test -d "${CHECKPOINT_PATH}"
test -f "${BASE_MODEL_PATH}/config.json"
if [[ -f "${CHECKPOINT_PATH}/adapter_config.json" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}/adapter_model.safetensors" && ! -f "${CHECKPOINT_PATH}/adapter_model.bin" ]]; then
    echo "LoRA checkpoint is missing adapter_model.safetensors/adapter_model.bin: ${CHECKPOINT_PATH}" >&2
    exit 2
  fi
elif [[ ! -f "${CHECKPOINT_PATH}/config.json" ]]; then
  echo "Checkpoint is neither a LoRA adapter nor a full Transformers model: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

case "${MODE}" in
  smoke)
    EPISODES="${HOLDOUT_EPISODES%%,*}"
    SAMPLE_ARGS=(--timestep-stride 10 --max-examples "${MAX_EXAMPLES:-5}")
    ;;
  holdout20)
    EPISODES="${HOLDOUT_EPISODES}"
    # Match the official MemER rollout: invoke the VLM at every raw timestep.
    # The recent visual context remains subsampled with --frame-subsample 5.
    # Formal deploy-style evaluation must use a causal fixed query cadence.
    # Ground-truth boundaries/keyframes are retained only for post-hoc scoring;
    # they must not trigger extra model calls that alter predicted memory.
    SAMPLE_ARGS=(--timestep-stride 1 --boundary-window 0)
    ;;
  *)
    echo "Unsupported mode: ${MODE}; expected smoke or holdout20." >&2
    exit 2
    ;;
esac

CHECKPOINT_NAME="$(basename "${CHECKPOINT_PATH}")"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/memer_eval/${TASK}_${CHECKPOINT_NAME}_${MODE}}"

echo "task=${TASK}"
echo "mode=${MODE}"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "base_model=${BASE_MODEL_PATH}"
echo "episodes=${EPISODES}"
echo "output=${OUTPUT_DIR}"

if [[ -n "${MEMER_PYTHON}" ]]; then
  test -x "${MEMER_PYTHON}"
  PYTHON_COMMAND=("${MEMER_PYTHON}")
else
  PYTHON_COMMAND=(conda run --no-capture-output -n "${MEMER_QWEN_ENV}" python)
fi

env HF_HOME="${MEMER_HF_HOME}" \
  PYTHONPATH="${ROOT}/policy/memer${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_COMMAND[@]}" "${ROOT}/policy/memer/scripts/eval_subtask_rollout.py" \
    --model-path "${CHECKPOINT_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --processor-path "${PROCESSOR_PATH}" \
    --system-role "${SYSTEM_ROLE}" \
    --lerobot-path "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --camera-layout-config "${CAMERA_LAYOUT}" \
    --keyframe-rule-file "${KEYFRAME_RULES}" \
    --episode-indices "${EPISODES}" \
    --frame-subsample 5 \
    --recent-frames-length 8 \
    --memory-length 8 \
    --prediction-horizon 2 \
    --merge-distance 5 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device cuda \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --save-raw-responses \
    "${INSTRUCTION_ARGS[@]}" \
    "${SAMPLE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo "summary=${OUTPUT_DIR}/summary.json"
echo "per_label=${OUTPUT_DIR}/label_metrics.json"
echo "confusion=${OUTPUT_DIR}/confusion_matrix.json"
echo "predictions=${OUTPUT_DIR}/predictions.jsonl"
