#!/usr/bin/env bash
  set -euo pipefail

  ROOT="/home/user/chensihang/membench"
  OUT="${ROOT}/subtask_bundles"
  STAGE="$(mktemp -d)"
  trap 'rm -rf "$STAGE"' EXIT

  mkdir -p "$OUT"
  shopt -s nullglob

  TASKS=(
    tm01 tm02 ts01 ts02 ts03 ts04
    wa01 wa02 wa03 wa05
    wr01 wr03 wr05 wr06 wx01
  )

  for task in "${TASKS[@]}"; do
    case "$task" in
      tm01)
        task_id="TM01"
        annotation="tm01_annotations_v2"
        boundary_script="planner_to_stage_boundaries_tm01.py"
        roots=(
          "${ROOT}"/outputs/datasets/TM01/tm01_compact_return_arm_v2_200seeds-part*/train/TM01
        )
        ;;
      tm02)
        task_id="TM02"
        annotation="tm02_annotations_v2"
        boundary_script="planner_to_stage_boundaries_tm02.py"
        roots=(
          "${ROOT}"/outputs/datasets/TM02/tm02_200seeds_part*/train/TM02
        )
        ;;
      ts01)
        task_id="TS01"
        annotation="ts01_annotations_v2"
        boundary_script="planner_to_stage_boundaries_ts01.py"
        roots=("${ROOT}/outputs/datasets/TS01/train/TS01")
        ;;
      ts02)
        task_id="TS02"
        annotation="ts02_annotations_v3"
        boundary_script="planner_to_stage_boundaries_ts02.py"
        roots=(
          "${ROOT}"/outputs/datasets/TS02/ts02_compact_return_arm_200seeds-part*/train/TS02
        )
        ;;
      ts03)
        task_id="TS03"
        annotation="ts03_annotations_v2"
        boundary_script="planner_to_stage_boundaries_ts03.py"
        roots=("${ROOT}/outputs/datasets/TS03/train/TS03")
        ;;
      ts04)
        task_id="TS04"
        annotation="ts04_annotations_v4"
        boundary_script="planner_to_stage_boundaries_ts04.py"
        roots=(
          "${ROOT}"/outputs/datasets/TS04/ts04_200seeds_part*/train/TS04
        )
        ;;
      wa01)
        task_id="WA-01"
        annotation="wa01_annotations_v3"
        boundary_script="planner_to_stage_boundaries.py"
        roots=(
          "${ROOT}"/outputs/datasets/WA01/wa01_unmarked_200seeds_part*/train/WA-01
        )
        ;;
      wa02)
        task_id="WA-02"
        annotation="wa02_annotations_v4"
        boundary_script="planner_to_stage_boundaries_wa02.py"
        roots=(
          "${ROOT}"/outputs/datasets/WA02/wa02_200seeds_v062_part*/train/WA-02
        )
        ;;
      wa03)
        task_id="WA-03"
        annotation="wa03_unmarked_annotations_v3"
        boundary_script="planner_to_stage_boundaries_wa03.py"
        roots=(
          "${ROOT}"/outputs/datasets/WA03/wa03_unmarked_200seeds_part*/train/WA-03
        )
        ;;
      wa05)
        task_id="WA-05"
        annotation="wa05_annotations_v2"
        boundary_script="planner_to_stage_boundaries_wa05.py"
        roots=("${ROOT}/outputs/datasets/WA05/train/WA-05")
        ;;
      wr01)
        task_id="WR-01"
        annotation="wr01_annotations_v2"
        boundary_script="planner_to_stage_boundaries_wr01.py"
        roots=("${ROOT}/outputs/datasets/wr01_v063_200seeds/train/WR-01")
        ;;
      wr03)
        task_id="WR-03"
        annotation="wr03_annotations_v2"
        boundary_script="planner_to_stage_boundaries_wr03.py"
        roots=("${ROOT}/outputs/datasets/wr03_v063_200seeds/train/WR-03")
        ;;
      wr05)
        task_id="WR-05"
        annotation="wr05_annotations_v3"
        boundary_script="planner_to_stage_boundaries_wr05.py"
        roots=("${ROOT}/outputs/datasets/WR05/train/WR-05")
        ;;
      wr06)
        task_id="WR-06"
        annotation="wr06_annotations_v2"
        boundary_script="planner_to_stage_boundaries_wr06.py"
        roots=("${ROOT}/outputs/datasets/WR06/train/WR-06")
        ;;
      wx01)
        task_id="WX-01"
        annotation="wx01_annotations_v2"
        boundary_script="planner_to_stage_boundaries_wx01.py"
        roots=("${ROOT}/outputs/datasets/WX01/train/WX-01")
        ;;
    esac

    if [[ "${#roots[@]}" -eq 0 ]]; then
      echo "No native episode root found for ${task}" >&2
      exit 1
    fi

    bundle="${STAGE}/${task}"
    mkdir -p \
      "${bundle}/native_episodes" \
      "${bundle}/configs/memer" \
      "${bundle}/scripts/annotate" \
      "${bundle}/reference"

    episode_count=0

    while IFS= read -r -d '' episode_dir; do
      seed_dir="$(basename "${episode_dir}")"
      destination="${bundle}/native_episodes/${seed_dir}"

      if [[ -e "${destination}" ]]; then
        echo "Duplicate episode directory: ${seed_dir}" >&2
        exit 1
      fi

      [[ -f "${episode_dir}/episode.json" ]] || {
        echo "Missing episode.json: ${episode_dir}" >&2
        exit 1
      }

      [[ -f "${episode_dir}/planner.json" ]] || {
        echo "Missing planner.json: ${episode_dir}" >&2
        exit 1
      }

      mkdir -p "${destination}"
      cp "${episode_dir}/episode.json" "${destination}/"
      cp "${episode_dir}/planner.json" "${destination}/"

      episode_count=$((episode_count + 1))
    done < <(
      find "${roots[@]}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name 'seed_*_attempt_00' \
        -print0 |
        sort -z
    )

    [[ "${episode_count}" -gt 0 ]] || {
      echo "No episodes found for ${task}" >&2
      exit 1
    }

    cp \
      "${ROOT}/configs/memer/${task}_subtasks.json" \
      "${bundle}/configs/memer/"

    cp \
      "${ROOT}/scripts/annotate/apply_subtask_labels.py" \
      "${ROOT}/scripts/annotate/add_subtask_to_lerobot.py" \
      "${ROOT}/scripts/annotate/convert_subtasks_to_pi05.py" \
      "${bundle}/scripts/annotate/"

    cp \
      "${ROOT}"/scripts/annotate/planner_to_stage_boundaries*.py \
      "${bundle}/scripts/annotate/"

    cp \
      "${ROOT}/outputs/memer/${annotation}/boundaries.jsonl" \
      "${bundle}/reference/"

    cp \
      "${ROOT}/outputs/memer/${annotation}/subtasks.jsonl" \
      "${bundle}/reference/"

    printf '%s\n' \
      "task_id=${task_id}" \
      "episode_count=${episode_count}" \
      "boundary_reference=reference/boundaries.jsonl" \
      "subtask_reference=reference/subtasks.jsonl" \
      "lerobot_data=not included; use the existing LeRobot dataset" \
      > "${bundle}/manifest.txt"

    tar \
      -czf "${OUT}/${task}_subtask_sources.tar.gz" \
      -C "${STAGE}" \
      "${task}"

    echo "created ${OUT}/${task}_subtask_sources.tar.gz (${episode_count} episodes)"
  done
