#!/usr/bin/env bash
# Resume WA02 training from a six-digit checkpoint step.
#
# Usage:
#   conda activate membench
#   bash scripts/train/train_gr00t_n1_7_wa02_v061_resume.sh 060000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_STEP="${1:-}"

if [[ ! "$CHECKPOINT_STEP" =~ ^[0-9]{6}$ ]]; then
    echo "Usage: $0 CHECKPOINT_STEP (for example: 060000)" >&2
    exit 2
fi

exec bash "$SCRIPT_DIR/train_gr00t_n1_7_wa02_v061.sh" --resume "$CHECKPOINT_STEP"
