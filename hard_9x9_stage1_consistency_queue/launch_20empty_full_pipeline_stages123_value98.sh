#!/usr/bin/env bash
# Full 20-empty baseline pipeline, matching the successful 10-empty procedure:
#   1) Stage-1 SFT to value precision/recall >= 0.98
#   2) Stage-1 GRPO
#   3) Stage-2 SFT
#   4) Stage-2 GRPO
#   5) Stage-3 SFT
#   6) Stage-3 GRPO
#
# This is a wrapper around:
#   - launch_20empty_sft_stage1_98p.sh
#   - launch_20empty_post_s1sft_stages123_value98.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_20empty_full_stages123_value98}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"

SFT_STAGE1_SCRIPT="${SCRIPT_DIR}/launch_20empty_sft_stage1_98p.sh"
POST_S1_SCRIPT="${SCRIPT_DIR}/launch_20empty_post_s1sft_stages123_value98.sh"
S1_DIR="${OUTPUT_ROOT}/20empty/stage01_sft_i1_20empty_sft98"

latest_checkpoint_in_dir() {
  local d="$1"
  shopt -s nullglob
  local checkpoints=("${d}"/checkpoint-step-*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} == 0 )); then
    printf ''
    return 1
  fi
  set +o pipefail
  printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1
  set -o pipefail
}

printf '=== 20-empty full baseline pipeline (stage1 SFT -> stages123) ===\n'
printf 'run_tag=%s\n' "${RUN_TAG}"
printf 'output_root=%s\n' "${OUTPUT_ROOT}"

OUTPUT_DIR="${S1_DIR}" \
RUN_TAG="${RUN_TAG}" \
CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
"${SFT_STAGE1_SCRIPT}"

STAGE1_SFT_ADAPTER_DIR="$(latest_checkpoint_in_dir "${S1_DIR}")"
if [[ -z "${STAGE1_SFT_ADAPTER_DIR}" ]]; then
  printf 'ERROR: No checkpoint-step-* found under %s\n' "${S1_DIR}" >&2
  exit 1
fi

printf '\nStage-1 SFT complete. Using checkpoint: %s\n' "${STAGE1_SFT_ADAPTER_DIR}"

STAGE1_SFT_ADAPTER_DIR="${STAGE1_SFT_ADAPTER_DIR}" \
RUN_TAG="${RUN_TAG}" \
CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
"${POST_S1_SCRIPT}"
