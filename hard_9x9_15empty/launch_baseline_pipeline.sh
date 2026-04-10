#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET_BUILDER="${ROOT}/simple_9x9_curriculum/build_dataset.py"
PIPELINE_LAUNCHER="${ROOT}/large_baseline_extension/launch_nonlocation_pipeline.sh"

TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/data/sudoku_t3_15empty_value_qwen_text.jsonl}"
NUM_PUZZLES="${NUM_PUZZLES:-20000}"
DATASET_SEED="${DATASET_SEED:-0}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MIN_STAGE="${MIN_STAGE:-1}"
MAX_STAGE="${MAX_STAGE:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_15empty_qwen05b/baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}/baseline_pipeline_15empty_4stage_hard9x9}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"
WAIT_FOR_EXISTING_TRAINING="${WAIT_FOR_EXISTING_TRAINING:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  mkdir -p "$(dirname "${TRAIN_JSONL}")"
  printf 'Building 15-empty dataset: %s\n' "${TRAIN_JSONL}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" \
    --output "${TRAIN_JSONL}" \
    --num_puzzles "${NUM_PUZZLES}" \
    --empties 15 \
    --seed "${DATASET_SEED}"
fi

if [[ "${WAIT_FOR_EXISTING_TRAINING}" == "1" ]]; then
  while pgrep -f "/home/ubuntu/curriculum_cot/.venv/bin/python.*(run_baseline_multi_output_pipeline_resume.py|run_latent_residual_projector_pipeline.py|sft_multi_output_train.py|grpo_multi_output_train.py|residual_projector_warmstart_sft_latent_multi_output_train.py|grpo_residual_projector_latent_train.py)" >/dev/null; do
    printf 'Existing training detected; waiting %ss before launching 15-empty baseline...\n' "${WAIT_SECONDS}"
    sleep "${WAIT_SECONDS}"
  done
fi

mkdir -p "${CHECKPOINT_ROOT}"

export TRAIN_JSONL
export TOTAL_EMPTIES_HINT=15
export GPU_IDS
export NUM_PROCESSES
export MIN_STAGE
export MAX_STAGE
export RUN_TAG
export CHECKPOINT_ROOT
export OUTPUT_ROOT
export WANDB_MODE
export WANDB_ENTITY

printf 'Launching 15-empty hard 9x9 baseline pipeline\n'
printf 'Dataset: %s\n' "${TRAIN_JSONL}"
printf 'Checkpoint root: %s\n' "${CHECKPOINT_ROOT}"
printf 'Output root: %s\n' "${OUTPUT_ROOT}"

exec "${PIPELINE_LAUNCHER}"
