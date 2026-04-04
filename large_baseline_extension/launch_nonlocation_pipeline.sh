#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
PIPELINE="${ROOT}/multi_output_cell_policy/run_baseline_multi_output_pipeline_resume.py"
TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/data/sudoku_t3_30empty_value_qwen_text.jsonl}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.hf_cache}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MIN_STAGE="${MIN_STAGE:-1}"
MAX_STAGE="${MAX_STAGE:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/large_baseline_extension/hard_9x9_qwen05b/baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}/baseline_pipeline_30empty_4stage_hard9x9}"

cmd=(
  "${PYTHON_BIN}" "${PIPELINE}"
  --python_executable "${PYTHON_BIN}"
  --train_jsonl "${TRAIN_JSONL}"
  --cache_dir "${CACHE_DIR}"
  --model_name "${MODEL_NAME}"
  --checkpoint_root "${CHECKPOINT_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --run_tag "${RUN_TAG}"
  --min_stage "${MIN_STAGE}"
  --max_stage "${MAX_STAGE}"
  --distributed_gpu_ids "${GPU_IDS}"
  --sft_num_processes "${NUM_PROCESSES}"
  --grpo_num_processes "${NUM_PROCESSES}"
  --total_empties_hint "${TOTAL_EMPTIES_HINT:-30}"
  --sft_num_epochs "${SFT_NUM_EPOCHS:-1.0}"
  --grpo_num_train_epochs "${GRPO_NUM_TRAIN_EPOCHS:-0.5}"
  --sft_gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
  --grpo_per_device_train_batch_size "${GRPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  --grpo_gradient_accumulation_steps "${GRPO_GRADIENT_ACCUMULATION_STEPS:-4}"
  --grpo_num_generations "${GRPO_NUM_GENERATIONS:-2}"
  --grpo_eval_solve_rate_stop "${GRPO_EVAL_SOLVE_RATE_STOP:-0.8}"
  --grpo_min_steps_before_stop "${GRPO_MIN_STEPS_BEFORE_STOP:-100}"
  --sft_enable_gradient_checkpointing
  --grpo_enable_gradient_checkpointing
  --sft_save_steps "${SFT_SAVE_STEPS:-100}"
  --sft_eval_steps "${SFT_EVAL_STEPS:-100}"
  --grpo_save_steps "${GRPO_SAVE_STEPS:-25}"
  --grpo_eval_steps "${GRPO_EVAL_STEPS:-25}"
  --phase_max_wall_clock_seconds "${PHASE_MAX_WALL_CLOCK_SECONDS:-21600}"
  --wandb_mode "${WANDB_MODE:-offline}"
)

if [[ -n "${BOOTSTRAP_ADAPTER_DIR:-}" ]]; then
  cmd+=(--bootstrap_adapter_dir "${BOOTSTRAP_ADAPTER_DIR}")
fi

if [[ -n "${LIMIT_TRAIN_ROWS:-}" ]]; then
  cmd+=(--limit_train_rows "${LIMIT_TRAIN_ROWS}")
fi

if [[ -n "${SFT_STAGE_MAX_STEPS:-}" ]]; then
  cmd+=(--sft_stage_max_steps "${SFT_STAGE_MAX_STEPS}")
fi

if [[ -n "${GRPO_STAGE_MAX_STEPS:-}" ]]; then
  cmd+=(--grpo_stage_max_steps "${GRPO_STAGE_MAX_STEPS}")
fi

if [[ "${WANDB_MODE:-offline}" != "offline" ]]; then
  cmd+=(--use_wandb)
fi

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi

printf 'Launching hard 9x9 baseline pipeline on GPUs %s\n' "${GPU_IDS}"
printf 'Output root: %s\n' "${OUTPUT_ROOT}"
printf 'Stages: %s -> %s, processes=%s\n' "${MIN_STAGE}" "${MAX_STAGE}" "${NUM_PROCESSES}"

"${cmd[@]}"
