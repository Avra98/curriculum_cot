#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
PIPELINE="${SCRIPT_DIR}/run_4x4_11empty_latent_pipeline.py"
TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/data/sudoku4x4_11empty_value_qwen_text.jsonl}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.hf_cache}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
GPU_IDS="${GPU_IDS:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIN_STAGE="${MIN_STAGE:-1}"
MAX_STAGE="${MAX_STAGE:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASELINE_CHECKPOINT_ROOT="${BASELINE_CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/sudoku4x4_11empty/baseline}"
LATENT_CHECKPOINT_ROOT="${LATENT_CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/sudoku4x4_11empty/latent}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-${BASELINE_CHECKPOINT_ROOT}/${RUN_TAG}/baseline_pipeline_11empty_4stage_4x4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LATENT_CHECKPOINT_ROOT}/${RUN_TAG}/latent_pipeline_11empty_4stage_4x4}"

cmd=(
  "${PYTHON_BIN}" "${PIPELINE}"
  --python_executable "${PYTHON_BIN}"
  --train_jsonl "${TRAIN_JSONL}"
  --cache_dir "${CACHE_DIR}"
  --model_name "${MODEL_NAME}"
  --checkpoint_root "${LATENT_CHECKPOINT_ROOT}"
  --baseline_output_root "${BASELINE_OUTPUT_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --run_tag "${RUN_TAG}"
  --min_stage "${MIN_STAGE}"
  --max_stage "${MAX_STAGE}"
  --distributed_gpu_ids "${GPU_IDS}"
  --sft_num_processes "${NUM_PROCESSES}"
  --grpo_num_processes "${NUM_PROCESSES}"
  --total_empties_hint "${TOTAL_EMPTIES_HINT:-11}"
  --sft_num_epochs "${SFT_NUM_EPOCHS:-1.0}"
  --grpo_num_train_epochs "${GRPO_NUM_TRAIN_EPOCHS:-1.0}"
  --sft_gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
  --grpo_per_device_train_batch_size "${GRPO_PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
  --grpo_gradient_accumulation_steps "${GRPO_GRADIENT_ACCUMULATION_STEPS:-4}"
  --grpo_num_generations "${GRPO_NUM_GENERATIONS:-2}"
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

if [[ -n "${STAGE1_INIT_ADAPTER_DIR:-}" ]]; then
  cmd+=(--stage1_init_adapter_dir "${STAGE1_INIT_ADAPTER_DIR}")
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

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=(--use_wandb --wandb_entity "${WANDB_ENTITY}")
fi

printf 'Launching 4x4 latent pipeline on GPUs %s\n' "${GPU_IDS}"
printf 'Baseline root: %s\n' "${BASELINE_OUTPUT_ROOT}"
printf 'Latent output root: %s\n' "${OUTPUT_ROOT}"
printf 'Stages: %s -> %s, processes=%s\n' "${MIN_STAGE}" "${MAX_STAGE}" "${NUM_PROCESSES}"

"${cmd[@]}"
