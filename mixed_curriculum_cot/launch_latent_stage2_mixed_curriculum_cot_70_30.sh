#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
SFT_TRAINER="${ROOT}/latent_multi_output_cell_policy/residual_projector_warmstart_sft_latent_multi_output_train.py"
GRPO_TRAINER="${ROOT}/latent_multi_output_cell_policy/grpo_residual_projector_latent_train.py"

BASE_JSONL="${BASE_JSONL:-${ROOT}/data/sudoku_t3_30empty_value_qwen_text.jsonl}"
TRAIN_JSONL_STAGE1="${TRAIN_JSONL_STAGE1:-${BASE_JSONL}}"
TRAIN_JSONL_STAGE2="${TRAIN_JSONL_STAGE2:-${BASE_JSONL}}"
TRAIN_JSONL="${TRAIN_JSONL:-${TRAIN_JSONL_STAGE2}}"

STAGE_I="${STAGE_I:-2}"
NUM_COT_TOKENS="${NUM_COT_TOKENS:-2}"
MIX_STAGE1_RATIO="${MIX_STAGE1_RATIO:-30}"
MIX_STAGE2_RATIO="${MIX_STAGE2_RATIO:-70}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

STAGE1_LATENT_GRPO_DIR="${STAGE1_LATENT_GRPO_DIR:-${ROOT}/final_checkpoint/large_latent_extension/hard_9x9_qwen05b/latent/grpo/i1_cot1_20260404_fixed_latent_grpo_i1/checkpoint-2740}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.hf_cache}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/final_checkpoint/mixed_curriculum_cot_70_30/latent}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${OUTPUT_ROOT}/stage02_sft_i2_cot2_${RUN_TAG}}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-${OUTPUT_ROOT}/stage02_grpo_i2_cot2_${RUN_TAG}}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${SFT_OUTPUT_DIR}" "${GRPO_OUTPUT_DIR}"

common_wandb_args=()
if [[ "${WANDB_MODE:-offline}" != "offline" ]]; then
  common_wandb_args+=(--use_wandb)
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  common_wandb_args+=(--wandb_entity "${WANDB_ENTITY}")
fi

sft_cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_TRAINER}"
  --model_name "${MODEL_NAME}"
  --train_jsonl "${TRAIN_JSONL}"
  --train_jsonl_stage1 "${TRAIN_JSONL_STAGE1}"
  --train_jsonl_stage2 "${TRAIN_JSONL_STAGE2}"
  --mixed_stage1_ratio "${MIX_STAGE1_RATIO}"
  --mixed_stage2_ratio "${MIX_STAGE2_RATIO}"
  --output_dir "${SFT_OUTPUT_DIR}"
  --init_adapter_dir "${STAGE1_LATENT_GRPO_DIR}"
  --cache_dir "${CACHE_DIR}"
  --gpu_id 0
  --stage_i "${STAGE_I}"
  --num_cot_tokens "${NUM_COT_TOKENS}"
  --total_empties_hint "${TOTAL_EMPTIES_HINT:-30}"
  --gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
  --num_epochs "${SFT_NUM_EPOCHS:-1.0}"
  --learning_rate "${SFT_LEARNING_RATE:-1e-6}"
  --weight_decay "${SFT_WEIGHT_DECAY:-0.0}"
  --enable_gradient_checkpointing
  --logging_steps "${SFT_LOGGING_STEPS:-10}"
  --save_steps "${SFT_SAVE_STEPS:-100}"
  --eval_steps "${SFT_EVAL_STEPS:-100}"
  --eval_rows "${SFT_EVAL_ROWS:-20}"
  --max_completion_length "${SFT_MAX_COMPLETION_LENGTH:-32}"
  --max_wall_clock_seconds "${SFT_MAX_WALL_CLOCK_SECONDS:-0}"
  --wandb_project "${SFT_WANDB_PROJECT:-sudoku-latent-multi-output-sft-residual-projector}"
  --wandb_run_name "${SFT_WANDB_RUN_NAME:-latent_stage02_sft_mixed_curriculum_cot_70_30_${RUN_TAG}}"
  --wandb_mode "${WANDB_MODE:-offline}"
)
sft_cmd+=("${common_wandb_args[@]}")

grpo_cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${GRPO_TRAINER}"
  --model_name "${MODEL_NAME}"
  --train_jsonl "${TRAIN_JSONL}"
  --train_jsonl_stage1 "${TRAIN_JSONL_STAGE1}"
  --train_jsonl_stage2 "${TRAIN_JSONL_STAGE2}"
  --mixed_stage1_ratio "${MIX_STAGE1_RATIO}"
  --mixed_stage2_ratio "${MIX_STAGE2_RATIO}"
  --output_dir "${GRPO_OUTPUT_DIR}"
  --init_adapter_dir "${SFT_OUTPUT_DIR}"
  --cache_dir "${CACHE_DIR}"
  --gpu_id 0
  --stage_i "${STAGE_I}"
  --num_cot_tokens "${NUM_COT_TOKENS}"
  --total_empties_hint "${TOTAL_EMPTIES_HINT:-30}"
  --per_device_train_batch_size "${GRPO_PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
  --gradient_accumulation_steps "${GRPO_GRADIENT_ACCUMULATION_STEPS:-2}"
  --num_train_epochs "${GRPO_NUM_TRAIN_EPOCHS:-0.5}"
  --learning_rate "${GRPO_LEARNING_RATE:-7e-7}"
  --logging_steps "${GRPO_LOGGING_STEPS:-5}"
  --save_steps "${GRPO_SAVE_STEPS:-25}"
  --eval_steps "${GRPO_EVAL_STEPS:-25}"
  --eval_rows "${GRPO_EVAL_ROWS:-20}"
  --num_generations "${GRPO_NUM_GENERATIONS:-2}"
  --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH:-1024}"
  --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH:-32}"
  --beta "${GRPO_BETA:-0.01}"
  --enable_gradient_checkpointing
  --max_wall_clock_seconds "${GRPO_MAX_WALL_CLOCK_SECONDS:-0}"
  --wandb_project "${GRPO_WANDB_PROJECT:-sudoku-latent-multi-output-grpo-residual-projector}"
  --wandb_run_name "${GRPO_WANDB_RUN_NAME:-latent_stage02_grpo_mixed_curriculum_cot_70_30_${RUN_TAG}}"
  --wandb_group "${GRPO_WANDB_GROUP:-mixed_curriculum_cot_70_30}"
  --wandb_mode "${WANDB_MODE:-offline}"
)
grpo_cmd+=("${common_wandb_args[@]}")

echo "Launching latent mixed curriculum_cot 70-30 stage-2 SFT"
"${sft_cmd[@]}"

echo "Launching latent mixed curriculum_cot 70-30 stage-2 GRPO"
"${grpo_cmd[@]}"

