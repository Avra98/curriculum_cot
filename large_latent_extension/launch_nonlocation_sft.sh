#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/curriculum-CoT"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRAINER="${ROOT}/sudoku/llm_policy_icon/latent_multi_output_cell_policy/residual_projector_warmstart_sft_latent_multi_output_train.py"
TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/sudoku/llm_policy_icon/data/sudoku_t3_30empty_value_qwen_text.jsonl}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.hf_cache}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_COT_TOKENS="${NUM_COT_TOKENS:?NUM_COT_TOKENS must be set}"
STAGE_I="${STAGE_I:-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sudoku/llm_policy_icon/final_checkpoint/large_latent_extension/nonlocation/sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/i${STAGE_I}_cot${NUM_COT_TOKENS}_${RUN_TAG}}"
INIT_ADAPTER_DIR="${INIT_ADAPTER_DIR:-${ROOT}/sudoku/llm_policy_icon/checkpoints/latent_multi_output_cell_policy/30_20260329_noloc_stage2_sft_from_stage1_grpo_i2_retry_gc/stage02_sft_i2_30empty_residual_projector/checkpoint-step-01100}"
WANDB_PROJECT="${WANDB_PROJECT:-sudoku-latent-multi-output-sft-residual-projector}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-large_latent_noloc_sft_i${STAGE_I}_cot${NUM_COT_TOKENS}_${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-large_latent_extension_noloc_sft_i${STAGE_I}}"

mkdir -p "${OUTPUT_DIR}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
else
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${TRAINER}"
  )
else
  cmd=(
    "${PYTHON_BIN}" -u "${TRAINER}"
  )
fi

cmd+=(
  --model_name "${MODEL_NAME}"
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --init_adapter_dir "${INIT_ADAPTER_DIR}"
  --cache_dir "${CACHE_DIR}"
  --gpu_id 0
  --stage_i "${STAGE_I}"
  --num_cot_tokens "${NUM_COT_TOKENS}"
  --total_empties_hint "${TOTAL_EMPTIES_HINT:-30}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --num_epochs "${NUM_EPOCHS:-0.75}"
  --learning_rate "${LEARNING_RATE:-7e-7}"
  --weight_decay "${WEIGHT_DECAY:-0.0}"
  --enable_gradient_checkpointing
  --logging_steps "${LOGGING_STEPS:-10}"
  --save_steps "${SAVE_STEPS:-100}"
  --eval_steps "${EVAL_STEPS:-100}"
  --eval_rows "${EVAL_ROWS:-20}"
  --max_completion_length "${MAX_COMPLETION_LENGTH:-32}"
  --use_wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE:-offline}"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ -n "${LIMIT_TRAIN_ROWS:-}" ]]; then
  cmd+=(--limit_train_rows "${LIMIT_TRAIN_ROWS}")
fi

printf 'Launching non-location latent SFT on GPUs %s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Init adapter: %s\n' "${INIT_ADAPTER_DIR}"
printf 'num_cot_tokens=%s stage_i=%s num_processes=%s\n' "${NUM_COT_TOKENS}" "${STAGE_I}" "${NUM_PROCESSES}"

"${cmd[@]}"
