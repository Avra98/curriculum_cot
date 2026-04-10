#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET_BUILDER="${ROOT}/simple_9x9_curriculum/build_dataset.py"
SFT_SCRIPT="${ROOT}/multi_output_cell_policy/sft_multi_output_train.py"

TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/data/sudoku_t3_15empty_value_qwen_text_stage1_train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-${ROOT}/data/sudoku_t3_15empty_value_qwen_text_stage1_eval.jsonl}"
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-2000}"
TRAIN_SEED="${TRAIN_SEED:-0}"
EVAL_SEED="${EVAL_SEED:-1}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_15empty_qwen05b/baseline_stage1_multivalue}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/${RUN_TAG}/stage01_sft_i1_15empty_size2only}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  mkdir -p "$(dirname "${TRAIN_JSONL}")"
  printf 'Building 15-empty train dataset: %s\n' "${TRAIN_JSONL}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" \
    --output "${TRAIN_JSONL}" \
    --num_puzzles "${TRAIN_PUZZLES}" \
    --empties 15 \
    --seed "${TRAIN_SEED}"
fi

if [[ ! -f "${EVAL_JSONL}" ]]; then
  mkdir -p "$(dirname "${EVAL_JSONL}")"
  printf 'Building 15-empty eval dataset: %s\n' "${EVAL_JSONL}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" \
    --output "${EVAL_JSONL}" \
    --num_puzzles "${EVAL_PUZZLES}" \
    --empties 15 \
    --seed "${EVAL_SEED}"
fi

mkdir -p "${CHECKPOINT_ROOT}"

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}"
  --model_name "Qwen/Qwen2.5-0.5B-Instruct"
  --train_jsonl "${TRAIN_JSONL}"
  --eval_jsonl "${EVAL_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --cache_dir "${ROOT}/.hf_cache"
  --seed 0
  --gpu_id 0
  --stage_i 1
  --total_empties_hint 15
  --per_device_train_batch_size 16
  --gradient_accumulation_steps 2
  --num_epochs 4.0
  --learning_rate 2e-4
  --enable_gradient_checkpointing
  --logging_steps 10
  --eval_steps 50
  --save_steps 50
  --eval_rows "${EVAL_PUZZLES}"
  --max_completion_length 24
  --limit_train_rows "${TRAIN_PUZZLES}"
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --multi_value_oversample_factor 1
  --train_target_size_min 2
  --train_target_size_max 2
  --eval_target_size_min 2
  --eval_target_size_max 2
  --eval_value_precision_stop 0.95
  --eval_value_recall_stop 0.95
  --min_steps_before_stop 100
  --max_wall_clock_seconds 7200
  --max_steps 600
  --use_wandb
  --wandb_project "sudoku-multi-output-sft"
  --wandb_run_name "baseline_stage01_sft_i1_15empty_size2only"
  --wandb_mode "${WANDB_MODE}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

printf 'Launching 15-empty stage-1 size-2-only SFT baseline\n'
printf 'Train dataset: %s (%s puzzles)\n' "${TRAIN_JSONL}" "${TRAIN_PUZZLES}"
printf 'Eval dataset: %s (%s puzzles)\n' "${EVAL_JSONL}" "${EVAL_PUZZLES}"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'GPUs: %s processes=%s\n' "${GPU_IDS}" "${NUM_PROCESSES}"

exec "${cmd[@]}"
