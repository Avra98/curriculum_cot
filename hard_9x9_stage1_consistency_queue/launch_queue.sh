#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET_BUILDER="${ROOT}/simple_9x9_curriculum/build_dataset.py"
SFT_SCRIPT="${ROOT}/multi_output_cell_policy/sft_multi_output_train.py"
GRPO_SCRIPT="${ROOT}/multi_output_cell_policy/grpo_multi_output_train.py"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-10800}"

SFT_TARGET="${SFT_TARGET:-0.99}"
GRPO_TARGET="${GRPO_TARGET:-0.99}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_stage1_consistency_qwen05b/baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"

mkdir -p "${CHECKPOINT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

run_sft() {
  local empties="$1"
  local train_jsonl="${ROOT}/data/sudoku_t3_${empties}empty_value_qwen_text_stage1_train.jsonl"
  local eval_jsonl="${ROOT}/data/sudoku_t3_${empties}empty_value_qwen_text_stage1_eval.jsonl"
  local sft_dir="${OUTPUT_ROOT}/${empties}empty/stage01_sft_i1_${empties}empty"

  if [[ ! -f "${train_jsonl}" ]]; then
    mkdir -p "$(dirname "${train_jsonl}")"
    printf 'Building %sempty train dataset: %s\n' "${empties}" "${train_jsonl}"
    "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${train_jsonl}" --num_puzzles "${TRAIN_PUZZLES}" --empties "${empties}" --seed 0
  fi
  if [[ ! -f "${eval_jsonl}" ]]; then
    mkdir -p "$(dirname "${eval_jsonl}")"
    printf 'Building %sempty eval dataset: %s\n' "${empties}" "${eval_jsonl}"
    "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${eval_jsonl}" --num_puzzles "${EVAL_PUZZLES}" --empties "${empties}" --seed 1
  fi

  mkdir -p "${sft_dir}"
  printf '\n=== %sempty stage1 SFT ===\n' "${empties}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --train_jsonl "${train_jsonl}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${sft_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i 1 \
    --total_empties_hint "${empties}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 8.0 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --enable_gradient_checkpointing \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 50 \
    --eval_rows "${EVAL_PUZZLES}" \
    --max_completion_length 24 \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --eval_value_precision_stop "${SFT_TARGET}" \
    --eval_value_recall_stop "${SFT_TARGET}" \
    --min_steps_before_stop 50 \
    --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
    --max_steps 4000 \
    --use_wandb \
    --wandb_project "sudoku-multi-output-sft" \
    --wandb_run_name "baseline_stage01_sft_i1_${empties}empty" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

latest_sft_checkpoint() {
  local empties="$1"
  local sft_dir="${OUTPUT_ROOT}/${empties}empty/stage01_sft_i1_${empties}empty"
  local latest=""
  shopt -s nullglob
  local checkpoints=("${sft_dir}"/checkpoint-step-*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} > 0 )); then
    latest="$(printf '%s\n' "${checkpoints[@]}" | sort | tail -n 1)"
  else
    latest="${sft_dir}"
  fi
  printf '%s\n' "${latest}"
}

run_grpo() {
  local empties="$1"
  local train_jsonl="${ROOT}/data/sudoku_t3_${empties}empty_value_qwen_text_stage1_train.jsonl"
  local eval_jsonl="${ROOT}/data/sudoku_t3_${empties}empty_value_qwen_text_stage1_eval.jsonl"
  local sft_init
  sft_init="$(latest_sft_checkpoint "${empties}")"
  local grpo_dir="${OUTPUT_ROOT}/${empties}empty/stage01_grpo_i1_${empties}empty"

  mkdir -p "${grpo_dir}"
  printf '\n=== %sempty stage1 GRPO ===\n' "${empties}"
  printf 'Init adapter: %s\n' "${sft_init}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${GRPO_SCRIPT}" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --train_jsonl "${train_jsonl}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${grpo_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${sft_init}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i 1 \
    --total_empties_hint "${empties}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 4.0 \
    --learning_rate 1e-6 \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 500 \
    --eval_rows "${EVAL_PUZZLES}" \
    --num_generations 4 \
    --max_prompt_length 1024 \
    --max_completion_length 24 \
    --beta 0.0 \
    --enable_gradient_checkpointing \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --reward_good_value 1.25 \
    --penalty_bad_value 1.0 \
    --penalty_malformed 4.0 \
    --penalty_empty 0.5 \
    --penalty_singleton 1.5 \
    --eval_value_precision_stop "${GRPO_TARGET}" \
    --eval_value_recall_stop "${GRPO_TARGET}" \
    --min_steps_before_stop 50 \
    --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
    --max_steps 4000 \
    --use_wandb \
    --wandb_project "sudoku-multi-output-grpo" \
    --wandb_run_name "baseline_stage01_grpo_i1_${empties}empty" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

for empties in 3 5 7 10; do
  run_sft "${empties}"
  run_grpo "${empties}"
done

printf '\nQueued stage1 consistency suite completed.\n'
