#!/usr/bin/env bash
# Continue 7-empty stage-1 SFT from a "fast" (~90%) checkpoint toward 0.99 value
# precision+recall (same metric as other launchers), then run stage-1 GRPO from the
# best SFT checkpoint in this run's output tree.
#
# You must point at the fast run either by explicit adapter dir or by the SFT output folder:
#   INIT_ADAPTER_DIR=/path/to/checkpoint-step-01200 ./launch_7empty_sft99_from_fast_then_grpo.sh
#   FAST_SFT_DIR=/path/to/stage01_sft_i1_7empty_fast ./launch_7empty_sft99_from_fast_then_grpo.sh
#
# Optional: SFT_TARGET=0.99 GRPO_TARGET=0.99 MAX_STEPS=15000 TRAIN_PUZZLES=10000 EVAL_PUZZLES=100
#           PHASE_WALL_CLOCK_SECONDS=0  USE_GC=1
#
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

EMPTIES=7
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
SFT_TARGET="${SFT_TARGET:-0.99}"
GRPO_TARGET="${GRPO_TARGET:-0.99}"
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"
MAX_STEPS="${MAX_STEPS:-15000}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_7empty_sft99_grpo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"
SFT_DIR="${OUTPUT_ROOT}/${EMPTIES}empty/stage01_sft_i1_${EMPTIES}empty_sft99"
GRPO_DIR="${OUTPUT_ROOT}/${EMPTIES}empty/stage01_grpo_i1_${EMPTIES}empty"

train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

latest_checkpoint_in_dir() {
  local d="$1"
  shopt -s nullglob
  local checkpoints=("${d}"/checkpoint-step-*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} == 0 )); then
    printf ''
    return 1
  fi
  printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1
}

resolve_init_adapter() {
  if [[ -n "${INIT_ADAPTER_DIR:-}" ]]; then
    if [[ ! -d "${INIT_ADAPTER_DIR}" ]]; then
      printf 'ERROR: INIT_ADAPTER_DIR is not a directory: %s\n' "${INIT_ADAPTER_DIR}" >&2
      exit 1
    fi
    printf '%s\n' "${INIT_ADAPTER_DIR}"
    return 0
  fi
  if [[ -n "${FAST_SFT_DIR:-}" ]]; then
    local latest
    latest="$(latest_checkpoint_in_dir "${FAST_SFT_DIR}" || true)"
    if [[ -z "${latest}" ]]; then
      printf 'ERROR: No checkpoint-step-* under FAST_SFT_DIR=%s\n' "${FAST_SFT_DIR}" >&2
      exit 1
    fi
    printf '%s\n' "${latest}"
    return 0
  fi
  printf 'ERROR: Set INIT_ADAPTER_DIR=/path/to/checkpoint-step-XXXXX or FAST_SFT_DIR=/path/to/stage01_sft_i1_7empty_fast\n' >&2
  exit 1
}

INIT_ADAPTER="$(resolve_init_adapter)"
printf 'SFT warm-start adapter: %s\n' "${INIT_ADAPTER}"

if [[ ! -f "${train_jsonl}" ]]; then
  mkdir -p "$(dirname "${train_jsonl}")"
  printf 'Building %s-empty train dataset: %s\n' "${EMPTIES}" "${train_jsonl}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${train_jsonl}" --num_puzzles "${TRAIN_PUZZLES}" --empties "${EMPTIES}" --seed 0
fi
if [[ ! -f "${eval_jsonl}" ]]; then
  mkdir -p "$(dirname "${eval_jsonl}")"
  printf 'Building %s-empty eval dataset: %s\n' "${EMPTIES}" "${eval_jsonl}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${eval_jsonl}" --num_puzzles 200 --empties "${EMPTIES}" --seed 1
fi

mkdir -p "${SFT_DIR}" "${GRPO_DIR}"

GC_FLAGS=()
if [[ "${USE_GC:-0}" == "1" ]]; then
  GC_FLAGS+=(--enable_gradient_checkpointing)
  printf 'NOTE: USE_GC=1 — slower, less VRAM.\n'
fi

printf '\n=== Phase 1: 7-empty SFT → prec+recall >= %s (from fast checkpoint) ===\n' "${SFT_TARGET}"
printf 'Output: %s\n' "${SFT_DIR}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}" \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_jsonl "${train_jsonl}" \
  --eval_jsonl "${eval_jsonl}" \
  --output_dir "${SFT_DIR}" \
  --cache_dir "${ROOT}/.hf_cache" \
  --init_adapter_dir "${INIT_ADAPTER}" \
  --seed 0 \
  --gpu_id 0 \
  --stage_i 1 \
  --total_empties_hint "${EMPTIES}" \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --num_epochs 24.0 \
  --learning_rate 2e-4 \
  --max_grad_norm 1.0 \
  "${GC_FLAGS[@]}" \
  --logging_steps 20 \
  --eval_steps 250 \
  --save_steps 100 \
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
  --max_steps "${MAX_STEPS}" \
  --use_wandb \
  --wandb_project "sudoku-multi-output-sft" \
  --wandb_run_name "stage01_sft99_i1_7empty_fromfast_${RUN_TAG}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_entity "${WANDB_ENTITY}"

SFT_FOR_GRPO="$(latest_checkpoint_in_dir "${SFT_DIR}")"
if [[ -z "${SFT_FOR_GRPO}" ]]; then
  printf 'ERROR: No SFT checkpoint under %s\n' "${SFT_DIR}" >&2
  exit 1
fi
printf '\n=== Phase 2: 7-empty GRPO (init from latest SFT in this run) ===\n'
printf 'Init adapter: %s\n' "${SFT_FOR_GRPO}"
printf 'Output: %s\n' "${GRPO_DIR}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${GRPO_SCRIPT}" \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_jsonl "${train_jsonl}" \
  --eval_jsonl "${eval_jsonl}" \
  --output_dir "${GRPO_DIR}" \
  --cache_dir "${ROOT}/.hf_cache" \
  --init_adapter_dir "${SFT_FOR_GRPO}" \
  --seed 0 \
  --gpu_id 0 \
  --stage_i 1 \
  --total_empties_hint "${EMPTIES}" \
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
  --wandb_run_name "stage01_grpo99_i1_7empty_fromfast_${RUN_TAG}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_entity "${WANDB_ENTITY}"

printf '\nDone. SFT: %s | GRPO: %s\n' "${SFT_DIR}" "${GRPO_DIR}"
