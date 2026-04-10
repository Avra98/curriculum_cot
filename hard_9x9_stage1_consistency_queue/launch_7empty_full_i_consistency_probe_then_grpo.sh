#!/usr/bin/env bash
# Long 7-empty run: warm-start SFT from fast checkpoint, no metric early-stop (thresholds 0),
# high max_steps — see how far I-consistency / value metrics go. Then GRPO from latest SFT
# checkpoint, also without precision/recall early-stop, bounded by max_steps only.
#
# Default init: fast run checkpoint-step-01200 (override with INIT_ADAPTER_DIR or FAST_SFT_DIR).
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
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"
# SFT: run until max_steps (no early stop on eval metrics).
SFT_MAX_STEPS="${SFT_MAX_STEPS:-40000}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-64}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-500}"
# GRPO: must set max_steps>0 when using limit_train_rows (trainer forces 1 otherwise).
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-8000}"
GRPO_EVAL_STEPS="${GRPO_EVAL_STEPS:-500}"

DEFAULT_FAST_ADAPTER="${ROOT}/final_checkpoint/hard_9x9_fast90_1h/20260407_174437/7empty/stage01_sft_i1_7empty_fast/checkpoint-step-01200"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_7empty_full_i_probe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"
SFT_DIR="${OUTPUT_ROOT}/${EMPTIES}empty/stage01_sft_i1_${EMPTIES}empty_fullprobe"
GRPO_DIR="${OUTPUT_ROOT}/${EMPTIES}empty/stage01_grpo_i1_${EMPTIES}empty_fullprobe"

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
    printf '%s\n' "${INIT_ADAPTER_DIR}"
    return 0
  fi
  if [[ -n "${FAST_SFT_DIR:-}" ]]; then
    local latest
    latest="$(latest_checkpoint_in_dir "${FAST_SFT_DIR}" 2>/dev/null || true)"
    if [[ -z "${latest}" ]]; then
      printf 'ERROR: No checkpoint-step-* under FAST_SFT_DIR=%s\n' "${FAST_SFT_DIR}" >&2
      exit 1
    fi
    printf '%s\n' "${latest}"
    return 0
  fi
  if [[ -d "${DEFAULT_FAST_ADAPTER}" ]]; then
    printf '%s\n' "${DEFAULT_FAST_ADAPTER}"
    return 0
  fi
  printf 'ERROR: No INIT_ADAPTER_DIR/FAST_SFT_DIR and default fast adapter missing: %s\n' "${DEFAULT_FAST_ADAPTER}" >&2
  exit 1
}

INIT_ADAPTER="$(resolve_init_adapter)"
if [[ ! -d "${INIT_ADAPTER}" ]]; then
  printf 'ERROR: adapter dir not found: %s\n' "${INIT_ADAPTER}" >&2
  exit 1
fi
printf 'SFT warm-start adapter: %s\n' "${INIT_ADAPTER}"

if [[ ! -f "${train_jsonl}" ]]; then
  mkdir -p "$(dirname "${train_jsonl}")"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${train_jsonl}" --num_puzzles "${TRAIN_PUZZLES}" --empties "${EMPTIES}" --seed 0
fi
if [[ ! -f "${eval_jsonl}" ]]; then
  mkdir -p "$(dirname "${eval_jsonl}")"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${eval_jsonl}" --num_puzzles 200 --empties "${EMPTIES}" --seed 1
fi

mkdir -p "${SFT_DIR}" "${GRPO_DIR}"

GC_FLAGS=()
if [[ "${USE_GC:-0}" == "1" ]]; then
  GC_FLAGS+=(--enable_gradient_checkpointing)
fi

printf '\n=== Phase 1: SFT full probe (no metric early-stop, max_steps=%s) ===\n' "${SFT_MAX_STEPS}"
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
  --num_epochs "${SFT_NUM_EPOCHS}" \
  --learning_rate 2e-4 \
  --max_grad_norm 1.0 \
  "${GC_FLAGS[@]}" \
  --logging_steps 50 \
  --eval_steps "${SFT_EVAL_STEPS}" \
  --save_steps 500 \
  --eval_rows "${EVAL_PUZZLES}" \
  --max_completion_length 24 \
  --limit_train_rows "${TRAIN_PUZZLES}" \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --eval_value_precision_stop 0 \
  --eval_value_recall_stop 0 \
  --eval_exact_set_match_stop 0 \
  --eval_solve_rate_stop 0 \
  --min_steps_before_stop 0 \
  --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
  --max_steps "${SFT_MAX_STEPS}" \
  --use_wandb \
  --wandb_project "sudoku-multi-output-sft" \
  --wandb_run_name "stage01_sft_fullprobe_i1_7empty_${RUN_TAG}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_entity "${WANDB_ENTITY}"

SFT_FOR_GRPO="$(latest_checkpoint_in_dir "${SFT_DIR}")"
if [[ -z "${SFT_FOR_GRPO}" ]]; then
  printf 'ERROR: No SFT checkpoint under %s\n' "${SFT_DIR}" >&2
  exit 1
fi

printf '\n=== Phase 2: GRPO full probe (no metric early-stop, max_steps=%s) ===\n' "${GRPO_MAX_STEPS}"
printf 'Init: %s\n' "${SFT_FOR_GRPO}"

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
  --num_train_epochs 8.0 \
  --learning_rate 1e-6 \
  --logging_steps 20 \
  --save_steps 200 \
  --eval_steps "${GRPO_EVAL_STEPS}" \
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
  --eval_value_precision_stop 0 \
  --eval_value_recall_stop 0 \
  --eval_solve_rate_stop 0 \
  --min_steps_before_stop 0 \
  --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
  --max_steps "${GRPO_MAX_STEPS}" \
  --use_wandb \
  --wandb_project "sudoku-multi-output-grpo" \
  --wandb_run_name "stage01_grpo_fullprobe_i1_7empty_${RUN_TAG}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_entity "${WANDB_ENTITY}"

printf '\nDone. SFT: %s | GRPO: %s\n' "${SFT_DIR}" "${GRPO_DIR}"
