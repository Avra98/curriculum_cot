#!/usr/bin/env bash
# Stage-1 fixed-slot latent SFT for 20-empty: train until eval value_precision AND
# value_recall both reach 0.98. This uses prompt + z1 + final_slot during stage 1,
# while still updating LoRA weights so the transformer can learn how to use z1.
#
# Fresh run:
#   ./launch_20empty_fixed_slot_sft_stage1_98p.sh
#
# Warm-start from a prior checkpoint:
#   INIT_ADAPTER_DIR=/path/to/checkpoint-step-XXXXX ./launch_20empty_fixed_slot_sft_stage1_98p.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET_BUILDER="${ROOT}/simple_9x9_curriculum/build_dataset.py"
SFT_SCRIPT="${ROOT}/latent_multi_output_cell_policy/sft_latent_multi_output_train.py"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
NUM_PROCESSES="${NUM_PROCESSES:-7}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

EMPTIES=20
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
SFT_TARGET="${SFT_TARGET:-0.98}"
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"
MAX_STEPS="${MAX_STEPS:-30000}"

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_LATENT_SLOTS="${MAX_LATENT_SLOTS:-3}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_20empty_fixed_slot_sft98_stage1}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/${RUN_TAG}/${EMPTIES}empty/stage01_fixed_slot_sft98_i1_${EMPTIES}empty}"

train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${train_jsonl}" ]]; then
  mkdir -p "$(dirname "${train_jsonl}")"
  printf 'Building %s-empty train dataset: %s\n' "${EMPTIES}" "${train_jsonl}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${train_jsonl}" --num_puzzles "${TRAIN_PUZZLES}" --empties "${EMPTIES}" --seed 0
fi
if [[ ! -f "${eval_jsonl}" ]]; then
  mkdir -p "$(dirname "${eval_jsonl}")"
  printf 'Building %s-empty eval dataset: %s\n' "${EMPTIES}" "${eval_jsonl}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${eval_jsonl}" --num_puzzles "${EVAL_PUZZLES}" --empties "${EMPTIES}" --seed 1
fi

mkdir -p "${OUTPUT_DIR}"

INIT_FLAGS=()
if [[ -n "${INIT_ADAPTER_DIR:-}" ]]; then
  INIT_FLAGS+=(--init_adapter_dir "${INIT_ADAPTER_DIR}")
  printf 'Warm-start from adapter: %s\n' "${INIT_ADAPTER_DIR}"
fi

GC_FLAGS=()
if [[ "${USE_GC:-1}" == "1" ]]; then
  GC_FLAGS+=(--enable_gradient_checkpointing)
  printf 'NOTE: USE_GC=1 - slower, less VRAM.\n'
fi

if [[ "${PHASE_WALL_CLOCK_SECONDS}" -gt 0 ]]; then
  printf '\n=== Stage1 fixed-slot SFT %s-empty (prec+recall >= %s, wall %ss) ===\n' "${EMPTIES}" "${SFT_TARGET}" "${PHASE_WALL_CLOCK_SECONDS}"
else
  printf '\n=== Stage1 fixed-slot SFT %s-empty (prec+recall >= %s, no wall cap) ===\n' "${EMPTIES}" "${SFT_TARGET}"
fi
printf 'Output: %s\n' "${OUTPUT_DIR}"
printf 'LoRA: r=%s alpha=%s dropout=%s | latent_mode=fixed_slots | active_z=1 | max_latent_slots=%s\n' "${LORA_R}" "${LORA_ALPHA}" "${LORA_DROPOUT}" "${MAX_LATENT_SLOTS}"
printf 'DDP: visible_gpus=%s nproc=%s | batch/device=%s grad_accum=%s\n' "${GPU_IDS}" "${NUM_PROCESSES}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}"

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}" \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_jsonl "${train_jsonl}" \
  --eval_jsonl "${eval_jsonl}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${ROOT}/.hf_cache" \
  "${INIT_FLAGS[@]}" \
  --seed 0 \
  --gpu_id 0 \
  --stage_i 1 \
  --num_cot_tokens 1 \
  --latent_mode fixed_slots \
  --max_latent_slots "${MAX_LATENT_SLOTS}" \
  --total_empties_hint "${EMPTIES}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_epochs 64.0 \
  --learning_rate 2e-4 \
  --max_grad_norm 1.0 \
  "${GC_FLAGS[@]}" \
  --logging_steps 20 \
  --eval_steps 250 \
  --save_steps 100 \
  --eval_rows "${EVAL_PUZZLES}" \
  --max_completion_length 24 \
  --limit_train_rows "${TRAIN_PUZZLES}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --eval_value_precision_stop "${SFT_TARGET}" \
  --eval_value_recall_stop "${SFT_TARGET}" \
  --eval_exact_set_match_stop 0 \
  --eval_solve_rate_stop 0 \
  --min_steps_before_stop 50 \
  --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
  --max_steps "${MAX_STEPS}" \
  --use_wandb \
  --wandb_project "sudoku-fixed-slot-sft" \
  --wandb_run_name "${WANDB_RUN_NAME:-stage01_fixed_slot_sft98_i1_${EMPTIES}empty_${RUN_TAG}}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_entity "${WANDB_ENTITY}"
