#!/usr/bin/env bash
# Full 7-empty stage-1 SFT retrain tuned for speed: aim for >=90% value precision+recall
# (stops early when both hit SFT_TARGET). For a 95% SFT goal on any empty count, use
# launch_sft_stage1_95p.sh instead (still SFT-only).
# (early stop when both hit SFT_TARGET). No wall-clock cap by default; set
# PHASE_WALL_CLOCK_SECONDS=3600 (or any positive value) if you want a time limit.
# Smaller eval sets than the default queue; disables gradient checkpointing for throughput
# (set USE_GC=1 if OOM).
#
# Usage:
#   ./launch_fast_7empty_sft_90p_1h.sh
#   RUN_TAG=myrun TRAIN_PUZZLES=10000 ./launch_fast_7empty_sft_90p_1h.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET_BUILDER="${ROOT}/simple_9x9_curriculum/build_dataset.py"
SFT_SCRIPT="${ROOT}/multi_output_cell_policy/sft_multi_output_train.py"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

# Slightly fewer train puzzles = fewer examples / epoch (faster iterations); still enough for ~90%.
TRAIN_PUZZLES="${TRAIN_PUZZLES:-8000}"
# Smaller held-out eval = much faster generate()-based eval passes.
EVAL_PUZZLES="${EVAL_PUZZLES:-64}"
# Stop once both metrics hit this (and min_steps_before_stop reached).
SFT_TARGET="${SFT_TARGET:-0.90}"
# 0 = no wall-clock stop (trainer treats 0 as disabled).
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"

# Separate tree so this never overwrites baseline queue checkpoints.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_fast90_1h}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/${RUN_TAG}/7empty/stage01_sft_i1_7empty_fast}"

EMPTIES=7
train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ ! -f "${train_jsonl}" ]]; then
  mkdir -p "$(dirname "${train_jsonl}")"
  printf 'Building %s-empty train dataset: %s\n' "${EMPTIES}" "${train_jsonl}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${train_jsonl}" --num_puzzles 10000 --empties "${EMPTIES}" --seed 0
fi
if [[ ! -f "${eval_jsonl}" ]]; then
  mkdir -p "$(dirname "${eval_jsonl}")"
  printf 'Building %s-empty eval dataset: %s\n' "${EMPTIES}" "${eval_jsonl}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" --output "${eval_jsonl}" --num_puzzles 200 --empties "${EMPTIES}" --seed 1
fi

mkdir -p "${OUTPUT_DIR}"

GC_FLAGS=()
if [[ "${USE_GC:-0}" == "1" ]]; then
  GC_FLAGS+=(--enable_gradient_checkpointing)
  printf 'NOTE: USE_GC=1 (gradient checkpointing on) — slower but uses less memory.\n'
fi

if [[ "${PHASE_WALL_CLOCK_SECONDS}" -gt 0 ]]; then
  printf '\n=== Fast 7-empty stage1 SFT (target prec+recall >= %s, wall cap %ss) ===\n' "${SFT_TARGET}" "${PHASE_WALL_CLOCK_SECONDS}"
else
  printf '\n=== Fast 7-empty stage1 SFT (target prec+recall >= %s, no wall cap) ===\n' "${SFT_TARGET}"
fi
printf 'Output: %s\n' "${OUTPUT_DIR}"
printf 'Train cap: %s puzzles | Eval: %s puzzles per pass\n' "${TRAIN_PUZZLES}" "${EVAL_PUZZLES}"

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}"
  --model_name "Qwen/Qwen2.5-0.5B-Instruct"
  --train_jsonl "${train_jsonl}"
  --eval_jsonl "${eval_jsonl}"
  --output_dir "${OUTPUT_DIR}"
  --cache_dir "${ROOT}/.hf_cache"
  --seed 0
  --gpu_id 0
  --stage_i 1
  --total_empties_hint "${EMPTIES}"
  --per_device_train_batch_size 16
  --gradient_accumulation_steps 2
  --num_epochs 16.0
  --learning_rate 2e-4
  --max_grad_norm 1.0
  "${GC_FLAGS[@]}"
  --logging_steps 20
  --eval_steps 200
  --save_steps 100
  --eval_rows "${EVAL_PUZZLES}"
  --max_completion_length 24
  --limit_train_rows "${TRAIN_PUZZLES}"
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --eval_value_precision_stop "${SFT_TARGET}"
  --eval_value_recall_stop "${SFT_TARGET}"
  --min_steps_before_stop 40
  --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}"
  --max_steps 5000
  --use_wandb
  --wandb_project "sudoku-multi-output-sft"
  --wandb_run_name "fast90_1h_stage01_sft_i1_7empty_${RUN_TAG}"
  --wandb_mode "${WANDB_MODE}"
  --wandb_entity "${WANDB_ENTITY}"
)

exec "${cmd[@]}"
