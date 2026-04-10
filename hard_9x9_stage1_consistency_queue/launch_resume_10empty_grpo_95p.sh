#!/usr/bin/env bash
# Resume 10-empty stage-1 GRPO from the latest HF checkpoint under the baseline run,
# continue training until value precision AND recall both reach GRPO_TARGET (default 0.95),
# or max_steps / optional wall clock.
#
# Usage:
#   ./launch_resume_10empty_grpo_95p.sh
#   GRPO_TARGET=0.96 MAX_STEPS=20000 ./launch_resume_10empty_grpo_95p.sh
#   RESUME_CKPT=/path/to/checkpoint-3500 ./launch_resume_10empty_grpo_95p.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
GRPO_SCRIPT="${ROOT}/multi_output_cell_policy/grpo_multi_output_train.py"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

# Baseline queue artifact tree (edit BASE_TAG if you use a different run tag)
BASE_TAG="${BASE_TAG:-20260407_093744}"
GRPO_DIR="${GRPO_DIR:-${ROOT}/final_checkpoint/hard_9x9_stage1_consistency_qwen05b/baseline/${BASE_TAG}/10empty/stage01_grpo_i1_10empty}"

TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/data/sudoku_t3_10empty_value_qwen_text_stage1_train.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-${ROOT}/data/sudoku_t3_10empty_value_qwen_text_stage1_eval.jsonl}"

TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
GRPO_TARGET="${GRPO_TARGET:-0.95}"
# Total optimizer steps (HF global step). Prior run stopped at 4000; raise so training can continue.
MAX_STEPS="${MAX_STEPS:-16000}"
# 0 = no wall-clock stop (same semantics as SFT script).
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"

if [[ -n "${RESUME_CKPT:-}" ]]; then
  resume_path="${RESUME_CKPT}"
else
  resume_path="$(ls -d "${GRPO_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -z "${resume_path}" || ! -d "${resume_path}" ]]; then
  printf 'ERROR: No GRPO checkpoint found under %s\n' "${GRPO_DIR}" >&2
  exit 1
fi
if [[ ! -f "${resume_path}/trainer_state.json" ]]; then
  printf 'ERROR: %s does not look like an HF trainer checkpoint (missing trainer_state.json)\n' "${resume_path}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

printf '\n=== Resume 10-empty GRPO → target prec+recall >= %s ===\n' "${GRPO_TARGET}"
printf 'Resume checkpoint: %s\n' "${resume_path}"
printf 'output_dir (same tree): %s\n' "${GRPO_DIR}"
printf 'max_steps (total): %s\n' "${MAX_STEPS}"
if [[ "${PHASE_WALL_CLOCK_SECONDS}" -gt 0 ]]; then
  printf 'wall cap: %ss\n' "${PHASE_WALL_CLOCK_SECONDS}"
else
  printf 'wall cap: none\n'
fi

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${GRPO_SCRIPT}" \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_jsonl "${TRAIN_JSONL}" \
  --eval_jsonl "${EVAL_JSONL}" \
  --output_dir "${GRPO_DIR}" \
  --cache_dir "${ROOT}/.hf_cache" \
  --init_adapter_dir "${resume_path}" \
  --resume_from_checkpoint "${resume_path}" \
  --seed 0 \
  --gpu_id 0 \
  --stage_i 1 \
  --total_empties_hint 10 \
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
  --max_steps "${MAX_STEPS}" \
  --use_wandb \
  --wandb_project "sudoku-multi-output-grpo" \
  --wandb_run_name "baseline_stage01_grpo_i1_10empty_resume95_${RUN_TAG}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_entity "${WANDB_ENTITY}"
