#!/usr/bin/env bash
# Run AFTER stage-1 SFT finishes (20-empty). Order:
#   1) Stage-1 GRPO   (init = your stage-1 SFT adapter)
#   2) Stage-2 SFT    (init = stage-1 GRPO adapter)
#   3) Stage-2 GRPO   (init = stage-2 SFT adapter)
#   4) Stage-3 SFT    (init = stage-2 GRPO adapter)
#   5) Stage-3 GRPO   (init = stage-3 SFT adapter)
#
# Each SFT/GRPO phase stops early only when BOTH eval value_precision AND value_recall
# are >= VALUE_TARGET (default 0.98). Other metric gates are disabled (0). Defaults use
# very large max_steps / epochs so in practice you exit on the 0.98 gate, not a low cap
# (override SFT_MAX_STEPS / GRPO_MAX_STEPS if you want a hard ceiling).
#
# Required (full pipeline from stage-1 SFT):
#   STAGE1_SFT_ADAPTER_DIR=/path/to/checkpoint-step-XXXXX
#
# Resume after stage-1 GRPO already ran (skip GRPO i=1, start at stage-2 SFT):
#   RESUME_FROM_STAGE1_GRPO_DIR=/path/to/stage01_grpo_i1_20empty
#   (OUTPUT_ROOT defaults to dirname of that dir.)
#
# Resume after stage-2 SFT already ran (skip through stage-2 SFT, start at stage-2 GRPO):
#   START_AT_STAGE2_GRPO_DIR=/path/to/stage02_sft_i2_20empty
#
# Resume after stage-2 GRPO finished (stage-3 SFT + stage-3 GRPO only):
#   START_AFTER_STAGE2_GRPO_DIR=/path/to/stage02_grpo_i2_20empty
#
# Optional:
#   VALUE_TARGET=0.98 SFT_MAX_STEPS=... GRPO_MAX_STEPS=... SFT_NUM_EPOCHS=... GRPO_NUM_TRAIN_EPOCHS=...
#   TRAIN_PUZZLES=10000 EVAL_PUZZLES=100 RUN_TAG=... CHECKPOINT_ROOT=... USE_GC=1 PHASE_WALL_CLOCK_SECONDS=0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
SFT_SCRIPT="${ROOT}/multi_output_cell_policy/sft_multi_output_train.py"
GRPO_SCRIPT="${ROOT}/multi_output_cell_policy/grpo_multi_output_train.py"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

EMPTIES=20
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
VALUE_TARGET="${VALUE_TARGET:-0.98}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-10000000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-10000000}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-512}"
GRPO_NUM_TRAIN_EPOCHS="${GRPO_NUM_TRAIN_EPOCHS:-200}"
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_20empty_stages123_value98}"
START_AT_STAGE2_GRPO_DIR="${START_AT_STAGE2_GRPO_DIR:-}"
START_AFTER_STAGE2_GRPO_DIR="${START_AFTER_STAGE2_GRPO_DIR:-}"
RESUME_FROM_STAGE1_GRPO_DIR="${RESUME_FROM_STAGE1_GRPO_DIR:-}"

if [[ -n "${START_AT_STAGE2_GRPO_DIR}" ]]; then
  if [[ ! -d "${START_AT_STAGE2_GRPO_DIR}" ]]; then
    printf 'ERROR: START_AT_STAGE2_GRPO_DIR is not a directory: %s\n' "${START_AT_STAGE2_GRPO_DIR}" >&2
    exit 1
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "${START_AT_STAGE2_GRPO_DIR}")}"
elif [[ -n "${START_AFTER_STAGE2_GRPO_DIR}" ]]; then
  if [[ ! -d "${START_AFTER_STAGE2_GRPO_DIR}" ]]; then
    printf 'ERROR: START_AFTER_STAGE2_GRPO_DIR is not a directory: %s\n' "${START_AFTER_STAGE2_GRPO_DIR}" >&2
    exit 1
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "${START_AFTER_STAGE2_GRPO_DIR}")}"
elif [[ -n "${RESUME_FROM_STAGE1_GRPO_DIR}" ]]; then
  if [[ ! -d "${RESUME_FROM_STAGE1_GRPO_DIR}" ]]; then
    printf 'ERROR: RESUME_FROM_STAGE1_GRPO_DIR is not a directory: %s\n' "${RESUME_FROM_STAGE1_GRPO_DIR}" >&2
    exit 1
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "${RESUME_FROM_STAGE1_GRPO_DIR}")}"
else
  if [[ -z "${STAGE1_SFT_ADAPTER_DIR:-}" ]] || [[ ! -d "${STAGE1_SFT_ADAPTER_DIR}" ]]; then
    printf 'ERROR: Set STAGE1_SFT_ADAPTER_DIR to a finished stage-1 SFT checkpoint directory, or RESUME_FROM_STAGE1_GRPO_DIR, START_AT_STAGE2_GRPO_DIR, or START_AFTER_STAGE2_GRPO_DIR.\n' >&2
    exit 1
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"
fi

train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

mkdir -p "${OUTPUT_ROOT}"

latest_sft_step_ckpt() {
  local d="$1"
  shopt -s nullglob
  local cks=("${d}"/checkpoint-step-*)
  shopt -u nullglob
  if (( ${#cks[@]} == 0 )); then
    printf ''
    return 1
  fi
  set +o pipefail
  printf '%s\n' "${cks[@]}" | sort -V | tail -n 1
  set -o pipefail
}

resolve_grpo_adapter() {
  local d="$1"
  if [[ -f "${d}/adapter_model.safetensors" ]]; then
    printf '%s\n' "${d}"
    return 0
  fi
  local best="" step=-1
  shopt -s nullglob
  local c
  for c in "${d}"/checkpoint-*; do
    [[ -d "${c}" ]] || continue
    [[ -f "${c}/adapter_model.safetensors" ]] || continue
    local n
    n="${c##*checkpoint-}"
    if [[ "${n}" =~ ^[0-9]+$ ]] && (( 10#${n} >= step )); then
      step=$((10#${n}))
      best="${c}"
    fi
  done
  shopt -u nullglob
  if [[ -n "${best}" ]]; then
    printf '%s\n' "${best}"
    return 0
  fi
  printf ''
  return 1
}

GC_FLAGS=()
if [[ "${USE_GC:-0}" == "1" ]]; then
  GC_FLAGS+=(--enable_gradient_checkpointing)
fi

run_sft() {
  local stage="$1"
  local init_adapter="$2"
  local out_dir="$3"
  local lr="$4"
  mkdir -p "${out_dir}"
  printf '\n=== Stage %s SFT -> stop when value prec+recall >= %s (max_steps=%s epochs=%s) ===\n' "${stage}" "${VALUE_TARGET}" "${SFT_MAX_STEPS}" "${SFT_NUM_EPOCHS}" >&2
  printf 'init=%s\nout=%s\n' "${init_adapter}" "${out_dir}" >&2
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --train_jsonl "${train_jsonl}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${out_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${init_adapter}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i "${stage}" \
    --total_empties_hint "${EMPTIES}" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_epochs "${SFT_NUM_EPOCHS}" \
    --learning_rate "${lr}" \
    --max_grad_norm 1.0 \
    "${GC_FLAGS[@]}" \
    --logging_steps 20 \
    --eval_steps 250 \
    --save_steps 200 \
    --eval_rows "${EVAL_PUZZLES}" \
    --max_completion_length 24 \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --eval_value_precision_stop "${VALUE_TARGET}" \
    --eval_value_recall_stop "${VALUE_TARGET}" \
    --eval_exact_set_match_stop 0 \
    --eval_solve_rate_stop 0 \
    --min_steps_before_stop 50 \
    --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
    --max_steps "${SFT_MAX_STEPS}" \
    --use_wandb \
    --wandb_project "sudoku-multi-output-sft" \
    --wandb_run_name "postS1_st${stage}_sft_i${stage}_${EMPTIES}empty_val${VALUE_TARGET}_${RUN_TAG}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

run_grpo() {
  local stage="$1"
  local init_adapter="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  printf '\n=== Stage %s GRPO -> stop when value prec+recall >= %s (max_steps=%s num_train_epochs=%s) ===\n' "${stage}" "${VALUE_TARGET}" "${GRPO_MAX_STEPS}" "${GRPO_NUM_TRAIN_EPOCHS}" >&2
  printf 'init=%s\nout=%s\n' "${init_adapter}" "${out_dir}" >&2
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${GRPO_SCRIPT}" \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --train_jsonl "${train_jsonl}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${out_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${init_adapter}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i "${stage}" \
    --total_empties_hint "${EMPTIES}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs "${GRPO_NUM_TRAIN_EPOCHS}" \
    --learning_rate 1e-6 \
    --logging_steps 20 \
    --save_steps 200 \
    --eval_steps 500 \
    --eval_rows "${EVAL_PUZZLES}" \
    --num_generations 4 \
    --max_prompt_length 1024 \
    --max_completion_length 24 \
    --beta 0.0 \
    --enable_gradient_checkpointing \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --reward_good_value 1.25 \
    --penalty_bad_value 1.0 \
    --penalty_malformed 4.0 \
    --penalty_empty 0.5 \
    --penalty_singleton 1.5 \
    --eval_value_precision_stop "${VALUE_TARGET}" \
    --eval_value_recall_stop "${VALUE_TARGET}" \
    --eval_solve_rate_stop 0 \
    --min_steps_before_stop 50 \
    --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
    --max_steps "${GRPO_MAX_STEPS}" \
    --use_wandb \
    --wandb_project "sudoku-multi-output-grpo" \
    --wandb_run_name "postS1_st${stage}_grpo_i${stage}_${EMPTIES}empty_val${VALUE_TARGET}_${RUN_TAG}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

if [[ ! -f "${train_jsonl}" ]] || [[ ! -f "${eval_jsonl}" ]]; then
  printf 'ERROR: Missing train/eval jsonl. Build stage-1 datasets first (see launch_sft_stage1_95p.sh / build_dataset.py).\n' >&2
  printf '  %s\n  %s\n' "${train_jsonl}" "${eval_jsonl}" >&2
  exit 1
fi

if [[ -n "${START_AT_STAGE2_GRPO_DIR}" ]]; then
  printf 'Fast-forward: stage-2 SFT dir %s -> stage-2 GRPO, then stage 3.\n' "${START_AT_STAGE2_GRPO_DIR}" >&2
  printf 'Pipeline root: %s\n' "${OUTPUT_ROOT}"
  S2_DIR="${START_AT_STAGE2_GRPO_DIR}"
  CKPT_S2="$(latest_sft_step_ckpt "${S2_DIR}")"
  if [[ -z "${CKPT_S2}" ]]; then
    printf 'ERROR: No checkpoint-step-* under %s\n' "${S2_DIR}" >&2
    exit 1
  fi
  printf 'Using SFT checkpoint: %s\n' "${CKPT_S2}" >&2
  G2_DIR="${OUTPUT_ROOT}/stage02_grpo_i2_${EMPTIES}empty"
  run_grpo 2 "${CKPT_S2}" "${G2_DIR}"
  A2="$(resolve_grpo_adapter "${G2_DIR}")"
  if [[ -z "${A2}" ]]; then
    printf 'ERROR: Could not resolve stage-2 GRPO adapter under %s\n' "${G2_DIR}" >&2
    exit 1
  fi
  S3_DIR="${OUTPUT_ROOT}/stage03_sft_i3_${EMPTIES}empty"
  run_sft 3 "${A2}" "${S3_DIR}" "5e-5"
  CKPT_S3="$(latest_sft_step_ckpt "${S3_DIR}")"
  if [[ -z "${CKPT_S3}" ]]; then
    printf 'ERROR: No SFT checkpoint-step-* under %s\n' "${S3_DIR}" >&2
    exit 1
  fi
  G3_DIR="${OUTPUT_ROOT}/stage03_grpo_i3_${EMPTIES}empty"
  run_grpo 3 "${CKPT_S3}" "${G3_DIR}"
  A3="$(resolve_grpo_adapter "${G3_DIR}")"
  if [[ -z "${A3}" ]]; then
    printf 'ERROR: Could not resolve stage-3 GRPO adapter under %s\n' "${G3_DIR}" >&2
    exit 1
  fi
  printf '\nAll phases finished (started at stage-2 GRPO).\n'
  printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
  printf 'Final GRPO adapter (stage 3): %s\n' "${A3}"
  exit 0
fi

if [[ -n "${START_AFTER_STAGE2_GRPO_DIR}" ]]; then
  printf 'Fast-forward: stage-2 GRPO dir %s -> stage-3 SFT + stage-3 GRPO.\n' "${START_AFTER_STAGE2_GRPO_DIR}" >&2
  printf 'Pipeline root: %s\n' "${OUTPUT_ROOT}"
  A2="$(resolve_grpo_adapter "${START_AFTER_STAGE2_GRPO_DIR}")"
  if [[ -z "${A2}" ]]; then
    printf 'ERROR: Could not resolve stage-2 GRPO adapter under %s\n' "${START_AFTER_STAGE2_GRPO_DIR}" >&2
    exit 1
  fi
  printf 'Using stage-2 GRPO adapter: %s\n' "${A2}" >&2
  S3_DIR="${OUTPUT_ROOT}/stage03_sft_i3_${EMPTIES}empty"
  run_sft 3 "${A2}" "${S3_DIR}" "5e-5"
  CKPT_S3="$(latest_sft_step_ckpt "${S3_DIR}")"
  if [[ -z "${CKPT_S3}" ]]; then
    printf 'ERROR: No SFT checkpoint-step-* under %s\n' "${S3_DIR}" >&2
    exit 1
  fi
  G3_DIR="${OUTPUT_ROOT}/stage03_grpo_i3_${EMPTIES}empty"
  run_grpo 3 "${CKPT_S3}" "${G3_DIR}"
  A3="$(resolve_grpo_adapter "${G3_DIR}")"
  if [[ -z "${A3}" ]]; then
    printf 'ERROR: Could not resolve stage-3 GRPO adapter under %s\n' "${G3_DIR}" >&2
    exit 1
  fi
  printf '\nAll phases finished (started after stage-2 GRPO).\n'
  printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
  printf 'Final GRPO adapter (stage 3): %s\n' "${A3}"
  exit 0
fi

printf 'Pipeline root: %s\n' "${OUTPUT_ROOT}"
if [[ -n "${RESUME_FROM_STAGE1_GRPO_DIR}" ]]; then
  printf 'Resume: using existing stage-1 GRPO dir %s\n' "${RESUME_FROM_STAGE1_GRPO_DIR}"
else
  printf 'Stage-1 SFT adapter: %s\n' "${STAGE1_SFT_ADAPTER_DIR}"
fi
printf 'Value gate: precision AND recall >= %s | SFT max_steps=%s epochs=%s | GRPO max_steps=%s train_epochs=%s | wall=%s\n' \
  "${VALUE_TARGET}" "${SFT_MAX_STEPS}" "${SFT_NUM_EPOCHS}" "${GRPO_MAX_STEPS}" "${GRPO_NUM_TRAIN_EPOCHS}" "${PHASE_WALL_CLOCK_SECONDS}"

G1_DIR="${OUTPUT_ROOT}/stage01_grpo_i1_${EMPTIES}empty"
if [[ -n "${RESUME_FROM_STAGE1_GRPO_DIR}" ]]; then
  A1="$(resolve_grpo_adapter "${RESUME_FROM_STAGE1_GRPO_DIR}")"
else
  run_grpo 1 "${STAGE1_SFT_ADAPTER_DIR}" "${G1_DIR}"
  A1="$(resolve_grpo_adapter "${G1_DIR}")"
fi
if [[ -z "${A1}" ]]; then
  printf 'ERROR: Could not resolve stage-1 GRPO adapter (resume dir or %s)\n' "${G1_DIR}" >&2
  exit 1
fi
printf 'Stage-1 GRPO adapter for stage-2 SFT init: %s\n' "${A1}"

S2_DIR="${OUTPUT_ROOT}/stage02_sft_i2_${EMPTIES}empty"
run_sft 2 "${A1}" "${S2_DIR}" "5e-5"
CKPT_S2="$(latest_sft_step_ckpt "${S2_DIR}")"
if [[ -z "${CKPT_S2}" ]]; then
  printf 'ERROR: No SFT checkpoint-step-* under %s\n' "${S2_DIR}" >&2
  exit 1
fi
G2_DIR="${OUTPUT_ROOT}/stage02_grpo_i2_${EMPTIES}empty"
run_grpo 2 "${CKPT_S2}" "${G2_DIR}"
A2="$(resolve_grpo_adapter "${G2_DIR}")"
if [[ -z "${A2}" ]]; then
  printf 'ERROR: Could not resolve stage-2 GRPO adapter under %s\n' "${G2_DIR}" >&2
  exit 1
fi

S3_DIR="${OUTPUT_ROOT}/stage03_sft_i3_${EMPTIES}empty"
run_sft 3 "${A2}" "${S3_DIR}" "5e-5"
CKPT_S3="$(latest_sft_step_ckpt "${S3_DIR}")"
if [[ -z "${CKPT_S3}" ]]; then
  printf 'ERROR: No SFT checkpoint-step-* under %s\n' "${S3_DIR}" >&2
  exit 1
fi
G3_DIR="${OUTPUT_ROOT}/stage03_grpo_i3_${EMPTIES}empty"
run_grpo 3 "${CKPT_S3}" "${G3_DIR}"
A3="$(resolve_grpo_adapter "${G3_DIR}")"
if [[ -z "${A3}" ]]; then
  printf 'ERROR: Could not resolve stage-3 GRPO adapter under %s\n' "${G3_DIR}" >&2
  exit 1
fi

printf '\nAll phases finished.\n'
printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
printf 'Final GRPO adapter (stage 3): %s\n' "${A3}"
