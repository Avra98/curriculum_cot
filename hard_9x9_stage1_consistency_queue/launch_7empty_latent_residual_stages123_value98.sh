#!/usr/bin/env bash
# Latent residual projector pipeline (7-empty), aligned with the text
# launch_7empty_post_s1sft_stages123_value98.sh order and value gate:
#   1) Stage-1 SFT    (default: init = STAGE1_INIT_ADAPTER_DIR or fresh LoRA + random residual)
#   2) Stage-1 GRPO   (init = stage-1 SFT checkpoint-step-* dir, or STAGE1_SFT_ADAPTER_DIR if set)
#   3) Stage-2 SFT    (init = stage-1 GRPO adapter)
#   4) Stage-2 GRPO
#   5) Stage-3 SFT
#   6) Stage-3 GRPO
#
# Legacy GRPO-first (skip training stage-1 SFT): STAGE1_GRPO_FIRST=1
#
# Latent structure (implemented in latent_multi_output_cell_policy/grpo_residual_projector_latent_train.py):
#   - attach_residual_projector_modules(): adds trainable special_thought_embed, latent_mix_logit,
#     and MLP latent_projector_in/out (hidden→4096→hidden) on the Peft-wrapped model.
#   - build_latent_hidden() / residual_next_token_logits_from_ids(): append num_cot_tokens "latent"
#     virtual tokens, run backbone, take (latent_hidden - base_hidden), project through the MLP,
#     mix with base hidden (sigmoid(latent_mix_logit)), then lm_head logits (with optional fallback).
#   - sample_latent_completion() / GRPO use this path for generation; SFT uses the same via
#     residual_projector_warmstart_sft_latent_multi_output_train.py (latent_residual_completion_ce_loss).
#   - latent_cot_state.pt saves/loads the projector + special_thought_embed + mix logit.
#
# Each phase stops when eval value_precision AND value_recall are both >= VALUE_TARGET
# (default 0.98), after MIN_STEPS_BEFORE_STOP optimizer steps (SFT) / GRPO steps (GRPO).
# Eval rows come from eval_jsonl (same held-out file as the text pipeline).
#
# Stage-1 SFT init (when not using STAGE1_SFT_ADAPTER_DIR or STAGE1_GRPO_FIRST):
#   Default: omit STAGE1_INIT_ADAPTER_DIR → fresh LoRA + random residual (same as trainers --init_adapter_dir "").
#   Optional: STAGE1_INIT_ADAPTER_DIR=/path/to/adapter
#
# Skip running stage-1 SFT (you already have a finished SFT checkpoint-step-*):
#   STAGE1_SFT_ADAPTER_DIR=/path/to/stage01_sft_.../checkpoint-step-XXXX
#   → first trained phase is stage-1 GRPO with that init.
#
# Resume:
#   RESUME_FROM_STAGE1_GRPO_DIR=/path/to/stage01_grpo_i1_7empty_latent_residual
#   START_AT_STAGE2_GRPO_DIR=/path/to/stage02_sft_i2_7empty_latent_residual
#   START_AFTER_STAGE2_GRPO_DIR=/path/to/stage02_grpo_i2_7empty_latent_residual
#
# Optional env: VALUE_TARGET, TRAIN_PUZZLES, EVAL_PUZZLES, RUN_TAG, CHECKPOINT_ROOT, GPU_IDS,
#   WANDB_MODE, WANDB_ENTITY, SFT_NUM_EPOCHS, GRPO_NUM_TRAIN_EPOCHS, SFT_MAX_STEPS, GRPO_MAX_STEPS,
#   STAGE1_SFT_LR (default 2e-4), STAGE1_GRPO_FIRST, STAGE1_SFT_ADAPTER_DIR
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
SFT_SCRIPT="${ROOT}/latent_multi_output_cell_policy/residual_projector_warmstart_sft_latent_multi_output_train.py"
GRPO_SCRIPT="${ROOT}/latent_multi_output_cell_policy/grpo_residual_projector_latent_train.py"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

EMPTIES=7
TAG_SUFFIX="latent_residual"
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
VALUE_TARGET="${VALUE_TARGET:-0.98}"
MIN_STEPS_BEFORE_STOP="${MIN_STEPS_BEFORE_STOP:-50}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-10000000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-10000000}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-512}"
GRPO_NUM_TRAIN_EPOCHS="${GRPO_NUM_TRAIN_EPOCHS:-200}"
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
# Fresh-LoRA defaults (match text 7-empty SFT scale); override if you use a different init checkpoint.
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_7empty_latent_residual_stages123_value98}"
START_AT_STAGE2_GRPO_DIR="${START_AT_STAGE2_GRPO_DIR:-}"
START_AFTER_STAGE2_GRPO_DIR="${START_AFTER_STAGE2_GRPO_DIR:-}"
RESUME_FROM_STAGE1_GRPO_DIR="${RESUME_FROM_STAGE1_GRPO_DIR:-}"
STAGE1_SFT_ADAPTER_DIR="${STAGE1_SFT_ADAPTER_DIR:-}"
STAGE1_GRPO_FIRST="${STAGE1_GRPO_FIRST:-0}"
STAGE1_SFT_LR="${STAGE1_SFT_LR:-2e-4}"

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
  if [[ -n "${STAGE1_INIT_ADAPTER_DIR:-}" ]] && [[ ! -d "${STAGE1_INIT_ADAPTER_DIR}" ]]; then
    printf 'ERROR: STAGE1_INIT_ADAPTER_DIR is not a directory: %s\n' "${STAGE1_INIT_ADAPTER_DIR}" >&2
    exit 1
  fi
  if [[ -n "${STAGE1_SFT_ADAPTER_DIR}" ]] && [[ ! -d "${STAGE1_SFT_ADAPTER_DIR}" ]]; then
    printf 'ERROR: STAGE1_SFT_ADAPTER_DIR is not a directory: %s\n' "${STAGE1_SFT_ADAPTER_DIR}" >&2
    exit 1
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"
fi

train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

resolve_latent_grpo_adapter() {
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

run_latent_sft() {
  local stage="$1"
  local init_adapter="$2"
  local out_dir="$3"
  local lr="$4"
  local cot="$5"
  # Stage-1 SFT must weight stage-1 rows only (mixed 1/0). Stages 2–3 use stage-i curriculum (mixed 0/1).
  local ms1=0 ms2=1
  if [[ "${stage}" == "1" ]]; then
    ms1=1
    ms2=0
  fi
  mkdir -p "${out_dir}"
  printf '\n=== Latent stage %s SFT (residual) → stop value prec+recall >= %s ===\n' "${stage}" "${VALUE_TARGET}" >&2
  printf 'init=%s\nout=%s num_cot_tokens=%s mixed_s1/s2=%s/%s\n' "${init_adapter}" "${out_dir}" "${cot}" "${ms1}" "${ms2}" >&2
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --train_jsonl "${train_jsonl}" \
    --output_dir "${out_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${init_adapter}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i "${stage}" \
    --num_cot_tokens "${cot}" \
    --total_empties_hint "${EMPTIES}" \
    --mixed_stage1_ratio "${ms1}" \
    --mixed_stage2_ratio "${ms2}" \
    --gradient_accumulation_steps 2 \
    --num_epochs "${SFT_NUM_EPOCHS}" \
    --learning_rate "${lr}" \
    --weight_decay 0.0 \
    --enable_gradient_checkpointing \
    --logging_steps 20 \
    --eval_steps 250 \
    --save_steps 200 \
    --eval_rows "${EVAL_PUZZLES}" \
    --eval_jsonl "${eval_jsonl}" \
    --max_completion_length 24 \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --eval_value_precision_stop "${VALUE_TARGET}" \
    --eval_value_recall_stop "${VALUE_TARGET}" \
    --eval_exact_set_match_stop 0 \
    --eval_solve_rate_stop 0 \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
    --max_steps "${SFT_MAX_STEPS}" \
    --reward_good_value 1.25 \
    --penalty_bad_value 1.0 \
    --penalty_malformed 4.0 \
    --penalty_empty 0.5 \
    --penalty_singleton 1.5 \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --use_wandb \
    --wandb_project "sudoku-latent-multi-output-sft-residual-projector" \
    --wandb_run_name "latent7_st${stage}_sft_i${stage}_${TAG_SUFFIX}_val${VALUE_TARGET}_${RUN_TAG}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

run_latent_grpo() {
  local stage="$1"
  local init_adapter="$2"
  local out_dir="$3"
  local cot="$4"
  mkdir -p "${out_dir}"
  printf '\n=== Latent stage %s GRPO (residual) → stop value prec+recall >= %s ===\n' "${stage}" "${VALUE_TARGET}" >&2
  printf 'init=%s\nout=%s num_cot_tokens=%s\n' "${init_adapter}" "${out_dir}" "${cot}" >&2
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${GRPO_SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --train_jsonl "${train_jsonl}" \
    --output_dir "${out_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${init_adapter}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i "${stage}" \
    --num_cot_tokens "${cot}" \
    --total_empties_hint "${EMPTIES}" \
    --mixed_stage1_ratio 0 \
    --mixed_stage2_ratio 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs "${GRPO_NUM_TRAIN_EPOCHS}" \
    --learning_rate 1e-6 \
    --logging_steps 20 \
    --save_steps 200 \
    --eval_steps 500 \
    --eval_rows "${EVAL_PUZZLES}" \
    --eval_jsonl "${eval_jsonl}" \
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
    --eval_value_precision_stop "${VALUE_TARGET}" \
    --eval_value_recall_stop "${VALUE_TARGET}" \
    --eval_solve_rate_stop 0 \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
    --max_steps "${GRPO_MAX_STEPS}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --use_wandb \
    --wandb_project "sudoku-latent-multi-output-grpo-residual-projector" \
    --wandb_run_name "latent7_st${stage}_grpo_i${stage}_${TAG_SUFFIX}_val${VALUE_TARGET}_${RUN_TAG}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

if [[ ! -f "${train_jsonl}" ]] || [[ ! -f "${eval_jsonl}" ]]; then
  printf 'ERROR: Missing train or eval jsonl.\n' >&2
  printf '  %s\n  %s\n' "${train_jsonl}" "${eval_jsonl}" >&2
  exit 1
fi

if [[ -n "${START_AT_STAGE2_GRPO_DIR}" ]]; then
  printf 'Fast-forward: stage-2 latent SFT dir %s → stage-2 GRPO, then stage 3.\n' "${START_AT_STAGE2_GRPO_DIR}" >&2
  printf 'Pipeline root: %s\n' "${OUTPUT_ROOT}"
  S2_DIR="${START_AT_STAGE2_GRPO_DIR}"
  CKPT_S2="$(latest_sft_step_ckpt "${S2_DIR}")"
  if [[ -z "${CKPT_S2}" ]]; then
    printf 'ERROR: No checkpoint-step-* under %s\n' "${S2_DIR}" >&2
    exit 1
  fi
  G2_DIR="${OUTPUT_ROOT}/stage02_grpo_i2_${EMPTIES}empty_${TAG_SUFFIX}"
  run_latent_grpo 2 "${CKPT_S2}" "${G2_DIR}" 2
  A2="$(resolve_latent_grpo_adapter "${G2_DIR}")"
  if [[ -z "${A2}" ]]; then
    printf 'ERROR: Could not resolve stage-2 latent GRPO adapter under %s\n' "${G2_DIR}" >&2
    exit 1
  fi
  S3_DIR="${OUTPUT_ROOT}/stage03_sft_i3_${EMPTIES}empty_${TAG_SUFFIX}"
  run_latent_sft 3 "${A2}" "${S3_DIR}" "5e-5" 3
  CKPT_S3="$(latest_sft_step_ckpt "${S3_DIR}")"
  if [[ -z "${CKPT_S3}" ]]; then
    printf 'ERROR: No checkpoint-step-* under %s\n' "${S3_DIR}" >&2
    exit 1
  fi
  G3_DIR="${OUTPUT_ROOT}/stage03_grpo_i3_${EMPTIES}empty_${TAG_SUFFIX}"
  run_latent_grpo 3 "${CKPT_S3}" "${G3_DIR}" 3
  A3="$(resolve_latent_grpo_adapter "${G3_DIR}")"
  if [[ -z "${A3}" ]]; then
    printf 'ERROR: Could not resolve stage-3 latent GRPO adapter under %s\n' "${G3_DIR}" >&2
    exit 1
  fi
  printf '\nAll latent phases finished (started at stage-2 GRPO).\n'
  printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
  printf 'Final latent GRPO adapter: %s\n' "${A3}"
  exit 0
fi

if [[ -n "${START_AFTER_STAGE2_GRPO_DIR}" ]]; then
  printf 'Fast-forward: stage-2 latent GRPO dir %s → stage-3 SFT + GRPO.\n' "${START_AFTER_STAGE2_GRPO_DIR}" >&2
  printf 'Pipeline root: %s\n' "${OUTPUT_ROOT}"
  A2="$(resolve_latent_grpo_adapter "${START_AFTER_STAGE2_GRPO_DIR}")"
  if [[ -z "${A2}" ]]; then
    printf 'ERROR: Could not resolve stage-2 latent GRPO adapter under %s\n' "${START_AFTER_STAGE2_GRPO_DIR}" >&2
    exit 1
  fi
  S3_DIR="${OUTPUT_ROOT}/stage03_sft_i3_${EMPTIES}empty_${TAG_SUFFIX}"
  run_latent_sft 3 "${A2}" "${S3_DIR}" "5e-5" 3
  CKPT_S3="$(latest_sft_step_ckpt "${S3_DIR}")"
  if [[ -z "${CKPT_S3}" ]]; then
    printf 'ERROR: No checkpoint-step-* under %s\n' "${S3_DIR}" >&2
    exit 1
  fi
  G3_DIR="${OUTPUT_ROOT}/stage03_grpo_i3_${EMPTIES}empty_${TAG_SUFFIX}"
  run_latent_grpo 3 "${CKPT_S3}" "${G3_DIR}" 3
  A3="$(resolve_latent_grpo_adapter "${G3_DIR}")"
  if [[ -z "${A3}" ]]; then
    printf 'ERROR: Could not resolve stage-3 latent GRPO adapter under %s\n' "${G3_DIR}" >&2
    exit 1
  fi
  printf '\nAll latent phases finished (started after stage-2 GRPO).\n'
  printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
  printf 'Final latent GRPO adapter: %s\n' "${A3}"
  exit 0
fi

printf 'Pipeline root: %s\n' "${OUTPUT_ROOT}"
printf 'Value gate: precision AND recall >= %s (min_steps=%s)\n' "${VALUE_TARGET}" "${MIN_STEPS_BEFORE_STOP}"

G1_DIR="${OUTPUT_ROOT}/stage01_grpo_i1_${EMPTIES}empty_${TAG_SUFFIX}"
S1_SFT_DIR="${OUTPUT_ROOT}/stage01_sft_i1_${EMPTIES}empty_${TAG_SUFFIX}"
STAGE1_INIT="${STAGE1_INIT_ADAPTER_DIR:-}"
if [[ -n "${RESUME_FROM_STAGE1_GRPO_DIR}" ]]; then
  A1="$(resolve_latent_grpo_adapter "${RESUME_FROM_STAGE1_GRPO_DIR}")"
elif [[ "${STAGE1_GRPO_FIRST}" == "1" ]]; then
  # Legacy: stage-1 GRPO first (fresh LoRA + random residual unless STAGE1_INIT_ADAPTER_DIR set).
  run_latent_grpo 1 "${STAGE1_INIT}" "${G1_DIR}" 1
  A1="$(resolve_latent_grpo_adapter "${G1_DIR}")"
else
  # Default: stage-1 SFT → stage-1 GRPO (matches text post-s1sft pipeline).
  if [[ -n "${STAGE1_SFT_ADAPTER_DIR}" ]]; then
    G1_SFT_CKPT="${STAGE1_SFT_ADAPTER_DIR}"
    printf 'Using existing stage-1 SFT checkpoint as GRPO init (skipping stage-1 SFT train): %s\n' "${G1_SFT_CKPT}" >&2
  else
    run_latent_sft 1 "${STAGE1_INIT}" "${S1_SFT_DIR}" "${STAGE1_SFT_LR}" 1
    G1_SFT_CKPT="$(latest_sft_step_ckpt "${S1_SFT_DIR}")"
    if [[ -z "${G1_SFT_CKPT}" ]]; then
      printf 'ERROR: No checkpoint-step-* under %s\n' "${S1_SFT_DIR}" >&2
      exit 1
    fi
  fi
  run_latent_grpo 1 "${G1_SFT_CKPT}" "${G1_DIR}" 1
  A1="$(resolve_latent_grpo_adapter "${G1_DIR}")"
fi
if [[ -z "${A1}" ]]; then
  printf 'ERROR: Could not resolve stage-1 latent GRPO adapter.\n' >&2
  exit 1
fi
printf 'Stage-1 latent GRPO adapter for stage-2 SFT init: %s\n' "${A1}"

S2_DIR="${OUTPUT_ROOT}/stage02_sft_i2_${EMPTIES}empty_${TAG_SUFFIX}"
run_latent_sft 2 "${A1}" "${S2_DIR}" "5e-5" 2
CKPT_S2="$(latest_sft_step_ckpt "${S2_DIR}")"
if [[ -z "${CKPT_S2}" ]]; then
  printf 'ERROR: No checkpoint-step-* under %s\n' "${S2_DIR}" >&2
  exit 1
fi
G2_DIR="${OUTPUT_ROOT}/stage02_grpo_i2_${EMPTIES}empty_${TAG_SUFFIX}"
run_latent_grpo 2 "${CKPT_S2}" "${G2_DIR}" 2
A2="$(resolve_latent_grpo_adapter "${G2_DIR}")"
if [[ -z "${A2}" ]]; then
  printf 'ERROR: Could not resolve stage-2 latent GRPO adapter under %s\n' "${G2_DIR}" >&2
  exit 1
fi

S3_DIR="${OUTPUT_ROOT}/stage03_sft_i3_${EMPTIES}empty_${TAG_SUFFIX}"
run_latent_sft 3 "${A2}" "${S3_DIR}" "5e-5" 3
CKPT_S3="$(latest_sft_step_ckpt "${S3_DIR}")"
if [[ -z "${CKPT_S3}" ]]; then
  printf 'ERROR: No checkpoint-step-* under %s\n' "${S3_DIR}" >&2
  exit 1
fi
G3_DIR="${OUTPUT_ROOT}/stage03_grpo_i3_${EMPTIES}empty_${TAG_SUFFIX}"
run_latent_grpo 3 "${CKPT_S3}" "${G3_DIR}" 3
A3="$(resolve_latent_grpo_adapter "${G3_DIR}")"
if [[ -z "${A3}" ]]; then
  printf 'ERROR: Could not resolve stage-3 latent GRPO adapter under %s\n' "${G3_DIR}" >&2
  exit 1
fi

printf '\nAll latent residual phases finished.\n'
printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
printf 'Final latent GRPO adapter (stage 3): %s\n' "${A3}"
