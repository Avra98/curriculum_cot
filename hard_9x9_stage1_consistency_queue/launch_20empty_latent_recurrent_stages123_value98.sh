#!/usr/bin/env bash
# Latent recurrent-hidden (Coconut-style) pipeline for 20-empty Sudoku.
#
# Per-stage latent token count grows with curriculum:
#   stage 1 -> num_cot_tokens = 1
#   stage 2 -> num_cot_tokens = 2
#   stage 3 -> num_cot_tokens = 3
#
# Pipeline:
#   Stage 1 SFT (cot=1, fresh LoRA + random latent state)
#     -> Stage 1 GRPO (cot=1)
#     -> Stage 2 SFT  (cot=2)
#     -> Stage 2 GRPO (cot=2)
#     -> Stage 3 SFT  (cot=3)
#     -> Stage 3 GRPO (cot=3)
#
# Mirrors the hyperparameters of the successful 20-empty recurrent-hidden stage-1
# run (bs=8 per-device, gradient accumulation 2, gradient checkpointing ON).
#
# Optional overrides:
#   STAGE1_INIT_ADAPTER_DIR=/path/to/adapter
#   STAGE1_SFT_ADAPTER_DIR=/path/to/stage01_sft/checkpoint-step-XXXX
#   VALUE_TARGET=0.98 TRAIN_PUZZLES=10000 EVAL_PUZZLES=100 RUN_TAG=... CHECKPOINT_ROOT=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
SFT_SCRIPT="${ROOT}/latent_multi_output_cell_policy/sft_latent_multi_output_train.py"
GRPO_SCRIPT="${ROOT}/latent_multi_output_cell_policy/grpo_residual_projector_latent_train.py"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

LATENT_MODE="recurrent_hidden"
EMPTIES=20
TAG_SUFFIX="latent_recurrent"
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
VALUE_TARGET="${VALUE_TARGET:-0.98}"
# Per-phase early-stop bars. Default behavior preserved: both phases use
# VALUE_TARGET unless explicitly overridden. Recommended: SFT_VALUE_TARGET=0.95
# (let SFT do bulk learning quickly) and GRPO_VALUE_TARGET=0.98 (let GRPO push
# the last few percent of value precision/recall).
SFT_VALUE_TARGET="${SFT_VALUE_TARGET:-${VALUE_TARGET}}"
GRPO_VALUE_TARGET="${GRPO_VALUE_TARGET:-${VALUE_TARGET}}"
MIN_STEPS_BEFORE_STOP="${MIN_STEPS_BEFORE_STOP:-50}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-10000000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-10000000}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-512}"
GRPO_NUM_TRAIN_EPOCHS="${GRPO_NUM_TRAIN_EPOCHS:-200}"
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
STAGE1_SFT_LR="${STAGE1_SFT_LR:-2e-4}"
SFT_PER_DEVICE_BS="${SFT_PER_DEVICE_BS:-8}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
GRPO_PER_DEVICE_BS="${GRPO_PER_DEVICE_BS:-8}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-2}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_20empty_latent_recurrent_stages123_value98}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"
STAGE1_INIT_ADAPTER_DIR="${STAGE1_INIT_ADAPTER_DIR:-}"
STAGE1_SFT_ADAPTER_DIR="${STAGE1_SFT_ADAPTER_DIR:-}"
# When set, skip both Stage-1 SFT and Stage-1 GRPO and use this adapter
# directly as the init for Stage-2 SFT. Useful for resuming after a Stage-1
# GRPO post-training eval hangs but the LoRA adapter is already on disk.
STAGE1_GRPO_ADAPTER_DIR="${STAGE1_GRPO_ADAPTER_DIR:-}"
STAGE2_SFT_ADAPTER_DIR="${STAGE2_SFT_ADAPTER_DIR:-}"
STAGE2_GRPO_ADAPTER_DIR="${STAGE2_GRPO_ADAPTER_DIR:-}"
# When set, skip Stage-3 SFT and use this adapter directly as the init for
# Stage-3 GRPO. Useful when SFT plateaus mid-training and we want GRPO to push
# the last few percentage points without burning more SFT compute.
STAGE3_SFT_ADAPTER_DIR="${STAGE3_SFT_ADAPTER_DIR:-}"
# KL anchor for GRPO. Setting > 0 keeps the policy close to the SFT reference
# and prevents singleton/mode collapse seen in Stage-2 GRPO. 0.0 = no KL.
GRPO_BETA="${GRPO_BETA:-0.0}"

train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${train_jsonl}" ]] || [[ ! -f "${eval_jsonl}" ]]; then
  printf 'ERROR: Missing train or eval jsonl.\n' >&2
  printf '  %s\n  %s\n' "${train_jsonl}" "${eval_jsonl}" >&2
  exit 1
fi

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
  local ms1=0 ms2=1
  if [[ "${stage}" == "1" ]]; then
    ms1=1
    ms2=0
  fi
  mkdir -p "${out_dir}"
  printf '\n=== Latent(recurrent) stage %s SFT -> stop value prec+recall >= %s (cot=%s) ===\n' "${stage}" "${SFT_VALUE_TARGET}" "${cot}" >&2
  printf 'init=%s\nout=%s num_cot_tokens=%s mixed_s1/s2=%s/%s\n' "${init_adapter}" "${out_dir}" "${cot}" "${ms1}" "${ms2}" >&2
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${SFT_SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --train_jsonl "${train_jsonl}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${out_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${init_adapter}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i "${stage}" \
    --num_cot_tokens "${cot}" \
    --latent_mode "${LATENT_MODE}" \
    --total_empties_hint "${EMPTIES}" \
    --mixed_stage1_ratio "${ms1}" \
    --mixed_stage2_ratio "${ms2}" \
    --per_device_train_batch_size "${SFT_PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
    --num_epochs "${SFT_NUM_EPOCHS}" \
    --learning_rate "${lr}" \
    --weight_decay 0.0 \
    --enable_gradient_checkpointing \
    --logging_steps 20 \
    --eval_steps 250 \
    --save_steps 200 \
    --eval_rows "${EVAL_PUZZLES}" \
    --max_completion_length 24 \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --eval_value_precision_stop "${SFT_VALUE_TARGET}" \
    --eval_value_recall_stop "${SFT_VALUE_TARGET}" \
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
    --wandb_project "sudoku-latent-multi-output-sft-recurrent" \
    --wandb_run_name "latent20_st${stage}_sft_i${stage}_${TAG_SUFFIX}_cot${cot}_val${SFT_VALUE_TARGET}_${RUN_TAG}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

run_latent_grpo() {
  local stage="$1"
  local init_adapter="$2"
  local out_dir="$3"
  local cot="$4"
  mkdir -p "${out_dir}"
  printf '\n=== Latent(recurrent) stage %s GRPO -> stop value prec+recall >= %s (cot=%s) ===\n' "${stage}" "${GRPO_VALUE_TARGET}" "${cot}" >&2
  printf 'init=%s\nout=%s num_cot_tokens=%s\n' "${init_adapter}" "${out_dir}" "${cot}" >&2
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_PROCESSES}" "${GRPO_SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --train_jsonl "${train_jsonl}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${out_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${init_adapter}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i "${stage}" \
    --num_cot_tokens "${cot}" \
    --latent_mode "${LATENT_MODE}" \
    --total_empties_hint "${EMPTIES}" \
    --mixed_stage1_ratio 0 \
    --mixed_stage2_ratio 1 \
    --per_device_train_batch_size "${GRPO_PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRPO_GRAD_ACCUM}" \
    --num_train_epochs "${GRPO_NUM_TRAIN_EPOCHS}" \
    --learning_rate 1e-6 \
    --logging_steps 20 \
    --save_steps 200 \
    --eval_steps 500 \
    --eval_rows "${EVAL_PUZZLES}" \
    --num_generations 4 \
    --max_prompt_length 1024 \
    --max_completion_length 24 \
    --beta "${GRPO_BETA}" \
    --enable_gradient_checkpointing \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --reward_good_value 1.25 \
    --penalty_bad_value 1.0 \
    --penalty_malformed 4.0 \
    --penalty_empty 0.5 \
    --penalty_singleton 1.5 \
    --eval_value_precision_stop "${GRPO_VALUE_TARGET}" \
    --eval_value_recall_stop "${GRPO_VALUE_TARGET}" \
    --eval_solve_rate_stop 0 \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --max_wall_clock_seconds "${PHASE_WALL_CLOCK_SECONDS}" \
    --max_steps "${GRPO_MAX_STEPS}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --use_wandb \
    --wandb_project "sudoku-latent-multi-output-grpo-recurrent" \
    --wandb_run_name "latent20_st${stage}_grpo_i${stage}_${TAG_SUFFIX}_cot${cot}_val${GRPO_VALUE_TARGET}_${RUN_TAG}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

printf 'Pipeline root: %s\n' "${OUTPUT_ROOT}"
printf 'Latent mode: %s (cot grows 1->2->3 per stage)\n' "${LATENT_MODE}"
printf 'Value gate: SFT prec+recall >= %s ; GRPO prec+recall >= %s (min_steps=%s) ; GRPO_BETA=%s\n' "${SFT_VALUE_TARGET}" "${GRPO_VALUE_TARGET}" "${MIN_STEPS_BEFORE_STOP}" "${GRPO_BETA}"
printf 'Stage-1 init adapter: %s\n' "${STAGE1_INIT_ADAPTER_DIR:-<fresh-lora-random-latent>}"

S1_SFT_DIR="${OUTPUT_ROOT}/stage01_sft_i1_${EMPTIES}empty_${TAG_SUFFIX}"
G1_DIR="${OUTPUT_ROOT}/stage01_grpo_i1_${EMPTIES}empty_${TAG_SUFFIX}"
if [[ -n "${STAGE1_GRPO_ADAPTER_DIR}" ]]; then
  A1="${STAGE1_GRPO_ADAPTER_DIR}"
  printf 'Using existing stage-1 GRPO adapter (skipping stage-1 SFT + GRPO): %s\n' "${A1}" >&2
elif [[ -n "${STAGE1_SFT_ADAPTER_DIR}" ]]; then
  G1_SFT_CKPT="${STAGE1_SFT_ADAPTER_DIR}"
  printf 'Using existing stage-1 SFT checkpoint as GRPO init (skipping stage-1 SFT train): %s\n' "${G1_SFT_CKPT}" >&2
  run_latent_grpo 1 "${G1_SFT_CKPT}" "${G1_DIR}" 1
  A1="$(resolve_latent_grpo_adapter "${G1_DIR}")"
else
  run_latent_sft 1 "${STAGE1_INIT_ADAPTER_DIR}" "${S1_SFT_DIR}" "${STAGE1_SFT_LR}" 1
  G1_SFT_CKPT="$(latest_sft_step_ckpt "${S1_SFT_DIR}")"
  if [[ -z "${G1_SFT_CKPT}" ]]; then
    printf 'ERROR: No checkpoint-step-* under %s\n' "${S1_SFT_DIR}" >&2
    exit 1
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
G2_DIR="${OUTPUT_ROOT}/stage02_grpo_i2_${EMPTIES}empty_${TAG_SUFFIX}"
if [[ -n "${STAGE2_GRPO_ADAPTER_DIR}" ]]; then
  A2="${STAGE2_GRPO_ADAPTER_DIR}"
  printf 'Using existing stage-2 GRPO adapter (skipping stage-2 SFT + GRPO): %s\n' "${A2}" >&2
elif [[ -n "${STAGE2_SFT_ADAPTER_DIR}" ]]; then
  CKPT_S2="${STAGE2_SFT_ADAPTER_DIR}"
  printf 'Using existing stage-2 SFT checkpoint as GRPO init (skipping stage-2 SFT train): %s\n' "${CKPT_S2}" >&2
  run_latent_grpo 2 "${CKPT_S2}" "${G2_DIR}" 2
  A2="$(resolve_latent_grpo_adapter "${G2_DIR}")"
else
  run_latent_sft 2 "${A1}" "${S2_DIR}" "5e-5" 2
  CKPT_S2="$(latest_sft_step_ckpt "${S2_DIR}")"
  if [[ -z "${CKPT_S2}" ]]; then
    printf 'ERROR: No checkpoint-step-* under %s\n' "${S2_DIR}" >&2
    exit 1
  fi
  run_latent_grpo 2 "${CKPT_S2}" "${G2_DIR}" 2
  A2="$(resolve_latent_grpo_adapter "${G2_DIR}")"
fi
 if [[ -z "${A2}" ]]; then
  printf 'ERROR: Could not resolve stage-2 latent GRPO adapter under %s\n' "${G2_DIR}" >&2
  exit 1
fi

S3_DIR="${OUTPUT_ROOT}/stage03_sft_i3_${EMPTIES}empty_${TAG_SUFFIX}"
G3_DIR="${OUTPUT_ROOT}/stage03_grpo_i3_${EMPTIES}empty_${TAG_SUFFIX}"
if [[ -n "${STAGE3_SFT_ADAPTER_DIR}" ]]; then
  CKPT_S3="${STAGE3_SFT_ADAPTER_DIR}"
  printf 'Using existing stage-3 SFT checkpoint as GRPO init (skipping stage-3 SFT train): %s\n' "${CKPT_S3}" >&2
else
  run_latent_sft 3 "${A2}" "${S3_DIR}" "5e-5" 3
  CKPT_S3="$(latest_sft_step_ckpt "${S3_DIR}")"
  if [[ -z "${CKPT_S3}" ]]; then
    printf 'ERROR: No checkpoint-step-* under %s\n' "${S3_DIR}" >&2
    exit 1
  fi
fi
run_latent_grpo 3 "${CKPT_S3}" "${G3_DIR}" 3
A3="$(resolve_latent_grpo_adapter "${G3_DIR}")"
if [[ -z "${A3}" ]]; then
  printf 'ERROR: Could not resolve stage-3 latent GRPO adapter under %s\n' "${G3_DIR}" >&2
  exit 1
fi

printf '\nAll latent(recurrent) phases finished.\n'
printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
printf 'Final latent GRPO adapter (stage 3): %s\n' "${A3}"
