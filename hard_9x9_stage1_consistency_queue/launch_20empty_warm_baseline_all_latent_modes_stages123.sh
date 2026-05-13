#!/usr/bin/env bash
# Full 20-empty latent comparison with baseline warm-up before latent stages.
#
# Required:
#   STAGE1_BASELINE_ADAPTER_DIR=/path/to/baseline/stage1/checkpoint-step-XXXXX
#
# Default mode split on 8 GPUs:
#   residual         -> GPUs 0,1
#   fixed_slots      -> GPUs 2,3
#   recurrent_hidden -> GPUs 4,5
#   latent_seeds     -> GPUs 6,7
#
# Per mode:
#   stage1 latent SFT -> stage1 latent GRPO
#   stage2 baseline SFT warm-up -> stage2 latent SFT -> stage2 latent GRPO
#   stage3 baseline SFT warm-up -> stage3 latent SFT -> stage3 latent GRPO
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
BASELINE_SFT_SCRIPT="${ROOT}/multi_output_cell_policy/sft_multi_output_train.py"
LATENT_SFT_SCRIPT="${ROOT}/latent_multi_output_cell_policy/sft_latent_multi_output_train.py"
LATENT_GRPO_SCRIPT="${ROOT}/latent_multi_output_cell_policy/grpo_multimode_latent_train.py"

EMPTIES="${EMPTIES:-20}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

MODES_SPEC="${MODES_SPEC:-residual fixed_slots recurrent_hidden latent_seeds}"
GPU_GROUPS_SPEC="${GPU_GROUPS_SPEC:-0,1 2,3 4,5 6,7}"
NPROC_PER_JOB="${NPROC_PER_JOB:-2}"

STAGE1_BASELINE_ADAPTER_DIR="${STAGE1_BASELINE_ADAPTER_DIR:-}"
if [[ -z "${STAGE1_BASELINE_ADAPTER_DIR}" ]] || [[ ! -d "${STAGE1_BASELINE_ADAPTER_DIR}" ]]; then
  printf 'ERROR: Set STAGE1_BASELINE_ADAPTER_DIR to a finished baseline SFT checkpoint directory.\n' >&2
  exit 1
fi

SFT_PER_DEVICE_BS="${SFT_PER_DEVICE_BS:-8}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
BASELINE_PER_DEVICE_BS="${BASELINE_PER_DEVICE_BS:-16}"
BASELINE_GRAD_ACCUM="${BASELINE_GRAD_ACCUM:-2}"
GRPO_PER_DEVICE_BS="${GRPO_PER_DEVICE_BS:-4}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-2}"

BASELINE_WARM_MAX_STEPS="${BASELINE_WARM_MAX_STEPS:-1000}"
LATENT_SFT_MAX_STEPS="${LATENT_SFT_MAX_STEPS:-1000}"
LATENT_GRPO_MAX_STEPS="${LATENT_GRPO_MAX_STEPS:-500}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-64}"
GRPO_NUM_TRAIN_EPOCHS="${GRPO_NUM_TRAIN_EPOCHS:-50}"

SOLVE_TARGET="${SOLVE_TARGET:-0.95}"
VALUE_TARGET="${VALUE_TARGET:-0}"
MIN_STEPS_BEFORE_STOP="${MIN_STEPS_BEFORE_STOP:-50}"
GRPO_BETA="${GRPO_BETA:-0.0}"

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_${EMPTIES}empty_warm_baseline_all_latent_modes_stages123}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"
train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

read -r -a MODES <<< "${MODES_SPEC}"
read -r -a GPU_GROUPS <<< "${GPU_GROUPS_SPEC}"
if [[ ${#MODES[@]} -ne ${#GPU_GROUPS[@]} ]]; then
  printf 'ERROR: expected one GPU group per mode. modes=%d gpu_groups=%d\n' "${#MODES[@]}" "${#GPU_GROUPS[@]}" >&2
  exit 1
fi

if [[ ! -f "${train_jsonl}" ]] || [[ ! -f "${eval_jsonl}" ]]; then
  printf 'ERROR: Missing train or eval jsonl.\n  %s\n  %s\n' "${train_jsonl}" "${eval_jsonl}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"

mode_tag() {
  case "$1" in
    residual) printf 'latent_residual' ;;
    fixed_slots) printf 'latent_fixed_slots' ;;
    recurrent_hidden) printf 'latent_recurrent_hidden' ;;
    latent_seeds) printf 'latent_seeds' ;;
    *) printf 'latent_%s' "$1" ;;
  esac
}

latest_checkpoint_or_dir() {
  local d="$1"
  shopt -s nullglob
  local checkpoints=("${d}"/checkpoint-step-*)
  shopt -u nullglob
  if (( ${#checkpoints[@]} > 0 )); then
    printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1
    return 0
  fi
  if [[ -f "${d}/adapter_model.safetensors" ]]; then
    printf '%s\n' "${d}"
    return 0
  fi
  printf ''
  return 1
}

run_baseline_sft() {
  local stage="$1" init_adapter="$2" out_dir="$3" lr="$4" run_name="$5"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_JOB}" "${BASELINE_SFT_SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --train_jsonl "${train_jsonl}" \
    --eval_jsonl "${eval_jsonl}" \
    --output_dir "${out_dir}" \
    --cache_dir "${ROOT}/.hf_cache" \
    --init_adapter_dir "${init_adapter}" \
    --seed 0 \
    --gpu_id 0 \
    --stage_i "${stage}" \
    --total_empties_hint "${EMPTIES}" \
    --per_device_train_batch_size "${BASELINE_PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${BASELINE_GRAD_ACCUM}" \
    --num_epochs "${SFT_NUM_EPOCHS}" \
    --learning_rate "${lr}" \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --eval_steps 250 \
    --save_steps 200 \
    --eval_rows "${EVAL_PUZZLES}" \
    --max_completion_length 24 \
    --limit_train_rows "${TRAIN_PUZZLES}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --eval_value_precision_stop "${VALUE_TARGET}" \
    --eval_value_recall_stop "${VALUE_TARGET}" \
    --eval_exact_set_match_stop 0 \
    --eval_solve_rate_stop "${SOLVE_TARGET}" \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --max_wall_clock_seconds 0 \
    --max_steps "${BASELINE_WARM_MAX_STEPS}" \
    --use_wandb \
    --wandb_project "sudoku-baseline-stage-warmups" \
    --wandb_run_name "${run_name}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

run_latent_sft() {
  local mode="$1" stage="$2" cot="$3" init_adapter="$4" out_dir="$5" lr="$6" run_name="$7"
  local ms1=0 ms2=1
  if [[ "${stage}" == "1" ]]; then
    ms1=1
    ms2=0
  fi
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_JOB}" "${LATENT_SFT_SCRIPT}" \
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
    --latent_mode "${mode}" \
    --max_latent_slots 8 \
    --max_latent_seeds 8 \
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
    --eval_value_precision_stop "${VALUE_TARGET}" \
    --eval_value_recall_stop "${VALUE_TARGET}" \
    --eval_exact_set_match_stop 0 \
    --eval_solve_rate_stop "${SOLVE_TARGET}" \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --max_wall_clock_seconds 0 \
    --max_steps "${LATENT_SFT_MAX_STEPS}" \
    --reward_good_value 1.25 \
    --penalty_bad_value 1.0 \
    --penalty_malformed 4.0 \
    --penalty_empty 0.5 \
    --penalty_singleton 1.5 \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --use_wandb \
    --wandb_project "sudoku-latent-stage-sft-warm-baseline" \
    --wandb_run_name "${run_name}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

run_latent_grpo() {
  local mode="$1" stage="$2" cot="$3" init_adapter="$4" out_dir="$5" run_name="$6"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_JOB}" "${LATENT_GRPO_SCRIPT}" \
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
    --latent_mode "${mode}" \
    --max_latent_seeds 8 \
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
    --eval_value_precision_stop "${VALUE_TARGET}" \
    --eval_value_recall_stop "${VALUE_TARGET}" \
    --eval_solve_rate_stop "${SOLVE_TARGET}" \
    --min_steps_before_stop "${MIN_STEPS_BEFORE_STOP}" \
    --max_wall_clock_seconds 0 \
    --max_steps "${LATENT_GRPO_MAX_STEPS}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --use_wandb \
    --wandb_project "sudoku-latent-stage-grpo-warm-baseline" \
    --wandb_run_name "${run_name}" \
    --wandb_mode "${WANDB_MODE}" \
    --wandb_entity "${WANDB_ENTITY}"
}

run_mode_pipeline() {
  local mode="$1" gpu_group="$2" tag
  tag="$(mode_tag "${mode}")"
  local mode_root="${OUTPUT_ROOT}/${tag}"
  local log="${OUTPUT_ROOT}/logs/${tag}.log"
  mkdir -p "${mode_root}"
  export CUDA_VISIBLE_DEVICES="${gpu_group}"
  printf 'Mode %s on GPUs %s\n' "${mode}" "${gpu_group}"

  local s1_lat="${mode_root}/stage01_latent_sft_i1_${EMPTIES}empty_${tag}"
  local g1="${mode_root}/stage01_latent_grpo_i1_${EMPTIES}empty_${tag}"
  run_latent_sft "${mode}" 1 1 "${STAGE1_BASELINE_ADAPTER_DIR}" "${s1_lat}" "2e-4" "warmfull_${mode}_st1_latent_sft_${RUN_TAG}" 2>&1 | tee -a "${log}"
  local a_s1_lat
  a_s1_lat="$(latest_checkpoint_or_dir "${s1_lat}")"
  run_latent_grpo "${mode}" 1 1 "${a_s1_lat}" "${g1}" "warmfull_${mode}_st1_latent_grpo_${RUN_TAG}" 2>&1 | tee -a "${log}"
  local a_g1
  a_g1="$(latest_checkpoint_or_dir "${g1}")"

  local b2="${mode_root}/stage02_baseline_warm_sft_i2_${EMPTIES}empty_${tag}"
  local s2_lat="${mode_root}/stage02_latent_sft_i2_${EMPTIES}empty_${tag}"
  local g2="${mode_root}/stage02_latent_grpo_i2_${EMPTIES}empty_${tag}"
  run_baseline_sft 2 "${a_g1}" "${b2}" "5e-5" "warmfull_${mode}_st2_baseline_warm_sft_${RUN_TAG}" 2>&1 | tee -a "${log}"
  local a_b2
  a_b2="$(latest_checkpoint_or_dir "${b2}")"
  run_latent_sft "${mode}" 2 2 "${a_b2}" "${s2_lat}" "5e-5" "warmfull_${mode}_st2_latent_sft_${RUN_TAG}" 2>&1 | tee -a "${log}"
  local a_s2_lat
  a_s2_lat="$(latest_checkpoint_or_dir "${s2_lat}")"
  run_latent_grpo "${mode}" 2 2 "${a_s2_lat}" "${g2}" "warmfull_${mode}_st2_latent_grpo_${RUN_TAG}" 2>&1 | tee -a "${log}"
  local a_g2
  a_g2="$(latest_checkpoint_or_dir "${g2}")"

  local b3="${mode_root}/stage03_baseline_warm_sft_i3_${EMPTIES}empty_${tag}"
  local s3_lat="${mode_root}/stage03_latent_sft_i3_${EMPTIES}empty_${tag}"
  local g3="${mode_root}/stage03_latent_grpo_i3_${EMPTIES}empty_${tag}"
  run_baseline_sft 3 "${a_g2}" "${b3}" "5e-5" "warmfull_${mode}_st3_baseline_warm_sft_${RUN_TAG}" 2>&1 | tee -a "${log}"
  local a_b3
  a_b3="$(latest_checkpoint_or_dir "${b3}")"
  run_latent_sft "${mode}" 3 3 "${a_b3}" "${s3_lat}" "5e-5" "warmfull_${mode}_st3_latent_sft_${RUN_TAG}" 2>&1 | tee -a "${log}"
  local a_s3_lat
  a_s3_lat="$(latest_checkpoint_or_dir "${s3_lat}")"
  run_latent_grpo "${mode}" 3 3 "${a_s3_lat}" "${g3}" "warmfull_${mode}_st3_latent_grpo_${RUN_TAG}" 2>&1 | tee -a "${log}"

  printf 'Mode %s finished. Output: %s\n' "${mode}" "${mode_root}" | tee -a "${log}"
}

printf 'Output root: %s\n' "${OUTPUT_ROOT}"
printf 'Stage-1 baseline adapter: %s\n' "${STAGE1_BASELINE_ADAPTER_DIR}"
printf 'Solve target: %s (value target: %s)\n' "${SOLVE_TARGET}" "${VALUE_TARGET}"

pids=()
for i in "${!MODES[@]}"; do
  (
    run_mode_pipeline "${MODES[$i]}" "${GPU_GROUPS[$i]}"
  ) >"${OUTPUT_ROOT}/logs/$(mode_tag "${MODES[$i]}").supervisor.log" 2>&1 &
  pids+=("$!")
  printf 'Launched mode=%s pid=%s gpus=%s\n' "${MODES[$i]}" "${pids[-1]}" "${GPU_GROUPS[$i]}"
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf 'DONE: %s\n' "${MODES[$i]}"
  else
    printf 'FAILED: %s (pid=%s). See %s/logs\n' "${MODES[$i]}" "${pids[$i]}" "${OUTPUT_ROOT}" >&2
    failed=1
  fi
done

exit "${failed}"
