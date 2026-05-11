#!/usr/bin/env bash
# Stage-1 SFT sweep over all latent modes for 20-empty Sudoku.
#
# Runs four independent SFT jobs in parallel:
#   residual, fixed_slots, recurrent_hidden, latent_seeds
#
# Default GPU split on an 8-GPU node:
#   residual         -> CUDA_VISIBLE_DEVICES=0,1
#   fixed_slots      -> CUDA_VISIBLE_DEVICES=2,3
#   recurrent_hidden -> CUDA_VISIBLE_DEVICES=4,5
#   latent_seeds     -> CUDA_VISIBLE_DEVICES=6,7
#
# Useful overrides:
#   RUN_TAG=... CHECKPOINT_ROOT=...
#   GPU_GROUPS_SPEC="0 1 2 3" NPROC_PER_JOB=1
#   TRAIN_PUZZLES=10000 EVAL_PUZZLES=100 SFT_VALUE_TARGET=0.98
#   STAGE1_INIT_ADAPTER_DIR=/path/to/init_adapter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
SFT_SCRIPT="${ROOT}/latent_multi_output_cell_policy/sft_latent_multi_output_train.py"

MODES=("residual" "fixed_slots" "recurrent_hidden" "latent_seeds")
MODE_TAGS=("latent_residual" "latent_fixed_slots" "latent_recurrent_hidden" "latent_seeds")

# Space-separated list of CUDA_VISIBLE_DEVICES groups, one per latent mode.
# Example for one GPU per method: GPU_GROUPS_SPEC="0 1 2 3" NPROC_PER_JOB=1
GPU_GROUPS_SPEC="${GPU_GROUPS_SPEC:-0,1 2,3 4,5 6,7}"
read -r -a GPU_GROUPS <<< "${GPU_GROUPS_SPEC}"

NPROC_PER_JOB="${NPROC_PER_JOB:-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"

EMPTIES="${EMPTIES:-20}"
TRAIN_PUZZLES="${TRAIN_PUZZLES:-10000}"
EVAL_PUZZLES="${EVAL_PUZZLES:-100}"
VALUE_TARGET="${VALUE_TARGET:-0.98}"
SFT_VALUE_TARGET="${SFT_VALUE_TARGET:-${VALUE_TARGET}}"
MIN_STEPS_BEFORE_STOP="${MIN_STEPS_BEFORE_STOP:-50}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-10000000}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-512}"
PHASE_WALL_CLOCK_SECONDS="${PHASE_WALL_CLOCK_SECONDS:-0}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
# Match the recurrent 20-empty launcher defaults: -1 resolves inside the
# trainer to hidden_size, and alpha=-1 resolves to 2 * resolved rank.
LORA_R="${LORA_R:--1}"
LORA_ALPHA="${LORA_ALPHA:--1}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
STAGE1_SFT_LR="${STAGE1_SFT_LR:-2e-4}"
SFT_PER_DEVICE_BS="${SFT_PER_DEVICE_BS:-8}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
NUM_COT_TOKENS="${NUM_COT_TOKENS:-1}"
MAX_LATENT_SLOTS="${MAX_LATENT_SLOTS:-8}"
MAX_LATENT_SEEDS="${MAX_LATENT_SEEDS:-8}"
STAGE1_INIT_ADAPTER_DIR="${STAGE1_INIT_ADAPTER_DIR:-}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_${EMPTIES}empty_stage1_sft_all_latent_modes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}}"

train_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_train.jsonl"
eval_jsonl="${ROOT}/data/sudoku_t3_${EMPTIES}empty_value_qwen_text_stage1_eval.jsonl"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ ${#GPU_GROUPS[@]} -ne ${#MODES[@]} ]]; then
  printf 'ERROR: expected %d GPU groups, got %d.\n' "${#MODES[@]}" "${#GPU_GROUPS[@]}" >&2
  printf 'Example: GPU_GROUPS_SPEC="0,1 2,3 4,5 6,7"\n' >&2
  exit 1
fi

if [[ ! -f "${train_jsonl}" ]] || [[ ! -f "${eval_jsonl}" ]]; then
  printf 'ERROR: Missing train or eval jsonl.\n' >&2
  printf '  %s\n  %s\n' "${train_jsonl}" "${eval_jsonl}" >&2
  exit 1
fi

if [[ -n "${STAGE1_INIT_ADAPTER_DIR}" ]] && [[ ! -d "${STAGE1_INIT_ADAPTER_DIR}" ]]; then
  printf 'ERROR: STAGE1_INIT_ADAPTER_DIR is not a directory: %s\n' "${STAGE1_INIT_ADAPTER_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

run_stage1_sft_for_mode() {
  local mode="$1"
  local tag="$2"
  local gpu_group="$3"
  local out_dir="${OUTPUT_ROOT}/stage01_sft_i1_${EMPTIES}empty_${tag}"
  local log_dir="${OUTPUT_ROOT}/logs"
  local log_file="${log_dir}/stage01_sft_${mode}.log"

  mkdir -p "${out_dir}" "${log_dir}"
  printf '\n=== launching stage-1 SFT: mode=%s gpus=%s out=%s ===\n' "${mode}" "${gpu_group}" "${out_dir}" >&2

  (
    export CUDA_VISIBLE_DEVICES="${gpu_group}"
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_JOB}" "${SFT_SCRIPT}" \
      --model_name "${MODEL_NAME}" \
      --train_jsonl "${train_jsonl}" \
      --eval_jsonl "${eval_jsonl}" \
      --output_dir "${out_dir}" \
      --cache_dir "${ROOT}/.hf_cache" \
      --init_adapter_dir "${STAGE1_INIT_ADAPTER_DIR}" \
      --seed 0 \
      --gpu_id 0 \
      --stage_i 1 \
      --num_cot_tokens "${NUM_COT_TOKENS}" \
      --latent_mode "${mode}" \
      --max_latent_slots "${MAX_LATENT_SLOTS}" \
      --max_latent_seeds "${MAX_LATENT_SEEDS}" \
      --total_empties_hint "${EMPTIES}" \
      --mixed_stage1_ratio 1 \
      --mixed_stage2_ratio 0 \
      --per_device_train_batch_size "${SFT_PER_DEVICE_BS}" \
      --gradient_accumulation_steps "${SFT_GRAD_ACCUM}" \
      --num_epochs "${SFT_NUM_EPOCHS}" \
      --learning_rate "${STAGE1_SFT_LR}" \
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
      --wandb_project "sudoku-latent-stage1-sft-all-modes" \
      --wandb_run_name "latent20_stage1_sft_${mode}_cot${NUM_COT_TOKENS}_val${SFT_VALUE_TARGET}_${RUN_TAG}" \
      --wandb_mode "${WANDB_MODE}" \
      --wandb_entity "${WANDB_ENTITY}"
  ) >"${log_file}" 2>&1 &

  printf '%s\n' "$!"
}

printf 'Output root: %s\n' "${OUTPUT_ROOT}"
printf 'Stage-1 init adapter: %s\n' "${STAGE1_INIT_ADAPTER_DIR:-<fresh-lora-random-latent>}"
printf 'Modes: %s\n' "${MODES[*]}"
printf 'GPU groups: %s\n' "${GPU_GROUPS[*]}"
printf 'Processes per job: %s\n' "${NPROC_PER_JOB}"

pids=()
names=()
for i in "${!MODES[@]}"; do
  pid="$(run_stage1_sft_for_mode "${MODES[$i]}" "${MODE_TAGS[$i]}" "${GPU_GROUPS[$i]}")"
  pids+=("${pid}")
  names+=("${MODES[$i]}")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf 'DONE: %s\n' "${names[$i]}"
  else
    printf 'FAILED: %s (pid=%s). See logs under %s/logs\n' "${names[$i]}" "${pids[$i]}" "${OUTPUT_ROOT}" >&2
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

printf '\nAll stage-1 latent SFT jobs finished.\n'
printf 'Outputs under: %s\n' "${OUTPUT_ROOT}"
