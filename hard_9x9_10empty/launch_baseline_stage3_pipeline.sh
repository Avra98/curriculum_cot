#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATASET_BUILDER="${ROOT}/simple_9x9_curriculum/build_dataset.py"
PIPELINE="${ROOT}/multi_output_cell_policy/run_baseline_multi_output_pipeline_resume.py"

TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/data/sudoku_t3_10empty_value_qwen_text_longrun.jsonl}"
NUM_PUZZLES="${NUM_PUZZLES:-5000}"
DATASET_SEED="${DATASET_SEED:-0}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/final_checkpoint/hard_9x9_10empty_qwen05b/baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/${RUN_TAG}/baseline_pipeline_10empty_3stage_hard9x9}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-training-dynamics}"
WAIT_FOR_EXISTING_TRAINING="${WAIT_FOR_EXISTING_TRAINING:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  mkdir -p "$(dirname "${TRAIN_JSONL}")"
  printf 'Building 10-empty dataset: %s\n' "${TRAIN_JSONL}"
  "${PYTHON_BIN}" "${DATASET_BUILDER}" \
    --output "${TRAIN_JSONL}" \
    --num_puzzles "${NUM_PUZZLES}" \
    --empties 10 \
    --seed "${DATASET_SEED}"
fi

if [[ "${WAIT_FOR_EXISTING_TRAINING}" == "1" ]]; then
  while pgrep -f "/home/ubuntu/curriculum_cot/.venv/bin/python.*(run_baseline_multi_output_pipeline_resume.py|run_latent_residual_projector_pipeline.py|sft_multi_output_train.py|grpo_multi_output_train.py|residual_projector_warmstart_sft_latent_multi_output_train.py|grpo_residual_projector_latent_train.py)" >/dev/null; do
    printf 'Existing training detected; waiting %ss before launching 10-empty baseline pipeline...\n' "${WAIT_SECONDS}"
    sleep "${WAIT_SECONDS}"
  done
fi

mkdir -p "${CHECKPOINT_ROOT}"

cmd=(
  "${PYTHON_BIN}" "${PIPELINE}"
  --python_executable "${PYTHON_BIN}"
  --train_jsonl "${TRAIN_JSONL}"
  --cache_dir "${ROOT}/.hf_cache"
  --model_name "Qwen/Qwen2.5-0.5B-Instruct"
  --checkpoint_root "${CHECKPOINT_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --run_tag "${RUN_TAG}"
  --min_stage 1
  --max_stage 3
  --distributed_gpu_ids "${GPU_IDS}"
  --sft_num_processes "${NUM_PROCESSES}"
  --grpo_num_processes "${NUM_PROCESSES}"
  --total_empties_hint 10
  --limit_train_rows 5000
  --sft_num_epochs 3.0
  --grpo_num_train_epochs 1.5
  --sft_gradient_accumulation_steps 8
  --grpo_per_device_train_batch_size 8
  --grpo_gradient_accumulation_steps 2
  --grpo_num_generations 4
  --sft_enable_gradient_checkpointing
  --grpo_enable_gradient_checkpointing
  --sft_eval_steps 100
  --sft_save_steps 100
  --grpo_eval_steps 50
  --grpo_save_steps 50
  --sft_eval_rows 100
  --grpo_eval_rows 100
  --sft_stage_max_steps "1:2000,2:2000,3:2000"
  --grpo_stage_max_steps "1:1200,2:1200,3:1200"
  --sft_eval_solve_rate_stop 0.8
  --sft_min_steps_before_stop 100
  --grpo_eval_solve_rate_stop 0.8
  --grpo_min_steps_before_stop 50
  --grpo_reward_good_value 1.25
  --grpo_penalty_bad_value 1.0
  --grpo_penalty_malformed 4.0
  --grpo_penalty_empty 0.5
  --grpo_penalty_singleton 1.0
  --phase_max_wall_clock_seconds 36000
  --wandb_mode "${WANDB_MODE}"
  --use_wandb
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi

printf 'Launching 10-empty baseline stage-3 pipeline\n'
printf 'Dataset: %s\n' "${TRAIN_JSONL}"
printf 'Checkpoint root: %s\n' "${CHECKPOINT_ROOT}"
printf 'Output root: %s\n' "${OUTPUT_ROOT}"
printf 'GPUs: %s processes=%s\n' "${GPU_IDS}" "${NUM_PROCESSES}"

exec "${cmd[@]}"
