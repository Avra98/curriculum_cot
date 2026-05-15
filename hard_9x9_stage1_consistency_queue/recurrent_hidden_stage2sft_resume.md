# Recurrent-Hidden 20-Empty Stage-2 SFT Resume

This note records the recovered recurrent-hidden run restarted on May 15, 2026.

## Source Checkpoints

Recovered adapters were downloaded from:

```text
https://huggingface.co/Avra98/sudoku-latent-recurrent-hidden-20empty-stages
```

Local snapshot path:

```text
/home/ubuntu/curriculum_cot/final_checkpoint/hf_sudoku_latent_recurrent_hidden_20empty_stages
```

Available recovered folders:

```text
stage01_latent_sft_i1_20empty_latent_recurrent_hidden
stage01_latent_grpo_i1_20empty_latent_recurrent_hidden
stage02_baseline_warm_sft_i2_20empty_latent_recurrent_hidden
stage02_latent_sft_i2_20empty_latent_recurrent_hidden
```

The uploaded stage-2 latent SFT checkpoint did not include `trainer_state.json`
or solve-rate metadata, so the restart intentionally resumes from the stage-2
baseline warm-up adapter and reruns stage-2 latent SFT instead of jumping to
stage-2 GRPO.

## Active Resume Run

Output root:

```text
/home/ubuntu/curriculum_cot/final_checkpoint/hard_9x9_20empty_warm_baseline_all_latent_modes_stages123/recurrent_hidden_resume_stage2sft_20260515_184858
```

W&B run:

```text
https://wandb.ai/training-dynamics/sudoku-latent-stage-sft-warm-baseline/runs/1vyq1a1n
```

Launch settings:

```text
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
MODES_SPEC=recurrent_hidden
GPU_GROUPS_SPEC=0,1,2,3,4,5,6,7
NPROC_PER_JOB=8
STAGE1_LATENT_GRPO_ADAPTER_DIR=<HF snapshot>/stage01_latent_grpo_i1_20empty_latent_recurrent_hidden
STAGE2_BASELINE_WARM_ADAPTER_DIR=<HF snapshot>/stage02_baseline_warm_sft_i2_20empty_latent_recurrent_hidden/checkpoint-step-01000
LATENT_SFT_MAX_STEPS=5000
LATENT_GRPO_MAX_STEPS=500
SOLVE_TARGET=0.95
VALUE_TARGET=0
MIN_STEPS_BEFORE_STOP=50
WANDB_MODE=online
WANDB_ENTITY=training-dynamics
```

## Backup Plan

Code changes are pushed to GitHub branch:

```text
llm-policy-icon-code
```

Checkpoint backups should be pushed periodically to the same Hugging Face repo
using:

```bash
HF_TOKEN=hf_xxx \
RUN_OUTPUT_DIR=/home/ubuntu/curriculum_cot/final_checkpoint/hard_9x9_20empty_warm_baseline_all_latent_modes_stages123/recurrent_hidden_resume_stage2sft_20260515_184858 \
bash hard_9x9_stage1_consistency_queue/sync_recurrent_hidden_checkpoints_to_hf.sh
```

The sync script uploads checkpoint folders, adapter files, tokenizer files, and
logs while ignoring W&B runtime directories and prepared-data caches.
