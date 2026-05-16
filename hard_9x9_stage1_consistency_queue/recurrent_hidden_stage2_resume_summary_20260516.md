# Recurrent Hidden Stage 2 Resume Summary

This note summarizes the May 16, 2026 stage-2 recurrent-hidden SFT recovery and monitoring changes.

## What Changed

- Added explicit eval lifecycle logging in `latent_multi_output_cell_policy/sft_latent_multi_output_train.py`.
- W&B now receives `eval/in_progress`, `eval/rows`, and `eval/duration_seconds`.
- Local logs now print `[latent sft eval start ...]` and `[latent sft eval end ...]` markers.

## Why

The previous resumed stage-2 run reached step 2000 but appeared silent during validation. The validation metrics only logged after the whole eval completed, and the old `eval_rows=100` setting made a single validation take roughly 35 minutes. The run then crashed before producing the step-2000 eval metrics or checkpoint.

## Probe Result

A one-GPU eval probe from `checkpoint-step-01800` measured validation cost:

- Eval rows: 20 puzzles
- Eval duration: 427.3 seconds, about 7.1 minutes
- Exact set match: 0.9225
- Value precision: 0.945
- Value recall: 0.934
- Solve rate: 0.15
- W&B run: `xudqbjqh`

## Active Resume Run

The main run was restarted from:

`final_checkpoint/hard_9x9_20empty_warm_baseline_all_latent_modes_stages123/recurrent_hidden_resume_stage2sft_from200_20260515_205857/latent_recurrent_hidden/stage02_latent_sft_i2_20empty_latent_recurrent_hidden/checkpoint-step-01800`

Run settings:

- Stage: 2
- Latent mode: recurrent_hidden
- GPUs: 8
- Eval rows: 20
- Eval interval: every 100 steps
- Checkpoint interval: every 100 steps
- Max steps: 5000
- Early stop: disabled for solve rate; precision and recall target set to 0.9999
- W&B run: `h3lxi62v`

At the first eval:

- Step: 100
- Eval duration: 427.3 seconds
- Exact set match: 0.935
- Value precision: 0.95875
- Value recall: 0.94875
- Solve rate: 0.25

## Checkpoint Sync

The run output is periodically synced to Hugging Face every 10 minutes:

`Avra98/sudoku-latent-recurrent-hidden-20empty-stages/resume_runs/recurrent_hidden_resume_stage2sft_from1800_eval20_long_20260516_090446`

Confirmed uploaded checkpoint:

- `checkpoint-step-00100/adapter_model.safetensors`
- `checkpoint-step-00100/adapter_config.json`
- `checkpoint-step-00100/tokenizer.json`
- `checkpoint-step-00100/tokenizer_config.json`
- `checkpoint-step-00100/chat_template.jinja`
- `checkpoint-step-00100/README.md`

