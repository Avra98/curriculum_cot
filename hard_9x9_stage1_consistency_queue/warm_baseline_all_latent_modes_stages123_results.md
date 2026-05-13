# Warm Baseline All-Latent Stages 1-3 Results

Run tag: `warmbaseline_alllatent_stages123_20260512_1620`

Base model: `Qwen/Qwen2.5-1.5B-Instruct`

Stage-1 warm baseline adapter:

```text
/home/ubuntu/curriculum_cot/final_checkpoint/hard_9x9_20empty_baseline_1p5b_warmup/baseline_1p5b_warmup_bs32_eval100_20260512_203845/20empty/stage01_sft_i1_20empty_1p5b_warmup/checkpoint-step-01000
```

This file records the solve-rate snapshot from the ongoing full pipeline. Later
stages should be updated when all modes finish.

## Current Phase Snapshot

| Mode | Current phase at snapshot |
| --- | --- |
| `residual` | Stage-2 latent SFT |
| `fixed_slots` | Stage-2 latent SFT |
| `recurrent_hidden` | Stage-2 baseline warm-up SFT |
| `latent_seeds` | Stage-3 baseline warm-up SFT |

## Latest Solve Rates By Phase

| Mode | Stage 1 latent SFT | Stage 1 latent GRPO | Stage 2 baseline warm-up | Stage 2 latent SFT | Stage 2 latent GRPO | Stage 3 baseline warm-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual` | 0.470 latest / 0.610 best | 0.620 | 0.110 latest / 0.150 best | in progress | not reached | not reached |
| `fixed_slots` | 0.770 latest / 0.770 best | 0.870 | 0.140 latest / 0.140 best | 0.100 latest / 0.100 best | not reached | not reached |
| `recurrent_hidden` | 0.860 latest / 0.860 best | 0.950 | 0.110 latest / 0.110 best | not reached | not reached | not reached |
| `latent_seeds` | 0.740 latest / 0.740 best | 0.860 | 0.090 latest / 0.100 best | 0.120 latest / 0.120 best | 0.090 | started, no eval yet |

## Stage 1 Solve Trajectories

| Mode | Latent SFT solve rates | Post-GRPO solve rate |
| --- | --- | ---: |
| `residual` | 0.320 -> 0.610 -> 0.520 -> 0.470 | 0.620 |
| `fixed_slots` | 0.650 -> 0.200 -> 0.660 -> 0.770 | 0.870 |
| `recurrent_hidden` | 0.400 -> 0.600 -> 0.800 -> 0.860 | 0.950 |
| `latent_seeds` | 0.290 -> 0.500 -> 0.640 -> 0.740 | 0.860 |

## Stage 2 Solve Trajectories So Far

| Mode | Baseline warm-up solve rates | Latent SFT solve rates | Post-GRPO solve rate |
| --- | --- | --- | ---: |
| `residual` | 0.050 -> 0.150 -> 0.110 -> 0.110 | in progress | not reached |
| `fixed_slots` | 0.090 -> 0.120 -> 0.080 -> 0.140 | 0.080 -> 0.100 | not reached |
| `recurrent_hidden` | 0.060 -> 0.090 -> 0.100 -> 0.110 | not reached | not reached |
| `latent_seeds` | 0.090 -> 0.100 -> 0.080 -> 0.090 | 0.080 -> 0.090 -> 0.110 -> 0.120 | 0.090 |

## W&B Links

Stage 1 latent SFT:

- `residual`: https://wandb.ai/training-dynamics/sudoku-latent-stage-sft-warm-baseline/runs/sp4seb59
- `fixed_slots`: https://wandb.ai/training-dynamics/sudoku-latent-stage-sft-warm-baseline/runs/d62aiu1g
- `recurrent_hidden`: https://wandb.ai/training-dynamics/sudoku-latent-stage-sft-warm-baseline/runs/cv3nr7ie
- `latent_seeds`: https://wandb.ai/training-dynamics/sudoku-latent-stage-sft-warm-baseline/runs/1f818jfg

Additional stage runs are logged under:

- SFT project: https://wandb.ai/training-dynamics/sudoku-latent-stage-sft-warm-baseline
- GRPO project: https://wandb.ai/training-dynamics/sudoku-latent-stage-grpo-warm-baseline
- Baseline warm-up project: https://wandb.ai/training-dynamics/sudoku-baseline-stage-warmups
