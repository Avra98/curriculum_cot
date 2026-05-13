# Stage-1 Latent SFT Mode Sweep

This folder contains launchers for the 9x9 Sudoku curriculum experiments. The
stage-1 latent sweep launcher is:

```bash
bash hard_9x9_stage1_consistency_queue/launch_20empty_stage1_sft_all_latent_modes_parallel.sh
```

The goal of this sweep is to compare the four latent implementations under the
same stage-1 SFT setup and measure which one gives the fastest useful
convergence. The main comparison should include training loss, held-out value
precision/recall, completion quality, wall-clock time, and GPU efficiency. In
particular, compare both loss vs. optimizer step and loss vs. elapsed time,
because some methods do more transformer forward passes per step.

## Four Latent Modes

### `residual`

The residual mode performs a dynamic latent hidden rollout, then projects the
difference between the latent hidden state and the base hidden state back into
the model hidden space. This projected delta is added to the base next-token
hidden state before computing logits. It is expressive, but it is slower because
the latent rollout requires repeated transformer passes.

### `fixed_slots`

The fixed-slots mode learns a bank of trainable latent slot embeddings plus a
separate final readout slot. For each prediction, the model runs once on:

```text
[prompt tokens, slot_1, ..., slot_k, final_slot]
```

The next token is predicted from the hidden state at `final_slot`. This is a
parallel latent method: all latent slots are inserted at once, so it avoids the
recursive pass used by recurrent methods.

### `recurrent_hidden`

The recurrent-hidden mode generates latent tokens dynamically from the current
example. It appends a hidden latent token, reruns the transformer, takes the new
last hidden state as the next latent token, and repeats for `num_cot_tokens`.
This is the closest to iterative hidden reasoning, but it is usually the
slowest because the latent steps are serial.

### `latent_seeds`

The latent-seeds mode learns a bank of trainable seed embeddings. For each
prediction, the model runs once on:

```text
[prompt tokens, seed_1, ..., seed_k]
```

The next token is predicted from the hidden state at the last seed position.
Like fixed slots, this is parallel and avoids recursive transformer passes. The
main difference from `fixed_slots` is that there is no separate final readout
slot; the last seed position acts as the readout.

## Experimental Strategy

Run all four modes in parallel on stage 1 with the same dataset, LoRA settings,
number of latent tokens, stopping rule, and evaluation set. The default launcher
splits an 8-GPU node into four two-GPU jobs:

```text
residual         -> GPUs 0,1
fixed_slots      -> GPUs 2,3
recurrent_hidden -> GPUs 4,5
latent_seeds     -> GPUs 6,7
```

Use the results to decide which one or two methods should be promoted to deeper
curriculum stages. The expected practical tradeoff is that `fixed_slots` and
`latent_seeds` should be much faster per wall-clock time, while `residual` and
`recurrent_hidden` test more iterative, example-dependent latent computation.

## Warm Baseline Stages 1-3 Pipeline

The full warm-baseline launcher is:

```bash
STAGE1_BASELINE_ADAPTER_DIR=/path/to/warmed/stage1/baseline/checkpoint \
  bash hard_9x9_stage1_consistency_queue/launch_20empty_warm_baseline_all_latent_modes_stages123.sh
```

It runs all four latent modes in parallel, two GPUs per mode:

```text
residual         -> GPUs 0,1
fixed_slots      -> GPUs 2,3
recurrent_hidden -> GPUs 4,5
latent_seeds     -> GPUs 6,7
```

For each mode, the intended sequence is:

```text
stage1 latent SFT
  -> stage1 latent GRPO
  -> stage2 baseline warm-up SFT
  -> stage2 latent SFT
  -> stage2 latent GRPO
  -> stage3 baseline warm-up SFT
  -> stage3 latent SFT
  -> stage3 latent GRPO
```

The run is capped by fixed step budgets by default (`1000` SFT steps and `500`
GRPO steps per phase) and can stop early when the configured solve-rate target
is reached. The current 1.5B run uses the warmed Stage-1 baseline adapter from
`hard_9x9_20empty_baseline_1p5b_warmup`.

See `warm_baseline_all_latent_modes_stages123_results.md` for the current
solve-rate snapshot from the ongoing full-pipeline run.
