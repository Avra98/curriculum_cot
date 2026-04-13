# Addition Carry Experiment

This folder contains a standalone PyTorch experiment for algorithmic decimal addition with carry on a one-layer decoder-only Transformer.

The comparison includes exactly three methods:

- `nocurr_nocot`: no curriculum, no latent chain-of-thought
- `curr_nocot`: digit-length curriculum, no latent chain-of-thought
- `curr_cot`: same one-layer backbone plus recurrent latent scratchpad tokens

## Task

Each example adds two reversed digit sequences. Stage `k` means only the first `k` least-significant positions vary and the rest are zero. The model is queried position by position and predicts:

- the sum digit at the queried position
- the carry bit leaving that position

The latent method reuses the same one-layer Transformer recurrently. After each pass, the final hidden state is appended as a continuous latent token and the same layer is run again.

## Files

- `config.py`: experiment config and CLI handling
- `data.py`: synthetic data generation, curriculum stages, carry-heavy subsets
- `model.py`: one-layer decoder-only Transformer and latent recurrence
- `train.py`: single-run training entrypoint
- `eval.py`: evaluation and diagnostics
- `plots.py`: local plotting
- `run_comparison.py`: multi-seed comparison across all three methods

## Outputs

Each run writes:

- `config.json`
- `artifacts/history.jsonl`
- `artifacts/summary.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- local plots under `plots/`

If W&B is enabled, the same run also logs metrics there.

## Run A Single Method

Default settings:

```bash
python addition/train.py --model nocurr_nocot --use_wandb
python addition/train.py --model curr_nocot --use_wandb
python addition/train.py --model curr_cot --use_wandb
```

Run offline or local-only:

```bash
python addition/train.py --model curr_cot --wandb_mode offline
python addition/train.py --model curr_cot --no_wandb
```

## Smoke Test

Use the smoke preset to verify the whole pipeline quickly:

```bash
python addition/train.py --model curr_cot --preset smoke --no_wandb --output_dir addition_runs/smoke_curr_cot
```

## Run The Full Comparison

This runs all three methods across multiple seeds and saves aggregate plots and JSON:

```bash
python addition/run_comparison.py --preset default --use_wandb --comparison_output_dir addition_runs/comparison_default
```

Small fast comparison:

```bash
python addition/run_comparison.py --preset smoke --no_wandb --comparison_output_dir addition_runs/comparison_smoke
```

## Main Metrics

The experiment reports:

- digit accuracy by queried position
- carry accuracy by queried position
- exact whole-sum accuracy by active length
- average digit accuracy by length
- in-distribution results up to `train_max_digits`
- OOD results on longer lengths
- separate uniform and carry-heavy evaluations

## Diagnostics

The evaluation also includes:

- a linear probe on hidden states for carry prediction
- attention summaries showing how strongly the final readout attends to current digits and latent tokens

## Notes

- The first version is intentionally small enough to iterate locally.
- The backbone depth stays fixed at one layer in all methods.
- The latent method gets more recurrent compute, not more layers.
