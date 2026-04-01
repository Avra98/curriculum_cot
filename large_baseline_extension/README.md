# Large Baseline Extension Launchers

This folder contains launch scripts for the non-location baseline multi-output runs.

- `launch_nonlocation_pipeline.sh`
- `launch_nonlocation_sft.sh`
- `launch_nonlocation_grpo.sh`

The main entry point for a full staged resume run is `launch_nonlocation_pipeline.sh`.

Useful environment variables:

- `MIN_STAGE`
- `MAX_STAGE`
- `NUM_PROCESSES`
- `GPU_IDS`
- `BOOTSTRAP_ADAPTER_DIR`
- `OUTPUT_ROOT`
- `RUN_TAG`
- `LIMIT_TRAIN_ROWS`
- `WANDB_MODE`
- `WANDB_ENTITY`

Example:

```bash
MIN_STAGE=3 \
MAX_STAGE=5 \
NUM_PROCESSES=8 \
GPU_IDS=0,1,2,3,4,5,6,7 \
BOOTSTRAP_ADAPTER_DIR=/path/to/stage02_grpo \
WANDB_MODE=online \
WANDB_ENTITY=training-dynamics \
bash launch_nonlocation_pipeline.sh
```
