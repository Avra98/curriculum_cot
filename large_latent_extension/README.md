# Large Latent Extension Launchers

This folder contains the launch scripts for the non-location latent CoT runs.

- `launch_nonlocation_sft.sh`
- `launch_nonlocation_grpo.sh`

These are the scripts used for the distributed multi-GPU non-location curriculum.

Useful environment variables:

- `NUM_COT_TOKENS`
- `STAGE_I`
- `NUM_PROCESSES`
- `GPU_IDS`
- `INIT_ADAPTER_DIR`
- `OUTPUT_DIR`
- `LIMIT_TRAIN_ROWS`
- `WANDB_MODE`
- `WANDB_ENTITY`

Example:

```bash
NUM_COT_TOKENS=3 \
STAGE_I=3 \
NUM_PROCESSES=8 \
GPU_IDS=0,1,2,3,4,5,6,7 \
WANDB_MODE=online \
WANDB_ENTITY=training-dynamics \
bash launch_nonlocation_sft.sh
```
