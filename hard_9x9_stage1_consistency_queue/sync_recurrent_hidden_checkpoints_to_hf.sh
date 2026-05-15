#!/usr/bin/env bash
# Periodically upload the active recurrent-hidden resume output to Hugging Face.
#
# Required:
#   RUN_OUTPUT_DIR=/path/to/recurrent_hidden_resume_stage2sft_...
#
# Optional:
#   HF_TOKEN=hf_...  # otherwise uses `hf auth login` / cached login
#   HF_REPO_ID=Avra98/sudoku-latent-recurrent-hidden-20empty-stages
#   HF_REPO_PREFIX=resume_runs/<run_name>
#   SYNC_INTERVAL_SECONDS=900

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
HF_REPO_ID="${HF_REPO_ID:-Avra98/sudoku-latent-recurrent-hidden-20empty-stages}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-900}"

if [[ -z "${RUN_OUTPUT_DIR}" ]] || [[ ! -d "${RUN_OUTPUT_DIR}" ]]; then
  printf 'ERROR: Set RUN_OUTPUT_DIR to an existing run output directory.\n' >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'ERROR: Python not found at %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

RUN_NAME="$(basename "${RUN_OUTPUT_DIR}")"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-resume_runs/${RUN_NAME}}"

upload_once() {
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, get_token

repo_id = os.environ["HF_REPO_ID"]
folder = Path(os.environ["RUN_OUTPUT_DIR"]).resolve()
path_in_repo = os.environ["HF_REPO_PREFIX"].strip("/")

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise SystemExit("No Hugging Face token found. Run `hf auth login` or set HF_TOKEN.")

api = HfApi(token=token)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder),
    path_in_repo=path_in_repo,
    commit_message=f"Sync recurrent-hidden resume checkpoints: {folder.name}",
    allow_patterns=[
        "logs/**",
        "**/checkpoint*/**",
        "**/adapter_config.json",
        "**/adapter_model.safetensors",
        "**/tokenizer.json",
        "**/tokenizer_config.json",
        "**/chat_template.jinja",
        "**/README.md",
        "**/training_args.bin",
    ],
    ignore_patterns=[
        "**/wandb_runtime/**",
        "**/.wandb/**",
        "**/wandb/**",
        "**/optimizer.pt",
        "**/scheduler.pt",
        "**/rng_state_*.pth",
    ],
)
print(f"Uploaded {folder} to {repo_id}/{path_in_repo}")
PY
}

while true; do
  date -Is
  upload_once
  sleep "${SYNC_INTERVAL_SECONDS}"
done
