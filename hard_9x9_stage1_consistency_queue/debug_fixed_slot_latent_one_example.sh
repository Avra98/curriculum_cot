#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_PATH="${DATA_PATH:-${ROOT}/data/sudoku_t3_20empty_value_qwen_text_stage1_train.jsonl}"
GPU_ID="${GPU_ID:-0}"
NUM_COT="${NUM_COT:-3}"
MAX_LATENT_SLOTS="${MAX_LATENT_SLOTS:-8}"
LIMIT_ROWS="${LIMIT_ROWS:-1}"
TRAIN_STEPS="${TRAIN_STEPS:-60}"
LR="${LR:-1e-1}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

exec "${PYTHON_BIN}" - <<'PY'
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from aligned_cell_policy.shared_cell_policy import build_cell_examples_from_row
from latent_multi_output_cell_policy.grpo_residual_projector_latent_train import (
    attach_fixed_latent_slot_modules,
    fixed_slot_next_token_logits_from_ids,
    load_jsonl_rows,
    load_trainable_adapter,
    pick_dtype,
    sample_fixed_slot_completion,
    unwrap_backbone,
)
from multi_output_cell_policy.prompt_builder import build_multi_output_cell_prompt
from multi_output_cell_policy.shared_multi_output_policy import build_supervised_completion


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
data_path = os.environ.get("DATA_PATH", "data/sudoku_t3_20empty_value_qwen_text_stage1_train.jsonl")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_cot = env_int("NUM_COT", 5)
max_latent_slots = env_int("MAX_LATENT_SLOTS", 8)
limit_rows = env_int("LIMIT_ROWS", 1)
train_steps = env_int("TRAIN_STEPS", 60)
lr = env_float("LR", 1e-1)
lora_r = env_int("LORA_R", 32)
lora_alpha = env_int("LORA_ALPHA", 64)
lora_dropout = env_float("LORA_DROPOUT", 0.05)

rows = load_jsonl_rows(data_path, limit_rows=limit_rows)
ex = build_cell_examples_from_row(rows[0])[0]

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

prompt = build_multi_output_cell_prompt(
    ex.grid,
    target_cell=ex.target_cell,
    stage_i=1,
    tokenizer=tokenizer,
    turn_idx=ex.turn_idx,
    total_turns=ex.total_turns,
    prev_output_flag=None,
    total_empties_hint=20,
)
target_text = build_supervised_completion(ex, stage_i=1) + (tokenizer.eos_token or "")
print("target_text", target_text)

base = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=pick_dtype(),
    low_cpu_mem_usage=True,
)
model = load_trainable_adapter(base, "", lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
attach_fixed_latent_slot_modules(
    model,
    hidden_size=int(unwrap_backbone(model).config.hidden_size),
    max_latent_slots=max_latent_slots,
)
if hasattr(model, "config"):
    model.config.use_cache = False
backbone = unwrap_backbone(model)
if hasattr(backbone, "config"):
    backbone.config.use_cache = False
model.to(device)

for p in model.parameters():
    p.requires_grad = False
model.fixed_latent_slots.requires_grad_(True)
model.fixed_final_slot_embed.requires_grad_(True)
optimizer = torch.optim.AdamW([model.fixed_latent_slots, model.fixed_final_slot_embed], lr=lr)

prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
completion_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


@torch.no_grad()
def sample_now(tag: str) -> None:
    model.eval()
    attn = torch.ones_like(prompt_ids, device=device)
    logits = fixed_slot_next_token_logits_from_ids(model, prompt_ids, attn, num_cot)
    probs = torch.softmax(logits[0].float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=5)
    out_ids = sample_fixed_slot_completion(
        model,
        tokenizer,
        prompt_ids,
        attn,
        num_cot_tokens=num_cot,
        max_new_tokens=12,
        do_sample=False,
    )
    top_next = [(tokenizer.decode([int(i)]), round(float(p), 4)) for i, p in zip(top_ids.tolist(), top_probs.tolist())]
    print(tag, tokenizer.decode(out_ids[0], skip_special_tokens=True), "top_next", top_next)


sample_now("before:")

for step in range(1, train_steps + 1):
    model.train()
    cur_ids = prompt_ids
    cur_mask = torch.ones_like(prompt_ids, device=device)
    losses = []
    for idx in range(int(completion_ids.shape[1])):
        logits = fixed_slot_next_token_logits_from_ids(model, cur_ids, cur_mask, num_cot)
        target = completion_ids[:, idx]
        losses.append(F.cross_entropy(logits.float(), target, reduction="mean"))
        cur_ids = torch.cat([cur_ids, completion_ids[:, idx : idx + 1]], dim=1)
        cur_mask = torch.cat(
            [
                cur_mask,
                torch.ones((cur_mask.shape[0], 1), dtype=cur_mask.dtype, device=cur_mask.device),
            ],
            dim=1,
        )
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step == 1 or step % 10 == 0 or step == train_steps:
        print(f"step={step} loss={float(loss.item()):.6f}")
        sample_now(f"after_step_{step}:")
PY
