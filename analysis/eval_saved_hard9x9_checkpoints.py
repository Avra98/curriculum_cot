from __future__ import annotations

import json
import os
from types import SimpleNamespace
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if ROOT := "/home/ubuntu/curriculum_cot":
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from multi_output_cell_policy import grpo_multi_output_train as baseline_grpo
from multi_output_cell_policy import sft_multi_output_train as baseline_sft
from latent_multi_output_cell_policy import grpo_residual_projector_latent_train as latent_grpo
from latent_multi_output_cell_policy import residual_projector_warmstart_sft_latent_multi_output_train as latent_sft

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_DIR = os.path.join(ROOT, ".hf_cache")
DATA_PATH = os.path.join(ROOT, "data", "sudoku_t3_30empty_value_qwen_text.jsonl")
EVAL_ROWS = 20
TOTAL_EMPTIES_HINT = 30


def make_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    return tokenizer


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_baseline_sft_model(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=baseline_sft.pick_dtype() if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint_dir, is_trainable=False)
    model.to(device)
    model.eval()
    return model


def make_baseline_grpo_model(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=baseline_grpo.pick_dtype() if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = baseline_grpo.load_trainable_adapter(base, checkpoint_dir)
    model.to(device)
    model.eval()
    return model


def make_latent_model(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=latent_grpo.pick_dtype() if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = latent_grpo.load_trainable_adapter(base, checkpoint_dir)
    projector_hidden = latent_grpo.infer_projector_hidden_from_state(checkpoint_dir) or latent_grpo.PROJECTOR_HIDDEN
    latent_grpo.attach_residual_projector_modules(
        model,
        hidden_size=int(latent_grpo.unwrap_backbone(model).config.hidden_size),
        projector_hidden=projector_hidden,
    )
    latent_grpo.maybe_load_projector_state(model, checkpoint_dir)
    model.to(device)
    model.eval()
    return model


def common_reward_args() -> dict:
    return {
        "reward_good_value": 1.0,
        "penalty_bad_value": 1.75,
        "penalty_malformed": 4.0,
        "penalty_empty": 0.5,
        "penalty_singleton": 1.5,
    }


def eval_baseline_sft(checkpoint_dir: str, stage_i: int) -> dict:
    device = make_device()
    tokenizer = make_tokenizer()
    model = make_baseline_sft_model(checkpoint_dir, device)
    rows = baseline_sft.load_jsonl_rows(DATA_PATH, limit_rows=EVAL_ROWS)
    args = SimpleNamespace(
        stage_i=int(stage_i),
        total_empties_hint=TOTAL_EMPTIES_HINT,
        max_completion_length=24,
        debug_print_limit=0,
    )
    metrics = baseline_sft.run_eval(args, rows, model, tokenizer, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def eval_baseline_grpo(checkpoint_dir: str, stage_i: int) -> dict:
    device = make_device()
    tokenizer = make_tokenizer()
    model = make_baseline_grpo_model(checkpoint_dir, device)
    rows = baseline_grpo.load_jsonl_rows(DATA_PATH, limit_rows=EVAL_ROWS)
    args = SimpleNamespace(
        stage_i=int(stage_i),
        total_empties_hint=TOTAL_EMPTIES_HINT,
        max_completion_length=24,
        debug_print_limit=0,
        **common_reward_args(),
    )
    metrics = baseline_grpo.run_eval(args=args, rows=rows, model=model, tokenizer=tokenizer, device=device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def eval_latent_sft(checkpoint_dir: str, stage_i: int, num_cot_tokens: int) -> dict:
    device = make_device()
    tokenizer = make_tokenizer()
    model = make_latent_model(checkpoint_dir, device)
    rows = baseline_sft.load_jsonl_rows(DATA_PATH, limit_rows=EVAL_ROWS)
    args = SimpleNamespace(
        stage_i=int(stage_i),
        num_cot_tokens=int(num_cot_tokens),
        total_empties_hint=TOTAL_EMPTIES_HINT,
        max_completion_length=32,
        debug_print_limit=0,
        **common_reward_args(),
    )
    metrics = latent_sft.run_eval(args=args, rows=rows, model=model, tokenizer=tokenizer, device=device, eval_stage_i=int(stage_i))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def eval_latent_grpo(checkpoint_dir: str, stage_i: int, num_cot_tokens: int) -> dict:
    device = make_device()
    tokenizer = make_tokenizer()
    model = make_latent_model(checkpoint_dir, device)
    rows = latent_grpo.load_jsonl_rows(DATA_PATH, limit_rows=EVAL_ROWS)
    args = SimpleNamespace(
        stage_i=int(stage_i),
        num_cot_tokens=int(num_cot_tokens),
        total_empties_hint=TOTAL_EMPTIES_HINT,
        max_completion_length=32,
        debug_print_limit=0,
        **common_reward_args(),
    )
    metrics = latent_grpo.run_eval(args=args, rows=rows, model=model, tokenizer=tokenizer, device=device, eval_stage_i=int(stage_i))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    # Explicit step dirs (not run roots) so metrics match the agreed endpoints.
    checkpoints = [
        {
            "label": "baseline_stage1_sft",
            "stage_i": 1,
            "kind": "baseline_sft",
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_baseline_extension/hard_9x9_qwen05b/baseline/20260404_023600_baseline30_clean/baseline_pipeline_30empty_4stage_hard9x9/stage01_sft_i1_30empty/checkpoint-step-01000",
            ),
        },
        {
            "label": "baseline_stage1_grpo",
            "stage_i": 1,
            "kind": "baseline_grpo",
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_baseline_extension/hard_9x9_qwen05b/baseline/grpo/i1_20260404_fixed_baseline_grpo_i1/checkpoint-5350",
            ),
        },
        {
            "label": "baseline_stage2_sft",
            "stage_i": 2,
            "kind": "baseline_sft",
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_baseline_extension/hard_9x9_qwen05b/baseline/sft/i2_20260404_stage2_baseline_sft_from_grpo5350/checkpoint-step-13100",
            ),
        },
        {
            "label": "baseline_stage2_grpo",
            "stage_i": 2,
            "kind": "baseline_grpo",
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_baseline_extension/hard_9x9_qwen05b/baseline/grpo/i2_20260405_stage2_baseline_grpo_from_sft13100/checkpoint-4325",
            ),
        },
        {
            "label": "latent_stage1_sft",
            "stage_i": 1,
            "kind": "latent_sft",
            "num_cot_tokens": 1,
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_latent_extension/hard_9x9_qwen05b/latent/20260404_013500_latent30_frombaseline/latent_pipeline_30empty_4stage_hard9x9/stage01_sft_i1_30empty_residual_projector/checkpoint-step-00200",
            ),
        },
        {
            "label": "latent_stage1_grpo",
            "stage_i": 1,
            "kind": "latent_grpo",
            "num_cot_tokens": 1,
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_latent_extension/hard_9x9_qwen05b/latent/grpo/i1_cot1_20260404_fixed_latent_grpo_i1/checkpoint-2740",
            ),
        },
        {
            "label": "latent_stage2_sft",
            "stage_i": 2,
            "kind": "latent_sft",
            "num_cot_tokens": 2,
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_latent_extension/hard_9x9_qwen05b/latent/sft/i2_cot2_20260404_stage2_latent_sft_from_grpo2740/checkpoint-step-00700",
            ),
        },
        {
            "label": "latent_stage2_grpo",
            "stage_i": 2,
            "kind": "latent_grpo",
            "num_cot_tokens": 2,
            "checkpoint_dir": os.path.join(
                ROOT,
                "final_checkpoint/large_latent_extension/hard_9x9_qwen05b/latent/grpo/i2_cot2_20260405_stage2_latent_grpo_from_sft00700/checkpoint-1620",
            ),
        },
    ]

    results: dict[str, dict] = {}
    for item in checkpoints:
        label = item["label"]
        print(f"[eval] starting {label}", flush=True)
        if item["kind"] == "baseline_sft":
            metrics = eval_baseline_sft(item["checkpoint_dir"], item["stage_i"])
        elif item["kind"] == "baseline_grpo":
            metrics = eval_baseline_grpo(item["checkpoint_dir"], item["stage_i"])
        elif item["kind"] == "latent_sft":
            metrics = eval_latent_sft(item["checkpoint_dir"], item["stage_i"], item["num_cot_tokens"])
        else:
            metrics = eval_latent_grpo(item["checkpoint_dir"], item["stage_i"], item["num_cot_tokens"])
        results[label] = metrics
        print(json.dumps({"label": label, "metrics": metrics}, sort_keys=True), flush=True)

    print("[eval] complete", flush=True)
    print(json.dumps(results, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
