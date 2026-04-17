from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from aligned_cell_policy.shared_cell_policy import build_cell_examples_from_row
from checkpoint_utils import ensure_final_checkpoint_dir, save_checkpoint_and_update_final
from multi_output_cell_policy.prompt_builder import build_multi_output_cell_prompt
from multi_output_cell_policy.rewards import score_prediction_text
from multi_output_cell_policy.shared_multi_output_policy import (
    build_supervised_completion,
    make_solved_grid_from_row,
    stage_i_consistent_values,
)
from latent_multi_output_cell_policy.grpo_residual_projector_latent_train import (
    PROJECTOR_HIDDEN,
    attach_fixed_latent_slot_modules,
    attach_latent_seed_modules,
    attach_residual_projector_modules,
    extend_attention_mask,
    fixed_slot_next_token_logits_from_ids,
    infer_latent_seed_count_from_state,
    infer_projector_hidden_from_state,
    infer_fixed_slot_count_from_state,
    latent_seed_next_token_logits_from_ids,
    load_trainable_adapter,
    maybe_load_fixed_slot_state,
    maybe_load_latent_seed_state,
    maybe_load_projector_state,
    recurrent_hidden_next_token_logits_from_ids,
    residual_next_token_logits_from_ids as shared_residual_next_token_logits_from_ids,
    sample_recurrent_hidden_completion,
    sample_fixed_slot_completion,
    sample_latent_completion,
    sample_latent_seed_completion,
    save_fixed_slot_latent_state,
    save_latent_projector_state,
    save_latent_seed_state,
    unwrap_backbone,
)


try:
    import wandb
except Exception:
    wandb = None


@dataclass
class Args:
    model_name: str
    train_jsonl: str
    train_jsonl_stage1: str
    train_jsonl_stage2: str
    eval_jsonl: str
    output_dir: str
    cache_dir: str
    init_adapter_dir: str
    seed: int
    gpu_id: int
    stage_i: int
    num_cot_tokens: int
    total_empties_hint: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_epochs: float
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    enable_gradient_checkpointing: bool
    logging_steps: int
    save_steps: int
    eval_steps: int
    eval_rows: int
    max_completion_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: str
    wandb_mode: str
    debug_print_limit: int
    limit_train_rows: int
    eval_exact_set_match_stop: float
    eval_value_precision_stop: float
    eval_value_recall_stop: float
    eval_solve_rate_stop: float
    min_steps_before_stop: int
    max_wall_clock_seconds: int
    max_steps: int
    mixed_stage1_ratio: float
    mixed_stage2_ratio: float
    reward_good_value: float
    penalty_bad_value: float
    penalty_malformed: float
    penalty_empty: float
    penalty_singleton: float
    multi_value_oversample_factor: int
    train_target_size_min: int
    train_target_size_max: int
    eval_target_size_min: int
    eval_target_size_max: int
    latent_mode: str
    max_latent_slots: int
    max_latent_seeds: int


def configure_hf_cache(cache_dir: str) -> str:
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    hub_dir = os.path.join(cache_dir, "hub")
    transformers_dir = os.path.join(cache_dir, "transformers")
    os.makedirs(hub_dir, exist_ok=True)
    os.makedirs(transformers_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = hub_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_dir
    os.environ["TRANSFORMERS_CACHE"] = transformers_dir
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return cache_dir


def configure_wandb_dirs(output_dir: str) -> None:
    wandb_dir = os.path.join(output_dir, "wandb_runtime")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", wandb_dir)
    os.environ.setdefault("WANDB_CACHE_DIR", wandb_dir)
    os.environ.setdefault("WANDB_CONFIG_DIR", wandb_dir)


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_jsonl_rows(path: str, limit_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit_rows > 0 and len(rows) >= limit_rows:
                break
    return rows


def target_size_allowed(target_size: int, min_size: int, max_size: int) -> bool:
    if int(min_size) > 0 and int(target_size) < int(min_size):
        return False
    if int(max_size) > 0 and int(target_size) > int(max_size):
        return False
    return True


def build_training_examples(
    rows: List[Dict[str, Any]],
    *,
    tokenizer: Any,
    stage_i: int,
    total_empties_hint: int,
    progress_every_rows: int = 10,
    progress_callback: Any = None,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    eos_text = getattr(tokenizer, "eos_token", None) or ""
    for row_idx, row in enumerate(rows, start=1):
        solved = make_solved_grid_from_row(row)
        for ex in build_cell_examples_from_row(row):
            target_values = stage_i_consistent_values(ex.grid, target_cell=ex.target_cell, stage_i=stage_i)
            if not target_size_allowed(
                len(target_values),
                getattr(tokenizer, "_train_target_size_min", 0),
                getattr(tokenizer, "_train_target_size_max", 0),
            ):
                continue
            prompt = build_multi_output_cell_prompt(
                ex.grid,
                target_cell=ex.target_cell,
                stage_i=stage_i,
                tokenizer=tokenizer,
                turn_idx=ex.turn_idx,
                total_turns=ex.total_turns,
                prev_output_flag=None,
                total_empties_hint=total_empties_hint,
            )
            target_text = build_supervised_completion(ex, stage_i=stage_i)
            if eos_text:
                target_text = target_text + eos_text
            repeat_count = max(1, int(getattr(tokenizer, "_multi_value_oversample_factor", 1))) if len(target_values) > 1 else 1
            for _ in range(repeat_count):
                examples.append(
                    {
                        "prompt_text": prompt,
                        "completion_text": target_text,
                        "target_values": list(target_values),
                        "grid": ex.grid,
                        "solved": solved,
                        "target_cell": ex.target_cell,
                    }
                )
        if progress_callback is not None and (
            row_idx == 1 or row_idx == len(rows) or row_idx % max(1, int(progress_every_rows)) == 0
        ):
            progress_callback(row_idx, len(rows), len(examples))
    return examples


def _prepared_data_dir(args: Args) -> str:
    path = os.path.join(PARENT_DIR, "_prepared_data", "multi_output_cell_policy")
    os.makedirs(path, exist_ok=True)
    return path


def _prepared_sft_cache_path(args: Args) -> str:
    payload = json.dumps(
        {
            "completion_format_version": 2,
            "train_jsonl": os.path.abspath(args.train_jsonl),
            "stage_i": int(args.stage_i),
            "total_empties_hint": int(args.total_empties_hint),
            "limit_train_rows": int(args.limit_train_rows),
            "model_name": str(args.model_name),
            "multi_value_oversample_factor": int(args.multi_value_oversample_factor),
            "train_target_size_min": int(args.train_target_size_min),
            "train_target_size_max": int(args.train_target_size_max),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:20]
    return os.path.join(_prepared_data_dir(args), f"sft_stage{int(args.stage_i):02d}_{digest}.jsonl")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    return value


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), separators=(",", ":")) + "\n")
    os.replace(tmp_path, path)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _wait_for_cache(path: str, timeout_s: float = 7200.0) -> None:
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for prepared cache: {path}")
        time.sleep(2.0)


def load_or_build_sft_examples(
    args: Args,
    *,
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    rank: int,
    world_size: int,
    progress_callback: Any = None,
) -> List[Dict[str, Any]]:
    cache_path = _prepared_sft_cache_path(args)
    if os.path.exists(cache_path):
        return _read_jsonl(cache_path)
    if rank == 0:
        print(f"[dataset build][sft stage {args.stage_i}] building prepared cache: {cache_path}", flush=True)
        examples = build_training_examples(
            rows,
            tokenizer=tokenizer,
            stage_i=args.stage_i,
            total_empties_hint=args.total_empties_hint,
            progress_every_rows=10,
            progress_callback=progress_callback,
        )
        _write_jsonl(cache_path, examples)
        return examples
    _wait_for_cache(cache_path)
    return _read_jsonl(cache_path)


def residual_next_token_logits_from_ids(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_cot_tokens: int
) -> torch.Tensor:
    # Keep SFT on the exact same latent rollout/logit path used by decoding and GRPO,
    # so training is supervising the same latent mechanism we later sample from.
    return shared_residual_next_token_logits_from_ids(model, input_ids, attention_mask, num_cot_tokens)


def fixed_slot_next_token_logits_for_sft(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_cot_tokens: int
) -> torch.Tensor:
    return fixed_slot_next_token_logits_from_ids(model, input_ids, attention_mask, num_cot_tokens)


def recurrent_hidden_next_token_logits_for_sft(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_cot_tokens: int
) -> torch.Tensor:
    return recurrent_hidden_next_token_logits_from_ids(model, input_ids, attention_mask, num_cot_tokens)


def latent_seed_next_token_logits_for_sft(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_cot_tokens: int
) -> torch.Tensor:
    return latent_seed_next_token_logits_from_ids(model, input_ids, attention_mask, num_cot_tokens)


def _tokenize_prompt_completion_pair(tokenizer: Any, prompt_text: str, completion_text: str) -> tuple[List[int], List[int]]:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    all_ids = tokenizer(prompt_text + completion_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    completion_ids = all_ids[int(prompt_ids.shape[0]) :]
    return prompt_ids.tolist(), completion_ids.tolist()


def prepare_tokenized_train_examples(examples: List[Dict[str, Any]], tokenizer: Any) -> List[Dict[str, Any]]:
    for ex in examples:
        if "prompt_ids" in ex and "completion_ids" in ex:
            continue
        prompt_ids, completion_ids = _tokenize_prompt_completion_pair(
            tokenizer,
            str(ex["prompt_text"]),
            str(ex["completion_text"]),
        )
        ex["prompt_ids"] = prompt_ids
        ex["completion_ids"] = completion_ids
    return examples


def latent_residual_completion_ce_loss(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    device: torch.device,
    *,
    num_cot_tokens: int,
    latent_mode: str,
) -> torch.Tensor:
    prompt_ids, completion_ids = _tokenize_prompt_completion_pair(tokenizer, prompt_text, completion_text)
    return latent_batched_completion_ce_loss(
        model,
        [{"prompt_ids": prompt_ids, "completion_ids": completion_ids}],
        device,
        num_cot_tokens=num_cot_tokens,
        latent_mode=latent_mode,
        pad_token_id=int(tokenizer.pad_token_id),
    )


def latent_batched_completion_ce_loss(
    model: nn.Module,
    batch_examples: List[Dict[str, Any]],
    device: torch.device,
    *,
    num_cot_tokens: int,
    latent_mode: str,
    pad_token_id: int,
) -> torch.Tensor:
    if not batch_examples:
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
    prompt_token_lists = [list(ex.get("prompt_ids", [])) for ex in batch_examples]
    completion_token_lists = [list(ex.get("completion_ids", [])) for ex in batch_examples]
    if not any(completion_token_lists):
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    batch_size = len(batch_examples)
    max_prompt_len = max(len(ids) for ids in prompt_token_lists)
    max_completion_len = max(len(ids) for ids in completion_token_lists)
    total_seq_len = max_prompt_len + max_completion_len

    full_ids = torch.full((batch_size, total_seq_len), int(pad_token_id), dtype=torch.long, device=device)
    full_mask = torch.zeros((batch_size, total_seq_len), dtype=torch.long, device=device)
    completion_tokens = torch.full((batch_size, max_completion_len), int(pad_token_id), dtype=torch.long, device=device)
    completion_mask = torch.zeros((batch_size, max_completion_len), dtype=torch.bool, device=device)

    for row_idx, (prompt_ids, completion_ids) in enumerate(zip(prompt_token_lists, completion_token_lists, strict=True)):
        prompt_len = len(prompt_ids)
        prompt_start = max_prompt_len - prompt_len
        if prompt_len > 0:
            full_ids[row_idx, prompt_start:max_prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            full_mask[row_idx, prompt_start:max_prompt_len] = 1
        if completion_ids:
            comp_len = len(completion_ids)
            completion_tokens[row_idx, :comp_len] = torch.tensor(completion_ids, dtype=torch.long, device=device)
            completion_mask[row_idx, :comp_len] = True

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=device, dtype=torch.float32)
    mode = str(latent_mode).strip().lower()
    for idx in range(max_completion_len):
        cur_len = max_prompt_len + idx
        cur_ids = full_ids[:, :cur_len]
        cur_mask = full_mask[:, :cur_len]
        if mode == "fixed_slots":
            logits = fixed_slot_next_token_logits_for_sft(model, cur_ids, cur_mask, num_cot_tokens)
        elif mode == "recurrent_hidden":
            logits = recurrent_hidden_next_token_logits_for_sft(model, cur_ids, cur_mask, num_cot_tokens)
        elif mode == "latent_seeds":
            logits = latent_seed_next_token_logits_for_sft(model, cur_ids, cur_mask, num_cot_tokens)
        else:
            logits = residual_next_token_logits_from_ids(model, cur_ids, cur_mask, num_cot_tokens)
        active = completion_mask[:, idx]
        if torch.any(active):
            targets = completion_tokens[:, idx]
            token_losses = F.cross_entropy(logits.float(), targets, reduction="none")
            total_loss = total_loss + token_losses[active].sum()
            total_tokens = total_tokens + active.to(dtype=torch.float32).sum()
        full_ids[:, cur_len] = completion_tokens[:, idx]
        full_mask[:, cur_len] = active.to(dtype=full_mask.dtype)
    if float(total_tokens.detach().item()) <= 0.0:
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
    return total_loss / total_tokens


@torch.no_grad()
def run_eval(args: Args, rows: List[Dict[str, Any]], model: torch.nn.Module, tokenizer: Any, device: torch.device):
    model.eval()
    total_cells = 0
    parse_ok = 0.0
    canonical_ok = 0.0
    exact_set_match = 0.0
    includes_gt = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    predicted_size_sum = 0.0
    good_count_sum = 0.0
    bad_count_sum = 0.0
    solve_ok = 0
    solve_rows = 0
    printed = 0
    for row in rows:
        solved = make_solved_grid_from_row(row)
        row_all_exact = True
        row_has_eval_cell = False
        row_debug_lines: List[str] = []
        for ex in build_cell_examples_from_row(row):
            target_values = stage_i_consistent_values(ex.grid, target_cell=ex.target_cell, stage_i=args.stage_i)
            if not target_size_allowed(len(target_values), int(args.eval_target_size_min), int(args.eval_target_size_max)):
                continue
            row_has_eval_cell = True
            prompt = build_multi_output_cell_prompt(
                ex.grid,
                target_cell=ex.target_cell,
                stage_i=args.stage_i,
                tokenizer=tokenizer,
                turn_idx=ex.turn_idx,
                total_turns=ex.total_turns,
                prev_output_flag=None,
                total_empties_hint=args.total_empties_hint,
            )
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            sft_eval_mode = str(args.latent_mode).strip().lower()
            if sft_eval_mode == "fixed_slots":
                completion_ids = sample_fixed_slot_completion(
                    model,
                    tokenizer,
                    prompt_ids,
                    attn,
                    num_cot_tokens=args.num_cot_tokens,
                    max_new_tokens=max(1, int(args.max_completion_length)),
                    do_sample=False,
                )
            elif sft_eval_mode == "recurrent_hidden":
                completion_ids = sample_recurrent_hidden_completion(
                    model,
                    tokenizer,
                    prompt_ids,
                    attn,
                    num_cot_tokens=args.num_cot_tokens,
                    max_new_tokens=max(1, int(args.max_completion_length)),
                    do_sample=False,
                )
            elif sft_eval_mode == "latent_seeds":
                completion_ids = sample_latent_seed_completion(
                    model,
                    tokenizer,
                    prompt_ids,
                    attn,
                    num_cot_tokens=args.num_cot_tokens,
                    max_new_tokens=max(1, int(args.max_completion_length)),
                    do_sample=False,
                )
            else:
                completion_ids = sample_latent_completion(
                    model,
                    tokenizer,
                    prompt_ids,
                    attn,
                    num_cot_tokens=args.num_cot_tokens,
                    max_new_tokens=max(1, int(args.max_completion_length)),
                    do_sample=False,
                )
            pred_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
            info = score_prediction_text(
                text=pred_text,
                grid=ex.grid,
                solved=solved,
                target_cell=ex.target_cell,
                stage_i=args.stage_i,
                reward_good_value=args.reward_good_value,
                penalty_bad_value=args.penalty_bad_value,
                penalty_malformed=args.penalty_malformed,
                penalty_empty=args.penalty_empty,
                penalty_singleton=args.penalty_singleton,
            )
            total_cells += 1
            parse_ok += float(info["parse_ok"])
            canonical_ok += float(info["strict_canonical"])
            exact_set_match += float(info["exact_set_match"])
            includes_gt += float(info["includes_ground_truth"])
            precision_sum += float(info["value_precision"])
            recall_sum += float(info["value_recall"])
            predicted_size_sum += float(info["num_predicted_values"])
            good_count_sum += float(info["num_i_consistent_values"])
            bad_count_sum += float(info["num_non_i_consistent_values"])
            if float(info["exact_set_match"]) < 0.5:
                row_all_exact = False
            if printed < int(args.debug_print_limit):
                row_debug_lines.append(
                    f"[latent sft eval debug] true_values={info['target_values']} "
                    f"predicted_values={info['predicted_values']} output={pred_text!r}"
                )
        if row_has_eval_cell:
            if printed < int(args.debug_print_limit) and row_debug_lines:
                print("[latent sft eval debug] puzzle_outputs_begin", flush=True)
                for line in row_debug_lines:
                    print(line, flush=True)
                print("[latent sft eval debug] puzzle_outputs_end", flush=True)
                printed += 1
            solve_ok += int(row_all_exact)
            solve_rows += 1
    out = {
        "parse_rate": float(parse_ok / max(1, total_cells)),
        "strict_canonical_rate": float(canonical_ok / max(1, total_cells)),
        "exact_set_match_rate": float(exact_set_match / max(1, total_cells)),
        "includes_ground_truth_rate": float(includes_gt / max(1, total_cells)),
        "value_precision": float(precision_sum / max(1, total_cells)),
        "value_recall": float(recall_sum / max(1, total_cells)),
        "avg_predicted_set_size": float(predicted_size_sum / max(1, total_cells)),
        "avg_num_i_consistent_values": float(good_count_sum / max(1, total_cells)),
        "avg_num_non_i_consistent_values": float(bad_count_sum / max(1, total_cells)),
        "solve_rate": float(solve_ok / max(1, solve_rows)),
    }
    print(
        f"[latent sft eval] parse={out['parse_rate']:.3f} canonical={out['strict_canonical_rate']:.3f} "
        f"exact={out['exact_set_match_rate']:.3f} precision={out['value_precision']:.3f} "
        f"recall={out['value_recall']:.3f} solve={out['solve_rate']:.3f}",
        flush=True,
    )
    model.train()
    return out


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--train_jsonl_stage1", type=str, default="")
    p.add_argument("--train_jsonl_stage2", type=str, default="")
    p.add_argument("--eval_jsonl", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/home/ubuntu/curriculum-CoT/.hf_cache")
    p.add_argument("--init_adapter_dir", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--stage_i", type=int, default=1)
    p.add_argument("--num_cot_tokens", type=int, default=1)
    p.add_argument("--total_empties_hint", type=int, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Clip global grad norm before each optimizer step (0 disables).",
    )
    p.add_argument("--enable_gradient_checkpointing", action="store_true")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--eval_rows", type=int, default=20)
    p.add_argument("--max_completion_length", type=int, default=24)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_project", type=str, default="sudoku-latent-multi-output-sft")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument("--debug_print_limit", type=int, default=3)
    p.add_argument("--limit_train_rows", type=int, default=0)
    p.add_argument("--eval_exact_set_match_stop", type=float, default=0.0)
    p.add_argument("--eval_value_precision_stop", type=float, default=0.0)
    p.add_argument("--eval_value_recall_stop", type=float, default=0.0)
    p.add_argument("--eval_solve_rate_stop", type=float, default=0.0)
    p.add_argument("--min_steps_before_stop", type=int, default=0)
    p.add_argument("--max_wall_clock_seconds", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--mixed_stage1_ratio", type=float, default=0.0)
    p.add_argument("--mixed_stage2_ratio", type=float, default=1.0)
    p.add_argument("--reward_good_value", type=float, default=1.0)
    p.add_argument("--penalty_bad_value", type=float, default=1.75)
    p.add_argument("--penalty_malformed", type=float, default=4.0)
    p.add_argument("--penalty_empty", type=float, default=0.5)
    p.add_argument("--penalty_singleton", type=float, default=1.5)
    p.add_argument("--multi_value_oversample_factor", type=int, default=1)
    p.add_argument("--train_target_size_min", type=int, default=0)
    p.add_argument("--train_target_size_max", type=int, default=0)
    p.add_argument("--eval_target_size_min", type=int, default=0)
    p.add_argument("--eval_target_size_max", type=int, default=0)
    p.add_argument(
        "--latent_mode",
        type=str,
        default="residual",
        choices=["residual", "fixed_slots", "recurrent_hidden", "latent_seeds"],
    )
    p.add_argument("--max_latent_slots", type=int, default=8)
    p.add_argument(
        "--max_latent_seeds",
        type=int,
        default=8,
        help="For --latent_mode latent_seeds: bank size of trainable seed vectors m_1..m_{max}. "
        "Each stage uses --num_cot_tokens of them; the bank persists across stages when loaded from init_adapter_dir.",
    )
    return Args(**vars(p.parse_args()))


def save_checkpoint(
    model: torch.nn.Module, tokenizer: Any, output_dir: str, step: int, extra_save_fn: Any | None = None
) -> None:
    save_checkpoint_and_update_final(
        model,
        tokenizer,
        output_dir,
        f"checkpoint-step-{step:05d}",
        extra_save_fn=extra_save_fn,
    )


def main() -> None:
    args = parse_args()
    preset_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    is_main_process = rank == 0

    if preset_visible_devices:
        if is_main_process:
            print(f"Respecting pre-set CUDA_VISIBLE_DEVICES={preset_visible_devices}", flush=True)
    elif int(args.gpu_id) >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))

    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)
    ensure_final_checkpoint_dir(args.output_dir)
    cache_dir = configure_hf_cache(args.cache_dir)
    configure_wandb_dirs(args.output_dir)

    wb_run = None
    if is_main_process and args.use_wandb and wandb is not None:
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name or None,
            "mode": args.wandb_mode,
        }
        if str(args.wandb_entity).strip():
            init_kwargs["entity"] = args.wandb_entity
        wb_run = wandb.init(**init_kwargs)
        print(f"W&B run id: {wb_run.id}", flush=True)
        print(f"W&B run URL: {wb_run.url}", flush=True)
        wandb.log({"prep/rows_done": 0.0, "prep/examples_built": 0.0, "prep/cache_hit": 0.0})

    train_source = args.train_jsonl
    if int(args.stage_i) <= 1 and str(args.train_jsonl_stage1).strip():
        train_source = args.train_jsonl_stage1
    elif int(args.stage_i) > 1 and str(args.train_jsonl_stage2).strip():
        train_source = args.train_jsonl_stage2
    rows = load_jsonl_rows(train_source, limit_rows=args.limit_train_rows)
    eval_source = args.eval_jsonl if str(args.eval_jsonl).strip() else train_source
    eval_rows = load_jsonl_rows(eval_source, limit_rows=max(1, int(args.eval_rows)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer._multi_value_oversample_factor = max(1, int(args.multi_value_oversample_factor))
    tokenizer._train_target_size_min = max(0, int(args.train_target_size_min))
    tokenizer._train_target_size_max = max(0, int(args.train_target_size_max))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else f"cuda:{max(0, int(args.gpu_id))}")
    else:
        device = torch.device("cpu")

    latent_mode = str(args.latent_mode).strip().lower()
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=cache_dir, torch_dtype=pick_dtype(), low_cpu_mem_usage=True
    )
    model = load_trainable_adapter(
        base,
        args.init_adapter_dir,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    init_ad = str(args.init_adapter_dir).strip()
    if latent_mode == "fixed_slots":
        max_latent_slots = max(1, int(args.max_latent_slots))
        if init_ad:
            max_latent_slots = infer_fixed_slot_count_from_state(init_ad) or max_latent_slots
        attach_fixed_latent_slot_modules(
            model,
            hidden_size=int(unwrap_backbone(model).config.hidden_size),
            max_latent_slots=max_latent_slots,
        )
        if init_ad:
            maybe_load_fixed_slot_state(model, init_ad)
        extra_save_fn = save_fixed_slot_latent_state
    elif latent_mode == "latent_seeds":
        max_latent_seeds = max(1, int(args.max_latent_seeds))
        if init_ad:
            max_latent_seeds = infer_latent_seed_count_from_state(init_ad) or max_latent_seeds
        attach_latent_seed_modules(
            model,
            hidden_size=int(unwrap_backbone(model).config.hidden_size),
            max_latent_seeds=max_latent_seeds,
        )
        if init_ad:
            maybe_load_latent_seed_state(model, init_ad)
        extra_save_fn = save_latent_seed_state
    elif latent_mode == "recurrent_hidden":
        extra_save_fn = None
    else:
        if init_ad:
            projector_hidden = infer_projector_hidden_from_state(init_ad) or PROJECTOR_HIDDEN
        else:
            projector_hidden = PROJECTOR_HIDDEN
        attach_residual_projector_modules(
            model,
            hidden_size=int(unwrap_backbone(model).config.hidden_size),
            projector_hidden=projector_hidden,
        )
        if init_ad:
            maybe_load_projector_state(model, init_ad)
        extra_save_fn = save_latent_projector_state
    if is_main_process:
        if init_ad:
            print(f"Init adapter: {init_ad}", flush=True)
        elif latent_mode == "fixed_slots":
            print(
                "init_adapter_dir empty: fresh LoRA (random) + fixed latent slot random init "
                f"(lora_r={args.lora_r} lora_alpha={args.lora_alpha} max_latent_slots={args.max_latent_slots}).",
                flush=True,
            )
        elif latent_mode == "recurrent_hidden":
            print(
                "init_adapter_dir empty: fresh LoRA (random) + recurrent hidden latent rollout "
                f"(lora_r={args.lora_r} lora_alpha={args.lora_alpha} num_cot_tokens={args.num_cot_tokens}).",
                flush=True,
            )
        elif latent_mode == "latent_seeds":
            print(
                "init_adapter_dir empty: fresh LoRA (random) + trainable latent seed bank "
                f"(lora_r={args.lora_r} lora_alpha={args.lora_alpha} "
                f"max_latent_seeds={args.max_latent_seeds} num_cot_tokens={args.num_cot_tokens}).",
                flush=True,
            )
        else:
            print(
                "init_adapter_dir empty: fresh LoRA (random) + residual projector random init "
                f"(lora_r={args.lora_r} lora_alpha={args.lora_alpha}).",
                flush=True,
            )
    model._latent_debug_tokenizer = tokenizer
    model._fixed_slot_debug_limit = int(os.environ.get("FIXED_SLOT_DEBUG_LIMIT", "0"))
    model._fixed_slot_debug_count = 0
    model._fixed_slot_decode_debug_limit = int(os.environ.get("FIXED_SLOT_DECODE_DEBUG_LIMIT", "0"))
    model._fixed_slot_decode_debug_count = 0
    model._latent_vocab_debug_topk = int(os.environ.get("LATENT_VOCAB_DEBUG_TOPK", "1"))
    model._attention_density_debug_limit = int(os.environ.get("ATTN_DENSITY_DEBUG_LIMIT", "0"))
    model._attention_density_debug_count = 0
    model._attention_density_threshold_mult = float(os.environ.get("ATTN_DENSITY_THRESHOLD_MULT", "1.0"))
    if args.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.enable_gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    backbone = unwrap_backbone(model)
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False
    model.to(device)
    model.train()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    def on_prep_progress(rows_done: int, total_rows: int, examples_built: int) -> None:
        if is_main_process:
            print(
                f"[dataset build][sft stage {args.stage_i}] rows={rows_done}/{total_rows} examples={examples_built}",
                flush=True,
            )
            if wb_run is not None:
                wandb.log({"prep/rows_done": float(rows_done), "prep/examples_built": float(examples_built)})

    train_examples = load_or_build_sft_examples(
        args,
        rows=rows,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        progress_callback=on_prep_progress,
    )
    train_examples = prepare_tokenized_train_examples(train_examples, tokenizer)
    if is_main_process and wb_run is not None:
        wandb.log(
            {
                "prep/cache_hit": float(os.path.exists(_prepared_sft_cache_path(args))),
                "prep/examples_final": float(len(train_examples)),
            }
        )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    denom = max(1, int(args.gradient_accumulation_steps)) * max(1, int(args.per_device_train_batch_size))
    total_steps = max(1, math.ceil(len(train_examples) * args.num_epochs / denom))
    if int(args.max_steps) > 0:
        total_steps = min(total_steps, int(args.max_steps))
    step = 0
    start_time = time.time()

    def average_scalar(value: float) -> float:
        if not is_distributed or not dist.is_initialized():
            return float(value)
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / float(world_size)).item())

    def all_reduce_gradients() -> None:
        if not is_distributed or not dist.is_initialized():
            return
        for param in model.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(float(world_size))

    def sync_stop(local_stop: bool) -> bool:
        if not is_distributed or not dist.is_initialized():
            return bool(local_stop)
        tensor = torch.tensor(1 if local_stop else 0, device=device, dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(int(tensor.item()) > 0)

    for epoch_idx in range(max(1, int(math.ceil(args.num_epochs)))):
        if is_distributed:
            sampler = DistributedSampler(
                train_examples,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            sampler.set_epoch(epoch_idx)
            order = list(iter(sampler))
        else:
            generator = torch.Generator()
            generator.manual_seed(args.seed + epoch_idx)
            order = torch.randperm(len(train_examples), generator=generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        accum_ce_sum = 0.0
        microbatch_size = max(1, int(args.per_device_train_batch_size))
        for batch_start in range(0, len(order), microbatch_size):
            batch_indices = order[batch_start : batch_start + microbatch_size]
            batch_examples = [train_examples[ex_idx] for ex_idx in batch_indices]
            ce_full = latent_batched_completion_ce_loss(
                model,
                batch_examples,
                device,
                num_cot_tokens=args.num_cot_tokens,
                latent_mode=latent_mode,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            loss = ce_full / max(1, int(args.gradient_accumulation_steps))
            loss.backward()
            accum_ce_sum += float(ce_full.detach().item())
            accum_count += 1
            if accum_count >= int(args.gradient_accumulation_steps):
                all_reduce_gradients()
                if float(args.max_grad_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                step += 1
                mean_ce = accum_ce_sum / max(1, int(args.gradient_accumulation_steps))
                accum_ce_sum = 0.0
                if step % int(args.logging_steps) == 0:
                    loss_value = average_scalar(mean_ce)
                    if is_main_process:
                        print(f"[latent sft train step {step:05d}] loss={loss_value:.4f}", flush=True)
                        if wb_run is not None:
                            wandb.log({"train/loss": loss_value, "step": step})
                if step % int(args.eval_steps) == 0:
                    if is_distributed and dist.is_initialized():
                        dist.barrier()
                    should_stop_eval = False
                    if is_main_process:
                        ev = run_eval(args, eval_rows, model, tokenizer, device)
                        if wb_run is not None:
                            wandb.log({f"eval/{k}": float(v) for k, v in ev.items()} | {"step": step})
                        if (
                            args.eval_exact_set_match_stop > 0.0
                            and float(ev["exact_set_match_rate"]) >= args.eval_exact_set_match_stop
                        ):
                            save_checkpoint(model, tokenizer, args.output_dir, step, extra_save_fn=extra_save_fn)
                            should_stop_eval = True
                        if (
                            not should_stop_eval
                            and step >= int(args.min_steps_before_stop)
                            and args.eval_value_precision_stop > 0.0
                            and args.eval_value_recall_stop > 0.0
                            and float(ev["value_precision"]) >= args.eval_value_precision_stop
                            and float(ev["value_recall"]) >= args.eval_value_recall_stop
                        ):
                            save_checkpoint(model, tokenizer, args.output_dir, step, extra_save_fn=extra_save_fn)
                            should_stop_eval = True
                        if (
                            not should_stop_eval
                            and args.eval_solve_rate_stop > 0.0
                            and step >= int(args.min_steps_before_stop)
                            and float(ev["solve_rate"]) >= args.eval_solve_rate_stop
                        ):
                            save_checkpoint(model, tokenizer, args.output_dir, step, extra_save_fn=extra_save_fn)
                            should_stop_eval = True
                    should_stop_eval = sync_stop(should_stop_eval)
                    if is_distributed and dist.is_initialized():
                        dist.barrier()
                    if should_stop_eval:
                        if is_main_process and wb_run is not None:
                            wb_run.finish()
                        if is_distributed and dist.is_initialized():
                            dist.destroy_process_group()
                        return
                if step % int(args.save_steps) == 0:
                    if is_distributed and dist.is_initialized():
                        dist.barrier()
                    if is_main_process:
                        save_checkpoint(model, tokenizer, args.output_dir, step, extra_save_fn=extra_save_fn)
                    if is_distributed and dist.is_initialized():
                        dist.barrier()
                reached_limit = step >= total_steps
                exceeded_wall = bool(args.max_wall_clock_seconds) and (
                    time.time() - start_time >= float(args.max_wall_clock_seconds)
                )
                should_stop = sync_stop(reached_limit or exceeded_wall)
                if should_stop:
                    break
        if sync_stop(step >= total_steps):
            break

    if is_distributed and dist.is_initialized():
        dist.barrier()
    if is_main_process:
        save_checkpoint(model, tokenizer, args.output_dir, step, extra_save_fn=extra_save_fn)
        if wb_run is not None:
            wb_run.finish()
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
