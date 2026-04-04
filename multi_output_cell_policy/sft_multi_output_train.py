from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
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
    completion_ce_loss,
    make_solved_grid_from_row,
)


try:
    import wandb
except Exception:
    wandb = None


@dataclass
class Args:
    model_name: str
    train_jsonl: str
    output_dir: str
    cache_dir: str
    init_adapter_dir: str
    seed: int
    gpu_id: int
    stage_i: int
    total_empties_hint: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_epochs: float
    learning_rate: float
    weight_decay: float
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
    max_wall_clock_seconds: int
    max_steps: int


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
            examples.append(
                {
                    "prompt_text": prompt,
                    "completion_text": target_text,
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
    printed = 0
    for row in rows:
        solved = make_solved_grid_from_row(row)
        row_all_exact = True
        for ex in build_cell_examples_from_row(row):
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
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=max(1, int(args.max_completion_length)),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred_text = tokenizer.decode(out[0][int(enc["input_ids"].shape[1]) :], skip_special_tokens=True).strip()
            info = score_prediction_text(
                text=pred_text,
                grid=ex.grid,
                solved=solved,
                target_cell=ex.target_cell,
                stage_i=args.stage_i,
                reward_good_value=1.0,
                penalty_bad_value=1.75,
                penalty_malformed=4.0,
                penalty_empty=0.5,
                penalty_singleton=1.5,
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
                rr, cc = ex.target_cell
                print(f"[baseline sft eval debug] target=({rr+1},{cc+1}) output={pred_text!r}", flush=True)
                print(
                    f"[baseline sft eval debug] target_values={info['target_values']} predicted_values={info['predicted_values']}",
                    flush=True,
                )
                printed += 1
        solve_ok += int(row_all_exact)
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
        "solve_rate": float(solve_ok / max(1, len(rows))),
    }
    print(
        f"[baseline sft eval] parse={out['parse_rate']:.3f} canonical={out['strict_canonical_rate']:.3f} "
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
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/home/ubuntu/curriculum-CoT/.hf_cache")
    p.add_argument("--init_adapter_dir", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--stage_i", type=int, default=1)
    p.add_argument("--total_empties_hint", type=int, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
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
    p.add_argument("--wandb_project", type=str, default="sudoku-multi-output-sft")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument("--debug_print_limit", type=int, default=3)
    p.add_argument("--limit_train_rows", type=int, default=0)
    p.add_argument("--eval_exact_set_match_stop", type=float, default=0.0)
    p.add_argument("--max_wall_clock_seconds", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    return Args(**vars(p.parse_args()))


def save_checkpoint(model: torch.nn.Module, tokenizer: Any, output_dir: str, step: int) -> None:
    save_checkpoint_and_update_final(model, tokenizer, output_dir, f"checkpoint-step-{step:05d}")


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

    rows = load_jsonl_rows(args.train_jsonl, limit_rows=args.limit_train_rows)
    eval_rows = rows[: max(1, int(args.eval_rows))]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else f"cuda:{max(0, int(args.gpu_id))}")
    else:
        device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=cache_dir, torch_dtype=pick_dtype(), low_cpu_mem_usage=True
    )
    if str(args.init_adapter_dir).strip():
        model = PeftModel.from_pretrained(model, args.init_adapter_dir, is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    if args.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.enable_gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(device)
    model.train()

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
    if is_main_process and wb_run is not None:
        wandb.log(
            {
                "prep/cache_hit": float(os.path.exists(_prepared_sft_cache_path(args))),
                "prep/examples_final": float(len(train_examples)),
            }
        )

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_examples) * args.num_epochs / max(1, args.gradient_accumulation_steps)))
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
        for ex_idx in order:
            ex = train_examples[ex_idx]
            loss = completion_ce_loss(
                model,
                tokenizer,
                ex["prompt_text"],
                ex["completion_text"],
                device,
            ) / max(1, int(args.gradient_accumulation_steps))
            loss.backward()
            accum_count += 1
            if accum_count >= int(args.gradient_accumulation_steps):
                all_reduce_gradients()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                step += 1
                if step % int(args.logging_steps) == 0:
                    loss_value = average_scalar(float(loss.item()) * args.gradient_accumulation_steps)
                    if is_main_process:
                        print(f"[baseline sft train step {step:05d}] loss={loss_value:.4f}", flush=True)
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
                            save_checkpoint(model, tokenizer, args.output_dir, step)
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
                        save_checkpoint(model, tokenizer, args.output_dir, step)
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
        save_checkpoint(model, tokenizer, args.output_dir, step)
        if wb_run is not None:
            wb_run.finish()
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
