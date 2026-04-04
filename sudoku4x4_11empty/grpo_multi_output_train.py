from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sudoku4x4_11empty.shared_cell_policy import build_cell_examples_from_row
from checkpoint_utils import ensure_final_checkpoint_dir, save_model_artifacts
from sudoku4x4_11empty.prompt_builder import build_multi_output_cell_prompt
from sudoku4x4_11empty.rewards import score_prediction_text
from sudoku4x4_11empty.shared_multi_output_policy import make_solved_grid_from_row


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
    num_train_epochs: float
    learning_rate: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    eval_rows: int
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    beta: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    enable_gradient_checkpointing: bool
    use_wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: str
    wandb_mode: str
    wandb_group: str
    wandb_run_id: str
    debug_print_limit: int
    limit_train_rows: int
    limit_train_examples: int
    reward_good_value: float
    penalty_bad_value: float
    penalty_malformed: float
    penalty_empty: float
    penalty_singleton: float
    eval_solve_rate_stop: float
    min_steps_before_stop: int
    max_wall_clock_seconds: int
    max_steps: int
    resume_from_checkpoint: str


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
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def ensure_trl_fsdp_compat() -> None:
    try:
        import torch.distributed.fsdp as fsdp

        if not hasattr(fsdp, "FSDPModule") and hasattr(fsdp, "FullyShardedDataParallel"):
            fsdp.FSDPModule = fsdp.FullyShardedDataParallel
    except Exception:
        pass


def load_trainable_adapter(base_model: torch.nn.Module, adapter_dir: str) -> torch.nn.Module:
    try:
        return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    except Exception:
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        model_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not (os.path.exists(config_path) and os.path.exists(model_path)):
            raise

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        lora = LoraConfig(
            r=int(cfg["r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias=str(cfg.get("bias", "none")),
            task_type=str(cfg.get("task_type", "CAUSAL_LM")),
            target_modules=list(cfg["target_modules"]),
        )
        model = get_peft_model(base_model, lora)
        state = load_safetensors_file(model_path)
        remapped: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            remapped[new_key] = value
        model.load_state_dict(remapped, strict=False)
        return model


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


def build_grpo_records(
    rows: List[Dict[str, Any]],
    *,
    tokenizer: Any,
    stage_i: int,
    total_empties_hint: int,
    max_records: int = 0,
    progress_every_rows: int = 10,
    progress_callback: Any = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
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
            records.append(
                {
                    "prompt": prompt,
                    "grid_json": json.dumps(ex.grid.tolist(), separators=(",", ":")),
                    "solved_json": json.dumps(solved.tolist(), separators=(",", ":")),
                    "target_row": int(ex.target_cell[0]),
                    "target_col": int(ex.target_cell[1]),
                    "stage_i": int(stage_i),
                }
            )
            if int(max_records) > 0 and len(records) >= int(max_records):
                break
        if progress_callback is not None and (
            row_idx == 1 or row_idx == len(rows) or row_idx % max(1, int(progress_every_rows)) == 0
        ):
            progress_callback(row_idx, len(rows), len(records))
        if int(max_records) > 0 and len(records) >= int(max_records):
            break
    return records


def _prepared_data_dir(args: Args) -> str:
    path = os.path.join(PARENT_DIR, "_prepared_data", "sudoku4x4_11empty")
    os.makedirs(path, exist_ok=True)
    return path


def _prepared_grpo_cache_path(args: Args) -> str:
    payload = json.dumps(
        {
            "train_jsonl": os.path.abspath(args.train_jsonl),
            "stage_i": int(args.stage_i),
            "total_empties_hint": int(args.total_empties_hint),
            "limit_train_rows": int(args.limit_train_rows),
            "limit_train_examples": int(args.limit_train_examples),
            "model_name": str(args.model_name),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:20]
    return os.path.join(_prepared_data_dir(args), f"grpo_stage{int(args.stage_i):02d}_{digest}.jsonl")


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
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


def load_or_build_grpo_records(
    args: Args,
    *,
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    rank: int,
    world_size: int,
    progress_callback: Any = None,
) -> List[Dict[str, Any]]:
    cache_path = _prepared_grpo_cache_path(args)
    if os.path.exists(cache_path):
        return _read_jsonl(cache_path)
    if rank == 0:
        print(f"[dataset build][grpo stage {args.stage_i}] building prepared cache: {cache_path}", flush=True)
        records = build_grpo_records(
            rows,
            tokenizer=tokenizer,
            stage_i=args.stage_i,
            total_empties_hint=args.total_empties_hint,
            max_records=int(args.limit_train_examples),
            progress_every_rows=10,
            progress_callback=progress_callback,
        )
        _write_jsonl(cache_path, records)
        return records
    _wait_for_cache(cache_path)
    return _read_jsonl(cache_path)


def make_reward_func(args: Args):
    def reward_func(completions, grid_json, solved_json, target_row, target_col, stage_i, **kwargs):
        rewards: List[float] = []
        for completion, grid_s, solved_s, rr, cc, stage_val in zip(
            completions, grid_json, solved_json, target_row, target_col, stage_i
        ):
            info = score_prediction_text(
                text=str(completion),
                grid=torch.tensor(json.loads(grid_s), dtype=torch.long).numpy(),
                solved=torch.tensor(json.loads(solved_s), dtype=torch.long).numpy(),
                target_cell=(int(rr), int(cc)),
                stage_i=int(stage_val),
                reward_good_value=args.reward_good_value,
                penalty_bad_value=args.penalty_bad_value,
                penalty_malformed=args.penalty_malformed,
                penalty_empty=args.penalty_empty,
                penalty_singleton=args.penalty_singleton,
            )
            rewards.append(float(info["reward"]))
        return rewards

    return reward_func


@torch.no_grad()
def run_eval(
    *,
    args: Args,
    rows: List[Dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, float]:
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
                rr, cc = ex.target_cell
                print(f"[baseline grpo eval debug] target=({rr+1},{cc+1}) output={pred_text!r}", flush=True)
                print(
                    f"[baseline grpo eval debug] target_values={info['target_values']} predicted_values={info['predicted_values']}",
                    flush=True,
                )
                printed += 1
        solve_ok += int(row_all_exact)

    return {
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
        "eval_cells": float(total_cells),
    }


def unwrap_training_model(model: Any) -> Any:
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current


class CustomEvalCallback(TrainerCallback):
    def __init__(
        self,
        args: Args,
        eval_rows: List[Dict[str, Any]],
        tokenizer: Any,
        device: torch.device,
        wb_run: Any,
        is_main_process: bool,
    ):
        self.args = args
        self.eval_rows = eval_rows
        self.tokenizer = tokenizer
        self.device = device
        self.wb_run = wb_run
        self.is_main_process = is_main_process
        self.last_logged_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_main_process:
            return control
        step = int(state.global_step)
        if step <= 0 or step == self.last_logged_step or step % int(self.args.eval_steps) != 0:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        metrics = run_eval(
            args=self.args,
            rows=self.eval_rows,
            model=unwrap_training_model(model),
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self.last_logged_step = step
        print(
            f"[baseline grpo custom eval step {step}] parse={metrics['parse_rate']:.3f} "
            f"avg_set_size={metrics['avg_predicted_set_size']:.3f} "
            f"good={metrics['avg_num_i_consistent_values']:.3f} "
            f"bad={metrics['avg_num_non_i_consistent_values']:.3f}",
            flush=True,
        )
        if self.args.use_wandb and self.wb_run is not None:
            payload = {f"custom_eval/{k}": float(v) for k, v in metrics.items()}
            payload["custom_eval/global_step"] = float(step)
            wandb.log(payload)
        if (
            int(step) >= int(self.args.min_steps_before_stop)
            and float(self.args.eval_solve_rate_stop) > 0.0
            and float(metrics["solve_rate"]) >= float(self.args.eval_solve_rate_stop)
        ):
            print(
                f"[baseline grpo custom eval step {step}] early stop: "
                f"solve_rate={metrics['solve_rate']:.3f} >= {float(self.args.eval_solve_rate_stop):.3f}",
                flush=True,
            )
            control.should_training_stop = True
        return control


class FinalCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, tokenizer: Any, is_main_process: bool):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.is_main_process = is_main_process

    def _save(self, model: Any) -> None:
        if self.is_main_process:
            save_model_artifacts(unwrap_training_model(model), self.tokenizer, ensure_final_checkpoint_dir(self.output_dir))

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            self._save(model)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            self._save(model)
        return control


class WallClockStopCallback(TrainerCallback):
    def __init__(self, max_wall_clock_seconds: int):
        self.max_wall_clock_seconds = int(max_wall_clock_seconds)
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_wall_clock_seconds > 0 and (time.time() - self.start_time) >= float(self.max_wall_clock_seconds):
            control.should_training_stop = True
        return control


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/home/ubuntu/curriculum_cot/.hf_cache")
    p.add_argument("--init_adapter_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--stage_i", type=int, default=1)
    p.add_argument("--total_empties_hint", type=int, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=0.5)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--eval_rows", type=int, default=20)
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=24)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--enable_gradient_checkpointing", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_project", type=str, default="sudoku-multi-output-grpo")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_run_id", type=str, default="")
    p.add_argument("--debug_print_limit", type=int, default=3)
    p.add_argument("--limit_train_rows", type=int, default=0)
    p.add_argument("--limit_train_examples", type=int, default=0)
    p.add_argument("--reward_good_value", type=float, default=1.0)
    p.add_argument("--penalty_bad_value", type=float, default=1.75)
    p.add_argument("--penalty_malformed", type=float, default=4.0)
    p.add_argument("--penalty_empty", type=float, default=0.5)
    p.add_argument("--penalty_singleton", type=float, default=1.5)
    p.add_argument("--eval_solve_rate_stop", type=float, default=0.0)
    p.add_argument("--min_steps_before_stop", type=int, default=0)
    p.add_argument("--max_wall_clock_seconds", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--resume_from_checkpoint", type=str, default="")
    ns = p.parse_args()
    return Args(**vars(ns))


def main() -> None:
    args = parse_args()
    preset_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main_process = rank == 0

    if preset_visible_devices:
        if is_main_process:
            print(f"Respecting pre-set CUDA_VISIBLE_DEVICES={preset_visible_devices}", flush=True)
    elif int(args.gpu_id) >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))

    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)
    ensure_final_checkpoint_dir(args.output_dir)
    cache_dir = configure_hf_cache(args.cache_dir)
    configure_wandb_dirs(args.output_dir)
    if is_main_process:
        print(f"Using Hugging Face cache dir: {cache_dir}", flush=True)

    wb_run = None
    if is_main_process and args.use_wandb and wandb is not None:
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name or None,
            "mode": args.wandb_mode,
            "group": args.wandb_group or None,
            "id": args.wandb_run_id or None,
        }
        if str(args.wandb_entity).strip():
            init_kwargs["entity"] = args.wandb_entity
        wb_run = wandb.init(**init_kwargs)
        print(f"W&B run id: {wb_run.id}", flush=True)
        print(f"W&B run URL: {wb_run.url}", flush=True)
        wandb.log({"prep/rows_done": 0.0, "prep/records_built": 0.0, "prep/cache_hit": 0.0})

    rows = load_jsonl_rows(args.train_jsonl, limit_rows=args.limit_train_rows)
    eval_rows = rows[: max(1, int(args.eval_rows))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process:
        print(f"Using device: {device}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        torch_dtype=pick_dtype(),
        low_cpu_mem_usage=True,
    )
    model = load_trainable_adapter(base, args.init_adapter_dir)
    if is_main_process:
        print(f"Loaded init adapter: {args.init_adapter_dir}", flush=True)
    if args.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.enable_gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if world_size <= 1:
        model.to(device)
    model.train()

    def on_prep_progress(rows_done: int, total_rows: int, records_built: int) -> None:
        if is_main_process:
            print(
                f"[dataset build][grpo stage {args.stage_i}] rows={rows_done}/{total_rows} records={records_built}",
                flush=True,
            )
            if wb_run is not None:
                wandb.log({"prep/rows_done": float(rows_done), "prep/records_built": float(records_built)})

    train_records = load_or_build_grpo_records(
        args,
        rows=rows,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        progress_callback=on_prep_progress,
    )
    if is_main_process and int(args.limit_train_examples) > 0:
        print(f"Limiting GRPO train records to {len(train_records)} examples", flush=True)
    if is_main_process and wb_run is not None:
        wandb.log(
            {
                "prep/cache_hit": float(os.path.exists(_prepared_grpo_cache_path(args))),
                "prep/records_final": float(len(train_records)),
            }
        )

    train_dataset = Dataset.from_list(train_records)
    reward_func = make_reward_func(args)

    if int(args.limit_train_rows) > 0 and int(args.max_steps) <= 0:
        args.max_steps = 1

    ensure_trl_fsdp_compat()
    from trl import GRPOConfig, GRPOTrainer

    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "bf16": (pick_dtype() == torch.bfloat16),
        "report_to": ["wandb"] if args.use_wandb and is_main_process else [],
        "remove_unused_columns": False,
        "max_steps": int(args.max_steps),
    }
    grpo_config_params = inspect.signature(GRPOConfig.__init__).parameters
    unsupported_keys = sorted(key for key in config_kwargs if key not in grpo_config_params)
    for key in unsupported_keys:
        config_kwargs.pop(key, None)
    if is_main_process and unsupported_keys:
        print(f"Skipping unsupported GRPOConfig args: {', '.join(unsupported_keys)}", flush=True)
    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=config,
        train_dataset=train_dataset,
        eval_dataset=train_dataset.select(range(min(len(train_dataset), max(1, int(args.eval_rows))))),
    )
    trainer.add_callback(CustomEvalCallback(args, eval_rows, tokenizer, device, wb_run, is_main_process))
    trainer.add_callback(FinalCheckpointCallback(args.output_dir, tokenizer, is_main_process))
    trainer.add_callback(WallClockStopCallback(args.max_wall_clock_seconds))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    final_model = unwrap_training_model(trainer.model)
    if is_main_process:
        eval_metrics = run_eval(args=args, rows=eval_rows, model=final_model, tokenizer=tokenizer, device=device)
        print(
            f"[baseline grpo final eval] parse={eval_metrics['parse_rate']:.3f} "
            f"canonical={eval_metrics['strict_canonical_rate']:.3f} "
            f"exact={eval_metrics['exact_set_match_rate']:.3f} precision={eval_metrics['value_precision']:.3f} "
            f"recall={eval_metrics['value_recall']:.3f} solve={eval_metrics['solve_rate']:.3f}",
            flush=True,
        )
        if wb_run is not None:
            wandb.log({f"custom_eval/{k}": float(v) for k, v in eval_metrics.items()})
        trainer.save_model(args.output_dir)
        save_model_artifacts(final_model, tokenizer, ensure_final_checkpoint_dir(args.output_dir))
        if wb_run is not None:
            wb_run.finish()


if __name__ == "__main__":
    main()
