from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from types import MethodType
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from transformers.modeling_outputs import CausalLMOutput

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from aligned_cell_policy.shared_cell_policy import build_cell_examples_from_row
from checkpoint_utils import ensure_final_checkpoint_dir, save_model_artifacts
from mixed_curriculum_cot.runtime_mixed_curriculum import build_two_stage_mixed_rows, training_stage_i_for_row
from multi_output_cell_policy.prompt_builder import build_multi_output_cell_prompt
from multi_output_cell_policy.rewards import score_prediction_text
from multi_output_cell_policy.shared_multi_output_policy import make_solved_grid_from_row


try:
    import wandb
except Exception:
    wandb = None


PROJECTOR_HIDDEN = 4096


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
    enable_gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_wandb: bool
    wandb_entity: str
    wandb_project: str
    wandb_run_name: str
    wandb_mode: str
    wandb_group: str
    wandb_run_id: str
    debug_print_limit: int
    limit_train_rows: int
    mixed_stage1_ratio: float
    mixed_stage2_ratio: float
    reward_good_value: float
    penalty_bad_value: float
    penalty_malformed: float
    penalty_empty: float
    penalty_singleton: float
    max_wall_clock_seconds: int
    max_steps: int
    resume_from_checkpoint: str
    eval_value_precision_stop: float
    eval_value_recall_stop: float
    eval_solve_rate_stop: float
    min_steps_before_stop: int


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
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            if int(getattr(props, "major", 0)) >= 8:
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


def load_trainable_adapter(
    base_model: torch.nn.Module,
    adapter_dir: str,
    *,
    lora_r: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
) -> torch.nn.Module:
    if not str(adapter_dir).strip():
        lora = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        return get_peft_model(base_model, lora)
    try:
        return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    except Exception:
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        model_path = os.path.join(adapter_dir, "adapter_model.safetensors")
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


def build_grpo_dataset(
    rows: List[Dict[str, Any]],
    *,
    tokenizer: Any,
    stage_i: int,
    total_empties_hint: int,
    progress_every_rows: int = 10,
    progress_callback: Any = None,
) -> Dataset:
    records: List[Dict[str, Any]] = []
    total_rows = len(rows)
    for row_idx, row in enumerate(rows, start=1):
        solved = make_solved_grid_from_row(row)
        row_stage_i = training_stage_i_for_row(row, stage_i)
        for ex in build_cell_examples_from_row(row):
            prompt = build_multi_output_cell_prompt(
                ex.grid,
                target_cell=ex.target_cell,
                stage_i=row_stage_i,
                tokenizer=tokenizer,
                turn_idx=ex.turn_idx,
                total_turns=ex.total_turns,
                prev_output_flag=None,
                total_empties_hint=total_empties_hint,
            )
            records.append(
                {
                    "prompt": prompt,
                    "grid_json": json.dumps(ex.grid.tolist()),
                    "solved_json": json.dumps(solved.tolist()),
                    "target_row": int(ex.target_cell[0]),
                    "target_col": int(ex.target_cell[1]),
                    "stage_i": int(row_stage_i),
                }
            )
        if progress_callback is not None and (
            row_idx == total_rows or row_idx % max(1, int(progress_every_rows)) == 0
        ):
            progress_callback(row_idx=row_idx, total_rows=total_rows, record_count=len(records))
    return Dataset.from_list(records)


def _prepared_data_dir() -> str:
    path = os.path.join(PARENT_DIR, "_prepared_data", "latent_multi_output_cell_policy")
    os.makedirs(path, exist_ok=True)
    return path


def _prepared_grpo_cache_path(args: Args) -> str:
    payload = {
        "kind": "grpo",
        "train_jsonl": os.path.abspath(args.train_jsonl),
        "train_jsonl_stage1": os.path.abspath(args.train_jsonl_stage1 or args.train_jsonl),
        "train_jsonl_stage2": os.path.abspath(args.train_jsonl_stage2 or args.train_jsonl),
        "stage_i": int(args.stage_i),
        "total_empties_hint": int(args.total_empties_hint),
        "limit_train_rows": int(args.limit_train_rows),
        "mixed_stage1_ratio": float(args.mixed_stage1_ratio),
        "mixed_stage2_ratio": float(args.mixed_stage2_ratio),
        "model_name": str(args.model_name),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return os.path.join(_prepared_data_dir(), f"grpo_stage{int(args.stage_i):02d}_{digest}.jsonl")


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    os.replace(tmp_path, path)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _wait_for_cache(path: str, timeout_seconds: int = 6 * 60 * 60) -> None:
    start = time.time()
    while not os.path.exists(path):
        if (time.time() - start) > float(timeout_seconds):
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
        if rank == 0:
            print(f"[dataset build][grpo stage {args.stage_i}] loading prepared cache: {cache_path}", flush=True)
        return _read_jsonl(cache_path)
    if rank == 0:
        print(f"[dataset build][grpo stage {args.stage_i}] building prepared cache: {cache_path}", flush=True)
        dataset = build_grpo_dataset(
            rows,
            tokenizer=tokenizer,
            stage_i=args.stage_i,
            total_empties_hint=args.total_empties_hint,
            progress_every_rows=10,
            progress_callback=progress_callback,
        )
        records = [dataset[int(i)] for i in range(len(dataset))]
        _write_jsonl(cache_path, records)
    elif world_size > 1:
        _wait_for_cache(cache_path)
    return _read_jsonl(cache_path)


def load_training_rows(args: Args) -> List[Dict[str, Any]]:
    stage1_path = str(args.train_jsonl_stage1 or "").strip()
    stage2_path = str(args.train_jsonl_stage2 or "").strip()
    use_mixed = bool(stage1_path or stage2_path)
    if not use_mixed:
        return load_jsonl_rows(args.train_jsonl, limit_rows=args.limit_train_rows)

    stage1_rows = load_jsonl_rows(stage1_path or args.train_jsonl, limit_rows=0)
    stage2_rows = load_jsonl_rows(stage2_path or args.train_jsonl, limit_rows=0)
    return build_two_stage_mixed_rows(
        stage1_rows,
        stage2_rows,
        stage1_ratio=float(args.mixed_stage1_ratio),
        stage2_ratio=float(args.mixed_stage2_ratio),
        seed=int(args.seed),
        target_stage=int(args.stage_i),
        total_rows=int(args.limit_train_rows),
    )


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


def unwrap_backbone(model: nn.Module) -> nn.Module:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def unwrap_training_model(model: Any) -> Any:
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current


def get_input_embeddings_module(model: nn.Module) -> nn.Module:
    return unwrap_backbone(model).get_input_embeddings()


def get_output_embeddings_module(model: nn.Module) -> nn.Module:
    base = unwrap_backbone(model)
    return base.get_output_embeddings() or base.lm_head


def get_last_hidden_state(model_output: Any) -> torch.Tensor:
    hidden = getattr(model_output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    return model_output.hidden_states[-1]


def run_backbone_from_embeds(backbone: nn.Module, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
    base = unwrap_backbone(backbone)
    inner = getattr(base, "model", base)
    return inner(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )


def extend_attention_mask(mask: torch.Tensor, extra_tokens: int) -> torch.Tensor:
    extra = torch.ones(mask.shape[0], int(extra_tokens), dtype=mask.dtype, device=mask.device)
    return torch.cat([mask, extra], dim=1)


def attach_residual_projector_modules(model: nn.Module, hidden_size: int, projector_hidden: int = PROJECTOR_HIDDEN) -> None:
    if hasattr(model, "latent_projector_in") and hasattr(model, "latent_projector_out") and hasattr(
        model, "special_thought_embed"
    ):
        return
    projector_hidden = int(projector_hidden)
    model.special_thought_embed = nn.Parameter(torch.randn(hidden_size) * 0.02)
    model.latent_mix_logit = nn.Parameter(torch.tensor(-8.0))
    model.latent_projector_in = nn.Linear(hidden_size, projector_hidden, bias=True)
    model.latent_projector_out = nn.Linear(projector_hidden, hidden_size, bias=True)
    nn.init.normal_(model.special_thought_embed, std=0.02)
    nn.init.xavier_uniform_(model.latent_projector_in.weight)
    nn.init.zeros_(model.latent_projector_in.bias)
    nn.init.xavier_uniform_(model.latent_projector_out.weight)
    nn.init.zeros_(model.latent_projector_out.bias)


def maybe_load_projector_state(model: nn.Module, path_or_dir: str) -> bool:
    state_path = str(path_or_dir)
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, "latent_cot_state.pt")
    if not os.path.exists(state_path):
        return False
    state = torch.load(state_path, map_location="cpu")
    with torch.no_grad():
        for name in [
            "special_thought_embed",
            "latent_mix_logit",
            "latent_projector_in_weight",
            "latent_projector_in_bias",
            "latent_projector_out_weight",
            "latent_projector_out_bias",
        ]:
            if name not in state:
                continue
            if name == "special_thought_embed":
                model.special_thought_embed.copy_(state[name].to(model.special_thought_embed))
            elif name == "latent_mix_logit":
                model.latent_mix_logit.copy_(state[name].to(model.latent_mix_logit))
            elif name == "latent_projector_in_weight":
                model.latent_projector_in.weight.copy_(state[name].to(model.latent_projector_in.weight))
            elif name == "latent_projector_in_bias":
                model.latent_projector_in.bias.copy_(state[name].to(model.latent_projector_in.bias))
            elif name == "latent_projector_out_weight":
                model.latent_projector_out.weight.copy_(state[name].to(model.latent_projector_out.weight))
            elif name == "latent_projector_out_bias":
                model.latent_projector_out.bias.copy_(state[name].to(model.latent_projector_out.bias))
    return True


def infer_projector_hidden_from_state(path_or_dir: str) -> int | None:
    state_path = str(path_or_dir)
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, "latent_cot_state.pt")
    if not os.path.exists(state_path):
        return None
    state = torch.load(state_path, map_location="cpu")
    weight = state.get("latent_projector_in_weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return None


def save_latent_projector_state(model: nn.Module, output_dir: str) -> None:
    state = {
        "special_thought_embed": model.special_thought_embed.detach().cpu(),
        "latent_mix_logit": model.latent_mix_logit.detach().cpu(),
        "latent_projector_in_weight": model.latent_projector_in.weight.detach().cpu(),
        "latent_projector_in_bias": model.latent_projector_in.bias.detach().cpu(),
        "latent_projector_out_weight": model.latent_projector_out.weight.detach().cpu(),
        "latent_projector_out_bias": model.latent_projector_out.bias.detach().cpu(),
    }
    torch.save(state, os.path.join(output_dir, "latent_cot_state.pt"))


def project_hidden(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden.dtype
    hidden = hidden.to(torch.float32)
    hidden = F.linear(
        hidden,
        model.latent_projector_in.weight.to(dtype=torch.float32),
        None if model.latent_projector_in.bias is None else model.latent_projector_in.bias.to(dtype=torch.float32),
    )
    hidden = F.gelu(hidden)
    hidden = F.linear(
        hidden,
        model.latent_projector_out.weight.to(dtype=torch.float32),
        None if model.latent_projector_out.bias is None else model.latent_projector_out.bias.to(dtype=torch.float32),
    )
    hidden = torch.nan_to_num(hidden, nan=0.0, posinf=50.0, neginf=-50.0)
    hidden = hidden.clamp(min=-50.0, max=50.0)
    return hidden.to(input_dtype)


def _sanitize_logits(logits: torch.Tensor, *, output_dtype: torch.dtype) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=50.0, neginf=-50.0)
    logits = logits.clamp(min=-50.0, max=50.0)
    return logits.to(dtype=output_dtype)


def _should_fallback_to_base(model: nn.Module, latent_logits: torch.Tensor) -> torch.Tensor:
    scores = torch.nan_to_num(latent_logits.float(), nan=0.0, posinf=50.0, neginf=-50.0)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    max_prob = probs.max(dim=-1).values
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    if int(scores.shape[-1]) > 1:
        top2 = torch.topk(scores, k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
    else:
        margin = torch.full_like(max_prob, float("inf"))
    fallback = (~torch.isfinite(scores)).any(dim=-1)
    fallback |= max_prob > float(getattr(model, "_latent_fallback_max_prob", 0.995))
    fallback |= entropy < float(getattr(model, "_latent_fallback_entropy_min", 0.02))
    fallback |= margin > float(getattr(model, "_latent_fallback_margin_max", 25.0))
    return fallback


def build_latent_hidden(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_cot_tokens: int):
    backbone = unwrap_backbone(model)
    inner_backbone = getattr(backbone, "model", backbone)
    input_embeds = get_input_embeddings_module(model)(input_ids)
    base_out = inner_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )
    base_hidden = get_last_hidden_state(base_out)[:, -1, :]
    if num_cot_tokens <= 0:
        return base_hidden, base_hidden

    cur_embeds = input_embeds
    cur_mask = attention_mask
    latent_token = None
    special = model.special_thought_embed.to(device=input_embeds.device, dtype=input_embeds.dtype).view(1, 1, -1)
    for _ in range(int(num_cot_tokens)):
        next_embed = special.expand(cur_embeds.shape[0], 1, -1) if latent_token is None else latent_token
        cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
        cur_mask = extend_attention_mask(cur_mask, 1)
        out = run_backbone_from_embeds(backbone, cur_embeds, cur_mask)
        latent_token = get_last_hidden_state(out)[:, -1:, :]
    latent_hidden = latent_token[:, 0, :]
    return base_hidden, latent_hidden


def residual_next_token_logits_from_ids(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_cot_tokens: int
) -> torch.Tensor:
    base_hidden, latent_hidden = build_latent_hidden(model, input_ids, attention_mask, num_cot_tokens)
    projected_delta = project_hidden(model, latent_hidden - base_hidden).float()
    mix = torch.sigmoid(model.latent_mix_logit.float()).to(projected_delta.device)
    projected_delta = projected_delta * float(getattr(model, "_latent_delta_scale", 1.0)) * mix
    base_hidden_fp32 = base_hidden.float()
    base_norm = base_hidden_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    delta_norm = projected_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    max_ratio = max(0.0, float(getattr(model, "_latent_delta_max_ratio", 0.5)))
    max_delta_norm = base_norm * max_ratio
    projected_delta = projected_delta * torch.clamp(max_delta_norm / delta_norm, max=1.0)
    final_hidden = torch.nan_to_num(base_hidden_fp32 + projected_delta, nan=0.0, posinf=50.0, neginf=-50.0)
    output_embeddings = get_output_embeddings_module(model)
    output_dtype = getattr(getattr(output_embeddings, "weight", None), "dtype", final_hidden.dtype)
    latent_logits = _sanitize_logits(output_embeddings(final_hidden.to(dtype=output_dtype)), output_dtype=output_dtype)
    fallback_mask = _should_fallback_to_base(model, latent_logits)
    if bool(fallback_mask.any()):
        warn_count = int(getattr(model, "_latent_fallback_warn_count", 0))
        if warn_count < 5:
            print(f"[latent grpo] falling back to base logits for {int(fallback_mask.sum().item())} rows")
        model._latent_fallback_warn_count = warn_count + 1
        fallback_hidden = base_hidden_fp32[fallback_mask].to(dtype=output_dtype)
        fallback_logits = _sanitize_logits(output_embeddings(fallback_hidden), output_dtype=output_dtype)
        latent_logits = latent_logits.clone()
        latent_logits[fallback_mask] = fallback_logits
    return latent_logits


def _apply_repetition_penalty(logits: torch.Tensor, tokens: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or tokens.numel() == 0:
        return logits
    adjusted = logits.clone()
    unique_tokens = torch.unique(tokens, sorted=False)
    seen_logits = adjusted.index_select(dim=-1, index=unique_tokens)
    seen_logits = torch.where(seen_logits < 0, seen_logits * penalty, seen_logits / penalty)
    adjusted.index_copy_(dim=-1, index=unique_tokens, source=seen_logits)
    return adjusted


def _sample_from_latent_logits(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    temperature = max(float(temperature), 1e-5)
    scores = logits / temperature

    if int(top_k) > 0 and int(top_k) < scores.shape[-1]:
        topk_values, _ = torch.topk(scores, k=int(top_k), dim=-1)
        cutoff = topk_values[:, -1:].expand_as(scores)
        scores = torch.where(scores < cutoff, torch.full_like(scores, float("-inf")), scores)

    if 0.0 < float(top_p) < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumulative_probs > float(top_p)
        nucleus_mask[:, 0] = False
        sorted_scores = sorted_scores.masked_fill(nucleus_mask, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def sample_latent_completion(
    model: nn.Module,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    num_cot_tokens: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    generated = prompt_ids
    mask = attention_mask
    eos = tokenizer.eos_token_id
    for _ in range(max(1, int(max_new_tokens))):
        logits = residual_next_token_logits_from_ids(model, generated, mask, num_cot_tokens)
        logits = _apply_repetition_penalty(logits, generated, float(repetition_penalty))
        next_id = _sample_from_latent_logits(
            logits.float(),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )
        generated = torch.cat([generated, next_id], dim=1)
        mask = extend_attention_mask(mask, 1)
        if eos is not None and bool((next_id == int(eos)).all()):
            break
    return generated[:, prompt_ids.shape[1] :]


def install_latent_grpo_model_interface(
    model: nn.Module,
    tokenizer: Any,
    *,
    num_cot_tokens: int,
    latent_delta_scale: float = 1.0,
    latent_delta_max_ratio: float = 0.5,
) -> nn.Module:
    if getattr(model, "_latent_grpo_interface_installed", False):
        model._latent_grpo_num_cot_tokens = int(num_cot_tokens)
        model._latent_grpo_tokenizer = tokenizer
        model._latent_delta_scale = float(latent_delta_scale)
        model._latent_delta_max_ratio = float(latent_delta_max_ratio)
        return model

    model._latent_grpo_interface_installed = True
    model._latent_grpo_num_cot_tokens = int(num_cot_tokens)
    model._latent_grpo_tokenizer = tokenizer
    model._latent_delta_scale = float(latent_delta_scale)
    model._latent_delta_max_ratio = float(latent_delta_max_ratio)
    model._latent_original_forward = model.forward
    model._latent_original_generate = model.generate

    def latent_forward(
        self,
        input_ids=None,
        attention_mask=None,
        logits_to_keep=None,
        use_cache=None,
        **kwargs,
    ):
        if input_ids is None or attention_mask is None or logits_to_keep is None:
            return self._latent_original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                **kwargs,
            )

        seq_len = int(input_ids.shape[1])
        keep = max(1, min(int(logits_to_keep), seq_len))
        start = max(1, seq_len - keep)
        logits = []
        for prefix_len in range(start, seq_len + 1):
            prefix_ids = input_ids[:, :prefix_len]
            prefix_mask = attention_mask[:, :prefix_len]
            step_logits = residual_next_token_logits_from_ids(
                self,
                prefix_ids,
                prefix_mask,
                int(self._latent_grpo_num_cot_tokens),
            )
            logits.append(step_logits.unsqueeze(1))
        return CausalLMOutput(logits=torch.cat(logits, dim=1))

    @torch.no_grad()
    def latent_generate(self, input_ids=None, attention_mask=None, generation_config=None, **kwargs):
        if input_ids is None or attention_mask is None:
            return self._latent_original_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs,
            )

        tokenizer_local = self._latent_grpo_tokenizer
        max_new_tokens = int(
            getattr(generation_config, "max_new_tokens", None) or kwargs.get("max_new_tokens") or 16
        )
        do_sample = bool(getattr(generation_config, "do_sample", True))
        temperature = float(getattr(generation_config, "temperature", 1.0))
        top_p = float(getattr(generation_config, "top_p", 1.0))
        top_k = int(getattr(generation_config, "top_k", 0))
        repetition_penalty = float(getattr(generation_config, "repetition_penalty", 1.0))
        pad_token_id = getattr(generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer_local, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer_local, "eos_token_id", 0)

        rows = []
        for row_ids, row_mask in zip(input_ids, attention_mask, strict=True):
            row_prompt = row_ids.unsqueeze(0)
            row_attn = row_mask.unsqueeze(0)
            completion = sample_latent_completion(
                self,
                tokenizer_local,
                row_prompt,
                row_attn,
                num_cot_tokens=int(self._latent_grpo_num_cot_tokens),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            rows.append(torch.cat([row_prompt, completion], dim=1).squeeze(0))

        max_len = max(int(row.shape[0]) for row in rows)
        padded = []
        for row in rows:
            if int(row.shape[0]) < max_len:
                pad = torch.full((max_len - int(row.shape[0]),), int(pad_token_id), device=row.device, dtype=row.dtype)
                row = torch.cat([row, pad], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)

    model.forward = MethodType(latent_forward, model)
    model.generate = MethodType(latent_generate, model)
    return model


@torch.no_grad()
def run_eval(
    *,
    args: Args,
    rows: List[Dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    eval_stage_i: int | None = None,
    log_prefix: str = "latent grpo eval",
) -> Dict[str, float]:
    model.eval()
    stage_i = int(eval_stage_i if eval_stage_i is not None else args.stage_i)
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
                stage_i=stage_i,
                tokenizer=tokenizer,
                turn_idx=ex.turn_idx,
                total_turns=ex.total_turns,
                prev_output_flag=None,
                total_empties_hint=args.total_empties_hint,
            )
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            completion_ids = sample_latent_completion(
                model,
                tokenizer,
                prompt_ids,
                attn,
                num_cot_tokens=args.num_cot_tokens,
                max_new_tokens=args.max_completion_length,
                do_sample=False,
            )
            pred_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()
            info = score_prediction_text(
                text=pred_text,
                grid=ex.grid,
                solved=solved,
                target_cell=ex.target_cell,
                stage_i=stage_i,
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
                print(f"[latent grpo eval debug] target=({rr+1},{cc+1}) output={pred_text!r}")
                print(f"[latent grpo eval debug] target_values={info['target_values']} predicted_values={info['predicted_values']}")
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
        "eval_cells": float(total_cells),
    }
    print(
        f"[{log_prefix}] parse={out['parse_rate']:.3f} "
        f"exact={out['exact_set_match_rate']:.3f} precision={out['value_precision']:.3f} "
        f"recall={out['value_recall']:.3f} solve={out['solve_rate']:.3f} "
        f"avg_set_size={out['avg_predicted_set_size']:.3f} "
        f"good={out['avg_num_i_consistent_values']:.3f} "
        f"bad={out['avg_num_non_i_consistent_values']:.3f}"
    )
    return out


def run_dual_eval(
    *,
    args: Args,
    eval_rows_stage1: List[Dict[str, Any]],
    eval_rows_stage2: List[Dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, float]:
    metrics_stage1 = run_eval(
        args=args,
        rows=eval_rows_stage1,
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_stage_i=1,
        log_prefix="latent grpo eval stage1",
    )
    metrics_stage2 = run_eval(
        args=args,
        rows=eval_rows_stage2,
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_stage_i=max(1, int(args.stage_i)),
        log_prefix=f"latent grpo eval stage{int(args.stage_i)}",
    )
    out = {f"stage1/{k}": float(v) for k, v in metrics_stage1.items()}
    out.update({f"stage{int(args.stage_i)}/{k}": float(v) for k, v in metrics_stage2.items()})
    return out


class ResidualProjectorEvalCallback(TrainerCallback):
    def __init__(
        self,
        args: Args,
        eval_rows_stage1: List[Dict[str, Any]],
        eval_rows_stage2: List[Dict[str, Any]],
        tokenizer: Any,
        device: torch.device,
        wb_run: Any,
        is_main_process: bool,
    ):
        self.args = args
        self.eval_rows_stage1 = eval_rows_stage1
        self.eval_rows_stage2 = eval_rows_stage2
        self.tokenizer = tokenizer
        self.device = device
        self.wb_run = wb_run
        self.is_main_process = is_main_process
        self.last_logged_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        eval_every = int(self.args.eval_steps)
        if step <= 0 or step % eval_every != 0:
            return control

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_dist = world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized()
        stop_tensor = torch.zeros(1, dtype=torch.int32, device=self.device)

        if self.is_main_process:
            if step != self.last_logged_step:
                model = kwargs.get("model")
                if model is not None:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    metrics = run_eval(
                        args=self.args,
                        rows=self.eval_rows_stage2,
                        model=unwrap_training_model(model),
                        tokenizer=self.tokenizer,
                        device=self.device,
                        eval_stage_i=max(1, int(self.args.stage_i)),
                        log_prefix=f"latent grpo callback eval stage{int(self.args.stage_i)}",
                    )
                    if self.eval_rows_stage1:
                        stage1_metrics = run_eval(
                            args=self.args,
                            rows=self.eval_rows_stage1,
                            model=unwrap_training_model(model),
                            tokenizer=self.tokenizer,
                            device=self.device,
                            eval_stage_i=1,
                            log_prefix="latent grpo callback eval stage1",
                        )
                        metrics = {f"stage1/{k}": float(v) for k, v in stage1_metrics.items()} | {
                            f"stage{int(self.args.stage_i)}/{k}": float(v) for k, v in metrics.items()
                        }
                    else:
                        metrics = {f"stage{int(self.args.stage_i)}/{k}": float(v) for k, v in metrics.items()}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.last_logged_step = step
                    si = int(self.args.stage_i)
                    pfx = f"stage{si}/"
                    print(
                        f"[latent grpo custom eval step {step}] "
                        f"stage1_exact={metrics.get('stage1/exact_set_match_rate', float('nan')):.3f} "
                        f"stage{si}_exact={metrics[f'{pfx}exact_set_match_rate']:.3f} "
                        f"stage{si}_prec={metrics[f'{pfx}value_precision']:.3f} "
                        f"stage{si}_rec={metrics[f'{pfx}value_recall']:.3f} "
                        f"stage{si}_solve={metrics[f'{pfx}solve_rate']:.3f}",
                        flush=True,
                    )
                    if self.args.use_wandb and self.wb_run is not None:
                        payload = {f"custom_eval/{k}": float(v) for k, v in metrics.items()}
                        payload["custom_eval/global_step"] = float(step)
                        wandb.log(payload)

                    if step >= int(self.args.min_steps_before_stop):
                        vp = float(metrics[f"{pfx}value_precision"])
                        vr = float(metrics[f"{pfx}value_recall"])
                        sr = float(metrics[f"{pfx}solve_rate"])
                        if (
                            float(self.args.eval_value_precision_stop) > 0.0
                            and float(self.args.eval_value_recall_stop) > 0.0
                            and vp >= float(self.args.eval_value_precision_stop)
                            and vr >= float(self.args.eval_value_recall_stop)
                        ):
                            print(
                                f"[latent grpo custom eval step {step}] stopping early: "
                                f"value_precision={vp:.3f} >= {float(self.args.eval_value_precision_stop):.3f} "
                                f"and value_recall={vr:.3f} >= {float(self.args.eval_value_recall_stop):.3f}",
                                flush=True,
                            )
                            stop_tensor[0] = 1
                        if (
                            int(stop_tensor.item()) == 0
                            and float(self.args.eval_solve_rate_stop) > 0.0
                            and sr >= float(self.args.eval_solve_rate_stop)
                        ):
                            print(
                                f"[latent grpo custom eval step {step}] stopping early: "
                                f"solve_rate={sr:.3f} >= {float(self.args.eval_solve_rate_stop):.3f}",
                                flush=True,
                            )
                            stop_tensor[0] = 1

        if use_dist:
            torch.distributed.broadcast(stop_tensor, src=0)

        if int(stop_tensor.item()) != 0:
            control.should_training_stop = True
        return control


class SaveLatentStateCallback(TrainerCallback):
    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process

    def on_save(self, args, state, control, **kwargs):
        if not self.is_main_process:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        step_dir = os.path.join(args.output_dir, f"checkpoint-{int(state.global_step)}")
        if os.path.isdir(step_dir):
            save_latent_projector_state(unwrap_training_model(model), step_dir)
        return control


class FinalCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, tokenizer: Any, is_main_process: bool):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.is_main_process = is_main_process

    def _save(self, model: Any) -> None:
        save_model_artifacts(
            unwrap_training_model(model),
            self.tokenizer,
            ensure_final_checkpoint_dir(self.output_dir),
            extra_save_fn=save_latent_projector_state,
        )

    def on_save(self, args, state, control, **kwargs):
        if not self.is_main_process:
            return control
        model = kwargs.get("model")
        if model is not None:
            self._save(model)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.is_main_process:
            return control
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
    p.add_argument(
        "--train_jsonl",
        type=str,
        default="/egr/research-slim/ghoshavr/curriculum-CoT/sudoku/llm_policy_icon/data/sudoku_t3_20empty_value_qwen_text.jsonl",
    )
    p.add_argument("--train_jsonl_stage1", type=str, default="")
    p.add_argument("--train_jsonl_stage2", type=str, default="")
    p.add_argument(
        "--eval_jsonl",
        type=str,
        default="",
        help="If set, first eval_rows lines are used for both stage1/stage2 eval (held-out). Else slice train files.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/egr/research-slim/ghoshavr/.hf_cache")
    p.add_argument(
        "--init_adapter_dir",
        type=str,
        default="",
        help="Peft adapter checkpoint dir, or empty string for fresh LoRA on the base model (random init).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--stage_i", type=int, default=1)
    p.add_argument("--num_cot_tokens", type=int, default=1)
    p.add_argument("--total_empties_hint", type=int, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=0.5)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--eval_rows", type=int, default=20)
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=24)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--enable_gradient_checkpointing", action="store_true")
    p.add_argument("--lora_r", type=int, default=192)
    p.add_argument("--lora_alpha", type=int, default=384)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_project", type=str, default="sudoku-latent-multi-output-grpo-residual-projector")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_run_id", type=str, default="")
    p.add_argument("--debug_print_limit", type=int, default=3)
    p.add_argument("--limit_train_rows", type=int, default=0)
    p.add_argument("--mixed_stage1_ratio", type=float, default=0.0)
    p.add_argument("--mixed_stage2_ratio", type=float, default=1.0)
    p.add_argument("--reward_good_value", type=float, default=1.0)
    p.add_argument("--penalty_bad_value", type=float, default=1.75)
    p.add_argument("--penalty_malformed", type=float, default=4.0)
    p.add_argument("--penalty_empty", type=float, default=0.5)
    p.add_argument("--penalty_singleton", type=float, default=1.5)
    p.add_argument("--max_wall_clock_seconds", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--resume_from_checkpoint", type=str, default="")
    p.add_argument(
        "--eval_value_precision_stop",
        type=float,
        default=0.0,
        help="If >0 and --eval_value_recall_stop>0, stop when both reached on current stage_i eval (with min_steps_before_stop).",
    )
    p.add_argument("--eval_value_recall_stop", type=float, default=0.0)
    p.add_argument(
        "--eval_solve_rate_stop",
        type=float,
        default=0.0,
        help="If >0, stop when stage_i solve_rate reaches this threshold (after min_steps_before_stop).",
    )
    p.add_argument("--min_steps_before_stop", type=int, default=0)
    return Args(**vars(p.parse_args()))


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main_process = rank == 0
    preset_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if preset_visible_devices:
        print(f"Respecting pre-set CUDA_VISIBLE_DEVICES={preset_visible_devices}")
    elif int(args.gpu_id) >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if world_size > 1 else max(0, int(args.gpu_id)))
    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)
    ensure_final_checkpoint_dir(args.output_dir)
    cache_dir = configure_hf_cache(args.cache_dir)
    configure_wandb_dirs(args.output_dir)
    print(f"Using Hugging Face cache dir: {cache_dir}")

    wb_run = None
    if is_main_process and args.use_wandb and wandb is not None:
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            mode=args.wandb_mode,
            group=args.wandb_group or None,
            id=args.wandb_run_id or None,
        )
        print(f"W&B run id: {wb_run.id}", flush=True)
        print(f"W&B run URL: {wb_run.url}", flush=True)
        wandb.log({"prep/rows_done": 0.0, "prep/records_built": 0.0, "prep/cache_hit": 0.0})

    rows = load_training_rows(args)
    eval_src = str(getattr(args, "eval_jsonl", "") or "").strip()
    if eval_src:
        _eval_slice = load_jsonl_rows(eval_src, limit_rows=0)[: max(1, int(args.eval_rows))]
        eval_rows_stage1 = _eval_slice
        eval_rows_stage2 = _eval_slice
    else:
        eval_rows_stage1 = load_jsonl_rows(args.train_jsonl_stage1 or args.train_jsonl, limit_rows=0)[
            : max(1, int(args.eval_rows))
        ]
        eval_rows_stage2 = load_jsonl_rows(args.train_jsonl_stage2 or args.train_jsonl, limit_rows=0)[
            : max(1, int(args.eval_rows))
        ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        torch_dtype=pick_dtype(),
        low_cpu_mem_usage=True,
    )
    model = load_trainable_adapter(
        base,
        args.init_adapter_dir,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    init_ad = str(args.init_adapter_dir).strip()
    if init_ad:
        print(f"Loaded init adapter: {init_ad}", flush=True)
        projector_hidden = infer_projector_hidden_from_state(init_ad) or PROJECTOR_HIDDEN
    else:
        print(
            "init_adapter_dir empty: fresh LoRA on base (weights random); matches --lora_r/--lora_alpha/--lora_dropout.",
            flush=True,
        )
        projector_hidden = PROJECTOR_HIDDEN
    attach_residual_projector_modules(
        model,
        hidden_size=int(unwrap_backbone(model).config.hidden_size),
        projector_hidden=projector_hidden,
    )
    if init_ad:
        if maybe_load_projector_state(model, init_ad):
            print(f"Loaded latent_cot_state.pt from: {init_ad}", flush=True)
        else:
            print(f"No latent_cot_state.pt under {init_ad}; residual projector kept at random init.", flush=True)
    else:
        print("Residual projector + special_thought_embed: random init (latent structure attached).", flush=True)
    if world_size <= 1:
        model.to(device)
    model.train()

    def on_prep_progress(*, row_idx: int, total_rows: int, record_count: int) -> None:
        if not is_main_process:
            return
        print(
            f"[dataset build][grpo stage {args.stage_i}] rows={row_idx}/{total_rows} records={record_count}",
            flush=True,
        )
        if wb_run is not None:
            wandb.log(
                {
                    "prep/rows_done": float(row_idx),
                    "prep/rows_total": float(total_rows),
                    "prep/records_built": float(record_count),
                }
            )

    train_records = load_or_build_grpo_records(
        args,
        rows=rows,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        progress_callback=on_prep_progress,
    )
    train_dataset = Dataset.from_list(train_records)
    if is_main_process and wb_run is not None:
        wandb.log(
            {
                "prep/cache_hit": float(os.path.exists(_prepared_grpo_cache_path(args))),
                "prep/records_final": float(len(train_records)),
            }
        )
    reward_func = make_reward_func(args)

    ensure_trl_fsdp_compat()
    from trl import GRPOConfig, GRPOTrainer

    if int(args.limit_train_rows) > 0 and int(args.max_steps) <= 0:
        args.max_steps = 1
    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_strategy": "no",
        "do_eval": False,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "beta": args.beta,
        "gradient_checkpointing": bool(args.enable_gradient_checkpointing),
        "bf16": (pick_dtype() == torch.bfloat16),
        "report_to": (["wandb"] if args.use_wandb and is_main_process else []),
        "remove_unused_columns": False,
    }
    if int(args.max_steps) > 0:
        config_kwargs["max_steps"] = int(args.max_steps)
    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=config,
        train_dataset=train_dataset,
    )
    trainer.add_callback(
        ResidualProjectorEvalCallback(
            args,
            eval_rows_stage1,
            eval_rows_stage2,
            tokenizer,
            device,
            wb_run,
            is_main_process,
        )
    )
    trainer.add_callback(SaveLatentStateCallback(is_main_process))
    trainer.add_callback(FinalCheckpointCallback(args.output_dir, tokenizer, is_main_process))
    trainer.add_callback(WallClockStopCallback(args.max_wall_clock_seconds))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()
    final_model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
    final_model = unwrap_training_model(final_model)
    if is_main_process:
        eval_metrics = run_dual_eval(
            args=args,
            eval_rows_stage1=eval_rows_stage1,
            eval_rows_stage2=eval_rows_stage2,
            model=final_model,
            tokenizer=tokenizer,
            device=device,
        )
        si = int(args.stage_i)
        print(
            f"[latent grpo final eval] "
            f"stage1_exact={eval_metrics.get('stage1/exact_set_match_rate', float('nan')):.3f} "
            f"stage{si}_exact={eval_metrics[f'stage{si}/exact_set_match_rate']:.3f} "
            f"stage{si}_prec={eval_metrics[f'stage{si}/value_precision']:.3f} "
            f"stage{si}_rec={eval_metrics[f'stage{si}/value_recall']:.3f} "
            f"stage{si}_solve={eval_metrics[f'stage{si}/solve_rate']:.3f}"
        )
        trainer.save_model(args.output_dir)
        save_latent_projector_state(final_model, args.output_dir)
        save_model_artifacts(
            final_model,
            tokenizer,
            ensure_final_checkpoint_dir(args.output_dir),
            extra_save_fn=save_latent_projector_state,
        )
        if wb_run is not None:
            wandb.log({f"final_eval/{k}": float(v) for k, v in eval_metrics.items()})
            wb_run.finish()


if __name__ == "__main__":
    main()
