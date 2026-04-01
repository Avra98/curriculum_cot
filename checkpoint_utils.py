from __future__ import annotations

import os
import shutil
from typing import Any, Callable

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file as save_safetensors_file

FINAL_CHECKPOINT_DIRNAME = "final_checkpoint"
_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)


def ensure_final_checkpoint_dir(output_dir: str) -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = os.path.abspath(output_dir)
    try:
        rel_output_dir = os.path.relpath(output_dir_abs, repo_root)
    except Exception:
        rel_output_dir = os.path.basename(output_dir_abs.rstrip(os.sep))
    rel_parts = [part for part in rel_output_dir.split(os.sep) if part not in ("", ".")]
    if rel_parts and rel_parts[0] == FINAL_CHECKPOINT_DIRNAME:
        rel_parts = rel_parts[1:]
    if rel_parts and rel_parts[0] == "checkpoints":
        rel_parts = rel_parts[1:]
    if not rel_parts:
        rel_parts = [os.path.basename(output_dir_abs.rstrip(os.sep)) or "run"]
    final_dir = os.path.join(repo_root, FINAL_CHECKPOINT_DIRNAME, *rel_parts)
    os.makedirs(final_dir, exist_ok=True)
    return final_dir


def final_checkpoint_root(*parts: str) -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(repo_root, FINAL_CHECKPOINT_DIRNAME, *parts)
    os.makedirs(root, exist_ok=True)
    return root


def normalize_to_final_checkpoint_root(path: str, *default_parts: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return final_checkpoint_root(*default_parts)
    abs_path = os.path.abspath(raw)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.relpath(abs_path, repo_root)
    rel_parts = [part for part in rel_path.split(os.sep) if part not in ("", ".")]
    if rel_parts[:1] == [FINAL_CHECKPOINT_DIRNAME]:
        return abs_path
    if rel_parts[:1] == ["checkpoints"]:
        rel_parts = rel_parts[1:]
        return final_checkpoint_root(*rel_parts)
    return abs_path


def _has_saved_weights(target_dir: str) -> bool:
    return any(os.path.exists(os.path.join(target_dir, name)) for name in _WEIGHT_FILENAMES)


def _fallback_save_adapter_weights(model: Any, target_dir: str) -> None:
    if _has_saved_weights(target_dir):
        return
    state = get_peft_model_state_dict(model)
    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    if cpu_state:
        save_safetensors_file(cpu_state, os.path.join(target_dir, "adapter_model.safetensors"))


def save_model_artifacts(
    model: Any,
    tokenizer: Any,
    target_dir: str,
    *,
    extra_save_fn: Callable[[Any, str], None] | None = None,
) -> str:
    os.makedirs(target_dir, exist_ok=True)
    model.save_pretrained(target_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(target_dir)
    _fallback_save_adapter_weights(model, target_dir)
    if extra_save_fn is not None:
        extra_save_fn(model, target_dir)
    return target_dir


def _replace_dir_contents(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    src_dir_abs = os.path.abspath(src_dir)
    for name in os.listdir(dst_dir):
        path = os.path.join(dst_dir, name)
        if os.path.abspath(path) == src_dir_abs:
            continue
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.isdir(src_path) and not os.path.islink(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def save_checkpoint_and_update_final(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    checkpoint_name: str,
    *,
    extra_save_fn: Callable[[Any, str], None] | None = None,
) -> str:
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    save_model_artifacts(model, tokenizer, checkpoint_dir, extra_save_fn=extra_save_fn)
    _replace_dir_contents(checkpoint_dir, ensure_final_checkpoint_dir(output_dir))
    return checkpoint_dir
