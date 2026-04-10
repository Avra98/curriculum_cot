from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from aligned_cell_policy.shared_cell_policy import CellExample
from formatting_icon import is_consistent_pair


def all_remaining_empties_have_legal_value(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int).reshape(9, 9)
    for r in range(9):
        for c in range(9):
            if int(g[r, c]) != 0:
                continue
            cell = r * 9 + c
            has_legal = any(is_consistent_pair(g, cell=cell, value=v, t=3, n=9) for v in range(1, 10))
            if not has_legal:
                return False
    return True


@dataclass(frozen=True)
class ParsedValues:
    values: tuple[int, ...]
    parse_ok: bool
    strict_canonical: bool


def all_digit_values() -> List[int]:
    return list(range(1, 10))


def make_solved_grid_from_row(row: Dict[str, Any]) -> np.ndarray:
    grid = parse_grid_from_tuple_prompt(str(row["prompt"]))
    solved = np.asarray(grid, dtype=int).copy()
    triples = row.get("metadata", {}).get("target_triples_1based", [])
    for rr, cc, value in triples:
        solved[int(rr) - 1, int(cc) - 1] = int(value)
    return solved


def _grid_state_key(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(grid, dtype=int).reshape(-1))


def _legal_values_for_cell(state: tuple[int, ...], cell: int) -> tuple[int, ...]:
    rr, cc = divmod(int(cell), 9)
    if int(state[cell]) != 0:
        return tuple()
    g = np.asarray(state, dtype=int).reshape(9, 9)
    return tuple(
        int(value)
        for value in all_digit_values()
        if is_consistent_pair(g, cell=int(cell), value=int(value), t=3, n=9)
    )


@lru_cache(maxsize=200000)
def _stage_i_consistent_values_for_grid(state: tuple[int, ...], stage_i: int) -> tuple[tuple[int, ...], ...]:
    stage_i = max(1, int(stage_i))
    out: List[tuple[int, ...]] = [tuple() for _ in range(81)]

    for cell in range(81):
        legal_values = _legal_values_for_cell(state, cell)
        if not legal_values:
            continue
        if stage_i <= 1:
            out[cell] = legal_values
            continue

        consistent_values: List[int] = []
        for value in legal_values:
            child = list(state)
            child[cell] = int(value)
            child_state = tuple(child)
            child_stage_values = _stage_i_consistent_values_for_grid(child_state, stage_i - 1)
            if all(int(child_state[idx]) != 0 or len(child_stage_values[idx]) > 0 for idx in range(81)):
                consistent_values.append(int(value))
        out[cell] = tuple(consistent_values)

    return tuple(out)


def stage_i_consistent_values(
    grid: np.ndarray,
    *,
    target_cell: tuple[int, int],
    stage_i: int,
) -> List[int]:
    g = np.asarray(grid, dtype=int).reshape(9, 9)
    rr, cc = int(target_cell[0]), int(target_cell[1])
    if int(g[rr, cc]) != 0:
        return []
    cell = rr * 9 + cc
    stage_values = _stage_i_consistent_values_for_grid(_grid_state_key(g), int(stage_i))
    return [int(value) for value in stage_values[cell]]


def canonicalize_values(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError("Boolean values are not allowed.")
        vv = int(value)
        if vv < 1 or vv > 9:
            raise ValueError(f"Value must be in [1,9], got {vv}.")
        if vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out


def values_json_text(values: Iterable[int]) -> str:
    return json.dumps({"values": canonicalize_values(values)}, separators=(",", ":"))


def parse_values_json(text: str) -> ParsedValues:
    raw = str(text).strip()
    if not raw:
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    try:
        obj = json.loads(raw)
    except Exception:
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    if not isinstance(obj, dict):
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    if set(obj.keys()) != {"values"}:
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    values_obj = obj.get("values")
    if not isinstance(values_obj, list):
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    try:
        values = canonicalize_values(values_obj)
    except Exception:
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    if len(values) != len(values_obj):
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    canonical = values_json_text(values)
    return ParsedValues(values=tuple(values), parse_ok=True, strict_canonical=(canonical == raw))


def compute_set_precision_recall(pred_values: Sequence[int], target_values: Sequence[int]) -> tuple[float, float]:
    pred = set(int(v) for v in pred_values)
    target = set(int(v) for v in target_values)
    precision = 0.0 if not pred else float(len(pred & target) / max(1, len(pred)))
    recall = 1.0 if not target else float(len(pred & target) / max(1, len(target)))
    return precision, recall


def completion_ce_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    all_ids = tokenizer(prompt_text + completion_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    labels = all_ids.clone()
    labels[:, : int(prompt_ids.shape[1])] = -100
    out = model(input_ids=all_ids, labels=labels)
    return out.loss


def batched_completion_ce_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_texts: Sequence[str],
    completion_texts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    if len(prompt_texts) != len(completion_texts):
        raise ValueError("prompt_texts and completion_texts must have the same length")
    if not prompt_texts:
        raise ValueError("batched_completion_ce_loss requires at least one example")

    full_texts = [str(p) + str(c) for p, c in zip(prompt_texts, completion_texts, strict=True)]
    batch = tokenizer(full_texts, return_tensors="pt", add_special_tokens=False, padding=True)
    prompt_batch = tokenizer(list(prompt_texts), return_tensors="pt", add_special_tokens=False, padding=True)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_attention = prompt_batch["attention_mask"]
    prompt_lengths = prompt_attention.sum(dim=1).tolist()

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for row_idx, prompt_len in enumerate(prompt_lengths):
        labels[row_idx, : int(prompt_len)] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return out.loss


def completion_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    all_ids = tokenizer(prompt_text + completion_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    labels = all_ids.clone()
    labels[:, : int(prompt_ids.shape[1])] = -100
    out = model(input_ids=all_ids, labels=labels)
    num_completion_tokens = int((labels != -100).sum().item())
    return -out.loss * max(1, num_completion_tokens)


def enumerate_value_permutations(
    values: Sequence[int],
    *,
    max_permutations: int,
    rng: Optional[random.Random] = None,
) -> List[tuple[int, ...]]:
    uniq = tuple(canonicalize_values(values))
    if len(uniq) <= 1:
        return [uniq]
    total = math.factorial(len(uniq))
    if total <= max(1, int(max_permutations)):
        return [tuple(p) for p in itertools.permutations(uniq)]

    rr = rng or random.Random(0)
    perms = set()
    base = list(uniq)
    max_needed = max(1, int(max_permutations))
    while len(perms) < max_needed:
        shuffled = list(base)
        rr.shuffle(shuffled)
        perms.add(tuple(shuffled))
    return list(perms)


def permutation_invariant_json_ce_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    values: Sequence[int],
    device: torch.device,
    *,
    max_permutations: int,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    permutations = enumerate_value_permutations(values, max_permutations=max_permutations, rng=rng)
    logps = [
        completion_logprob(model, tokenizer, prompt_text, values_json_text(perm), device) for perm in permutations
    ]
    stacked = torch.stack(logps, dim=0)
    return -(torch.logsumexp(stacked, dim=0) - math.log(float(len(permutations))))


def build_supervised_completion(
    ex: CellExample,
    *,
    stage_i: int,
    rng: Optional[random.Random] = None,
    randomize_order: bool = False,
) -> str:
    values = stage_i_consistent_values(ex.grid, target_cell=ex.target_cell, stage_i=stage_i)
    if randomize_order and len(values) > 1:
        shuffled = list(values)
        (rng or random).shuffle(shuffled)
        values = shuffled
    return values_json_text(values)


def summarize_values(values: Iterable[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


_TUPLE_PROMPT_RE = re.compile(r"\((\d+),(\d+),(\d+)\)")


def parse_grid_from_tuple_prompt(prompt_text: str) -> np.ndarray:
    triples = _TUPLE_PROMPT_RE.findall(str(prompt_text))
    if len(triples) < 81:
        raise ValueError("Could not recover 81 (row,col,value) tuples from prompt.")
    grid = np.zeros((9, 9), dtype=int)
    for rr, cc, vv in triples[:81]:
        r = int(rr) - 1
        c = int(cc) - 1
        grid[r, c] = int(vv)
    return grid
