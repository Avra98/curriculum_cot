from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from sudoku4x4_11empty.shared_cell_policy import CellExample, parse_grid_from_tuple_prompt
from formatting_icon import is_consistent_pair


GRID_SIZE = 4
BOX_SIZE = 2
ALL_VALUES = (1, 2, 3, 4)
NUM_CELLS = GRID_SIZE * GRID_SIZE


def all_remaining_empties_have_legal_value(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int).reshape(GRID_SIZE, GRID_SIZE)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if int(g[r, c]) != 0:
                continue
            cell = r * GRID_SIZE + c
            has_legal = any(is_consistent_pair(g, cell=cell, value=v, t=BOX_SIZE, n=GRID_SIZE) for v in ALL_VALUES)
            if not has_legal:
                return False
    return True


@dataclass(frozen=True)
class ParsedValues:
    values: tuple[int, ...]
    parse_ok: bool
    strict_canonical: bool


def all_digit_values() -> List[int]:
    return list(ALL_VALUES)


def make_solved_grid_from_row(row: Dict[str, Any]) -> np.ndarray:
    grid = parse_grid_from_tuple_prompt(str(row['prompt']))
    solved = np.asarray(grid, dtype=int).copy()
    triples = row.get('metadata', {}).get('target_triples_1based', [])
    for rr, cc, value in triples:
        solved[int(rr) - 1, int(cc) - 1] = int(value)
    return solved


def _grid_state_key(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(grid, dtype=int).reshape(-1))


def _legal_values_for_cell(state: tuple[int, ...], cell: int) -> tuple[int, ...]:
    rr, cc = divmod(int(cell), GRID_SIZE)
    if int(state[cell]) != 0:
        return tuple()
    g = np.asarray(state, dtype=int).reshape(GRID_SIZE, GRID_SIZE)
    return tuple(
        int(value)
        for value in all_digit_values()
        if is_consistent_pair(g, cell=int(cell), value=int(value), t=BOX_SIZE, n=GRID_SIZE)
    )


@lru_cache(maxsize=200000)
def _stage_i_consistent_values_for_grid(state: tuple[int, ...], stage_i: int) -> tuple[tuple[int, ...], ...]:
    stage_i = max(1, int(stage_i))
    out: List[tuple[int, ...]] = [tuple() for _ in range(NUM_CELLS)]

    for cell in range(NUM_CELLS):
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
            if all(int(child_state[idx]) != 0 or len(child_stage_values[idx]) > 0 for idx in range(NUM_CELLS)):
                consistent_values.append(int(value))
        out[cell] = tuple(consistent_values)

    return tuple(out)


def stage_i_consistent_values(
    grid: np.ndarray,
    *,
    target_cell: tuple[int, int],
    stage_i: int,
) -> List[int]:
    g = np.asarray(grid, dtype=int).reshape(GRID_SIZE, GRID_SIZE)
    rr, cc = int(target_cell[0]), int(target_cell[1])
    if int(g[rr, cc]) != 0:
        return []
    cell = rr * GRID_SIZE + cc
    stage_values = _stage_i_consistent_values_for_grid(_grid_state_key(g), int(stage_i))
    return [int(value) for value in stage_values[cell]]


def canonicalize_values(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError('Boolean values are not allowed.')
        vv = int(value)
        if vv < 1 or vv > GRID_SIZE:
            raise ValueError(f'Value must be in [1,{GRID_SIZE}], got {vv}.')
        if vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out


def values_json_text(values: Iterable[int]) -> str:
    return json.dumps({'values': canonicalize_values(values)}, separators=(',', ':'))


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
    if set(obj.keys()) != {'values'}:
        return ParsedValues(values=tuple(), parse_ok=False, strict_canonical=False)
    values_obj = obj.get('values')
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
    prompt_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    all_ids = tokenizer(prompt_text + completion_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    labels = all_ids.clone()
    labels[:, : int(prompt_ids.shape[1])] = -100
    out = model(input_ids=all_ids, labels=labels)
    return out.loss


def completion_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    all_ids = tokenizer(prompt_text + completion_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
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
        rr.shuffle(base)
        perms.add(tuple(base))
    return list(perms)


def build_supervised_completion(ex: CellExample, *, stage_i: int) -> str:
    values = stage_i_consistent_values(ex.grid, target_cell=ex.target_cell, stage_i=stage_i)
    return values_json_text(values)
