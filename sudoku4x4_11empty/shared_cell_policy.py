from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


_INT_RE = re.compile(r"-?\d+")
_TUPLE_PROMPT_RE = re.compile(r"\((\d+),(\d+),(\d+)\)")


@dataclass(frozen=True)
class CellExample:
    grid: np.ndarray
    target_cell: tuple[int, int]
    target_value: int
    turn_idx: int
    total_turns: int


def parse_n_value_prediction(text: str, n: int) -> Tuple[List[int] | None, bool]:
    raw = str(text or '').strip()
    if not raw:
        return None, False

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get('values'), list):
            values = [int(v) for v in parsed['values']]
            if len(values) == int(n):
                return values, True
        if isinstance(parsed, list):
            values = [int(v) for v in parsed]
            if len(values) == int(n):
                return values, True
    except Exception:
        pass

    values = [int(match.group(0)) for match in _INT_RE.finditer(raw)]
    if len(values) == int(n):
        return values, True
    return None, False


def parse_grid_from_tuple_prompt(prompt_text: str) -> np.ndarray:
    triples = _TUPLE_PROMPT_RE.findall(str(prompt_text))
    if len(triples) < 16:
        raise ValueError('Could not recover 16 (row,col,value) tuples from prompt.')
    grid = np.zeros((4, 4), dtype=int)
    for rr, cc, vv in triples[:16]:
        grid[int(rr) - 1, int(cc) - 1] = int(vv)
    return grid


def build_cell_examples_from_row(row: Dict[str, Any]) -> List[CellExample]:
    prompt = str(row['prompt'])
    grid = parse_grid_from_tuple_prompt(prompt)
    metadata = dict(row.get('metadata', {}))
    empty_locs = metadata.get('empty_locs_1based')
    target_triples = metadata.get('target_triples_1based')

    if not empty_locs or not target_triples:
        completion = str(row.get('completion', ''))
        parsed, _ = parse_n_value_prediction(completion, int(metadata.get('empties', 0) or 0))
        if parsed is None:
            raise ValueError('Row is missing metadata and completion could not be parsed.')
        empty_locs = [(r + 1, c + 1) for r, c in np.argwhere(grid == 0).tolist()]
        target_triples = [(int(r), int(c), int(v)) for (r, c), v in zip(empty_locs, parsed)]

    total_turns = len(target_triples)
    out: List[CellExample] = []
    for idx, triple in enumerate(target_triples, start=1):
        rr, cc, value = int(triple[0]) - 1, int(triple[1]) - 1, int(triple[2])
        out.append(
            CellExample(
                grid=np.asarray(grid, dtype=int).copy(),
                target_cell=(rr, cc),
                target_value=value,
                turn_idx=idx,
                total_turns=total_turns,
            )
        )
    return out
