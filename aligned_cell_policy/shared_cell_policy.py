from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from format_utils_icon import parse_n_value_prediction


@dataclass(frozen=True)
class CellExample:
    grid: np.ndarray
    target_cell: tuple[int, int]
    target_value: int
    turn_idx: int
    total_turns: int


_TUPLE_PROMPT_RE = re.compile(r"\((\d+),(\d+),(\d+)\)")


def parse_grid_from_tuple_prompt(prompt_text: str) -> np.ndarray:
    triples = _TUPLE_PROMPT_RE.findall(str(prompt_text))
    if len(triples) < 81:
        raise ValueError("Could not recover 81 (row,col,value) tuples from prompt.")
    grid = np.zeros((9, 9), dtype=int)
    for rr, cc, vv in triples[:81]:
        grid[int(rr) - 1, int(cc) - 1] = int(vv)
    return grid


def build_cell_examples_from_row(row: Dict[str, Any]) -> List[CellExample]:
    prompt = str(row["prompt"])
    grid = parse_grid_from_tuple_prompt(prompt)
    metadata = dict(row.get("metadata", {}))
    empty_locs = metadata.get("empty_locs_1based")
    target_triples = metadata.get("target_triples_1based")

    if not empty_locs or not target_triples:
        completion = str(row.get("completion", ""))
        parsed, _ = parse_n_value_prediction(completion, int(metadata.get("empties", 0) or 0))
        if parsed is None:
            raise ValueError("Row is missing metadata and completion could not be parsed.")
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
