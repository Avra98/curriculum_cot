from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from format_utils_icon import grid_to_text


def make_multi_output_system_prompt(*, stage_i: int, total_empties_hint: int = 10) -> str:
    i = max(1, int(stage_i))
    return (
        "You are a Sudoku value policy.\n"
        f"This setup uses puzzles with about {int(total_empties_hint)} empty cells.\n"
        "You will be given one target empty cell.\n"
        'Return ONLY one JSON object of the form {"values":[...]}.\n'
        'The JSON object must contain exactly one key named "values".\n'
        'The "values" field must be a JSON array of unique integers in [1,9].\n'
        "You may return as many candidate values as you want, including one, several, or many values.\n"
        "Choose the number of returned values yourself based on which values seem i-consistent.\n"
        "The order of the values does not matter.\n"
        "Do not output any explanation, markdown, punctuation outside JSON, or extra text.\n"
        f"Current stage objective: i={i} consistency.\n"
    )


def build_multi_output_cell_prompt(
    grid_9x9: np.ndarray,
    *,
    target_cell: tuple[int, int],
    stage_i: int,
    tokenizer: Any,
    turn_idx: int,
    total_turns: int,
    prev_output_flag: Optional[str] = None,
    total_empties_hint: int = 10,
) -> str:
    g = np.asarray(grid_9x9, dtype=int).reshape(9, 9)
    empties = int(np.sum(g == 0))
    rr, cc = int(target_cell[0]), int(target_cell[1])
    system_msg = make_multi_output_system_prompt(
        stage_i=stage_i, total_empties_hint=total_empties_hint
    ).strip()
    empty_locs = [(int(r) + 1, int(c) + 1) for r, c in np.argwhere(g == 0).tolist()]
    empty_locs_text = ", ".join(f"({r},{c})" for r, c in empty_locs)
    user_msg = (
        "Sudoku grid (0 means empty):\n"
        + grid_to_text(g)
        + "\n"
        + f"Empty cells in row-major order ({empties} total): {empty_locs_text}\n\n"
        + f"Target cell to fill now: ({rr + 1},{cc + 1}).\n"
        + f"Turn: {int(turn_idx)}/{int(total_turns)}.\n"
        + 'Return only JSON with candidate values for this target cell: {"values":[...]}'
    )
    if prev_output_flag is not None:
        user_msg += f"\nPrevious output_flag (context only): {prev_output_flag}"

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return system_msg + "\n\n" + user_msg + "\n"
