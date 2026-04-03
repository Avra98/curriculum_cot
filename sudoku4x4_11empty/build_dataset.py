from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


GRID_SIZE = 4
BOX_SIZE = 2
ALL_VALUES = (1, 2, 3, 4)

BASE_GRID = np.array(
    [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ],
    dtype=int,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_output = root / "data" / "sudoku4x4_11empty_value_qwen_text.jsonl"
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default=str(default_output))
    p.add_argument("--num_puzzles", type=int, default=20000)
    p.add_argument("--empties", type=int, default=11)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def permute_groups(rng: random.Random, values: Sequence[int], group_size: int) -> List[int]:
    groups = [list(values[idx : idx + group_size]) for idx in range(0, len(values), group_size)]
    rng.shuffle(groups)
    out: List[int] = []
    for group in groups:
        rng.shuffle(group)
        out.extend(group)
    return out


def random_solved_grid(rng: random.Random) -> np.ndarray:
    grid = np.asarray(BASE_GRID, dtype=int).copy()

    digits = list(ALL_VALUES)
    rng.shuffle(digits)
    digit_map = {src: dst for src, dst in zip(ALL_VALUES, digits, strict=True)}
    grid = np.vectorize(lambda value: digit_map[int(value)], otypes=[int])(grid)

    row_order = permute_groups(rng, list(range(GRID_SIZE)), BOX_SIZE)
    col_order = permute_groups(rng, list(range(GRID_SIZE)), BOX_SIZE)
    grid = grid[row_order, :]
    grid = grid[:, col_order]

    if rng.random() < 0.5:
        grid = grid.T
    return np.asarray(grid, dtype=int)


def row_major_empty_locs(grid: np.ndarray) -> List[Tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(np.asarray(grid, dtype=int) == 0).tolist()]


def make_prompt(grid: np.ndarray) -> str:
    tuples = [
        f"({r + 1},{c + 1},{int(grid[r, c])})"
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
    ]
    return (
        "4x4 Sudoku board encoded as (row,col,value) tuples in row-major order.\n"
        "Value 0 means the cell is empty.\n"
        + " ".join(tuples)
    )


def make_example(solved: np.ndarray, *, empties: int, rng: random.Random) -> dict:
    if empties <= 0 or empties >= GRID_SIZE * GRID_SIZE:
        raise ValueError(f"empties must be between 1 and {GRID_SIZE * GRID_SIZE - 1}")

    cells = list(range(GRID_SIZE * GRID_SIZE))
    rng.shuffle(cells)
    masked_cells = sorted(cells[:empties])

    puzzle = np.asarray(solved, dtype=int).copy()
    for cell in masked_cells:
        rr, cc = divmod(int(cell), GRID_SIZE)
        puzzle[rr, cc] = 0

    empty_locs_1based = [(rr + 1, cc + 1) for rr, cc in row_major_empty_locs(puzzle)]
    target_triples_1based = [
        (rr + 1, cc + 1, int(solved[rr, cc]))
        for rr, cc in row_major_empty_locs(puzzle)
    ]
    completion_values = [int(value) for _, _, value in target_triples_1based]

    return {
        "prompt": make_prompt(puzzle),
        "completion": json.dumps(completion_values, separators=(",", ":")),
        "metadata": {
            "grid_size": GRID_SIZE,
            "box_size": BOX_SIZE,
            "empties": int(empties),
            "empty_locs_1based": empty_locs_1based,
            "target_triples_1based": target_triples_1based,
        },
    }


def generate_examples(num_puzzles: int, *, empties: int, seed: int) -> Iterable[dict]:
    rng = random.Random(int(seed))
    for _ in range(int(num_puzzles)):
        solved = random_solved_grid(rng)
        yield make_example(solved, empties=int(empties), rng=rng)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in generate_examples(args.num_puzzles, empties=args.empties, seed=args.seed):
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(f"Wrote {int(args.num_puzzles)} puzzles to {output_path}")


if __name__ == "__main__":
    main()
