from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from formatting_icon import is_consistent_pair
from multi_output_cell_policy.shared_multi_output_policy import stage_i_consistent_values


GRID_SIZE = 9
BOX_SIZE = 3
ALL_VALUES = tuple(range(1, 10))
DEFAULT_BASE_NAME = "sudoku_t3_30empty_stage3hard"


@dataclass(frozen=True)
class DifficultyProfile:
    stage1_solved: bool
    stage2_solved: bool
    stage3_solved: bool
    stage1_steps: int
    stage2_steps: int
    stage3_steps: int


@dataclass(frozen=True)
class SeedMask:
    mask_cells: tuple[int, ...]
    profile: DifficultyProfile


def parse_args() -> argparse.Namespace:
    root = PARENT_DIR
    default_train = root / "data" / f"{DEFAULT_BASE_NAME}_value_qwen_text.jsonl"
    default_eval = root / "data" / f"{DEFAULT_BASE_NAME}_eval_value_qwen_text.jsonl"
    default_manifest = root / "data" / f"{DEFAULT_BASE_NAME}_manifest.json"
    p = argparse.ArgumentParser()
    p.add_argument("--train_output", type=str, default=str(default_train))
    p.add_argument("--eval_output", type=str, default=str(default_eval))
    p.add_argument("--manifest_output", type=str, default=str(default_manifest))
    p.add_argument("--num_train_puzzles", type=int, default=4000)
    p.add_argument("--num_eval_puzzles", type=int, default=200)
    p.add_argument("--empties", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_attempts", type=int, default=200000)
    p.add_argument("--progress_every", type=int, default=250)
    p.add_argument("--num_seed_masks", type=int, default=8)
    return p.parse_args()


def permute_groups(rng: random.Random, values: Sequence[int], group_size: int) -> List[int]:
    groups = [list(values[idx : idx + group_size]) for idx in range(0, len(values), group_size)]
    rng.shuffle(groups)
    out: List[int] = []
    for group in groups:
        rng.shuffle(group)
        out.extend(group)
    return out


def base_solved_grid() -> np.ndarray:
    return np.asarray(
        [[((rr * BOX_SIZE + rr // BOX_SIZE + cc) % GRID_SIZE) + 1 for cc in range(GRID_SIZE)] for rr in range(GRID_SIZE)],
        dtype=int,
    )


def row_major_empty_locs(grid: np.ndarray) -> List[Tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(np.asarray(grid, dtype=int) == 0).tolist()]


def make_prompt(grid: np.ndarray) -> str:
    tuples = [f"({r + 1},{c + 1},{int(grid[r, c])})" for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    return (
        "9x9 Sudoku board encoded as (row,col,value) tuples in row-major order.\n"
        "Value 0 means the cell is empty.\n"
        + " ".join(tuples)
    )


def legal_values(grid: np.ndarray, row: int, col: int) -> List[int]:
    cell = int(row) * GRID_SIZE + int(col)
    return [int(value) for value in ALL_VALUES if is_consistent_pair(grid, cell=cell, value=int(value), t=3, n=9)]


def count_solutions(grid: np.ndarray, *, limit: int = 2) -> int:
    board = np.asarray(grid, dtype=int).copy()
    solutions = 0

    def backtrack() -> None:
        nonlocal solutions
        if solutions >= int(limit):
            return
        best_cell: Tuple[int, int] | None = None
        best_values: List[int] | None = None
        for rr, cc in row_major_empty_locs(board):
            values = legal_values(board, rr, cc)
            if not values:
                return
            if best_values is None or len(values) < len(best_values):
                best_cell = (rr, cc)
                best_values = values
                if len(best_values) == 1:
                    break
        if best_cell is None:
            solutions += 1
            return
        rr, cc = best_cell
        for value in best_values or []:
            board[rr, cc] = int(value)
            backtrack()
            board[rr, cc] = 0
            if solutions >= int(limit):
                return

    backtrack()
    return int(solutions)


def propagate_stage(grid: np.ndarray, *, stage_i: int) -> Tuple[np.ndarray | None, int]:
    board = np.asarray(grid, dtype=int).copy()
    num_assignments = 0
    while True:
        chosen: Tuple[int, int, int] | None = None
        for rr, cc in row_major_empty_locs(board):
            values = stage_i_consistent_values(board, target_cell=(rr, cc), stage_i=int(stage_i))
            if not values:
                return None, num_assignments
            if len(values) == 1:
                chosen = (rr, cc, int(values[0]))
                break
        if chosen is None:
            return board, num_assignments
        rr, cc, value = chosen
        board[rr, cc] = int(value)
        num_assignments += 1


def build_difficulty_profile(puzzle: np.ndarray, solved: np.ndarray) -> DifficultyProfile | None:
    stage1_board, stage1_steps = propagate_stage(puzzle, stage_i=1)
    if stage1_board is None:
        return None
    stage2_board, stage2_steps = propagate_stage(puzzle, stage_i=2)
    if stage2_board is None:
        return None
    stage3_board, stage3_steps = propagate_stage(puzzle, stage_i=3)
    if stage3_board is None:
        return None
    return DifficultyProfile(
        stage1_solved=bool(np.array_equal(stage1_board, solved)),
        stage2_solved=bool(np.array_equal(stage2_board, solved)),
        stage3_solved=bool(np.array_equal(stage3_board, solved)),
        stage1_steps=int(stage1_steps),
        stage2_steps=int(stage2_steps),
        stage3_steps=int(stage3_steps),
    )


def qualifies(profile: DifficultyProfile) -> bool:
    return (not profile.stage1_solved) and (not profile.stage2_solved) and profile.stage3_solved


def build_puzzle_from_mask(solved: np.ndarray, mask_cells: Sequence[int]) -> np.ndarray:
    puzzle = np.asarray(solved, dtype=int).copy()
    for cell in mask_cells:
        rr, cc = divmod(int(cell), GRID_SIZE)
        puzzle[rr, cc] = 0
    return puzzle


def sample_mask_cells(*, empties: int, rng: random.Random) -> tuple[int, ...]:
    cells = list(range(GRID_SIZE * GRID_SIZE))
    rng.shuffle(cells)
    return tuple(sorted(int(cell) for cell in cells[: int(empties)]))


def greedy_find_seed_mask(
    *,
    empties: int,
    max_attempts: int,
    rng: random.Random,
    progress_every: int,
) -> Tuple[SeedMask | None, Dict[str, int]]:
    solved = base_solved_grid()
    attempts = 0
    restarts = 0
    while attempts < int(max_attempts):
        restarts += 1
        mask: List[int] = []
        remaining = list(range(GRID_SIZE * GRID_SIZE))
        rng.shuffle(remaining)
        current_profile: DifficultyProfile | None = None

        while len(mask) < int(empties) and attempts < int(max_attempts):
            best_cell: int | None = None
            best_profile: DifficultyProfile | None = None
            best_score: Tuple[int, int, int] | None = None
            candidate_cells = list(remaining[: min(len(remaining), 12)])
            if not candidate_cells:
                break

            for cell in candidate_cells:
                attempts += 1
                trial_mask = tuple(sorted(mask + [int(cell)]))
                puzzle = build_puzzle_from_mask(solved, trial_mask)
                profile = build_difficulty_profile(puzzle, solved)
                if profile is None or not profile.stage3_solved:
                    continue
                score = (
                    int(not profile.stage2_solved),
                    int(not profile.stage1_solved),
                    int(profile.stage3_steps - profile.stage2_steps),
                )
                if best_score is None or score > best_score:
                    best_cell = int(cell)
                    best_profile = profile
                    best_score = score

                if attempts == 1 or attempts % max(1, int(progress_every)) == 0:
                    print(
                        f"[search hard 9x9 masks] attempts={attempts} restarts={restarts} current_empties={len(mask)}",
                        flush=True,
                    )

            if best_cell is None or best_profile is None:
                break

            mask.append(int(best_cell))
            mask.sort()
            remaining.remove(int(best_cell))
            current_profile = best_profile

        if len(mask) != int(empties) or current_profile is None:
            continue

        final_mask = tuple(sorted(int(cell) for cell in mask))
        final_puzzle = build_puzzle_from_mask(solved, final_mask)
        final_profile = build_difficulty_profile(final_puzzle, solved)
        if final_profile is None or not qualifies(final_profile):
            continue
        if count_solutions(final_puzzle, limit=2) != 1:
            continue
        return SeedMask(mask_cells=final_mask, profile=final_profile), {
            "attempts": int(attempts),
            "restarts": int(restarts),
        }

    return None, {"attempts": int(attempts), "restarts": int(restarts)}


def random_symmetry(
    rng: random.Random, *, solved: np.ndarray, mask_cells: Sequence[int]
) -> Tuple[np.ndarray, tuple[int, ...]]:
    digits = list(ALL_VALUES)
    rng.shuffle(digits)
    digit_map = {src: dst for src, dst in zip(ALL_VALUES, digits, strict=True)}
    transformed = np.vectorize(lambda value: digit_map[int(value)], otypes=[int])(np.asarray(solved, dtype=int).copy())

    row_order = permute_groups(rng, list(range(GRID_SIZE)), BOX_SIZE)
    col_order = permute_groups(rng, list(range(GRID_SIZE)), BOX_SIZE)
    inverse_row = {old: new for new, old in enumerate(row_order)}
    inverse_col = {old: new for new, old in enumerate(col_order)}

    transformed = transformed[row_order, :]
    transformed = transformed[:, col_order]

    transformed_cells: List[int] = []
    for cell in mask_cells:
        rr, cc = divmod(int(cell), GRID_SIZE)
        new_r = int(inverse_row[int(rr)])
        new_c = int(inverse_col[int(cc)])
        transformed_cells.append(new_r * GRID_SIZE + new_c)

    if rng.random() < 0.5:
        transformed = transformed.T
        transformed_cells = [int(cc) * GRID_SIZE + int(rr) for rr, cc in (divmod(cell, GRID_SIZE) for cell in transformed_cells)]

    return np.asarray(transformed, dtype=int), tuple(sorted(int(cell) for cell in transformed_cells))


def make_example(solved: np.ndarray, mask_cells: Sequence[int], *, empties: int, profile: DifficultyProfile) -> Dict[str, object]:
    puzzle = build_puzzle_from_mask(solved, mask_cells)
    empty_locs_1based = [(rr + 1, cc + 1) for rr, cc in row_major_empty_locs(puzzle)]
    target_triples_1based = [(rr + 1, cc + 1, int(solved[rr, cc])) for rr, cc in row_major_empty_locs(puzzle)]
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
            "required_consistency_stage": 3,
            "difficulty_profile": asdict(profile),
        },
    }


def search_seed_masks(
    *,
    num_seed_masks: int,
    empties: int,
    max_attempts: int,
    seed: int,
    progress_every: int,
) -> Tuple[List[SeedMask], Dict[str, int]]:
    rng = random.Random(int(seed))
    seeds: List[SeedMask] = []
    seen = set()
    total_attempts = 0
    total_restarts = 0

    while len(seeds) < int(num_seed_masks) and total_attempts < int(max_attempts):
        mask_seed, stats = greedy_find_seed_mask(
            empties=int(empties),
            max_attempts=max(1, int(max_attempts) - int(total_attempts)),
            rng=rng,
            progress_every=int(progress_every),
        )
        total_attempts += int(stats.get("attempts", 0))
        total_restarts += int(stats.get("restarts", 0))
        if mask_seed is None:
            break
        if mask_seed.mask_cells in seen:
            continue
        seen.add(mask_seed.mask_cells)
        seeds.append(mask_seed)
        print(
            f"[search hard 9x9 masks] attempts={total_attempts} accepted={len(seeds)}/{num_seed_masks}",
            flush=True,
        )

    stats = {
        "attempts": int(total_attempts),
        "restarts": int(total_restarts),
        "accepted_seed_masks": int(len(seeds)),
    }
    return seeds, stats


def generate_examples(
    *,
    num_examples: int,
    empties: int,
    seed_masks: Sequence[SeedMask],
    seed: int,
) -> List[Dict[str, object]]:
    if not seed_masks:
        raise ValueError("seed_masks must not be empty")
    rng = random.Random(int(seed) + 1)
    solved = base_solved_grid()
    rows: List[Dict[str, object]] = []
    for idx in range(int(num_examples)):
        seed_mask = seed_masks[idx % len(seed_masks)]
        transformed_solved, transformed_mask = random_symmetry(
            rng, solved=solved, mask_cells=seed_mask.mask_cells
        )
        rows.append(
            make_example(
                transformed_solved,
                transformed_mask,
                empties=int(empties),
                profile=seed_mask.profile,
            )
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    total_needed = int(args.num_train_puzzles) + int(args.num_eval_puzzles)
    num_seed_masks = min(max(1, int(args.num_seed_masks)), total_needed)
    seed_masks, search_stats = search_seed_masks(
        num_seed_masks=num_seed_masks,
        empties=int(args.empties),
        max_attempts=int(args.max_attempts),
        seed=int(args.seed),
        progress_every=int(args.progress_every),
    )
    if len(seed_masks) < num_seed_masks:
        raise RuntimeError(
            f"Only found {len(seed_masks)} qualifying seed masks out of requested {num_seed_masks}. "
            f"Try increasing --max_attempts or reducing --num_seed_masks."
        )
    rows = generate_examples(
        num_examples=total_needed,
        empties=int(args.empties),
        seed_masks=seed_masks,
        seed=int(args.seed),
    )

    eval_rows = rows[: int(args.num_eval_puzzles)]
    train_rows = rows[int(args.num_eval_puzzles) :]

    train_output = Path(args.train_output).resolve()
    eval_output = Path(args.eval_output).resolve()
    manifest_output = Path(args.manifest_output).resolve()

    write_jsonl(train_output, train_rows)
    write_jsonl(eval_output, eval_rows)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(
            {
                "train_output": str(train_output),
                "eval_output": str(eval_output),
                "num_train_puzzles": int(len(train_rows)),
                "num_eval_puzzles": int(len(eval_rows)),
                "empties": int(args.empties),
                "seed": int(args.seed),
                "required_consistency_stage": 3,
                "num_seed_masks": int(num_seed_masks),
                "search_stats": search_stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(train_rows)} train puzzles to {train_output}")
    print(f"Wrote {len(eval_rows)} eval puzzles to {eval_output}")
    print(f"Wrote manifest to {manifest_output}")


if __name__ == "__main__":
    main()
