from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_base = root / "data" / "sudoku_t3_30empty_value_qwen_text.jsonl"
    default_output = root / "data" / "mixed_curriculum_cot" / "30empty"
    p = argparse.ArgumentParser()
    p.add_argument("--base_jsonl", type=str, default=str(default_base))
    p.add_argument("--output_dir", type=str, default=str(default_output))
    p.add_argument("--max_stage", type=int, default=4)
    p.add_argument("--rows_per_stage", type=int, default=0)
    p.add_argument("--current_stage_fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def stage_mix(stage: int, current_fraction: float) -> List[tuple[int, float]]:
    stage = int(stage)
    if stage <= 1:
        return [(1, 1.0)]
    current_fraction = max(0.0, min(1.0, float(current_fraction)))
    prev_fraction = 1.0 - current_fraction
    return [(stage, current_fraction), (stage - 1, prev_fraction)]


def sample_rows(
    base_rows: Sequence[Dict[str, object]],
    *,
    count: int,
    rng: random.Random,
    offset: int,
) -> List[Dict[str, object]]:
    total = len(base_rows)
    if total == 0:
        return []
    if count <= total:
        indices = list(range(total))
        rng.shuffle(indices)
        chosen = indices[:count]
        return [dict(base_rows[idx]) for idx in chosen]
    rows: List[Dict[str, object]] = []
    for ii in range(int(count)):
        idx = (int(offset) + ii) % total
        rows.append(dict(base_rows[idx]))
    rng.shuffle(rows)
    return rows


def build_stage_rows(
    base_rows: Sequence[Dict[str, object]],
    *,
    target_stage: int,
    rows_per_stage: int,
    current_stage_fraction: float,
    rng: random.Random,
) -> List[Dict[str, object]]:
    mixed_rows: List[Dict[str, object]] = []
    mix = stage_mix(target_stage, current_stage_fraction)
    assigned = 0
    for item_idx, (source_stage, fraction) in enumerate(mix):
        if item_idx == len(mix) - 1:
            count = int(rows_per_stage - assigned)
        else:
            count = int(round(float(rows_per_stage) * float(fraction)))
            assigned += count
        sampled = sample_rows(
            base_rows,
            count=count,
            rng=rng,
            offset=target_stage * 100003 + source_stage * 1009 + item_idx * 17,
        )
        for row in sampled:
            metadata = dict(row.get("metadata", {}))
            metadata["mixed_curriculum_target_stage"] = int(target_stage)
            metadata["mixed_curriculum_source_stage"] = int(source_stage)
            metadata["mixed_curriculum_fraction"] = float(fraction)
            row["metadata"] = metadata
            mixed_rows.append(row)
    rng.shuffle(mixed_rows)
    return mixed_rows


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(base_path)
    if not base_rows:
        raise RuntimeError(f"No rows found in {base_path}")

    rows_per_stage = int(args.rows_per_stage) if int(args.rows_per_stage) > 0 else len(base_rows)
    rng = random.Random(int(args.seed))
    manifest: Dict[str, object] = {
        "base_jsonl": str(base_path),
        "output_dir": str(output_dir),
        "max_stage": int(args.max_stage),
        "rows_per_stage": int(rows_per_stage),
        "current_stage_fraction": float(args.current_stage_fraction),
        "stages": {},
    }

    for stage in range(1, int(args.max_stage) + 1):
        stage_rows = build_stage_rows(
            base_rows,
            target_stage=stage,
            rows_per_stage=rows_per_stage,
            current_stage_fraction=float(args.current_stage_fraction),
            rng=rng,
        )
        stage_file = output_dir / f"stage{stage:02d}_mixed.jsonl"
        write_jsonl(stage_file, stage_rows)
        manifest["stages"][f"stage{stage:02d}"] = {
            "path": str(stage_file),
            "mix": [
                {"source_stage": int(source_stage), "fraction": float(fraction)}
                for source_stage, fraction in stage_mix(stage, float(args.current_stage_fraction))
            ],
            "rows": int(len(stage_rows)),
        }
        print(
            f"Wrote stage {stage} mixed dataset to {stage_file} "
            f"with mix={manifest['stages'][f'stage{stage:02d}']['mix']}",
            flush=True,
        )

    manifest_path = output_dir / "mixed_curriculum_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
