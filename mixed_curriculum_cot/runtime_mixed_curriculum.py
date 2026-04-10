from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence


def sample_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    count: int,
    rng: random.Random,
    offset: int,
) -> List[Dict[str, Any]]:
    total = len(rows)
    if total == 0 or int(count) <= 0:
        return []
    if int(count) <= total:
        indices = list(range(total))
        rng.shuffle(indices)
        chosen = indices[: int(count)]
        return [dict(rows[idx]) for idx in chosen]
    out: List[Dict[str, Any]] = []
    for item_idx in range(int(count)):
        out.append(dict(rows[(int(offset) + item_idx) % total]))
    rng.shuffle(out)
    return out


def annotate_mixed_curriculum_row(
    row: Dict[str, Any],
    *,
    source_stage: int,
    target_stage: int,
    fraction: float,
) -> Dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata", {}))
    metadata["mixed_curriculum_target_stage"] = int(target_stage)
    metadata["mixed_curriculum_source_stage"] = int(source_stage)
    metadata["mixed_curriculum_fraction"] = float(fraction)
    metadata["mixed_curriculum_runtime"] = True
    out["metadata"] = metadata
    return out


def build_two_stage_mixed_rows(
    stage1_rows: Sequence[Dict[str, Any]],
    stage2_rows: Sequence[Dict[str, Any]],
    *,
    stage1_ratio: float,
    stage2_ratio: float,
    seed: int,
    target_stage: int,
    total_rows: int = 0,
) -> List[Dict[str, Any]]:
    weight1 = max(0.0, float(stage1_ratio))
    weight2 = max(0.0, float(stage2_ratio))
    weight_sum = weight1 + weight2
    if weight_sum <= 0.0:
        raise ValueError("At least one mixed curriculum ratio must be positive.")

    if int(total_rows) > 0:
        target_count = int(total_rows)
    else:
        target_count = max(len(stage1_rows), len(stage2_rows))
    target_count = max(1, int(target_count))

    count1 = int(round(target_count * (weight1 / weight_sum)))
    count1 = min(max(count1, 0), target_count)
    count2 = int(target_count - count1)
    rng = random.Random(int(seed))

    mixed: List[Dict[str, Any]] = []
    for row in sample_rows(stage1_rows, count=count1, rng=rng, offset=1009):
        mixed.append(
            annotate_mixed_curriculum_row(
                row,
                source_stage=1,
                target_stage=int(target_stage),
                fraction=(weight1 / weight_sum),
            )
        )
    for row in sample_rows(stage2_rows, count=count2, rng=rng, offset=2003):
        mixed.append(
            annotate_mixed_curriculum_row(
                row,
                source_stage=2,
                target_stage=int(target_stage),
                fraction=(weight2 / weight_sum),
            )
        )
    rng.shuffle(mixed)
    return mixed


def training_stage_i_for_row(row: Dict[str, Any], default_stage_i: int) -> int:
    metadata = dict(row.get("metadata", {}))
    source_stage = metadata.get("mixed_curriculum_source_stage")
    if source_stage is not None:
        return max(1, int(source_stage))
    row_stage = row.get("stage_i")
    if row_stage is not None:
        return max(1, int(row_stage))
    return max(1, int(default_stage_i))
