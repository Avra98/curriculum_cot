from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from multi_output_cell_policy.shared_multi_output_policy import (
    compute_set_precision_recall,
    parse_values_json,
    stage_i_consistent_values,
)


def triangular_number(n: int) -> float:
    nn = max(0, int(n))
    return float(nn * (nn + 1) // 2)


def score_prediction_text(
    *,
    text: str,
    grid: np.ndarray,
    solved: np.ndarray,
    target_cell: tuple[int, int],
    stage_i: int,
    reward_good_value: float,
    penalty_bad_value: float,
    penalty_malformed: float,
    penalty_empty: float,
    penalty_singleton: float,
) -> Dict[str, float | List[int] | str]:
    parsed = parse_values_json(text)
    target_values = stage_i_consistent_values(grid, target_cell=target_cell, stage_i=stage_i)
    solved_value = int(np.asarray(solved, dtype=int).reshape(9, 9)[int(target_cell[0]), int(target_cell[1])])
    singleton_penalty = 0.0 if int(stage_i) >= 2 else float(penalty_singleton)

    if not parsed.parse_ok:
        return {
            "reward": -float(penalty_malformed),
            "parse_ok": 0.0,
            "strict_canonical": 0.0,
            "num_predicted_values": 0.0,
            "num_i_consistent_values": 0.0,
            "num_non_i_consistent_values": 0.0,
            "includes_ground_truth": 0.0,
            "value_precision": 0.0,
            "value_recall": 0.0,
            "exact_set_match": 0.0,
            "predicted_values": [],
            "target_values": [int(v) for v in target_values],
            "format_error": "parse_failed",
        }

    predicted_values = [int(v) for v in parsed.values]
    target_set = set(int(v) for v in target_values)
    num_good = sum(1 for v in predicted_values if v in target_set)
    num_bad = sum(1 for v in predicted_values if v not in target_set)

    # Encourage larger all-good sets while making extra wrong values expensive.
    reward = triangular_number(num_good) * float(reward_good_value) - float(num_bad) * float(
        penalty_bad_value
    )
    if not predicted_values:
        reward -= float(penalty_empty)
    if len(predicted_values) == 1 and len(target_values) > 1:
        reward -= singleton_penalty

    precision, recall = compute_set_precision_recall(predicted_values, target_values)
    return {
        "reward": float(reward),
        "parse_ok": 1.0,
        "strict_canonical": 1.0 if parsed.strict_canonical else 0.0,
        "num_predicted_values": float(len(predicted_values)),
        "num_i_consistent_values": float(num_good),
        "num_non_i_consistent_values": float(num_bad),
        "includes_ground_truth": 1.0 if solved_value in predicted_values else 0.0,
        "value_precision": float(precision),
        "value_recall": float(recall),
        "exact_set_match": 1.0 if set(predicted_values) == target_set else 0.0,
        "predicted_values": predicted_values,
        "target_values": [int(v) for v in target_values],
        "format_error": "",
    }
