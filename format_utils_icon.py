from __future__ import annotations

import json
import re
from typing import List, Tuple

import numpy as np


_INT_RE = re.compile(r"-?\d+")


def grid_to_text(grid_9x9: np.ndarray) -> str:
    grid = np.asarray(grid_9x9, dtype=int).reshape(9, 9)
    return "\n".join(" ".join(str(int(value)) for value in row) for row in grid.tolist())


def parse_n_value_prediction(text: str, n: int) -> Tuple[List[int] | None, bool]:
    raw = str(text or "").strip()
    if not raw:
        return None, False

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("values"), list):
            values = [int(v) for v in parsed["values"]]
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
