from __future__ import annotations

import numpy as np


def is_consistent_pair(grid, *, cell: int, value: int, t: int = 3, n: int = 9) -> bool:
    g = np.asarray(grid, dtype=int).reshape(int(n), int(n))
    cell = int(cell)
    value = int(value)
    if value < 1 or value > int(n):
        return False
    rr, cc = divmod(cell, int(n))
    current = int(g[rr, cc])
    if current != 0 and current != value:
        return False

    row = g[rr, :]
    for idx, existing in enumerate(row):
        if idx != cc and int(existing) == value:
            return False

    col = g[:, cc]
    for idx, existing in enumerate(col):
        if idx != rr and int(existing) == value:
            return False

    box_r = (rr // int(t)) * int(t)
    box_c = (cc // int(t)) * int(t)
    for r in range(box_r, box_r + int(t)):
        for c in range(box_c, box_c + int(t)):
            if (r != rr or c != cc) and int(g[r, c]) == value:
                return False

    return True
