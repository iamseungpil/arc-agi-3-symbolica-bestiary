"""Probe ft09's L+1 mechanic by trying many 4-card click combinations.

Question: is L+1 triggered by
  (a) exact 4 cards matching a specific reference position pattern
  (b) any 4 cards (count: 5 maroon + 4 blue)
  (c) clicking 4 distinct target-zone cards in any order
  (d) something else
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARC_USE_LOCAL_ENV_ONLY", "1")
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", "environment_files")


# 9 cards in target zone. Each card center coord = (x, y).
TARGET_CARDS = [
    (38, 38), (46, 38), (54, 38),  # row 0
    (38, 46), (46, 46), (54, 46),  # row 1
    (38, 54), (46, 54), (54, 54),  # row 2
]


def _trial(arc, path):
    from arcengine import GameAction
    env = arc.make("ft09-9ab2447a")
    raw = env.reset()
    lvls0 = int(getattr(raw, "levels_completed", 0) or 0)
    for (x, y) in path:
        raw = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return int(getattr(raw, "levels_completed", 0) or 0) - lvls0


def main():
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE)

    print("[probing all 126 combinations of 4 cards from 9 target cards]")
    successes = []
    failures_count = 0
    for combo in combinations(range(9), 4):
        path = [TARGET_CARDS[i] for i in combo]
        delta = _trial(arc, path)
        if delta >= 1:
            successes.append((combo, path, delta))
            # Build 3x3 binary pattern (which positions clicked)
            grid = [[0]*3 for _ in range(3)]
            for i in combo:
                r, c = divmod(i, 3)
                grid[r][c] = 1
            pattern_str = "\n  ".join(" ".join(map(str, row)) for row in grid)
            print(f"✅ L+{delta} combo={combo} path={path}")
            print(f"  click pattern:\n  {pattern_str}")
        else:
            failures_count += 1

    print(f"\n=== summary: {len(successes)} success / "
          f"{failures_count} fail = {len(successes)+failures_count} total")
    print("\nclicked-cell counts across all winners:")
    pos_freq = {i: 0 for i in range(9)}
    for combo, _, _ in successes:
        for i in combo:
            pos_freq[i] += 1
    for i in range(9):
        r, c = divmod(i, 3)
        print(f"  position ({r},{c}) [coord {TARGET_CARDS[i]}]: {pos_freq[i]} wins")


if __name__ == "__main__":
    main()
