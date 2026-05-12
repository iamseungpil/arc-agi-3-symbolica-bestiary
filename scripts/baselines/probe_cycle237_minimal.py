"""Empirically discover the MINIMAL cycle237 subset that reaches L+1.

cycle237 7-step path: [(6,38), (34,34), (6,38), (38,38), (38,46), (54,46), (38,54)]
Step diffs show t0-t2 have ZERO visible pixel change. Question: are they
necessary state-setters or red herrings? Test all sub-suffixes and
single-skip variants.
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


CYCLE237 = [(6, 38), (34, 34), (6, 38), (38, 38),
            (38, 46), (54, 46), (38, 54)]


def _trial(env, path):
    from arcengine import GameAction
    raw = env.reset()
    levels_before = int(getattr(raw, "levels_completed", 0) or 0)
    for (x, y) in path:
        raw = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return (int(getattr(raw, "levels_completed", 0) or 0) - levels_before, path)


def main():
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE)

    trials = []
    # Baseline: full cycle237
    trials.append(("full_7", CYCLE237))
    # Suffix starting from each step
    for start in range(1, 7):
        trials.append((f"suffix_t{start}", CYCLE237[start:]))
    # Skip each single step
    for skip in range(7):
        path = CYCLE237[:skip] + CYCLE237[skip+1:]
        trials.append((f"skip_t{skip}", path))
    # Just steps 3-6 (the visible-effect ones)
    trials.append(("only_effective", CYCLE237[3:7]))
    # Just step 6 (the L+1 trigger)
    trials.append(("only_t6", [CYCLE237[6]]))
    # Just steps 3 + 7 (most economic)
    trials.append(("t3_t6", [CYCLE237[3], CYCLE237[6]]))
    # Just t3,t4,t6 (skip t5)
    trials.append(("t3_t4_t6", [CYCLE237[3], CYCLE237[4], CYCLE237[6]]))
    # Reverse order of t3-t6
    trials.append(("rev_t3_t6", list(reversed(CYCLE237[3:7]))))

    results = []
    for name, path in trials:
        env = arc.make("ft09-9ab2447a")
        delta, p = _trial(env, path)
        results.append((name, delta, len(path), p))
        marker = "✅ L+1" if delta >= 1 else f"  L+{delta}" if delta > 0 else "  no advance"
        print(f"{marker} | {name:<20s} len={len(path)} path={p}")

    print("\n=== minimal subsets that reach L+1 ===")
    successes = sorted([r for r in results if r[1] >= 1], key=lambda r: r[2])
    for name, delta, n, p in successes:
        print(f"  len={n}: {name} -> {p}")


if __name__ == "__main__":
    main()
