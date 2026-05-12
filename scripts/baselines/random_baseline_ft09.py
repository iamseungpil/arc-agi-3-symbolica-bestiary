"""Random-click baseline on ft09 — does ANY L+1 happen in N actions?

Per codex 2026-05-12 Q4: if random hits L+1 within 1000 actions, our
framework is over-constraining (transition is sparse but reachable).
If 0 across 5000 actions, structure is genuinely needed.

Pure env.step(ACTION6, data={'x':int,'y':int}) — NO framework, NO LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARC_USE_LOCAL_ENV_ONLY", "1")
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", "environment_files")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-actions", type=int, default=1000)
    parser.add_argument("--game", default="ft09-9ab2447a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snap-grid", action="store_true",
                        help="constrain coords to x in {2,6,...,62} y in {0,2,...,62}")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    random.seed(args.seed)

    from arcengine import GameAction
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make(args.game)

    if args.snap_grid:
        xs = list(range(2, 64, 4))   # 16
        ys = list(range(0, 64, 2))   # 32
    else:
        xs = list(range(0, 64))
        ys = list(range(0, 64))

    t0 = time.time()
    raw = env.reset()
    levels_initial = int(getattr(raw, "levels_completed", 0) or 0)
    levels = levels_initial
    l_plus_events = []

    for i in range(args.n_actions):
        x = random.choice(xs)
        y = random.choice(ys)
        raw = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        new_lvls = int(getattr(raw, "levels_completed", 0) or 0)
        if new_lvls > levels:
            l_plus_events.append({
                "action_idx": i, "coord": [x, y],
                "levels_before": levels, "levels_after": new_lvls,
                "delta": new_lvls - levels,
                "wall_seconds": time.time() - t0,
            })
            levels = new_lvls

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[t+{elapsed:.1f}s] i={i+1}/{args.n_actions} "
                  f"levels={levels} l_plus_events={len(l_plus_events)}",
                  flush=True)

    total = time.time() - t0
    summary = {
        "game": args.game,
        "n_actions": args.n_actions,
        "snap_grid": args.snap_grid,
        "seed": args.seed,
        "levels_initial": levels_initial,
        "levels_final": levels,
        "total_l_plus_events": len(l_plus_events),
        "wall_seconds": total,
        "actions_per_sec": args.n_actions / total,
        "l_plus_events": l_plus_events,
    }

    out = args.out or f"reports/random_baseline_ft09_{int(t0)}.json"
    out_full = (REPO_ROOT / out).resolve()
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.write_text(json.dumps(summary, indent=2))

    print(f"\n[VERDICT] {len(l_plus_events)} L+ events in {args.n_actions}"
          f" random actions ({total:.1f}s)")
    print(f"  levels: {levels_initial} -> {levels}")
    print(f"  wrote: {out_full}")

    return 0 if len(l_plus_events) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
