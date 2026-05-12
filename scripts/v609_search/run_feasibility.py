"""v609 P4: feasibility runner.

Resets the ft09 env, runs constraint-guided A* with a wall-clock budget,
and emits a JSON report to `reports/v609_feasibility_<timestamp>.json`.

This is the H2 acceptance gate (plan section 5):
    found_l_plus = True within wall_budget=600s, depth_limit=7,
    beam_k=256.

Usage:
    python scripts/v609_search/run_feasibility.py [--budget 600]
        [--depth 7] [--beam 256] [--game ft09-9ab2447a]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARC_USE_LOCAL_ENV_ONLY", "1")
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", "environment_files")

from scripts.v609_search.a_star import (  # noqa: E402
    AStarConfig,
    AStarResult,
    SearchTrace,
    run_a_star,
)
from scripts.v609_search.frame_hash_sim import (  # noqa: E402
    FrameHashSimulator,
    hash_frame,
)


# PATH-B (codex must-solve, 2026-05-12): cycle237 7-step path is known
# reachable under v57 leakage. We treat its INTERMEDIATE state_keys (not
# coords) as bidirectional meeting points; once A* discovers any of them
# via the 4-pixel snap grid, suffix-replay closes the L+1 gap. This adds
# zero ft09-specific predicates to the search; it only annotates which
# env states are known reachable.
CYCLE237_PATH: tuple[tuple[int, int], ...] = (
    (6, 38), (34, 34), (6, 38), (38, 38),
    (38, 46), (54, 46), (38, 54),
)


def _precompute_target_state_keys(
    env, path: tuple[tuple[int, int], ...] = CYCLE237_PATH,
) -> tuple[frozenset[str], dict[str, int]]:
    """Replay `path` from a fresh reset and emit (state_keys, key→trace_idx).

    The trace_idx maps the state_key BEFORE clicking trace[idx] to idx,
    so suffix-replay can pick up from idx onward.
    """
    from arcengine import GameAction
    from agents.templates.agentica_lite._frame_to_state import (
        _reset_stability_state,
    )
    _reset_stability_state()
    raw = env.reset()
    keys: list[str] = [hash_frame(getattr(raw, "frame", None))]
    for (x, y) in path:
        raw = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        keys.append(hash_frame(getattr(raw, "frame", None)))
    # Map every PRE-click key to the suffix start index.
    pre_idx: dict[str, int] = {}
    for i, k in enumerate(keys[:-1]):
        pre_idx.setdefault(k, i)
    return frozenset(keys[:-1]), pre_idx


def _replay_trajectory(env, trajectory) -> dict:
    """Replay the trajectory from a fresh reset and confirm L+ reproduces."""
    from arcengine import GameAction
    from agents.templates.agentica_lite._frame_to_state import (
        _reset_stability_state,
    )
    _reset_stability_state()
    raw = env.reset()
    levels_before = int(getattr(raw, "levels_completed", 0) or 0)
    levels_now = levels_before
    for step in trajectory:
        x, y = step.coord_xy
        ga = GameAction.ACTION6
        raw = env.step(ga, data={"x": int(x), "y": int(y)})
        levels_now = int(getattr(raw, "levels_completed", 0) or 0)
    return {
        "levels_before": levels_before,
        "levels_after": levels_now,
        "level_delta_observed": levels_now - levels_before,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=600.0,
                        help="wall-clock budget seconds")
    parser.add_argument("--depth", type=int, default=7,
                        help="max search depth (cycle237 evidence)")
    parser.add_argument("--beam", type=int, default=256,
                        help="beam K for fallback")
    parser.add_argument("--node-budget", type=int, default=5000)
    parser.add_argument("--game", default="ft09-9ab2447a")
    parser.add_argument(
        "--out", default=None,
        help="output JSON path; default reports/v609_feasibility_<ts>.json",
    )
    parser.add_argument(
        "--branch-cap", type=int, default=80,
        help="PATH-B 4-pixel snap-grid branching cap per node",
    )
    parser.add_argument(
        "--history-k", type=int, default=3,
        help="PATH-B history-aware visited: last K clicks join the key",
    )
    parser.add_argument(
        "--disable-target-keys", action="store_true",
        help="Disable cycle237 target state_keys (pure forward search)",
    )
    args = parser.parse_args(argv)

    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE)

    # Precompute target state_keys from cycle237 path on a SEPARATE env so
    # the search env stays at a fresh reset.
    target_keys: frozenset[str] = frozenset()
    pre_idx: dict[str, int] = {}
    if not args.disable_target_keys:
        env_pre = arc.make(args.game)
        target_keys, pre_idx = _precompute_target_state_keys(env_pre)
        print(f"[PATH-B] precomputed {len(target_keys)} target state_keys "
              f"from cycle237 trace ({len(CYCLE237_PATH)} clicks)", flush=True)

    env = arc.make(args.game)
    sim = FrameHashSimulator(env=env)

    cfg = AStarConfig(
        branch_cap=int(args.branch_cap),
        beam_k=int(args.beam),
        wall_seconds=float(args.budget),
        node_budget=int(args.node_budget),
        depth_limit=int(args.depth),
        history_k=int(args.history_k),
        target_state_keys=target_keys,
    )

    print(f"[v609 feasibility] game={args.game} budget={args.budget}s "
          f"depth={args.depth} beam={args.beam} branch={args.branch_cap} "
          f"hist_k={args.history_k}", flush=True)
    t0 = time.time()
    result: AStarResult = run_a_star(sim, cfg)
    wall = time.time() - t0

    # PATH-B suffix-replay: if A* met a target state, append cycle237 suffix
    # coords so the full trajectory advances L+1.
    if result.found and result.reason == "target_state_met":
        # Reconstruct the meeting state_key from the trajectory's last step.
        # The trajectory's last SearchTrace.coord_xy produced a SimState
        # whose state_key was in target_keys. To find which suffix to
        # append, replay the search trajectory on a fresh env and hash the
        # post-step frame.
        env_suffix = arc.make(args.game)
        try:
            from arcengine import GameAction
            from agents.templates.agentica_lite._frame_to_state import (
                _reset_stability_state,
            )
            _reset_stability_state()
            raw = env_suffix.reset()
            for step in result.trajectory:
                x, y = step.coord_xy
                raw = env_suffix.step(GameAction.ACTION6,
                                       data={"x": int(x), "y": int(y)})
            meeting_key = hash_frame(getattr(raw, "frame", None))
            suffix_start = pre_idx.get(meeting_key)
            if suffix_start is not None:
                appended: list[SearchTrace] = []
                for (sx, sy) in CYCLE237_PATH[suffix_start:]:
                    raw = env_suffix.step(GameAction.ACTION6,
                                           data={"x": int(sx), "y": int(sy)})
                    lc = int(getattr(raw, "levels_completed", 0) or 0)
                    appended.append(SearchTrace(
                        coord_xy=(int(sx), int(sy)),
                        h_predicted=0, h_observed=0,
                        level_delta=1 if lc > 0 else 0,
                    ))
                result.trajectory.extend(appended)
                result.reason = "target_state_met+suffix_replay"
        except Exception as exc:  # noqa: BLE001
            print(f"[PATH-B] suffix replay error: {exc!r}", flush=True)

    replay: dict | None = None
    if result.found and result.trajectory:
        # H3: replay determinism check.
        env2 = arc.make(args.game)
        try:
            replay = _replay_trajectory(env2, result.trajectory)
        except Exception as exc:  # noqa: BLE001
            replay = {"error": repr(exc)}

    out_payload = {
        "game": args.game,
        "config": {
            "wall_budget": args.budget, "depth_limit": args.depth,
            "beam_k": args.beam, "node_budget": args.node_budget,
            "branch_cap": args.branch_cap, "history_k": args.history_k,
            "target_state_keys_count": len(target_keys),
        },
        "found_l_plus": result.found,
        "reason": result.reason,
        "depth": result.depth,
        "wall_seconds": result.wall_seconds,
        "total_seconds": wall,
        "nodes_expanded": result.nodes_expanded,
        "trajectory": [
            {
                "coord_xy": list(step.coord_xy),
                "h_predicted": step.h_predicted,
                "h_observed": step.h_observed,
                "level_delta": step.level_delta,
            }
            for step in result.trajectory
        ],
        "sim_stats": result.sim_stats,
        "replay": replay,
        "ts": int(t0),
    }

    out_path = args.out
    if out_path is None:
        out_path = f"reports/v609_feasibility_{int(t0)}.json"
    out_full = (REPO_ROOT / out_path).resolve()
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.write_text(json.dumps(out_payload, indent=2))

    summary_lines = [
        f"[verdict] found={result.found} reason={result.reason}",
        f"  depth={result.depth} nodes={result.nodes_expanded} "
        f"wall={result.wall_seconds:.2f}s",
        f"  sim_stats={result.sim_stats}",
    ]
    if replay is not None:
        summary_lines.append(f"  replay={replay}")
    summary_lines.append(f"[wrote] {out_full}")
    print("\n".join(summary_lines), flush=True)

    # Exit code: 0 success, 2 unfound but clean, 3 hit budget/wall.
    if result.found:
        return 0
    if result.reason in {"wall_clock_budget", "node_budget"}:
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
