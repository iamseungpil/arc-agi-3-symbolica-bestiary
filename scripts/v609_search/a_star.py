"""v609 P3: constraint-guided A* with beam fallback.

Plan section 6 P3. The search expands nodes from an LRU-cached frame
simulator (`scripts.v609_search.frame_hash_sim.FrameHashSimulator`) and
ranks by `f(n) = g(n) + h(n)` where:

- `g(n)` = click count so far.
- `h(n)` = `marker_constraint_summary.high_unsatisfied` (codex Q1
  validated as the load-bearing signal). Fallback to total `unsatisfied`
  when `high_unsatisfied == 0`.

Branching policy (plan section 8 C2):
1. **primary**: regions in any unsatisfied high-quality marker constraint.
2. **compass**: every marker's compass neighbors (cycle237 turn 0-3 lived
   in this band).
3. **exploratory**: top-5 visible regions by area not already covered.

Hard cap branching at 12 to keep memory bounded.

Termination:
- `level_delta >= 1` (success).
- open list empty.
- wall clock budget exceeded.
- nodes expanded budget exceeded.
- beam width K exceeded -> drop tail of open list (codex Q3 caveat).
"""

from __future__ import annotations

import heapq
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARC_USE_LOCAL_ENV_ONLY", "1")

from scripts.v609_search.frame_hash_sim import (  # noqa: E402
    FrameHashSimulator,
    SimResult,
    SimState,
)

logger = logging.getLogger(__name__)


DEFAULT_BRANCH_CAP = 80
DEFAULT_BEAM_K = 256
DEFAULT_WALL_SECONDS = 600.0
DEFAULT_NODE_BUDGET = 5000
DEFAULT_DEPTH_LIMIT = 7
# PATH-B (codex must-solve, 2026-05-12): ft09 click grid is 4-pixel granular
# with x offset 2 and y offsets {0, 2}. cycle237 7-step prereq coords ALL lie
# on x%4==2 ∧ y%4==2. Snap-grid candidate generation guarantees enumerate.
SNAP_X_VALUES: tuple[int, ...] = tuple(range(2, 64, 4))      # 16 vals
SNAP_Y_VALUES: tuple[int, ...] = tuple(range(0, 64, 2))       # 32 vals (any %2==0)
# History-aware visited: key = (state_key, last_K_clicks) — codex Q2 (b).
# Pure frame-hash dedup collapses order-sensitive prereq chains.
DEFAULT_HISTORY_K = 3


@dataclass
class AStarConfig:
    branch_cap: int = DEFAULT_BRANCH_CAP
    beam_k: int = DEFAULT_BEAM_K
    wall_seconds: float = DEFAULT_WALL_SECONDS
    node_budget: int = DEFAULT_NODE_BUDGET
    depth_limit: int = DEFAULT_DEPTH_LIMIT
    history_k: int = DEFAULT_HISTORY_K
    # PATH-B: subgoal-augmented termination (meet-in-the-middle proxy under
    # step-only env). If the expanded state's state_key matches any target,
    # treat as found. Empty set disables.
    target_state_keys: frozenset[str] = frozenset()


@dataclass
class SearchTrace:
    coord_xy: tuple[int, int]
    h_predicted: int
    h_observed: int
    level_delta: int


@dataclass
class AStarResult:
    found: bool
    reason: str
    depth: int = 0
    wall_seconds: float = 0.0
    nodes_expanded: int = 0
    trajectory: list[SearchTrace] = field(default_factory=list)
    sim_stats: dict[str, Any] = field(default_factory=dict)


def _region_view_centroid(view: dict[str, Any]) -> tuple[int, int]:
    bbox = view.get("bbox") or [0, 0, 0, 0]
    try:
        x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]),
                          int(bbox[2]), int(bbox[3]))
    except (TypeError, ValueError):
        return 0, 0
    return (x0 + x1) // 2, (y0 + y1) // 2


def _bbox_or_none(view: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = view.get("bbox") or []
    if len(bbox) < 4:
        return None
    try:
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except (TypeError, ValueError):
        return None


def _snap_coords_in_bbox(
    bbox: tuple[int, int, int, int],
    buffer: int = 2,
) -> list[tuple[int, int]]:
    """Enumerate 4-pixel snap coords (x%4==2, y%2==0) inside bbox+buffer.

    cycle237 trace pattern: ALL prereq coords have x%4==2 ∧ y%4==2; full
    trace uses x%4==2 ∧ y%4∈{0,2}. We enumerate the looser set (y%2==0) to
    keep coverage broad while still pruning blind clicks 8x.
    """
    x0, y0, x1, y1 = bbox
    lo_x, hi_x = max(0, x0 - buffer), min(63, x1 + buffer)
    lo_y, hi_y = max(0, y0 - buffer), min(63, y1 + buffer)
    out: list[tuple[int, int]] = []
    for x in SNAP_X_VALUES:
        if x < lo_x or x > hi_x:
            continue
        for y in SNAP_Y_VALUES:
            if y < lo_y or y > hi_y:
                continue
            out.append((x, y))
    return out


def candidate_coords(state_payload: dict, branch_cap: int) -> list[tuple[int, int]]:
    """Return a deduplicated, capped list of click coords to expand.

    PATH-B (codex must-solve, 2026-05-12): emits 4-pixel snap coords inside
    visible region bboxes, ranked by (primary | compass | exploratory). The
    previous region-centroid-only generator missed all cycle237 prereq
    coords because ft09 centroids do not align with the 4-pixel snap grid.

    `state_payload` is the dict returned by
    `agents.templates.agentica_lite._frame_to_state.frame_to_state`.
    """
    if not isinstance(state_payload, dict):
        return []
    visible = state_payload.get("visible_regions") or []
    constraints = state_payload.get("marker_constraints") or []
    markers = state_payload.get("marker_neighbor_states") or []

    primary_ids: set[str] = set()
    for c in constraints:
        if not isinstance(c, dict) or bool(c.get("satisfied", True)):
            continue
        if c.get("evidence_quality") != "high":
            continue
        rid = c.get("neighbor_region_id")
        if rid:
            primary_ids.add(str(rid))

    compass_ids: set[str] = set()
    for m in markers:
        comp = m.get("compass") or {}
        for slot, entry in comp.items():
            rid = entry.get("region_id") if isinstance(entry, dict) else None
            if rid:
                compass_ids.add(str(rid))

    by_id: dict[str, dict] = {}
    for r in visible:
        rid = r.get("region_id") or r.get("id")
        if rid:
            by_id[str(rid)] = r

    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add_many(items: Iterable[tuple[int, int]]):
        for c in items:
            if c in seen:
                continue
            seen.add(c)
            coords.append(c)
            if len(coords) >= branch_cap:
                return True
        return False

    # Tier 1: snap coords inside any primary (unsatisfied high) bbox.
    for rid in primary_ids:
        bbox = _bbox_or_none(by_id.get(str(rid), {}))
        if bbox is None:
            continue
        if _add_many(_snap_coords_in_bbox(bbox)):
            return coords[:branch_cap]

    # Tier 2: snap coords inside compass neighbor bboxes.
    for rid in compass_ids - primary_ids:
        bbox = _bbox_or_none(by_id.get(str(rid), {}))
        if bbox is None:
            continue
        if _add_many(_snap_coords_in_bbox(bbox)):
            return coords[:branch_cap]

    # Tier 3: snap coords inside top-N visible by area.
    others = sorted(
        [r for rid, r in by_id.items()
         if rid not in primary_ids and rid not in compass_ids],
        key=lambda r: -int(r.get("size") or 0),
    )[:8]
    for r in others:
        bbox = _bbox_or_none(r)
        if bbox is None:
            continue
        if _add_many(_snap_coords_in_bbox(bbox)):
            return coords[:branch_cap]

    # Tier 4 (only if branch_cap not filled): full 4-pixel snap grid fallback.
    # Guards against hidden affordances outside visible bboxes (codex Q1).
    if len(coords) < branch_cap:
        full = [(x, y) for x in SNAP_X_VALUES for y in SNAP_Y_VALUES]
        _add_many(full)

    return coords[:branch_cap]


def _heuristic(result_or_payload) -> int:
    """h(n) = high_unsatisfied; fallback to unsatisfied when high == 0."""
    if isinstance(result_or_payload, SimResult):
        if result_or_payload.high_unsatisfied > 0:
            return result_or_payload.high_unsatisfied
        return result_or_payload.unsatisfied
    if isinstance(result_or_payload, dict):
        mcs = result_or_payload.get("marker_constraint_summary") or {}
        h = int(mcs.get("high_unsatisfied", 0) or 0)
        if h > 0:
            return h
        return int(mcs.get("unsatisfied", 0) or 0)
    return 0


def _state_payload_for(state: SimState, sim: FrameHashSimulator) -> dict:
    """Replay the env to the state's clicks and return frame_to_state."""
    proxy = sim._replay(state.clicks)  # uses cached counter inside sim
    return sim._summary_for_proxy(proxy)


def _history_key(state: SimState, k: int) -> tuple[str, tuple[tuple[int, int], ...]]:
    """PATH-B visited key: (state_key, last-K clicks). Codex Q2 (b).

    Pure state_key dedup collapses order-sensitive prereq chains (e.g. the
    cycle237 [6,38] → ... → [38,54] sequence). Pairing the content hash
    with a short click suffix preserves history while still pruning loops.
    """
    if k <= 0 or not state.clicks:
        return (state.state_key, ())
    return (state.state_key, tuple(state.clicks[-k:]))


def run_a_star(
    sim: FrameHashSimulator,
    cfg: AStarConfig | None = None,
) -> AStarResult:
    """Constraint-guided A* with beam fallback.

    PATH-B additions (codex must-solve, 2026-05-12):
    - 4-pixel snap candidate generation via candidate_coords (above).
    - history-aware visited via `_history_key(state, cfg.history_k)`.
    - target_state_keys early termination as bidirectional meet-in-middle
      proxy (full backward expansion not possible under step-only env).

    Returns an `AStarResult` describing whether L+1 was found (or a target
    state was met), the trajectory, and search-cost telemetry.
    """
    cfg = cfg or AStarConfig()
    t0 = time.time()
    root = sim.reset()
    root_payload = _state_payload_for(root, sim)
    h_root = _heuristic(root_payload)

    open_heap: list[tuple[int, int, int, SimState, list[SearchTrace], dict]] = []
    counter = 0
    heapq.heappush(open_heap,
                   (h_root, counter, 0, root, [], root_payload))

    visited: set[tuple[str, tuple[tuple[int, int], ...]]] = {
        _history_key(root, cfg.history_k)
    }
    nodes_expanded = 0
    targets = cfg.target_state_keys or frozenset()

    while open_heap:
        if time.time() - t0 > cfg.wall_seconds:
            return AStarResult(
                found=False, reason="wall_clock_budget",
                wall_seconds=time.time() - t0,
                nodes_expanded=nodes_expanded,
                sim_stats=sim.stats(),
            )
        if nodes_expanded > cfg.node_budget:
            return AStarResult(
                found=False, reason="node_budget",
                wall_seconds=time.time() - t0,
                nodes_expanded=nodes_expanded,
                sim_stats=sim.stats(),
            )

        f, _, g, state, traj, payload = heapq.heappop(open_heap)
        if state.depth() >= cfg.depth_limit:
            continue

        cands = candidate_coords(payload, cfg.branch_cap)
        if not cands:
            continue

        # PATH-B Q3 fix: pass parent unsatisfied so delta is observable.
        parent_unsat = int(((payload or {}).get("marker_constraint_summary")
                            or {}).get("unsatisfied", 0) or 0)

        for coord in cands:
            res = sim.step(state, coord, parent_unsatisfied=parent_unsat)
            nodes_expanded += 1
            new_traj = traj + [SearchTrace(
                coord_xy=coord,
                h_predicted=_heuristic(payload),
                h_observed=_heuristic(res),
                level_delta=res.level_delta,
            )]
            if res.level_delta >= 1:
                return AStarResult(
                    found=True, reason="level_advance",
                    depth=res.state.depth(),
                    wall_seconds=time.time() - t0,
                    nodes_expanded=nodes_expanded,
                    trajectory=new_traj,
                    sim_stats=sim.stats(),
                )
            if targets and res.state.state_key in targets:
                return AStarResult(
                    found=True, reason="target_state_met",
                    depth=res.state.depth(),
                    wall_seconds=time.time() - t0,
                    nodes_expanded=nodes_expanded,
                    trajectory=new_traj,
                    sim_stats=sim.stats(),
                )
            vk = _history_key(res.state, cfg.history_k)
            if vk in visited:
                continue
            visited.add(vk)
            new_payload = _state_payload_for(res.state, sim)
            h_next = _heuristic(new_payload)
            # PATH-B Q3 (codex e) v2: stronger delta weights. Null branches
            # get heavy penalty; constraint-reducing branches get strong
            # bonus that dominates h(state) for promising prefixes.
            progress_score = 0
            if not res.frame_changed:
                progress_score += 30   # strong null penalty
            if res.observed_unsat_delta < 0:
                progress_score -= 15   # strong repair bonus
            elif res.observed_unsat_delta > 0:
                progress_score += 3
            counter += 1
            heapq.heappush(open_heap,
                           (g + 1 + h_next + progress_score, counter,
                            g + 1, res.state, new_traj, new_payload))

        # Beam fallback: if open list exceeds K, keep the K best.
        if len(open_heap) > cfg.beam_k:
            open_heap = heapq.nsmallest(cfg.beam_k, open_heap)
            heapq.heapify(open_heap)

    return AStarResult(
        found=False, reason="open_list_empty",
        wall_seconds=time.time() - t0,
        nodes_expanded=nodes_expanded,
        sim_stats=sim.stats(),
    )
