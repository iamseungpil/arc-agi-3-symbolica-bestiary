"""ArcgenticaSimple v14: three-isolated-subagent orchestrator.

Plan v14 §4.9. Per turn:
  M1 (state-grounded hypothesis cards) ->
  validate / score ->
  M2 (single-action choice) ->
  execute one action ->
  extract_observation -> append to DiffMemory -> evaluate_falsifier -> GoalBoard ->
  M3 (analogy compression) every 3 turns or on falsify (after 5 actions).

No ledgers, no reasoning bundle, no shared closures. All state lives in
GoalBoard + DiffMemory. Falls back to coordinate-region exploration if any
subagent fails to parse.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from arcengine import GameAction
from agentica.logging import AgentListener

from ..agentica.agent import Arcgentica
from ..agentica.logging.logger import EventServer, WsLogger
from ..agentica.logging.tracker import UsageTracker
from .diff_memory import DiffMemory
from .goal_board import (
    AbstractSkill,
    ChosenAction,
    GoalBoard,
    HypothesisCard,
    LessonCard,
    _NullTrace,
    _parse_action_str,
    archetype_alignment,
    evaluate_falsifier,
    jaccard_words,
    parse_card_list,
    parse_lesson_card,
    parse_or_retry,
    parse_with_critic,
    precision_score,
    validate_card,
)
from .m1_prompt import M1_SYSTEM_PROMPT, M1_TASK_INSTRUCTIONS
from .m2_prompt import M2_SYSTEM_PROMPT, M2_TASK_INSTRUCTIONS
from .m3_prompt import M3_SYSTEM_PROMPT, M3_TASK_INSTRUCTIONS
from .m4_prompt import M4_SYSTEM_PROMPT, M4_TASK_INSTRUCTIONS
from .reflexion_prompt import (
    REFLEXION_SYSTEM_PROMPT,
    REFLEXION_TASK_INSTRUCTIONS,
)

# v41 minimum: disable M3/M4 by default; route corrective text through
# Reflexion module instead. Set ARC_V41_LEGACY_M34=1 to keep old behaviour
# (e.g., for ablation runs).
_V41_LEGACY_M34 = os.environ.get("ARC_V41_LEGACY_M34", "0") == "1"
from .state import (
    SemanticPackets,
    _change_bbox,
    _change_pattern_summary,
    _grid_to_list,
)

logger = logging.getLogger(__name__)


_SCHEMAS_PATH = Path(__file__).parent / "schemas.json"


def _load_arc_archetypes() -> list[dict]:
    """Load static ARC archetype library (v19). Returns [] on any error so
    a missing/corrupt file degrades gracefully without crashing the loop."""
    try:
        data = json.loads(_SCHEMAS_PATH.read_text(encoding="utf-8"))
        schemas = data.get("schemas", [])
        return schemas if isinstance(schemas, list) else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("schemas.json load failed: %s", exc)
        return []


_ARC_ARCHETYPES = _load_arc_archetypes()


def _infer_gqb_pair(diff_memory) -> tuple[int, int] | None:
    """v29: infer the toggle color cycle (gqb) from observed dominant transitions.
    Returns (a, b) where (a, b) is the most-common bidirectional transition pair.
    Returns None if no transitions observed yet.

    v30 Fix A: only consider entries since the last level_delta>0 observation
    so a level transition resets the inferred pair (e.g. ft09 L0={8,9} but
    L4={14,15}). Stale pre-transition pairs would otherwise dominate counts
    and corrupt per_neighbor_target.target_color downstream.
    """
    pair_counts: dict[tuple[int, int], int] = {}
    try:
        all_entries = diff_memory.entries
        boundary = diff_memory.last_level_boundary() if hasattr(
            diff_memory, "last_level_boundary"
        ) else 0
        entries = all_entries[boundary:]
    except AttributeError:
        return None
    for e in entries:
        dt = e.dominant_transition if hasattr(e, "dominant_transition") else None
        if isinstance(dt, dict):
            f = dt.get("from")
            t = dt.get("to")
            try:
                if f is not None and t is not None and int(f) != int(t):
                    key = tuple(sorted((int(f), int(t))))
                    pair_counts[key] = pair_counts.get(key, 0) + 1
            except (TypeError, ValueError):
                continue
    if not pair_counts:
        return None
    best = max(pair_counts.items(), key=lambda kv: kv[1])
    return best[0]


def _region_for_coord(
    name: str, x: int, y: int, regions: list[dict]
) -> str | None:
    """Return the region_id whose bbox contains (y, x) for an ACTION6 click,
    else None. Non-ACTION6 actions return None (no click point)."""
    if name != "ACTION6":
        return None
    for r in regions:
        if not isinstance(r, dict):
            continue
        bbox = r.get("bbox") or {}
        if _point_in_bbox((y, x), bbox):
            rid = r.get("id")
            if rid:
                return str(rid)
    return None


def _changed_cells_centroid(
    before: list[list[int]], after: list[list[int]]
) -> tuple[int, int] | None:
    """Returns (y, x) centroid of changed cells, or None if no diff."""
    rows = min(len(before), len(after))
    cols = min(len(before[0]) if before else 0, len(after[0]) if after else 0)
    sx = sy = n = 0
    for y in range(rows):
        br = before[y]
        ar = after[y]
        for x in range(cols):
            if int(br[x]) != int(ar[x]):
                sx += x
                sy += y
                n += 1
    if n == 0:
        return None
    return (sy // n, sx // n)


def _point_in_bbox(yx: tuple[int, int], bbox: dict) -> bool:
    if not isinstance(bbox, dict):
        return False
    y, x = yx
    return (
        int(bbox.get("min_y", 0)) <= y <= int(bbox.get("max_y", -1))
        and int(bbox.get("min_x", 0)) <= x <= int(bbox.get("max_x", -1))
    )


def _step_mismatch(observed: dict, expected: dict) -> bool:
    """v40: decide whether an executed step diverged from M2's prediction.

    Returns True (mismatch -> abort remaining plan) when ANY of:
      * observed.changed_cells == 0 (the click was a no-op when M2 expected
        a transition).
      * expected has a region_id and observed.primary_region_id differs
        (the click affected a different region than predicted).
      * expected has a dominant_transition and the observed transition's
        from/to disagrees (e.g., reversed toggle).
    Returns False when expected is empty/sparse (treat as no prediction
    -> never abort) or when the step matches.
    """
    if not isinstance(expected, dict) or not expected:
        return False
    if not isinstance(observed, dict):
        return True
    if int(observed.get("changed_cells", 0) or 0) == 0:
        return True
    exp_region = str(expected.get("region_id", "") or "").strip()
    if exp_region and exp_region != "_outside_":
        obs_region = str(observed.get("primary_region_id", "") or "").strip()
        if obs_region and obs_region != exp_region:
            return True
    exp_dt = expected.get("dominant_transition") or {}
    obs_dt = observed.get("dominant_transition") or {}
    if isinstance(exp_dt, dict) and isinstance(obs_dt, dict):
        try:
            ef, et = exp_dt.get("from"), exp_dt.get("to")
            of, ot = obs_dt.get("from"), obs_dt.get("to")
            if ef is not None and et is not None and of is not None and ot is not None:
                if int(ef) != int(of) or int(et) != int(ot):
                    return True
        except (TypeError, ValueError):
            pass
    return False


def _compute_marker_progress(visible_regions: list[dict]) -> dict:
    """v38: derive per-marker satisfaction directly from per_neighbor_target.

    A marker is satisfied iff all 8 directions have needs_toggle=False.
    Aggregates global progress so M1/M2/M3 can reason about whether the
    last click moved closer to win, not just whether the diff signature
    matched. This is pure data exposure — the same per_neighbor_target
    field already drives the click choice; we just count it.
    """
    markers: list[dict] = []
    total_unsat = 0
    for r in visible_regions or []:
        if not isinstance(r, dict):
            continue
        pnt = r.get("per_neighbor_target") or {}
        if not pnt:
            continue
        n_unsat = 0
        n_known = 0
        unsat_dirs: list[str] = []
        for d, info in pnt.items():
            nt = (info or {}).get("needs_toggle")
            if nt is True:
                n_unsat += 1
                unsat_dirs.append(d)
                n_known += 1
            elif nt is False:
                n_known += 1
            # nt is None → unknown (grid_sample missing); skip
        markers.append({
            "marker_id": r.get("id"),
            "n_unsatisfied": n_unsat,
            "n_known": n_known,
            "satisfied": (n_unsat == 0 and n_known > 0),
            "unsatisfied_dirs": unsat_dirs,
        })
        total_unsat += n_unsat
    n_total = len(markers)
    n_sat = sum(1 for m in markers if m["satisfied"])
    return {
        "markers": markers,
        "markers_total": n_total,
        "markers_satisfied": n_sat,
        "total_unsatisfied_neighbors": total_unsat,
        "win_when_zero": True,
    }


def _compute_joint_neighbors(visible_regions: list[dict]) -> list[dict]:
    """v38: detect Hkx neighbors shared across multiple markers.

    For each (neighbor_xy or neighbor_id), aggregate which markers depend
    on it and what target_color each requires. If multiple markers share
    a neighbor with DISAGREEING target_colors, that neighbor is a joint
    conflict — toggling it solves one marker but breaks another. M2 needs
    this to break the L4 inconclusive loop where every click that helps
    marker A regresses marker B.
    """
    by_key: dict[str, dict] = {}
    for r in visible_regions or []:
        if not isinstance(r, dict):
            continue
        mid = r.get("id")
        pnt = r.get("per_neighbor_target") or {}
        for d, info in pnt.items():
            if not isinstance(info, dict):
                continue
            nb_xy = info.get("neighbor_xy")
            nb_id = info.get("neighbor_id")
            key = None
            if isinstance(nb_xy, (list, tuple)) and len(nb_xy) == 2:
                key = f"xy:{int(nb_xy[0])},{int(nb_xy[1])}"
            elif nb_id:
                key = f"id:{nb_id}"
            if not key:
                continue
            entry = by_key.setdefault(key, {
                "key": key,
                "neighbor_xy": list(nb_xy) if isinstance(nb_xy, (list, tuple)) else None,
                "neighbor_id": nb_id,
                "depends": [],
            })
            entry["depends"].append({
                "marker_id": mid,
                "direction": d,
                "current_color": info.get("current_color"),
                "target_color": info.get("target_color"),
                "needs_toggle": info.get("needs_toggle"),
            })
    joints: list[dict] = []
    for entry in by_key.values():
        deps = entry["depends"]
        if len(deps) < 2:
            continue  # not shared
        targets = {d.get("target_color") for d in deps if d.get("target_color") is not None}
        conflict = len(targets) > 1
        any_unsat = any(d.get("needs_toggle") is True for d in deps)
        joints.append({
            "neighbor_xy": entry["neighbor_xy"],
            "neighbor_id": entry["neighbor_id"],
            "shared_by": [d["marker_id"] for d in deps],
            "target_colors_requested": sorted(t for t in targets if t is not None),
            "is_conflict": conflict,
            "any_marker_needs_toggle": any_unsat,
        })
    joints.sort(key=lambda j: (not j["is_conflict"], -len(j["shared_by"])))
    return joints


class ArcgenticaSimple(Arcgentica):
    """v14: three-isolated-subagent orchestrator with single-action M2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        namespace = os.environ.get("ARC_SIMPLE_NAMESPACE", "").strip()
        if not namespace:
            namespace = f"simple_{int(time.time())}_{os.getpid()}"
        root = Path(os.environ.get("ARC_SIMPLE_LOG_DIR", "simple_logs"))
        self.simple_namespace = namespace
        self.simple_workdir = root / self.game_id / namespace
        self.simple_workdir.mkdir(parents=True, exist_ok=True)
        self._full_reset_id: int = 0

    def _init_research_runtime(self) -> None:
        self.shared_namespace = None
        self.research_workdir = None
        self._registry = None
        self._research_scope = None

    def _research_active(self) -> bool:
        return False

    def _augment_system_prompt(self, system_prompt: str | None) -> str | None:
        return system_prompt

    def _make_listener(self):
        server = self._server
        tracker = self._tracker
        if server is None and tracker is None:
            return None
        return lambda: AgentListener(WsLogger(server, tracker=tracker))

    def _build_shared_objects(self):
        """Legacy hook kept for parent-contract compatibility (now unused)."""
        return None

    async def _delegate_once(
        self,
        system_prompt: str,
        return_type: Any,
        task: str,
        **objects: Any,
    ) -> Any:
        agent = await self.spawn_agent(system_prompt)
        return await agent.call(return_type, task, **objects)

    # ------------------------------------------------------------------
    # extract_observation: plan v11 §4.5 / v14
    # ------------------------------------------------------------------

    def extract_observation(
        self,
        prev_frame: Any,
        curr_frame: Any,
        choice: ChosenAction,
        semantic_packet: dict,
    ) -> dict:
        """Translate a (prev_frame, curr_frame, choice) tuple into a diff snapshot.

        Increments ``self._full_reset_id`` on the False->True ``full_reset`` edge.
        Resolves the primary region by inspecting the changed-cells centroid
        against the visible_regions list from the semantic packet.
        """
        prev_full_reset = bool(getattr(getattr(prev_frame, "_data", None), "full_reset", False))
        curr_full_reset = bool(getattr(getattr(curr_frame, "_data", None), "full_reset", False))
        if curr_full_reset and not prev_full_reset:
            self._full_reset_id += 1

        before_grid = _grid_to_list(getattr(prev_frame, "grid", []))
        after_grid = _grid_to_list(getattr(curr_frame, "grid", []))
        summary = _change_pattern_summary(before_grid, after_grid)

        visible_regions = (
            semantic_packet.get("visible_regions", [])
            if isinstance(semantic_packet, dict)
            else []
        )
        primary_region_id = "_outside_"
        if int(summary.get("changed_cells", 0) or 0) > 0:
            centroid = _changed_cells_centroid(before_grid, after_grid)
            if centroid is not None:
                for region in visible_regions:
                    if not isinstance(region, dict):
                        continue
                    if _point_in_bbox(centroid, region.get("bbox", {})):
                        primary_region_id = str(region.get("id", "_outside_"))
                        break

        prev_levels = int(getattr(prev_frame, "levels_completed", 0) or 0)
        curr_levels = int(getattr(curr_frame, "levels_completed", 0) or 0)
        action_label = (
            choice.action_sequence[0]
            if choice and choice.action_sequence
            else ""
        )

        # v24: absolute change_bbox enables click→Hkx mapping. Without this,
        # M2 cannot learn which click coords toggle which Hkx and must guess.
        abs_bbox = _change_bbox(before_grid, after_grid) if int(summary.get("changed_cells", 0) or 0) > 0 else None
        return {
            "action": action_label,
            "changed_cells": int(summary.get("changed_cells", 0) or 0),
            "dominant_transition": summary.get("dominant_transition"),
            "transitions": summary.get("transitions", []),
            "relative_bbox": summary.get("relative_bbox"),
            "change_bbox": abs_bbox,  # v24: absolute (min_x, min_y, max_x, max_y)
            "primary_region_id": primary_region_id,
            "level_delta": curr_levels - prev_levels,
            "full_reset_id": self._full_reset_id,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """v14 three-subagent orchestrator with single-action M2."""
        tracker = UsageTracker()
        self._tracker = tracker

        server: EventServer | None = None
        if self.visualize:
            server = EventServer(game_id=self.game_id)
            await server.start()
            self._server = server

        submit_action_raw, history, full_history = self._make_submit_action(server=server)

        initial_raw = self.take_action(GameAction.RESET)
        if initial_raw:
            self.append_frame(initial_raw)
        initial_frame = submit_action_raw("RESET")

        # SemanticPackets with _NullTrace shim (R8-#S1) and stub sprite supplier.
        semantic_packets = SemanticPackets(
            history,
            _NullTrace(),
            game_id=self.game_id,
            live_object_supplier=lambda limit=12: [],
        )

        self._full_reset_id = 0
        board = GoalBoard(self.simple_workdir)
        diff_memory = DiffMemory()

        prev_frame = initial_frame
        actions_used = 0

        while True:
            if actions_used >= self.MAX_ACTIONS:
                break
            latest_frame = self.frames[-1] if self.frames else initial_frame
            if (
                latest_frame.win_levels > 0
                and latest_frame.levels_completed >= latest_frame.win_levels
            ):
                break

            # v29: infer the toggle color cycle (gqb) from observed transitions
            # so per_neighbor_target uses correct target colors (e.g. level 1
            # of ft09 uses gqb=[9,12], not [9,8] like level 0).
            inferred_gqb = _infer_gqb_pair(diff_memory)
            packet = semantic_packets.current(gqb_pair=inferred_gqb)
            current_region_ids = {
                str(r.get("id", ""))
                for r in packet.get("visible_regions", [])
                if isinstance(r, dict) and r.get("id")
            }

            # ---- v41: Reflexion trigger ----
            # Decay stale buffer; if stagnation conditions met, run a single
            # Reflexion call and store the corrective text in board.
            board.clear_reflection_if_stale()
            if not _V41_LEGACY_M34 and board.should_reflect():
                try:
                    reflex_agent = await self.spawn_agent(REFLEXION_SYSTEM_PROMPT)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Reflexion spawn failed: %s", exc)
                    reflex_agent = None
                if reflex_agent is not None:
                    # Build a compact recent-trace input. Recent obs + clicks
                    # joined by index. lesson_seeds from M4 if present (legacy
                    # ablation may run with M4 enabled).
                    recent_obs = board.observation_log[-8:]
                    recent_choices = board.choice_history[-8:]
                    recent_clicks = []
                    for o, c in zip(recent_obs, recent_choices):
                        recent_clicks.append({
                            "action": (c.get("action_sequence") or [None])[0],
                            "primary_region_id": o.get("primary_region_id"),
                            "verdict": o.get("verdict"),
                        })
                    skill_anchor_hist = dict(Counter(
                        e.get("skill_anchor", "none")
                        for e in board.choice_history[-30:]
                    ))
                    lesson_seeds = []
                    seen_seeds = set()
                    for L in reversed(board.lesson_log[-12:]):
                        s = (getattr(L, "skill_seed", None) or "").strip()
                        if s and s not in seen_seeds:
                            seen_seeds.add(s)
                            lesson_seeds.append(s)
                        if len(lesson_seeds) >= 4:
                            break
                    # v41-b: enrich Reflexion input with state + skills + cards
                    # so corrective text is grounded in actual visible regions,
                    # active hypotheses, and prior wins.
                    region_summary = [
                        {
                            "id": r.get("id"),
                            "size": r.get("size"),
                            "is_multicolor": r.get("is_multicolor", False),
                            "has_per_neighbor_target": bool(r.get("per_neighbor_target")),
                            "dominant_color": r.get("dominant_color"),
                        }
                        for r in packet.get("visible_regions", [])
                        if isinstance(r, dict) and r.get("id")
                    ]
                    active_card_summary = [
                        {
                            "id": c.id,
                            "predicate": (c.predicate or "")[:160],
                            "expected_region": (c.expected_signature or {}).get("region_id"),
                            "expected_transition": (c.expected_signature or {}).get(
                                "dominant_transition"
                            ),
                        }
                        for c in board.cards[:5]
                    ]
                    falsified_summary = [
                        {
                            "id": f.get("id"),
                            "predicate": (f.get("predicate") or "")[:160],
                            "axis_failed": (f.get("falsified_by") or {}).get("axis", "unknown"),
                        }
                        for f in board.falsified_cards[-3:]
                    ]
                    refl_marker_progress = _compute_marker_progress(
                        packet.get("visible_regions", [])
                    )
                    refl_joint = _compute_joint_neighbors(
                        packet.get("visible_regions", [])
                    )
                    reflex_input = {
                        "recent_obs": [
                            {
                                "card_id": e.get("card_id"),
                                "primary_region_id": e.get("primary_region_id"),
                                "changed_cells": e.get("changed_cells"),
                                "dominant_transition": e.get("dominant_transition"),
                                "level_delta": e.get("level_delta"),
                                "verdict": e.get("verdict"),
                            }
                            for e in recent_obs
                        ],
                        "recent_clicks": recent_clicks,
                        "stagnation_window": board.stagnation_window,
                        "skill_anchor_histogram": skill_anchor_hist,
                        "lesson_seeds": lesson_seeds,
                        "marker_progress": refl_marker_progress,
                        # v41-b: state visibility
                        "visible_regions_summary": region_summary,
                        "joint_neighbors": refl_joint[:8],  # cap for prompt size
                        # v41-b: skills + cards + cross-level memory
                        "top_skills": board.top_skills(3),
                        "cross_level_confirmed": list(
                            board.cross_level_confirmed[-5:]
                        ),
                        "active_cards": active_card_summary,
                        "falsified_recent": falsified_summary,
                    }
                    try:
                        raw = await parse_or_retry(
                            reflex_agent,
                            REFLEXION_TASK_INSTRUCTIONS
                            + "\n\nINPUT:\n"
                            + json.dumps(reflex_input, default=str),
                            lambda d: d if isinstance(d, dict) else None,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Reflexion call failed: %s", exc)
                        raw = None
                    if isinstance(raw, dict):
                        corrective = str(raw.get("corrective", "") or "").strip()
                        if corrective:
                            board.set_reflection(corrective)
                            logger.info(
                                "v41 reflexion set (turn=%d, stag=%d): %s",
                                board.turn_index, board.stagnation_window,
                                corrective[:160],
                            )

            # ---- M1: hypothesis cards ----
            try:
                m1_agent = await self.spawn_agent(M1_SYSTEM_PROMPT)
            except Exception as exc:  # noqa: BLE001
                logger.warning("M1 spawn failed: %s", exc)
                consumed, prev_frame = await self._fallback_explore(
                    submit_action_raw, packet, board, prev_frame
                )
                actions_used += consumed
                continue
            # Sort visible_regions by size desc (largest first) so M1 sees the
            # likely play area at the top of the list. Also flag edge_touch and
            # thin_edge_like so M1 can deprioritize indicator strips.
            sorted_regions_raw = sorted(
                packet.get("visible_regions", []),
                key=lambda r: (-int(r.get("size", 0) or 0),
                               r.get("id", "")),
            )
            # Annotate with click_response so M1 can label markers vs targets
            # without mutating the underlying packet (state.py reuses it).
            sorted_regions: list[dict] = []
            for r in sorted_regions_raw:
                if not isinstance(r, dict):
                    continue
                rid = str(r.get("id", ""))
                stats = board.region_click_stats.get(rid, {})
                clicks = int(stats.get("clicks", 0) or 0)
                responses = int(stats.get("responses", 0) or 0)
                annotated = dict(r)
                annotated["click_response"] = {
                    "clicks": clicks,
                    "responses": responses,
                    "likely_marker": clicks >= 2 and responses == 0,
                    "likely_target": responses > 0,
                }
                sorted_regions.append(annotated)
            # Indicator pattern: SMALL upper-row regions (y < 15) sorted by x.
            # These typically encode the win-state target color sequence in
            # ARC-AGI-3 puzzles. Surface them explicitly so M1 can hypothesize
            # the play area should match this pattern.
            indicator_pattern = []
            for r in sorted(packet.get("visible_regions", []),
                            key=lambda r: (r.get("bbox", {}).get("min_y", 99),
                                           r.get("bbox", {}).get("min_x", 99))):
                bbox = r.get("bbox", {})
                size = int(r.get("size", 0) or 0)
                # heuristic: small (<200 cells) AND in upper third of grid
                if size > 0 and size < 200 and bbox.get("min_y", 99) < 15:
                    indicator_pattern.append({
                        "id": r.get("id"),
                        "dominant_color": r.get("dominant_color"),
                        "x_position": bbox.get("min_x"),
                    })
            # v20: per-turn novelty = facts seen in last 10 turns whose
            # last_seen_turn >= turn_index - 10. fresh_facts is a SET so
            # precision_score / M1 can compare cheaply.
            fresh_facts = {
                k for k, t in board.discovered_facts.items()
                if t >= max(1, board.turn_index - 4)  # last 5 turns
            }
            falsified_recent = [
                str(f.get("predicate", ""))
                for f in board.falsified_cards[-12:]
                if f.get("predicate")
            ]
            # v38: marker-progress and joint-neighbor maps. Same per_neighbor_target
            # already used by M2 for click choice, just aggregated so the LLMs can
            # reason about *progress toward win* (markers_satisfied/total) and
            # *joint conflicts* (one Hkx that multiple markers depend on with
            # different target colors → unsolvable in one click). This is what
            # L4 needs: the inconclusive loop comes from L4's joint constraint
            # invisible at single-click level.
            marker_progress = _compute_marker_progress(packet.get("visible_regions", []))
            joint_neighbors = _compute_joint_neighbors(packet.get("visible_regions", []))
            # v44: PROACTIVE novel sprite detection. Compare each visible
            # region's dominant_color against the union of colors observed
            # in cross_level_confirmed transitions (= colors we've already
            # learned to behave as standard Hkx in prior levels). Regions
            # with novel dominant_color are flagged so M1 emits exploration
            # cards on them BEFORE assuming standard Hkx mechanics.
            # Reactive anomaly (changed_cells>50 after click) already exists;
            # this complements it with pre-click detection.
            known_colors: set[int] = set()
            for clc in board.cross_level_confirmed:
                dt = (clc.get("expected_signature") or {}).get("dominant_transition") or {}
                if isinstance(dt, dict):
                    for k in ("from", "to"):
                        v = dt.get(k)
                        if isinstance(v, int):
                            known_colors.add(v)
            novel_region_ids: list[dict] = []
            if known_colors:  # only meaningful after we have prior-level evidence
                for r in packet.get("visible_regions", []):
                    if not isinstance(r, dict):
                        continue
                    dc = r.get("dominant_color")
                    rid = r.get("id")
                    if rid and isinstance(dc, int) and dc not in known_colors:
                        novel_region_ids.append({
                            "id": rid,
                            "dominant_color": dc,
                            "size": r.get("size"),
                            "is_multicolor": r.get("is_multicolor", False),
                        })
            m1_input = {
                "grid_render": packet.get("render_rows", {}),
                "visible_regions": sorted_regions,
                "indicator_pattern": indicator_pattern,
                "arc_archetypes": _ARC_ARCHETYPES,
                "diff_memory": diff_memory.snapshot_for_m1(),
                "promoted_skills": board.top_skills(3),
                # v32 Fix D: confirmed predicates from prior levels — positive
                # transfer signal so M1 can ANALOGISE rather than rediscover.
                "cross_level_confirmed": list(board.cross_level_confirmed[-10:]),
                "marker_progress": marker_progress,
                "joint_neighbors": joint_neighbors,
                # v20 additions
                "falsified_predicates_recent": falsified_recent,
                "stagnation_window": board.stagnation_window,
                "discovered_facts_summary": {
                    "regions": sorted({v for (k, v) in board.discovered_facts if k == "region"}),
                    "transitions": sorted(
                        list({v for (k, v) in board.discovered_facts if k == "transition"})
                    ),
                    "colors": sorted({v for (k, v) in board.discovered_facts if k == "color"}),
                    "fresh_count_last5": len(fresh_facts),
                },
                "archetype_stagnation": dict(board.archetype_stagnation),
                # v21: recent M4 lessons — gives M1 explicit rationale to act on.
                "recent_lessons": [asdict(L) for L in board.lesson_log[-5:]],
                # v44: novel sprite candidates — regions whose dominant_color
                # has not appeared in any cross_level_confirmed transition.
                # Empty list pre-L1 (no prior colors yet) or when current
                # frame contains no new colors.
                "novel_region_ids": novel_region_ids,
                "known_colors_so_far": sorted(known_colors),
                # v41-b: reflexion buffer — corrective from prior stagnation.
                # Empty string when no buffer is active. M1 must respect this
                # when generating new cards (e.g., reflexion says "stop
                # archetype cards on R28" → M1 omits color_cycle archetype
                # cards targeting R28 from this batch).
                "reflexion_buffer": board.reflexion_buffer,
                # v42 (Q3): self-memory — M1 sees its OWN previous turn cards
                # AND the actual outcome of M2's pick on those cards. Lets M1
                # do REFLECT-DECIDE: which of my cards landed? which axis
                # missed? then emit a refined batch.
                "last_self_cards": list(board.last_m1_cards or []),
                "last_outcome": board.last_outcome or {},
                "last_m2_pick": board.last_m2_choice or {},
            }
            m1_task_v1 = (
                M1_TASK_INSTRUCTIONS + "\n\nINPUT:\n" + json.dumps(m1_input, default=str)
            )
            cards = await parse_with_critic(m1_agent, m1_task_v1, parse_card_list, num_passes=1)
            # Reflexion: audit each card and ask M1 to revise weak ones.
            # MODULAR per-card check — no new module, same M1 sub-agent, 2-pass.
            if cards:
                v1_audit = []
                for c in cards:
                    sig = c.expected_signature or {}
                    issues = []
                    if not sig:
                        issues.append("expected_signature is empty")
                    if not sig.get("is_target_state") and sig.get("level_delta") in (None, 0):
                        # Not a goal card; mechanism cards are OK without target_state
                        pass
                    if not sig.get("region_id") and sig:
                        issues.append("expected_signature.region_id missing")
                    if issues:
                        v1_audit.append({"id": c.id, "issues": issues, "predicate": c.predicate[:120]})
                goal_count_v1 = sum(
                    1 for c in cards
                    if (c.expected_signature or {}).get("is_target_state")
                    or (c.expected_signature or {}).get("level_delta") not in (None, 0)
                )
                # If <2 goal cards or any audited issues, do reflexion call
                if goal_count_v1 < 2 or v1_audit:
                    feedback = {
                        "card_count": len(cards),
                        "goal_card_count": goal_count_v1,
                        "missing_target_state_cards": max(0, 2 - goal_count_v1),
                        "audit_issues": v1_audit,
                        "instruction": (
                            "Review your previous cards above. Issues: too few goal cards, "
                            "and some have missing/empty expected_signature fields. "
                            "Re-emit a CORRECTED JSON array with: "
                            "(1) at least 2 cards having is_target_state=true and level_delta=1, "
                            "(2) every card has fully populated expected_signature with region_id, "
                            "(3) goal cards reference indicator_pattern from INPUT explicitly. "
                            "Output ONLY the revised JSON array."
                        ),
                    }
                    m1_task_v2 = m1_task_v1 + (
                        "\n\nPREVIOUS_OUTPUT:\n"
                        + json.dumps([asdict(c) for c in cards], default=str)
                        + "\n\nREFLEXION_FEEDBACK:\n"
                        + json.dumps(feedback, default=str)
                    )
                    revised = await parse_or_retry(m1_agent, m1_task_v2, parse_card_list)
                    if revised:
                        cards = revised
            if not cards:
                consumed, prev_frame = await self._fallback_explore(
                    submit_action_raw, packet, board, prev_frame
                )
                actions_used += consumed
                continue
            cards = [c for c in cards if validate_card(c, current_region_ids)]
            # v20: hard jaccard filter — drop any new card whose predicate
            # has token-jaccard > 0.7 with any of the recent falsified ones.
            # Calibrated against cycle-10 falsified set (12 paraphrase pairs
            # in the 0.70-0.85 band; 0.7 is the cleanest cut). Soft penalty
            # for the 0.5-0.7 band lives in precision_score.
            if falsified_recent:
                cards = [
                    c for c in cards
                    if max(
                        (jaccard_words(c.predicate, fp) for fp in falsified_recent),
                        default=0.0,
                    ) <= 0.7
                ]
            # Post-filter for type balance: enforce mechanism_count <= goal_count
            # so M2 doesn't drown in mechanism cards. Goal cards are
            # is_target_state OR have level_delta != 0 set.
            def _is_goal(c):
                sig = (c.expected_signature or {})
                return bool(sig.get("is_target_state")) or (
                    sig.get("level_delta") not in (None, 0)
                )
            goal_cards = [c for c in cards if _is_goal(c)]
            mech_cards = [c for c in cards if not _is_goal(c)]
            # Drop excess mechanism cards beyond max(1, len(goal_cards))
            cap = max(1, len(goal_cards))
            mech_cards = mech_cards[:cap]
            cards = goal_cards + mech_cards
            if not cards:
                consumed, prev_frame = await self._fallback_explore(
                    submit_action_raw, packet, board, prev_frame
                )
                actions_used += consumed
                continue
            for c in cards:
                c.precision_score = precision_score(
                    c,
                    current_region_ids,
                    falsified_predicates=falsified_recent,
                    fresh_facts=fresh_facts,
                )
            cards.sort(key=lambda c: c.precision_score, reverse=True)
            # v42 (Q3): snapshot M1's emission so NEXT turn's M1 sees its own
            # prior cards. Combined with last_outcome (set after step-loop)
            # this gives M1 the "what I said vs what happened" reflection.
            board.record_m1_emit(cards)

            # ---- M2: single action ----
            try:
                m2_agent = await self.spawn_agent(M2_SYSTEM_PROMPT)
            except Exception as exc:  # noqa: BLE001
                logger.warning("M2 spawn failed: %s", exc)
                consumed, prev_frame = await self._fallback_explore(
                    submit_action_raw, packet, board, prev_frame
                )
                actions_used += consumed
                continue
            m2_input = {
                "cards": [asdict(c) for c in cards],
                "tried_coords": [list(t) for t in board.tried_coords],
                "visible_regions": packet.get("visible_regions", []),
                # v24: chain-connection telemetry — lets M2 learn click→Hkx
                # mapping and avoid gap zones (changed_cells=0 misses).
                "click_history": diff_memory.click_history_for_m2(last_n=30),
                "hkx_states": diff_memory.known_hkx_states(),
                # v34 framework: M2 sees skills with explicit S1..S5 IDs so
                # the M2 output's `skill_anchor` field can reference them.
                "active_skills": [
                    {"id": f"S{i+1}", **sk}
                    for i, sk in enumerate(board.top_skills(5))
                ],
                "marker_progress": marker_progress,
                "joint_neighbors": joint_neighbors,
                # v41: corrective text from Reflexion module. Empty string when
                # no buffer is active. M2 prompt rule treats this as MUST-FOLLOW
                # imperative when present.
                "reflexion_buffer": board.reflexion_buffer,
                # v41-b: cross-level memory so M2 can directly reuse confirmed
                # mechanics from prior levels (e.g., "L1 confirmed R10 click
                # toggles 14->15") without waiting for M1 to re-emit a card.
                "cross_level_confirmed": list(board.cross_level_confirmed[-5:]),
                # v42 (Q3): self-memory — M2 sees its OWN previous turn plan
                # AND the actual per-step outcome. Includes plan_emitted vs
                # plan_executed (abort detection), expected_step_diffs vs
                # observed regions/transitions, plan_aborted flag.
                "last_self_emit": board.last_m2_choice or {},
                "last_outcome": board.last_outcome or {},
            }
            m2_task = M2_TASK_INSTRUCTIONS
            if board.reflexion_buffer:
                m2_task = (
                    "REFLEXION (must-follow corrective from prior stagnation):\n"
                    f"  >>> {board.reflexion_buffer}\n\n"
                    + M2_TASK_INSTRUCTIONS
                )
            choice = await parse_with_critic(
                m2_agent,
                m2_task + "\n\nINPUT:\n" + json.dumps(m2_input, default=str),
                lambda d: ChosenAction.from_dict(d),
                num_passes=1,
            )
            if not choice or not choice.action_sequence:
                consumed, prev_frame = await self._fallback_explore(
                    submit_action_raw, packet, board, prev_frame
                )
                actions_used += consumed
                continue
            # v40: cap multi-step plan to 3 actions. If M2 emitted >1 step
            # but didn't supply matching expected_step_diffs, fall back to
            # single-step (data-flow contract violation = unsafe to chain).
            plan = list(choice.action_sequence[:3])
            expected_steps = list(choice.expected_step_diffs or [])
            if len(plan) > 1 and len(expected_steps) < len(plan):
                logger.info(
                    "v40: plan length %d but only %d expected_step_diffs; truncating to 1",
                    len(plan), len(expected_steps),
                )
                plan = plan[:1]
                expected_steps = []
            choice.action_sequence = plan
            choice.expected_step_diffs = expected_steps

            chosen_card = next((c for c in cards if c.id == choice.card_id), cards[0])
            current_marker_progress = marker_progress
            verdict = "inconclusive"  # last step's verdict (used for M3 trigger below)
            observed = None
            plan_aborted = False
            steps_executed = 0
            # v42 (Q3): per-step trace for record_turn_outcome.
            per_step_observed: list[dict] = []
            per_step_verdict: list[str] = []
            plan_executed_strs: list[str] = []

            for step_idx, action_str in enumerate(plan):
                name, x, y = _parse_action_str(action_str)
                try:
                    if name == "RESET":
                        new_frame = submit_action_raw(name)
                    else:
                        new_frame = submit_action_raw(name, x, y)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "submit_action_raw raised %s for %s; abort plan", exc, action_str
                    )
                    plan_aborted = True
                    break
                curr_frame = new_frame
                actions_used += 1
                steps_executed += 1
                board.tried_coords.add((name, x, y))

                # ---- Observe + falsifier verdict (per step) ----
                observed = self.extract_observation(
                    prev_frame, curr_frame, choice, packet
                )
                diff_memory.append(observed)
                # v39: compute post-step marker_progress against current_marker_progress
                # (which is the BEFORE-step value for this step).
                try:
                    packet_after = semantic_packets.current(gqb_pair=inferred_gqb)
                    marker_progress_after = _compute_marker_progress(
                        packet_after.get("visible_regions", [])
                    )
                    marker_progress_delta = (
                        int(current_marker_progress.get("total_unsatisfied_neighbors", 0))
                        - int(marker_progress_after.get("total_unsatisfied_neighbors", 0))
                    )
                except Exception:  # noqa: BLE001
                    marker_progress_after = current_marker_progress
                    marker_progress_delta = None
                # v20: stagnation-falsify telemetry.
                chosen_aid = archetype_alignment(chosen_card)
                arch_stag = (
                    board.archetype_stagnation.get(chosen_aid, 0) if chosen_aid else 0
                )
                arch_fals = (
                    sum(
                        1 for f in board.falsified_cards
                        if chosen_aid and chosen_aid in str(f.get("predicate", "")).lower()
                    ) if chosen_aid else 0
                )
                arch_conf = (
                    sum(
                        1 for e in board.observation_log
                        if e.get("verdict") == "confirm"
                        and chosen_aid and chosen_aid in str(e.get("predicate", "")).lower()
                    ) if chosen_aid else 0
                )
                verdict = evaluate_falsifier(
                    chosen_card, choice, observed,
                    archetype_stagnation_window=arch_stag,
                    archetype_falsified_count=arch_fals,
                    archetype_confirm_count=arch_conf,
                    marker_progress_delta=marker_progress_delta,
                )
                clicked_region = _region_for_coord(
                    name, x, y, packet.get("visible_regions", [])
                )
                board.update(
                    cards, choice, observed, verdict, clicked_region=clicked_region
                )
                # v42 (Q3): track per-step trace for next-turn self-reflection.
                per_step_observed.append(observed or {})
                per_step_verdict.append(verdict)
                plan_executed_strs.append(action_str)
                if verdict == "falsify" and (
                    chosen_card.expected_signature or {}
                ).get("is_archetype") and chosen_aid:
                    evicted = board.evict_skills_matching_archetype(chosen_aid)
                    if evicted:
                        logger.info(
                            "v20 stagnation: archetype %s falsified, evicted %d skill(s)",
                            chosen_aid, evicted,
                        )

                # ---- v21: M4 reflector (per step) ----
                # v41: gated. Default off; reflexion module replaces this path.
                if _V41_LEGACY_M34 and actions_used > 2:
                    try:
                        m4_agent = await self.spawn_agent(M4_SYSTEM_PROMPT)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("M4 spawn failed: %s", exc)
                        m4_agent = None
                    if m4_agent is not None:
                        m4_input = {
                            "chosen_card": {
                                "id": chosen_card.id,
                                "predicate": chosen_card.predicate,
                                "expected_signature": chosen_card.expected_signature,
                            },
                            "observed": observed,
                            "verdict": verdict,
                            "recent_falsified": [
                                {"predicate": f.get("predicate", "")[:200]}
                                for f in board.falsified_cards[-3:]
                            ],
                            "turn_index": board.turn_index,
                        }
                        lesson = await parse_or_retry(
                            m4_agent,
                            M4_TASK_INSTRUCTIONS
                            + "\n\nINPUT:\n"
                            + json.dumps(m4_input, default=str),
                            parse_lesson_card,
                        )
                        if lesson:
                            board.add_lesson(lesson)

                prev_frame = curr_frame
                current_marker_progress = marker_progress_after

                # ---- v40: mismatch / level-rise check before next step ----
                if step_idx + 1 < len(plan):
                    exp_step = (
                        expected_steps[step_idx]
                        if step_idx < len(expected_steps)
                        else (choice.expected_diff_signature or {})
                    )
                    if _step_mismatch(observed, exp_step):
                        logger.info(
                            "v40 abort: step %d/%d mismatch exp=%s obs_region=%s obs_dt=%s",
                            step_idx + 1, len(plan), exp_step,
                            observed.get("primary_region_id"),
                            observed.get("dominant_transition"),
                        )
                        plan_aborted = True
                        break
                    if int(observed.get("level_delta", 0) or 0) > 0:
                        logger.info(
                            "v40 stop: level rose at step %d/%d, dropping remaining",
                            step_idx + 1, len(plan),
                        )
                        break

            if steps_executed == 0:
                # All steps failed at submit_action_raw — fall back to explore.
                consumed, prev_frame = await self._fallback_explore(
                    submit_action_raw, packet, board, prev_frame
                )
                actions_used += consumed
                continue
            if plan_aborted and steps_executed < len(plan):
                logger.info(
                    "v40: plan aborted after %d/%d steps", steps_executed, len(plan)
                )
            # v42 (Q3): snapshot turn outcome so NEXT turn's M2 sees its own
            # plan vs actual. Combined with last_m1_cards (set above) this
            # gives both modules "self memory".
            board.record_turn_outcome(
                choice,
                plan_executed_strs,
                per_step_observed,
                per_step_verdict,
                plan_aborted,
            )

            # ---- M3: analogy compression ----
            # v41: gated. Default off (skill bank promote=0 in cycle33 across
            # 85 turns; LLM cost 36 calls of skip/paraphrase). Will be replaced
            # by ContractSkill module in a later step.
            if _V41_LEGACY_M34 and (
                (verdict == "falsify" and actions_used >= 5)
                or (actions_used % 3 == 0 and actions_used >= 5)
            ):
                try:
                    m3_agent = await self.spawn_agent(M3_SYSTEM_PROMPT)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("M3 spawn failed: %s", exc)
                    m3_agent = None
                if m3_agent is not None:
                    # v32 Fix E/F: feed M3 mechanic-rich payload so it can
                    # anchor skill in game nouns (marker/mask/neighbor) and
                    # generalise across confirmed predicates from prior levels.
                    vr_summary = []
                    for r in packet.get("visible_regions", []):
                        if not isinstance(r, dict):
                            continue
                        if r.get("is_multicolor") or r.get("per_neighbor_target"):
                            vr_summary.append({
                                "id": r.get("id"),
                                "is_multicolor": r.get("is_multicolor"),
                                "bsT_center_color": r.get("bsT_center_color"),
                                "per_neighbor_target": r.get("per_neighbor_target"),
                            })
                    # v35 framework: derive skill usage from choice_history.
                    # Skills with usage_count=0 over recent window are dead
                    # — M3 sees this and naturally varies its next emission.
                    skill_usage = Counter(
                        e.get("skill_anchor", "none")
                        for e in board.choice_history[-30:]
                    )
                    m3_input = {
                        "obs_log": board.observation_log_summary(),
                        "promoted_skills": board.top_skills(5),
                        "fresh_facts_last5_count": sum(
                            1 for _, t in board.discovered_facts.items()
                            if t >= max(1, board.turn_index - 4)
                        ),
                        "lesson_log": [asdict(L) for L in board.lesson_log[-15:]],
                        "visible_regions_summary": vr_summary,
                        "cross_level_confirmed": list(board.cross_level_confirmed[-10:]),
                        "skill_usage_last_30_actions": dict(skill_usage),
                        "marker_progress": marker_progress,
                        "joint_neighbors": joint_neighbors,
                    }
                    skill = await parse_with_critic(
                        m3_agent,
                        M3_TASK_INSTRUCTIONS
                        + "\n\nINPUT:\n"
                        + json.dumps(m3_input, default=str),
                        lambda d: AbstractSkill.from_dict(d),
                        num_passes=1,
                    )
                    if skill:
                        board.add_skill(skill)

            prev_frame = curr_frame

        # Persist final summary
        try:
            (self.simple_workdir / "summary.json").write_text(
                json.dumps(board.summary(), indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary.json write failed: %s", exc)

    # ------------------------------------------------------------------
    # Fallback exploration
    # ------------------------------------------------------------------

    async def _fallback_explore(
        self, submit_fn, packet: dict, board: GoalBoard, prev_frame: Any
    ) -> tuple[int, Any]:
        """Coordinate-driven fallback. Returns (1, latest_frame).

        The returned frame is the post-fallback ``Frame`` so the orchestrator
        can refresh ``prev_frame`` and avoid stale-diff corruption (W1).
        Returns ``prev_frame`` unchanged if every submit attempt raised.
        """
        regions = sorted(
            [r for r in packet.get("visible_regions", []) if isinstance(r, dict)],
            key=lambda r: -int(r.get("size", 0) or 0),
        )
        for region in regions:
            bbox = region.get("bbox", {})
            if not isinstance(bbox, dict):
                continue
            cx = (int(bbox.get("min_x", 0)) + int(bbox.get("max_x", 0))) // 2
            cy = (int(bbox.get("min_y", 0)) + int(bbox.get("max_y", 0))) // 2
            if ("ACTION6", cx, cy) not in board.tried_coords:
                board.tried_coords.add(("ACTION6", cx, cy))
                try:
                    new_frame = submit_fn("ACTION6", cx, cy)
                    return (1, new_frame)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fallback submit_fn raised %s", exc)
                    continue
        cx, cy = random.randint(0, 63), random.randint(0, 63)
        board.tried_coords.add(("ACTION6", cx, cy))
        try:
            new_frame = submit_fn("ACTION6", cx, cy)
            return (1, new_frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fallback random submit_fn raised %s", exc)
            return (1, prev_frame)
