"""v57 agent: Action-as-main + Hypothesize sub + Reflexion compressor.

Per-turn flow:
  1. board collects state (image + visible_regions + click_history).
  2. If active hypotheses < 3: call HYPOTHESIZE (sub-agent).
  3. Action picks next coord (LLM call 1× per turn).
  4. Execute ACTION6, collect observation, update board.
  5. Every 5 turns OR on level rise: call REFLEXION to compress
     verbose history into a tight summary that future Action turns use
     INSTEAD OF the raw log (context-bound architecture).
  6. Append one line to trace.jsonl per turn (immediate fsync).

This module reuses agentica_simple/state.py for visible_regions and
diff_memory.py for click_history bookkeeping. It does NOT reuse the
M1-M4 multi-module pipeline — v57 is a clean rewrite focused on
context compression.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from agents.templates.agentica.agent import Arcgentica
from agents.templates.agentica.compat import spawn  # TRAPI bridge
from tools.chain_compress import compress_action_state_chain  # v588 B16
from tools.candidate_generator import (  # v589 B17
    generate_candidates,
    update_candidate_log_with_observation,
)

from .prompts import (
    ACTION_SYSTEM_PROMPT,
    ACTION_TASK_INSTRUCTIONS,
    HYPOTHESIZE_SYSTEM_PROMPT,
    HYPOTHESIZE_TASK_INSTRUCTIONS,
    REFLEXION_SYSTEM_PROMPT,
    REFLEXION_TASK_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)

_REFLEXION_CADENCE = int(os.environ.get("V57_REFLEXION_CADENCE", "5"))
_HYPOTHESIS_POOL_TARGET = int(os.environ.get("V57_HYPOTHESIS_POOL_TARGET", "3"))
_VERBOSE_WINDOW = int(os.environ.get("V57_VERBOSE_WINDOW", "3"))
_MAX_ACTIONS = int(os.environ.get("ARC_SMOKE_MAX_ACTIONS", "32"))
# H_RECYCLE: after K consecutive inconclusive verdicts, evict the
# most-tested card so HYPOTHESIZE can re-fire and inject fresh cards.
# Min tested_count required for eviction (don't drop unproven cards).
_INCONCLUSIVE_EVICT_K = int(os.environ.get("V57_INCONCLUSIVE_EVICT_K", "5"))
_EVICT_MIN_TESTED = int(os.environ.get("V57_EVICT_MIN_TESTED", "3"))
# v587 B14: hierarchical memory parameters (empirically derived from
# tools/analyze_lp_events.py — see reports/plan_v587_buckets.json).
_TURN_DIFF_MAX_RAW = int(os.environ.get("V57_TURN_DIFF_MAX_RAW", "24"))
_K_STUCK = int(os.environ.get("V57_K_STUCK", "15"))
_STUCK_HYSTERESIS_OFF = int(os.environ.get("V57_STUCK_HYSTERESIS_OFF", "7"))
_CLUSTER_OBS_CAP = int(os.environ.get("V57_CLUSTER_OBS_CAP", "5"))
_CPSR_K_PER_CLASS_DEFAULT = 1
_CPSR_K_PER_CLASS_PROMOTED = 2
_CPSR_K_PROMOTE_THRESHOLD = 10  # promote to k=2 when ≥10 segments per class
# v588 round-2 review D-3: separate chain-verbose buffer (24 entries
# of trimmed verbose: turn + obs + expected_observation only). Avoids
# bloating M1 prompt while feeding compute_prediction_errors a deeper
# history.
_CHAIN_VERBOSE_WINDOW = int(os.environ.get("V57_CHAIN_VERBOSE_WINDOW", "24"))
# v588 round-2 review N-2: chain_rule_log cap. LRU-by-confirmed
# eviction (not pure FIFO — preserves rules that are accumulating
# evidence even if old).
_CHAIN_RULE_LOG_MAX = int(os.environ.get("V57_CHAIN_RULE_LOG_MAX", "30"))
# v589 B17: with typed-candidate path live, M3 dormant by default.
# Set V57_LEGACY_M3=1 to re-enable old M3 spawn (back-compat).
_V57_LEGACY_M3 = int(os.environ.get("V57_LEGACY_M3", "0")) == 1
# Per round-3 C-12 + plan §4.1: per-marker beam=2, global beam=8.
_CAND_K_PER_MARKER = int(os.environ.get("V57_CAND_K_PER_MARKER", "2"))
_CAND_K_GLOBAL = int(os.environ.get("V57_CAND_K_GLOBAL", "8"))
# Per round-3 C-13: role_history.json race-safe persistence.
# Per round-2 C-3 + §4.2 verdict table: max in-flight candidate log.
_CANDIDATE_LOG_MAX = int(os.environ.get("V57_CANDIDATE_LOG_MAX", "40"))
# Per round-3 C-9 (M1 textual compliance): re-emit ignored candidates
# up to N times before dropping. M4 enforces alignment.
_CANDIDATE_REEMIT_MAX = int(os.environ.get("V57_CANDIDATE_REEMIT_MAX", "3"))


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove first fence line and trailing fence.
        lines = text.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
    return text


def _compact_regions(regions: list[dict]) -> list[dict]:
    """Strip visible_regions to essential fields — context-budget critical.

    Game-AGNOSTIC. Keeps purely descriptive fields:
      id, bbox=[min_x,min_y,max_x,max_y], size, dominant_color,
      is_multicolor, y_band, click_response, is_marker_neighbor
      (derived: appears in some multicolor region's neighbors_3x3),
      neighbors_3x3 (multicolor only), crop (multicolor only — pixel
      pattern that may encode game-specific structure).
    Drops: width/height, color histograms, edge flags."""
    # First pass: collect R-ids that appear as geometric neighbors of any
    # multicolor region. In many ARC games these are the most likely
    # interactive cells, but the prompt treats this only as a prior — the
    # agent still tests the assumption via clicks.
    marker_neighbors: set[str] = set()
    for r in regions:
        if r.get("is_multicolor"):
            n3 = r.get("neighbors_3x3") or {}
            for rid in n3.values():
                if isinstance(rid, str):
                    marker_neighbors.add(rid)

    # cycle88: REVERT primary_marker enforcement. cycle86/87 attempts to
    # deterministically pick "the right" multicolor were leaning toward
    # task-specific priors that risked guiding the agent. Reverting to
    # cycle83-style: LLM picks multicolor itself via prompt heuristics,
    # accept variance as inherent to no-leak exploration.
    primary_marker_id: str | None = None

    compact = []
    for r in regions[:30]:
        b = r.get("bbox") or {}
        entry = {
            "id": r.get("id"),
            "bbox": [b.get("min_x"), b.get("min_y"), b.get("max_x"), b.get("max_y")],
            "size": r.get("size"),
            "color": r.get("dominant_color"),
            "is_multicolor": r.get("is_multicolor", False),
            "y_band": r.get("y_band"),
            "click_response": r.get("click_response") or {"clicks": 0, "responses": 0},
            "is_marker_neighbor": r.get("id") in marker_neighbors,
            "is_primary_marker": r.get("id") == primary_marker_id,
        }
        if r.get("is_multicolor"):
            n3 = r.get("neighbors_3x3")
            if n3:
                entry["neighbors_3x3"] = n3
            crop = r.get("crop")
            if crop and len(crop) <= 8:
                entry["crop"] = crop
        compact.append(entry)
    return compact


def _safe_json_parse(text: str, *, default: Any = None) -> Any:
    """Tolerant JSON parser for LLM outputs — strips fences, finds first {}."""
    text = _strip_json_fence(text)
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall through to fragment extraction.
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return default
    return default


# ---------------------------------------------------------------------------
# v587 B13: Level-bridge composition + anonymised priors.
# ---------------------------------------------------------------------------


def _compose_level_bridge_entry(
    *, board: "V57Board", from_level_delta: int, recent_diffs_serialised: list[dict]
) -> dict | None:
    """Distil the most recent turn-diffs into a level-bridge entry.
    Anti-leak: store only deterministic structural metrics + a short
    abstract_signature constructed from those metrics. Never quote
    LLM thoughts, rule names, or game-specific vocab."""
    if not recent_diffs_serialised:
        return None
    # Use only the diffs from THIS level (since last L+ event in our
    # buffer). recent_turn_diffs is at most 6 entries — for L0→L1 the
    # whole buffer is "this level". For higher levels we just take the
    # tail.
    diffs = recent_diffs_serialised[-15:]
    click_count = len(diffs)
    region_ids = [d.get("click_region_id") for d in diffs if d.get("click_region_id")]
    valid_ids = [r for r in region_ids if r and r != "_outside_"]
    unique_region_ids = list(dict.fromkeys(valid_ids))
    repeat_clicks = len(valid_ids) - len(unique_region_ids)
    kind_counts: dict[str, int] = {}
    for d in diffs:
        k = d.get("region_kind_pre") or "outside"
        kind_counts[k] = kind_counts.get(k, 0) + 1
    compass_change_traj = []
    for d in diffs:
        cc = d.get("compass_changes")
        compass_change_traj.append(len(cc) if isinstance(cc, list) else 0)
    abstract_signature = (
        f"{click_count} clicks · {len(unique_region_ids)} unique non-outside regions · "
        f"{repeat_clicks} repeat clicks · "
        f"kind_dist={kind_counts} · "
        f"compass_change_traj={compass_change_traj}"
    )
    return {
        "from_level_delta": from_level_delta,
        "run_namespace": board.namespace,
        "click_count": click_count,
        "unique_regions_clicked": len(unique_region_ids),
        "repeat_clicks": repeat_clicks,
        "kind_distribution": kind_counts,
        "compass_change_traj": compass_change_traj,
        "abstract_signature": abstract_signature,
        "recorded_at_turn": board.turn_index,
    }


def _approx_current_metrics(serialised_turn_diffs: list[dict]) -> dict:
    """Compute approximate {click_count, repeat_clicks, kind_distribution,
    compass_change_traj} from the agent's recent serialised turn-diffs.
    Used to build a current-trajectory signature for CPSR matching."""
    diffs = serialised_turn_diffs or []
    cc = len(diffs)
    region_ids = [d.get("click_region_id") for d in diffs if d.get("click_region_id")]
    valid = [r for r in region_ids if r and r != "_outside_"]
    unique = len(dict.fromkeys(valid))
    rc = len(valid) - unique
    kd: dict = {}
    for d in diffs:
        k = d.get("region_kind_pre") or "outside"
        kd[k] = kd.get(k, 0) + 1
    traj = []
    for d in diffs:
        cc_changes = d.get("compass_changes")
        traj.append(len(cc_changes) if isinstance(cc_changes, list) else 0)
    return {
        "click_count": cc,
        "repeat_clicks": rc,
        "kind_distribution": kd,
        "compass_change_traj": traj,
    }


def _retrieve_analogous_segments(
    *,
    current_signature: str,
    segment_index: list,
    k_per_class: int = 1,
) -> list[dict]:
    """CPSR: pick top-k success + top-k failure segments closest to the
    current trajectory signature. Returns anonymised payload list (no
    R-ids or coords). Empty list when index is empty.

    Distance: Hamming-style — count matching tokens in the pipe-split
    signature. Higher score = closer match."""
    if not segment_index:
        return []

    def _score(seg_sig: str) -> int:
        cur_tokens = (current_signature or "").split("|")
        seg_tokens = (seg_sig or "").split("|")
        return sum(1 for a, b in zip(cur_tokens, seg_tokens) if a == b)

    scored = [
        (
            _score(s.get("abstract_signature", "")),
            s,
        )
        for s in segment_index
        if isinstance(s, dict)
    ]
    successes = sorted(
        [(sc, s) for sc, s in scored if s.get("did_progress")],
        key=lambda x: -x[0],
    )[:k_per_class]
    failures = sorted(
        [(sc, s) for sc, s in scored if not s.get("did_progress")],
        key=lambda x: -x[0],
    )[:k_per_class]
    out: list[dict] = []
    for sc, s in successes + failures:
        out.append({
            "kind": s.get("kind"),
            "did_progress": bool(s.get("did_progress")),
            "abstract_signature": s.get("abstract_signature"),
            "what_changed_in_next_5turns": s.get("what_changed_in_next_5turns"),
            "match_score": int(sc),
        })
    return out


def _build_level_bridge_priors(bridges: list[dict]) -> list[dict]:
    """Return at most 6 anonymised level-bridge priors. Strips run-
    specific fields (id, namespace, recorded_at_turn) and keeps only
    the structural shape of past successful transitions."""
    if not bridges:
        return []
    # Most recent first, dedup by abstract_signature so a single
    # successful shape doesn't dominate priors.
    seen_sigs: set[str] = set()
    out: list[dict] = []
    for b in reversed(bridges):
        sig = b.get("abstract_signature") or ""
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        out.append({
            "from_level_delta": b.get("from_level_delta"),
            "click_count": b.get("click_count"),
            "unique_regions_clicked": b.get("unique_regions_clicked"),
            "repeat_clicks": b.get("repeat_clicks"),
            "kind_distribution": b.get("kind_distribution"),
            "compass_change_traj": b.get("compass_change_traj"),
            "abstract_signature": sig,
        })
        if len(out) >= 6:
            break
    return out


class V57Board:
    """Lightweight state holder. Replaces GoalBoard's lifecycle bookkeeping
    with just the fields v57 needs: active hypothesis pool, falsified set,
    rolling summary, recent verbose turns, trace log."""

    def __init__(self, namespace: str, game_id: str, workdir: Path) -> None:
        self.namespace = namespace
        self.game_id = game_id
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.trace_path = workdir / "trace.jsonl"
        # State.
        self.turn_index = 0
        self.active_hypotheses: list[dict] = []
        self.falsified: list[dict] = []  # rolling 30
        self.summary: str = ""  # produced by Reflexion every 5 turns
        self.recent_verbose: list[dict] = []  # rolling _VERBOSE_WINDOW
        self.gqb_pair: tuple[int, int] | None = None
        self._next_card_id = 1
        # v57.3: per-region click outcome tally for deterministic dead-region
        # exclusion. {region_id: {"outside": int, "transitions": int}}.
        self.region_click_tally: dict[str, dict[str, int]] = {}
        # Regions where >=2 _outside_ hits — auto-excluded from new cards.
        self.dead_regions: set[str] = set()
        # v585zz: precondition-gated trigger registry. Each entry =
        # {turn, region_id, coord, dominant_transition, level_at_event}.
        # Per-turn filter: include only entries whose region_id is in
        # current visible_region_ids → matched_prior_triggers.
        self.l_plus_events: list[dict] = []
        # H_RECYCLE: count of consecutive inconclusive verdicts. When this
        # exceeds _INCONCLUSIVE_EVICT_K and pool is at/over target, evict
        # the most-tested card so HYPOTHESIZE re-fires next turn.
        self.consecutive_inconclusive: int = 0
        # v586 B4 v2: deferred post-L+ eviction. When level_delta >= 1
        # fires inside a turn, the visible_regions argument we already
        # computed reflects the *pre-click* frame (which still shows the
        # marker the agent just clicked). The reseed only takes effect on
        # the *next* turn's visible_regions. We therefore set this flag
        # and run the eviction at the start of the next turn.
        self.pending_post_lp_eviction: bool = False
        # v586 B9: cross-run memory port — shared atomic-mechanism memory
        # across runs of the same game. Schema {schema_version: 1,
        # abstract_mechanics: [{id, text, confirmed_runs, last_seen_run,
        # refuted_runs, promoted_at_turn}]}. Loaded from
        # <workdir.parent>/cross_run_memory.json on construction; saved
        # whenever add_to_1A is called. Empty list on first run for a
        # given game.
        self.cross_run_memory: dict = self._load_cross_run_memory()
        # v586 B10: per-region click count since the last L+ event.
        # Resets to {} on level_delta >= 1. Exposed to M1 as the
        # actionable per-region parity counter so the agent can avoid
        # re-clicking a region whose count is already at the right
        # parity for its current colour. cycle179 traces showed 5/11
        # post-L+ regions ended at "cancelled" (even click count) due
        # to lack of this signal at decision time.
        self.region_clicks_this_level: dict[str, int] = {}
        # v587 B12 (Symbolica turn-diff buffer / Layer 1):
        # Rolling window of structured per-turn diffs. Each entry shows
        # what changed between the pre-click frame and the post-click
        # frame. compass_changes is filled lazily at the START of the
        # NEXT turn (need both pre + post snapshots). Schema:
        #   {turn, click, click_region_id, region_color_pre,
        #    color_transitions, compass_changes, level_delta,
        #    _pre_compass_snapshot (internal, dropped on serialise)}
        self.recent_turn_diffs: list[dict] = []
        # v587 B13 (Hermes-style episodic level-bridge memory / Layer 2):
        # Cross-run accumulating record of past level transitions. On
        # level_delta >= 1, snapshot the last K=15 turn diffs and
        # compute a Symbolica-derived abstract_signature. Persisted to
        # <workdir.parent>/level_bridges.json with race-safe reload-
        # merge-save (mirrors cross_run_memory pattern).
        self.level_bridges: dict = self._load_level_bridges()
        # v587 B14 (hierarchical memory):
        # ASMW — turn index of the most recent L+ event in this run. Used
        # to compute stuck severity. Initialise to 0 so cold-start does
        # not register as already-stuck (severity=0 until the first L+ or
        # until turn>K_STUCK, whichever first).
        self.last_lp_event_turn: int = 0
        # ASMW — sticky stuck-mode flag with hysteresis (avoid thrashing
        # at threshold boundary). Turns ON at gap >= _K_STUCK. Turns OFF
        # only after L+ event AND gap drops to <= _STUCK_HYSTERESIS_OFF.
        self.stuck_mode: bool = False
        # CPSR — segment_index loaded once per board init from
        # <workdir.parent>/segment_index.json (built offline by
        # tools/index_traces.py). Empty list when file absent.
        self.segment_index: list = self._load_segment_index()
        # v588 round-2 D-3: separate chain-verbose buffer for the chain
        # layer's prediction_errors channel. Each entry is TRIMMED:
        # {turn, observation, action.expected_observation} only —
        # drops thought + action.coord + cards to keep state size small.
        # Max _CHAIN_VERBOSE_WINDOW=24 (vs recent_verbose=3 for M1).
        self.chain_verbose: list[dict] = []
        # v588 round-2 Q-E + N-2: chain_rule emission log. Each entry
        # carries a unique id, full rule fields, evidence_turns, and
        # confirmed_count (incremented when a rule's
        # evidence_turn matches a later observed L+ event in the same
        # cycle). Cap _CHAIN_RULE_LOG_MAX=30 with LRU-by-confirmed
        # eviction (preferring to drop low-confirmed entries).
        self.chain_rule_log: list[dict] = []
        self._next_chain_rule_id: int = 1
        # v589 B17: candidate-test infrastructure.
        # candidate_log holds in-flight typed candidates with their
        # verdict status. Capped at _CANDIDATE_LOG_MAX, oldest evicted
        # when over cap (with verdict-aware eviction: ignored first).
        self.candidate_log: list[dict] = []
        # recent_candidate_emissions tracks (role, anchor_marker_id)
        # tuples per turn so the score's novelty term can penalise
        # repeats within the last 5 turns.
        self.recent_candidate_emissions: list[dict] = []
        # recent_clicks (rolling window of last 10 region_ids the
        # agent actually clicked). Used by score's causal_proximity.
        self.recent_clicks: list[str] = []
        # role_history: cross-run persistence of per-role outcome
        # statistics (supported / refuted / ignored). Loaded once at
        # board init from <workdir.parent>/role_history.json. Used by
        # the score's delta_correlation_prior term.
        self.role_history: dict = self._load_role_history()

    def _cross_run_memory_path(self) -> Path:
        return self.workdir.parent / "cross_run_memory.json"

    def _load_cross_run_memory(self) -> dict:
        path = self._cross_run_memory_path()
        if not path.exists():
            return {"schema_version": 1, "abstract_mechanics": []}
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {"schema_version": 1, "abstract_mechanics": []}
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            return {"schema_version": 1, "abstract_mechanics": []}
        data.setdefault("abstract_mechanics", [])
        return data

    def _save_cross_run_memory(self) -> None:
        try:
            self._cross_run_memory_path().write_text(
                json.dumps(self.cross_run_memory, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def add_to_1A(self, text: str, turn: int, signature: str | None = None) -> None:
        """Append/update an atomic mechanism observation in cross-run
        memory. Race-safe (B9 v2): reload-merge-save.

        v587 B14 (ESC) — when `signature` is provided:
          1. Look up by signature first; if found, increment
             confirmed_runs and append text into cluster_observations
             (capped at _CLUSTER_OBS_CAP).
          2. If not found by signature, also search the legacy-text
             pool (entries without `signature` field) for an exact text
             match; if matched, MIGRATE that legacy entry into a new
             cluster (carry forward confirmed_runs+1 + cluster_observations
             with the legacy text + this text).
          3. Otherwise, create a new cluster entry with both signature
             and text.

        When `signature` is None (legacy callers), keep the v586 B9 v2
        text-dedup behaviour exactly so existing tests still pass."""
        text = (text or "").strip()
        if not text:
            return
        on_disk = self._load_cross_run_memory()
        on_disk_am = on_disk.setdefault("abstract_mechanics", [])
        norm = text.lower()

        if signature is not None:
            # ESC: signature-based dedup.
            for entry in on_disk_am:
                if entry.get("signature") == signature:
                    entry["confirmed_runs"] = int(entry.get("confirmed_runs", 0)) + 1
                    entry["last_seen_run"] = self.namespace
                    obs = entry.setdefault("cluster_observations", [])
                    if text not in obs and len(obs) < _CLUSTER_OBS_CAP:
                        obs.append(text)
                    self.cross_run_memory = on_disk
                    self._save_cross_run_memory()
                    return
            # Look in legacy-text pool (entries without signature).
            for i, entry in enumerate(on_disk_am):
                if entry.get("signature") is None and (
                    (entry.get("text") or "").strip().lower() == norm
                ):
                    # Migrate legacy entry into cluster.
                    legacy_text = entry.get("text") or text
                    cfr = int(entry.get("confirmed_runs", 0)) + 1
                    new_entry = {
                        "id": entry.get("id") or f"A{len(on_disk_am)}",
                        "text": text,
                        "signature": signature,
                        "confirmed_runs": cfr,
                        "last_seen_run": self.namespace,
                        "refuted_runs": int(entry.get("refuted_runs", 0)),
                        "promoted_at_turn": turn,
                        "cluster_observations": [legacy_text],
                    }
                    if text != legacy_text and len(new_entry["cluster_observations"]) < _CLUSTER_OBS_CAP:
                        new_entry["cluster_observations"].append(text)
                    on_disk_am[i] = new_entry
                    self.cross_run_memory = on_disk
                    self._save_cross_run_memory()
                    return
            # No match — create new cluster entry.
            new_id = f"A{len(on_disk_am) + 1}"
            on_disk_am.append({
                "id": new_id,
                "text": text,
                "signature": signature,
                "confirmed_runs": 1,
                "last_seen_run": self.namespace,
                "refuted_runs": 0,
                "promoted_at_turn": turn,
                "cluster_observations": [text],
            })
            self.cross_run_memory = on_disk
            self._save_cross_run_memory()
            return

        # Legacy text-dedup path (signature is None).
        for entry in on_disk_am:
            if (entry.get("text") or "").strip().lower() == norm:
                entry["confirmed_runs"] = int(entry.get("confirmed_runs", 0)) + 1
                entry["last_seen_run"] = self.namespace
                self.cross_run_memory = on_disk
                self._save_cross_run_memory()
                return
        new_id = f"A{len(on_disk_am) + 1}"
        on_disk_am.append({
            "id": new_id,
            "text": text,
            "confirmed_runs": 1,
            "last_seen_run": self.namespace,
            "refuted_runs": 0,
            "promoted_at_turn": turn,
        })
        self.cross_run_memory = on_disk
        self._save_cross_run_memory()

    # ---------------- v587 B14: hierarchical-memory helpers --------------

    @staticmethod
    def _bucket_click_count(cc: int) -> str:
        """Empirical buckets from reports/plan_v587_buckets.json:
        click_count quartiles 4 / 6 over 26 events → {<=4, 5-6, >6}."""
        if cc <= 4:
            return "cc_le4"
        if cc <= 6:
            return "cc_5to6"
        return "cc_gt6"

    @staticmethod
    def _bucket_repeat_clicks(rc: int) -> str:
        """Manual buckets — empirical distribution sparse (21x@0, 4x@1,
        1x@2). Use {0, 1, 2+} for stability across small samples."""
        if rc <= 0:
            return "rc_0"
        if rc == 1:
            return "rc_1"
        return "rc_2plus"

    @staticmethod
    def _bucket_compass_traj_pattern(traj: list[int]) -> str:
        """Compress compass_change_traj into a coarse pattern label so
        runs differing only in length collapse into the same signature.
        Buckets: 'flat' (all 0), 'sparse' (mean<1), 'active' (mean>=1)."""
        if not traj:
            return "tr_flat"
        nz = [t for t in traj if isinstance(t, int) and t > 0]
        if not nz:
            return "tr_flat"
        mean = sum(traj) / max(1, len(traj))
        if mean < 1.0:
            return "tr_sparse"
        return "tr_active"

    @staticmethod
    def _bucket_kind_distribution(kd: dict) -> str:
        """Compress kind_distribution into 'mixed' / 'all_non_marker' /
        'has_marker_clicks' / 'has_outside'. Captures the *kind* of
        trajectory shape without leaking specific R-ids."""
        if not isinstance(kd, dict):
            return "kd_unknown"
        keys = set(kd.keys())
        if keys == {"non_marker"}:
            return "kd_all_non_marker"
        if "marker_multicolor" in keys and kd.get("marker_multicolor", 0) >= 1:
            return "kd_has_marker"
        if "outside" in keys and kd.get("outside", 0) >= 1:
            return "kd_has_outside"
        return "kd_mixed"

    @classmethod
    def compute_event_signature(cls, bridge_metrics: dict) -> str:
        """ESC: 4-tuple structural signature for L+ events. Two events
        with the same signature collapse into one cluster."""
        cc = int(bridge_metrics.get("click_count", 0))
        rc = int(bridge_metrics.get("repeat_clicks", 0))
        traj = bridge_metrics.get("compass_change_traj", []) or []
        kd = bridge_metrics.get("kind_distribution", {}) or {}
        return "|".join([
            cls._bucket_click_count(cc),
            cls._bucket_repeat_clicks(rc),
            cls._bucket_compass_traj_pattern(traj),
            cls._bucket_kind_distribution(kd),
        ])

    def stuck_severity(self) -> int:
        """ASMW: how many turns past _K_STUCK we are. 0 if not stuck."""
        gap = self.turn_index - int(self.last_lp_event_turn or 0)
        return max(0, gap - _K_STUCK)

    def update_stuck_mode(self, level_delta: int) -> None:
        """ASMW: hysteresis. ON at gap≥_K_STUCK; OFF after L+ event AND
        gap<=_STUCK_HYSTERESIS_OFF."""
        gap = self.turn_index - int(self.last_lp_event_turn or 0)
        if not self.stuck_mode:
            if gap >= _K_STUCK:
                self.stuck_mode = True
        else:
            if level_delta >= 1 and gap <= _STUCK_HYSTERESIS_OFF:
                self.stuck_mode = False
        # Update last_lp_event_turn AFTER hysteresis decision so the
        # gap measured at this turn reflects the pre-event state.
        if level_delta >= 1:
            self.last_lp_event_turn = self.turn_index

    def serialised_recent_turn_diffs(self, expand: int = 0) -> list[dict]:
        """ASMW: return up to `min(6 + 2*expand, _TURN_DIFF_MAX_RAW)`
        most recent turn-diff entries (oldest at index 0). When expand=0
        (not stuck), returns the most recent 6 entries — same as v587
        B12 behaviour. When stuck, expands the visible window."""
        target_n = min(6 + 2 * max(0, int(expand)), _TURN_DIFF_MAX_RAW)
        slice_diffs = self.recent_turn_diffs[-target_n:]
        out = []
        for e in slice_diffs:
            d = {k: v for k, v in e.items() if not k.startswith("_")}
            out.append(d)
        return out

    # ---------------- v587 B13: level_bridges (race-safe) ----------------

    def _level_bridges_path(self) -> Path:
        return self.workdir.parent / "level_bridges.json"

    def _load_level_bridges(self) -> dict:
        path = self._level_bridges_path()
        if not path.exists():
            return {"schema_version": 1, "bridges": []}
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {"schema_version": 1, "bridges": []}
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            return {"schema_version": 1, "bridges": []}
        data.setdefault("bridges", [])
        return data

    # ---------------- v589 B17: role_history (race-safe) -------------

    def _role_history_path(self) -> Path:
        return self.workdir.parent / "role_history.json"

    def _load_role_history(self) -> dict:
        path = self._role_history_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_role_history(self) -> None:
        try:
            self._role_history_path().write_text(
                json.dumps(self.role_history, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def update_role_history(self, role: str, outcome: str) -> None:
        """outcome ∈ {supported, refuted, ignored}. Race-safe: reload,
        merge, save. Mirrors B9v2 pattern."""
        if outcome not in ("supported", "refuted", "ignored"):
            return
        on_disk = self._load_role_history()
        entry = on_disk.setdefault(role, {
            "observed_supported": 0, "observed_refuted": 0,
            "observed_ignored": 0, "last_seen_run": "",
            "last_seen_turn": 0,
        })
        entry[f"observed_{outcome}"] = int(entry.get(f"observed_{outcome}", 0)) + 1
        entry["last_seen_run"] = self.namespace
        entry["last_seen_turn"] = int(self.turn_index)
        self.role_history = on_disk
        self._save_role_history()

    # ---------------- v589 B17: candidate_log management ------------

    def append_candidate_emissions(self, candidates: list[dict]) -> None:
        """Track per-turn (role, anchor) for novelty score."""
        for c in candidates or []:
            st = c.get("suggested_test") or {}
            self.recent_candidate_emissions.append({
                "turn": self.turn_index,
                "role": st.get("role"),
                "anchor_marker_id": st.get("anchor_marker_id"),
            })
        # Trim to last 5 turns × beam=8 = 40 entries.
        if len(self.recent_candidate_emissions) > 40:
            self.recent_candidate_emissions = self.recent_candidate_emissions[-40:]

    def register_candidate_log(self, candidates: list[dict]) -> None:
        """Append fresh candidates to in-flight log with emitted_at_turn
        and verdict=None. LRU-by-verdict eviction at cap (ignored
        before resolved before unverified)."""
        for c in candidates or []:
            entry = {
                "candidate_id": c.get("candidate_id"),
                "suggested_test": c.get("suggested_test") or {},
                "expected_observable_signature":
                    c.get("expected_observable_signature") or {},
                "refutation_signature":
                    c.get("refutation_signature") or {},
                "score": c.get("score", 0),
                "emitted_at_turn": int(self.turn_index),
                "verdict": None,
                "reemit_count": 0,
            }
            self.candidate_log.append(entry)
        # Evict at cap.
        if len(self.candidate_log) > _CANDIDATE_LOG_MAX:
            # Sort by eviction priority: ignored first, then resolved,
            # then unverified. Within each, oldest first.
            def _key(e):
                v = e.get("verdict")
                v_pri = {"ignored": 0, "supported": 1,
                         "refuted": 1, "inconclusive": 1}.get(v, 2)
                return (v_pri, int(e.get("emitted_at_turn", 0)))
            self.candidate_log.sort(key=_key)
            # Drop the oldest of lowest-priority.
            self.candidate_log = self.candidate_log[
                -_CANDIDATE_LOG_MAX:
            ]

    def serialised_candidate_tests_for_m1(self) -> list[dict]:
        """Round-3 C-9 re-emission rule: present unresolved candidates
        (verdict=None) AND ignored ones whose reemit_count<MAX. Sort
        by score desc."""
        out = []
        for e in self.candidate_log:
            v = e.get("verdict")
            if v in ("supported", "refuted", "inconclusive"):
                continue
            if v == "ignored" and int(e.get("reemit_count", 0)) >= _CANDIDATE_REEMIT_MAX:
                continue
            # Build leak-safe payload (drop internal fields).
            out.append({
                "candidate_id": e.get("candidate_id"),
                "suggested_test": e.get("suggested_test"),
                "expected_observable_signature":
                    e.get("expected_observable_signature"),
                "refutation_signature":
                    e.get("refutation_signature"),
                "score": e.get("score"),
                "emitted_at_turn": e.get("emitted_at_turn"),
                "reemit_count": int(e.get("reemit_count", 0)),
            })
        out.sort(key=lambda c: -c.get("score", 0))
        return out[:_CAND_K_GLOBAL]

    def _segment_index_path(self) -> Path:
        return self.workdir.parent / "segment_index.json"

    def _load_segment_index(self) -> list:
        path = self._segment_index_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except Exception:
            return []
        if isinstance(data, dict) and "segments" in data:
            return data.get("segments", []) or []
        if isinstance(data, list):
            return data
        return []

    def _save_level_bridges(self) -> None:
        try:
            self._level_bridges_path().write_text(
                json.dumps(self.level_bridges, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def append_level_bridge(self, bridge_entry: dict) -> None:
        """Race-safe append: reload from disk, append, save. Multiple
        parallel cycles writing simultaneously will all retain their
        contributions (mirrors B9 v2 add_to_1A pattern)."""
        on_disk = self._load_level_bridges()
        bridges = on_disk.setdefault("bridges", [])
        bridge_entry = dict(bridge_entry)
        bridge_entry["id"] = f"B{len(bridges) + 1}"
        bridges.append(bridge_entry)
        self.level_bridges = on_disk
        self._save_level_bridges()

    # ---------------- v587 B12: turn-diff buffer (in-memory only) -------

    @staticmethod
    def _compass_snapshot_from_marker_states(
        marker_neighbor_states: list[dict],
    ) -> dict[str, dict[str, int | None]]:
        """Compact snapshot suitable for cross-turn diffing:
        {marker_id: {direction: current_color}}"""
        snap: dict[str, dict[str, int | None]] = {}
        for m in marker_neighbor_states or []:
            mid = m.get("marker_id")
            compass = m.get("compass") or {}
            if not mid or not compass:
                continue
            snap[mid] = {
                d: (v.get("current_color") if isinstance(v, dict) else None)
                for d, v in compass.items()
            }
        return snap

    @staticmethod
    def _diff_compass_snapshots(
        prev: dict[str, dict[str, int | None]],
        cur: dict[str, dict[str, int | None]],
    ) -> list[dict]:
        """Return list of {marker_id, direction, from, to} change entries."""
        changes: list[dict] = []
        for mid, p_compass in (prev or {}).items():
            c_compass = (cur or {}).get(mid) or {}
            for direction, p_color in p_compass.items():
                c_color = c_compass.get(direction)
                if c_color is not None and c_color != p_color:
                    changes.append({
                        "marker_id": mid,
                        "direction": direction,
                        "from": p_color,
                        "to": c_color,
                    })
        return changes

    def fill_pending_compass_changes(
        self,
        current_compass_snapshot: dict[str, dict[str, int | None]],
    ) -> None:
        """At START of each turn (after marker_neighbor_states is built),
        fill compass_changes for the most recent recent_turn_diffs entry
        whose compass_changes is None (i.e. the previous turn's diff
        which was waiting for the post-click snapshot)."""
        if not self.recent_turn_diffs:
            return
        last = self.recent_turn_diffs[-1]
        if last.get("compass_changes") is not None:
            return
        prev_snap = last.pop("_pre_compass_snapshot", {}) or {}
        last["compass_changes"] = self._diff_compass_snapshots(
            prev_snap, current_compass_snapshot
        )

    # ---------------- v588 round-2 D-3: chain_verbose buffer ----------

    def append_chain_verbose(self, full_verbose: dict) -> None:
        """Append a TRIMMED verbose entry to chain_verbose buffer.
        Strips thought, full action body, cards. Keeps turn,
        observation, action.expected_observation only — exactly what
        compute_prediction_errors needs."""
        if not isinstance(full_verbose, dict):
            return
        action = full_verbose.get("action") or {}
        if isinstance(action, dict):
            expected = action.get("expected_observation")
        else:
            expected = None
        trimmed = {
            "turn": full_verbose.get("turn"),
            "observation": full_verbose.get("observation") or {},
            "action": (
                {"expected_observation": expected}
                if expected is not None else {}
            ),
        }
        self.chain_verbose.append(trimmed)
        if len(self.chain_verbose) > _CHAIN_VERBOSE_WINDOW:
            self.chain_verbose.pop(0)

    # ---------------- v588 round-2 Q-E + N-2: chain_rule_log ----------

    def alloc_chain_rule_id(self) -> str:
        rid = f"CR{self._next_chain_rule_id}"
        self._next_chain_rule_id += 1
        return rid

    @staticmethod
    def _validate_chain_rule_entry(entry: dict) -> bool:
        """Server-side hard floor on chain_rule fields per round-2 D-4.
        Returns True if entry is well-formed; False to drop.

        Rules:
          - All required fields present.
          - rule must contain a T<int> reference (not just
            falsification_condition — content check, not Goodhart bait).
          - falsification_condition contains T<int>.
          - rule and falsification_condition each ≥30 chars.
          - evidence_turns is a non-empty list of ints.
        """
        if not isinstance(entry, dict):
            return False
        rule = entry.get("rule") or ""
        falsif = entry.get("falsification_condition") or ""
        evidence = entry.get("evidence_turns")
        if len(rule) < 30 or len(falsif) < 30:
            return False
        if not isinstance(evidence, list) or not evidence:
            return False
        # D-4: T<int> reference required in BOTH rule and falsif.
        import re as _re
        if not _re.search(r"T\d+", rule):
            return False
        if not _re.search(r"T\d+", falsif):
            return False
        return True

    def serialised_chain_rule_log(self, k: int = 8) -> list[dict]:
        """Round-3 reviewer fix — close the chain_rule loop. Returns
        last `k` entries suitable for M3 input. Drops internal fields.

        Round-4 reviewer fix D: falsified rules MUST appear in the top-k
        so M3 can cite them and propose replacements (per the prompt's
        instruction). Sort key now: falsified-first within bucket so
        recently-falsified rules surface before they get evicted."""
        log = list(self.chain_rule_log)
        # Order:
        #   (1) falsified rules first (they MUST be cited+replaced),
        #   (2) within each falsified bucket, highest confirmed_count first,
        #   (3) then most recent emitted_at_turn first.
        log.sort(
            key=lambda r: (
                # falsified=True (1) sorts BEFORE falsified=False (0)
                # under (-int(r.get("falsified",False))) — descending bool.
                -int(r.get("falsified", False)),
                -int(r.get("confirmed_count", 0)),
                -int(r.get("emitted_at_turn", 0)),
            )
        )
        out = []
        for r in log[:k]:
            out.append({
                "id": r.get("id"),
                "rule": r.get("rule"),
                "evidence_turns": r.get("evidence_turns"),
                "predicted_outcome_next_turn": r.get("predicted_outcome_next_turn"),
                "falsification_condition": r.get("falsification_condition"),
                "temporal_scope": r.get("temporal_scope"),
                "confirmed_count": int(r.get("confirmed_count", 0)),
                "falsified": bool(r.get("falsified", False)),
                "emitted_at_turn": int(r.get("emitted_at_turn", 0)),
            })
        return out

    def update_chain_rule_log_with_observation(
        self, *, observation: dict, turn_now: int
    ) -> None:
        """Round-3 fix + round-4 corrections.

        Per round-4 reviewer:
          - B FAIL: broadened falsification — also catches negated L+
            phrasings ("not", "no", "must not", "should not", "never",
            "remains") within 30 chars of L+/level/transition tokens.
            Plus T<int> matches against turn_now.
          - F1: confirmed_count saturates at 5 — generic rules can
            inflate, but capped so they don't lock newer rules out.
          - F2: confirm + falsify mutually exclusive in single turn —
            falsify takes precedence (early-return).
          - A WARN: tighter prediction match — "level"/"L+" alone is
            too broad. Require co-occurrence with transition_str OR
            obs_rid substring OR explicit "fire"/"rise"/"increase"
            verb to reduce Goodhart inflation.
        """
        if not isinstance(observation, dict):
            return
        import re as _re
        obs_rid = observation.get("primary_region_id") or "_outside_"
        dt = observation.get("dominant_transition") or {}
        if isinstance(dt, dict) and dt.get("from") is not None:
            transition_str = f"{dt.get('from')}->{dt.get('to')}"
        else:
            transition_str = ""
        ld = int(observation.get("level_delta") or 0)
        turn_marker = f"t{turn_now}"  # case-insensitive scan

        for entry in self.chain_rule_log:
            if entry.get("falsified"):
                continue
            predicted = (entry.get("predicted_outcome_next_turn") or "").lower()
            falsif = (entry.get("falsification_condition") or "").lower()

            # ---------- F2: FALSIFICATION FIRST (early-return if matched) ----
            # Round-5 reviewer: drop `T?` from regex (numeric-collision
            # risk) — match `T<int>` only, not bare integers like
            # "5 turns later" or "Region5". Plus add the unfulfilled-L+
            # branch (rule predicts L+ but L+ did not fire).
            falsified = False
            # 1. Direct transition string in falsification_condition.
            if transition_str and transition_str in falsif:
                falsified = True
            # 2. Strict T<turn_now> reference (round-5 numeric-collision fix).
            elif _re.search(rf"\bT{turn_now}\b", falsif):
                falsified = True
            # 3. Negated-L+ pattern in falsif when L+ fired this turn.
            elif ld >= 1:
                lp_tokens = ["l+", "level", "fire", "rise", "increase",
                             "advance", "progress"]
                neg_tokens = ["not", "no ", "no_", "never", "must not",
                              "should not", "doesn't", "remains",
                              "stays", "fails to", "won't", "without"]
                for lp in lp_tokens:
                    idx = falsif.find(lp)
                    if idx < 0:
                        continue
                    window = falsif[max(0, idx - 30): idx + len(lp) + 30]
                    if any(neg in window for neg in neg_tokens):
                        falsified = True
                        break
            # 4. Round-5 reviewer asymmetric-falsification fix.
            #    Rule PREDICTS L+ (positive phrasing in
            #    predicted_outcome_next_turn) but L+ did NOT fire (ld=0)
            #    → unfulfilled prediction = falsification.
            if not falsified and ld == 0:
                positive_lp_tokens = ["l+", "level rise", "level rises",
                                      "level will rise", "fire", "rises",
                                      "advances", "increments"]
                if any(t in predicted for t in positive_lp_tokens):
                    # Require an anchor (transition / region / verb) so
                    # generic "level" mentions in unrelated contexts
                    # don't trigger falsification — matches the
                    # confirmation-anchor gate (A WARN fix).
                    has_anchor = (
                        (transition_str and transition_str in predicted)
                        or (obs_rid != "_outside_" and obs_rid.lower() in predicted)
                        or any(a in predicted for a in
                               ["fire", "rise", "advance", "increment"])
                    )
                    if has_anchor:
                        falsified = True
            if falsified:
                entry["falsified"] = True
                continue  # F2: do not double-credit

            # ---------- CONFIRMATION (only if not falsified) -----------
            confirmed = False
            # 1. Transition substring is the strongest signal.
            if transition_str and transition_str in predicted:
                confirmed = True
            # 2. Region-id substring is also strong.
            if obs_rid != "_outside_" and obs_rid.lower() in predicted:
                confirmed = True
            # 3. A WARN fix: "L+"/"level" mention requires co-occurrence
            #    with a *specific* anchor (transition / region / verb)
            #    to count as confirmed by an L+ event. Pure "level"
            #    keyword alone is too broad.
            if ld >= 1 and not confirmed:
                generic_tokens = ["l+", "level"]
                anchor_tokens = ["fire", "rise", "advance", "increment"]
                if any(g in predicted for g in generic_tokens):
                    if (any(a in predicted for a in anchor_tokens)
                        or (transition_str and transition_str in predicted)
                        or (obs_rid != "_outside_" and obs_rid.lower() in predicted)):
                        confirmed = True
            if confirmed:
                # F1: cap at 5 to prevent generic rules dominating top-k.
                cur = int(entry.get("confirmed_count", 0))
                entry["confirmed_count"] = min(5, cur + 1)

    def append_chain_rule_log(self, entry: dict, turn_emitted: int) -> str | None:
        """Validate + append entry. Returns the assigned rule id, or
        None when the entry was rejected by the validator (logged)."""
        if not self._validate_chain_rule_entry(entry):
            return None
        rid = self.alloc_chain_rule_id()
        record = {
            "id": rid,
            "rule": entry.get("rule"),
            "evidence_turns": entry.get("evidence_turns"),
            "predicted_outcome_next_turn": entry.get("predicted_outcome_next_turn"),
            "falsification_condition": entry.get("falsification_condition"),
            "temporal_scope": entry.get("temporal_scope") or "always",
            "emitted_at_turn": int(turn_emitted),
            "confirmed_count": 0,
            "falsified": False,
        }
        self.chain_rule_log.append(record)
        # N-2: LRU-by-confirmed eviction. When over cap, evict the
        # lowest-confirmed-count entry first; ties broken by oldest
        # emitted_at_turn.
        if len(self.chain_rule_log) > _CHAIN_RULE_LOG_MAX:
            self.chain_rule_log.sort(
                key=lambda r: (r.get("confirmed_count", 0),
                               r.get("emitted_at_turn", 0))
            )
            self.chain_rule_log.pop(0)
        return rid

    def append_turn_diff(self, entry: dict) -> None:
        """Append a per-turn diff entry; trim to max _TURN_DIFF_MAX_RAW.
        v587 B14: raised cap from 6 to _TURN_DIFF_MAX_RAW so ASMW can
        slice a longer window when stuck. Default exposure is still 6
        via serialised_recent_turn_diffs(expand=0)."""
        self.recent_turn_diffs.append(entry)
        if len(self.recent_turn_diffs) > _TURN_DIFF_MAX_RAW:
            self.recent_turn_diffs.pop(0)

    def alloc_card_id(self) -> str:
        cid = f"C{self._next_card_id}"
        self._next_card_id += 1
        return cid

    def append_trace(self, entry: dict) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def push_recent(self, entry: dict) -> None:
        self.recent_verbose.append(entry)
        if len(self.recent_verbose) > _VERBOSE_WINDOW:
            self.recent_verbose.pop(0)

    def push_falsified(self, region_id: str, transition: dict | None) -> None:
        self.falsified.append({"region_id": region_id, "transition": transition})
        if len(self.falsified) > 30:
            self.falsified.pop(0)

    def remove_hypothesis(self, card_id: str) -> dict | None:
        for i, c in enumerate(self.active_hypotheses):
            if c.get("id") == card_id:
                return self.active_hypotheses.pop(i)
        return None


# ---------------------------------------------------------------------------
# Sub-agent dispatchers (pure async functions; tests mock these via spawn).
# ---------------------------------------------------------------------------


async def call_hypothesize(
    *,
    summary: str,
    visible_regions: list[dict],
    falsified_recent: list[dict],
    gqb_pair: tuple[int, int] | None,
    image_b64: str | None,
    next_card_id_seed: int,
    cross_run_priors: list[dict] | None = None,
    action_state_chain: dict | None = None,
    chain_rule_log: list[dict] | None = None,
) -> dict:
    """Invoke the HYPOTHESIZE sub-agent. Returns parsed JSON dict."""
    compact_regions = _compact_regions(visible_regions)
    # B9: trim cross-run priors to avoid blowing up the prompt — keep
    # only id/text/confirmed_runs/refuted_runs and cap entries at 12.
    priors_compact = []
    for entry in (cross_run_priors or [])[:12]:
        if not isinstance(entry, dict):
            continue
        priors_compact.append({
            "id": entry.get("id"),
            "text": entry.get("text"),
            "confirmed_runs": int(entry.get("confirmed_runs", 0)),
            "refuted_runs": int(entry.get("refuted_runs", 0)),
        })
    payload = {
        "summary": summary,
        "visible_regions": compact_regions,
        "falsified_recent": falsified_recent,
        "gqb_pair": list(gqb_pair) if gqb_pair else None,
        "next_card_id_seed": next_card_id_seed,
        "cross_run_priors": priors_compact,
        "action_state_chain": action_state_chain or {
            "chain_tokens": [], "trajectory_features": {},
            "prediction_errors": [], "causal_table": [],
        },
        "chain_rule_log": chain_rule_log or [],
    }
    agent = await spawn(
        system=HYPOTHESIZE_SYSTEM_PROMPT,
        model="gpt-5.5",
    )
    task = HYPOTHESIZE_TASK_INSTRUCTIONS.format(
        input_json=json.dumps(payload, ensure_ascii=False, default=str)[:14000]
    )
    raw = await agent.call(
        str,
        task,
        _image_b64=image_b64,
    )
    return _safe_json_parse(raw, default={"thought": "", "cards": []})


async def call_action(
    *,
    summary: str,
    visible_regions: list[dict],
    active_hypotheses: list[dict],
    recent_turns: list[dict],
    gqb_pair: tuple[int, int] | None,
    image_b64: str | None,
    matched_prior_triggers: list[dict] | None = None,
    region_clicks_this_level: dict[str, int] | None = None,
    marker_neighbor_states: list[dict] | None = None,
    recent_turn_diffs: list[dict] | None = None,
    level_bridge_priors: list[dict] | None = None,
    stuck_mode: bool = False,
    analogous_past_segments: list[dict] | None = None,
    action_state_chain_compact: dict | None = None,
    candidate_tests: list[dict] | None = None,
) -> dict:
    """Invoke the ACTION main reasoner."""
    compact_regions = _compact_regions(visible_regions)
    payload = {
        "summary": summary,
        "visible_regions": compact_regions,
        "active_hypotheses": active_hypotheses,
        "recent_turns": recent_turns,
        "gqb_pair": list(gqb_pair) if gqb_pair else None,
        "matched_prior_triggers": matched_prior_triggers or [],
        "region_clicks_this_level": region_clicks_this_level or {},
        "marker_neighbor_states": marker_neighbor_states or [],
        "recent_turn_diffs": recent_turn_diffs or [],
        "level_bridge_priors": level_bridge_priors or [],
        "stuck_mode": bool(stuck_mode),
        "analogous_past_segments": analogous_past_segments or [],
        "action_state_chain": action_state_chain_compact or {
            "chain_tokens": [], "trajectory_features": {},
            "prediction_errors": [], "causal_table": [],
        },
        "candidate_tests": candidate_tests or [],
    }
    agent = await spawn(
        system=ACTION_SYSTEM_PROMPT,
        model="gpt-5.5",
    )
    task = ACTION_TASK_INSTRUCTIONS.format(
        input_json=json.dumps(payload, ensure_ascii=False, default=str)[:12000]
    )
    raw = await agent.call(
        str,
        task,
        _image_b64=image_b64,
    )
    return _safe_json_parse(
        raw,
        default={
            "thought": "(parse failed)",
            "chosen_hypothesis_id": None,
            "action": {"type": "ACTION6", "coord": [32, 32]},
            "expected_observation": {
                "primary_region_id": None,
                "dominant_transition": None,
                "level_delta": 0,
            },
        },
    )


async def call_reflexion(
    *,
    previous_summary: str,
    recent_turns: list[dict],
    active_hypotheses: list[dict],
    falsified: list[dict],
) -> dict:
    """Invoke the REFLEXION compressor sub-agent."""
    payload = {
        "previous_summary": previous_summary,
        "recent_turns": recent_turns,
        "active_hypotheses": active_hypotheses,
        "falsified": falsified,
    }
    agent = await spawn(
        system=REFLEXION_SYSTEM_PROMPT,
        model="gpt-5.5",
    )
    task = REFLEXION_TASK_INSTRUCTIONS.format(
        input_json=json.dumps(payload, ensure_ascii=False, default=str)[:12000]
    )
    raw = await agent.call(str, task)
    return _safe_json_parse(raw, default={"thought": "", "summary": previous_summary})


# ---------------------------------------------------------------------------
# Verdict logic (pure, deterministic — testable without LLM).
# ---------------------------------------------------------------------------


def evaluate_card_against_observation(
    card: dict, observation: dict
) -> str:
    """Return one of: 'confirm' | 'falsify' | 'inconclusive' |
    'plan_in_progress' | 'plan_confirm' | 'plan_falsify'.

    Pure function — no I/O, fully testable. Plan cards only fire
    confirm/falsify when the sequence completes; mid-sequence steps
    return 'plan_in_progress' so the card stays in the active pool."""
    obs_region = observation.get("primary_region_id")
    obs_trans = observation.get("dominant_transition")
    obs_dl = int(observation.get("level_delta") or 0)

    # ---- Plan-card path -------------------------------------------------
    if card.get("type") == "plan":
        seq = card.get("click_sequence") or []
        progress = int(card.get("sequence_progress", 0))
        # Mid-sequence: any level rise = early plan_confirm; otherwise stay open.
        if obs_dl > 0:
            return "plan_confirm"
        if progress < len(seq):
            # Each step optionally checks expected_color_after via transition.
            step = seq[progress]
            expected_to = step.get("expected_color_after")
            if expected_to is not None and obs_trans:
                if obs_trans.get("to") != expected_to:
                    # A single step disagreed with rule_hypothesis prediction.
                    return "plan_falsify"
            # Step OK or no expectation set; sequence continues.
            return "plan_in_progress"
        # Sequence exhausted without level rise → rule_hypothesis was wrong.
        expected_dl = int((card.get("expected_outcome") or {}).get("level_delta", 1))
        return "plan_confirm" if expected_dl == 0 else "plan_falsify"

    # ---- Single-card path (unchanged) ----------------------------------
    sig = card.get("expected_signature") or {}
    expected_region = card.get("region_id")
    if expected_region and obs_region and obs_region != expected_region:
        return "falsify"
    if obs_region == "_outside_":
        return "falsify"
    expected_trans = sig.get("dominant_transition")
    if expected_trans and not obs_trans:
        return "falsify"
    if expected_trans and obs_trans:
        if (
            expected_trans.get("from") != obs_trans.get("from")
            or expected_trans.get("to") != obs_trans.get("to")
        ):
            return "falsify"
    expected_dl = int(sig.get("level_delta") or 0)
    if expected_dl > 0 and obs_dl == 0:
        return "inconclusive"
    if expected_dl == 0 and obs_dl > 0:
        return "confirm"
    if expected_trans and obs_trans:
        return "confirm"
    return "inconclusive"


def compose_observation(
    *, region_id: str | None, transition: dict | None, level_delta: int
) -> dict:
    """Build a normalized observation dict (pure helper for tests)."""
    return {
        "primary_region_id": region_id,
        "dominant_transition": transition,
        "level_delta": int(level_delta),
    }


# ---------------------------------------------------------------------------
# Per-turn orchestration (no I/O — tests can drive this with fake spawn).
# ---------------------------------------------------------------------------


async def run_turn(
    *,
    board: V57Board,
    visible_regions: list[dict],
    image_b64: str | None,
    spawn_action,
    spawn_hypothesize,
    spawn_reflexion,
    execute_action,
) -> dict:
    """Execute one v57 turn. spawn_*/execute_action are dependency-injected
    so tests can run this loop without TRAPI."""
    # 0. v586 B4 v2: process deferred post-L+ eviction now that we have
    # the new visible_regions (post-reseed). Evict every active card
    # whose region_id (or any click_sequence step region_id) is absent
    # from the new visible layout. Forces HYPOTHESIZE to refire below.
    if board.pending_post_lp_eviction and board.active_hypotheses:
        visible_rids: set = set()
        for r in (visible_regions or []):
            if isinstance(r, dict):
                rid = r.get("region_id") or r.get("id")
                if rid:
                    visible_rids.add(rid)
        for c in list(board.active_hypotheses):
            cand_rids: set = set()
            cr = c.get("region_id")
            if cr:
                cand_rids.add(cr)
            for step in (c.get("click_sequence") or []):
                if isinstance(step, dict):
                    sr = step.get("region_id")
                    if sr:
                        cand_rids.add(sr)
            if cand_rids and cand_rids.isdisjoint(visible_rids):
                board.push_falsified(
                    c.get("region_id") or f"post_lp_stale:{c.get('id')}",
                    "post_level_reseed",
                )
                board.remove_hypothesis(c["id"])
        board.consecutive_inconclusive = 0
        board.pending_post_lp_eviction = False

    # 1. Top up hypothesis pool if below target.
    falsified_recent = list(board.falsified[-15:])
    # v57.3: include deterministically-detected dead regions in
    # FALSIFIED_RECENT so HYPOTHESIZE never re-emits cards on them.
    for rid in board.dead_regions:
        falsified_recent.append({"region_id": rid, "transition": None,
                                  "reason": "auto: 2+ _outside_ misses"})
    hypothesize_log = None
    # v588 B16: build the action-state chain payload using the
    # CHAIN_VERBOSE buffer (round-2 D-3) for prediction_errors depth.
    # build_chain_tokens still iterates union-of-turns from chain_verbose
    # and recent_turn_diffs.
    chain_full = compress_action_state_chain(
        recent_verbose=board.chain_verbose,
        recent_turn_diffs=board.recent_turn_diffs,
        level="full",
    )
    chain_compact = compress_action_state_chain(
        recent_verbose=board.chain_verbose,
        recent_turn_diffs=board.recent_turn_diffs,
        level="compact",
    )
    # v589 B17: candidate generation deferred until AFTER
    # marker_neighbor_states is built (downstream). Initialise here
    # to avoid forward-reference; populate at the deferred site.
    new_candidates: list = []
    candidate_tests_for_m1: list = []

    # v589 B17: M3 dormant by default. Re-enable via V57_LEGACY_M3=1.
    spawn_m3 = _V57_LEGACY_M3 and len(board.active_hypotheses) < _HYPOTHESIS_POOL_TARGET
    if spawn_m3:
        # B9: pass cross-run abstract mechanics to HYPOTHESIZE as priors.
        cross_run_priors = list(
            (board.cross_run_memory or {}).get("abstract_mechanics", [])
        )
        hresp = await spawn_hypothesize(
            summary=board.summary,
            visible_regions=visible_regions,
            falsified_recent=falsified_recent,
            gqb_pair=board.gqb_pair,
            image_b64=image_b64,
            next_card_id_seed=board._next_card_id,
            cross_run_priors=cross_run_priors,
            action_state_chain=chain_full,
            chain_rule_log=board.serialised_chain_rule_log(k=8),
        )
        hypothesize_log = hresp
        for raw_card in (hresp or {}).get("cards", [])[:3]:
            cid = raw_card.get("id") or board.alloc_card_id()
            # Ensure unique
            if any(c["id"] == cid for c in board.active_hypotheses):
                cid = board.alloc_card_id()
            else:
                # Sync next_card_id with what we just consumed.
                if cid.startswith("C") and cid[1:].isdigit():
                    n = int(cid[1:])
                    if n >= board._next_card_id:
                        board._next_card_id = n + 1
            card_type = (raw_card.get("type") or "single").lower()
            card_obj = {
                "id": cid,
                "type": card_type,
                "region_id": raw_card.get("region_id"),
                "abstraction_level": raw_card.get("abstraction_level", "low"),
                "predicate": raw_card.get("predicate", ""),
                "expected_signature": raw_card.get("expected_signature") or {},
                "tested_count": 0,
            }
            # v57.1: pass through plan-card fields for the joint-constraint path.
            if card_type == "plan":
                card_obj["rule_hypothesis"] = raw_card.get("rule_hypothesis", "")
                seq = list(raw_card.get("click_sequence") or [])
                # v57.3: validate plan steps reference EXISTING regions in
                # the current visible_regions snapshot. Drop hallucinated
                # ids and dead regions. Allow markers themselves
                # (commit-step heuristic) and play-zone neighbors.
                valid_region_ids: set[str] = set()
                for r in visible_regions:
                    rid = r.get("id")
                    if rid:
                        valid_region_ids.add(rid)
                # cycle88: revert HARD primary_marker filtering — let the
                # LLM target any visible region; coherence is enforced via
                # prompt only (single-marker sweep).
                seq_valid = [
                    step for step in seq
                    if isinstance(step, dict)
                    and step.get("region_id") in valid_region_ids
                    and step.get("region_id") not in board.dead_regions
                ]
                if not seq_valid:
                    # All steps invalid — drop this plan card silently.
                    continue
                card_obj["click_sequence"] = seq_valid
                card_obj["expected_outcome"] = raw_card.get("expected_outcome") or {"level_delta": 1}
                card_obj["sequence_progress"] = 0  # which step is next
            else:
                # Single-card path: skip if region is dead.
                if raw_card.get("region_id") in board.dead_regions:
                    continue
            board.active_hypotheses.append(card_obj)

    # 2. Action picks next coord.
    # v585zz: precondition-gated trigger registry — pass only events
    # whose region_id is currently visible.
    visible_rids = {r.get("id") for r in visible_regions if r.get("id")}
    matched_prior_triggers = [
        e for e in board.l_plus_events if e.get("region_id") in visible_rids
    ]
    # B11: per-marker 8-neighbor collective state vector. For each visible
    # multicolor marker, build {direction: (neighbor_R-id, current_color,
    # click_count_this_level)} so the agent can reason about JOINT state
    # across the 8 compass neighbors of one marker. cycle181 trace showed
    # the agent toggling all 8 neighbors of R12 to 12, then having no
    # signal that "all 8 toggled = saturation" because the per-region
    # click counter is too local. This collective view exposes saturation
    # without leaking the trigger condition.
    region_color_map: dict = {}
    for r in (visible_regions or []):
        if isinstance(r, dict):
            rid = r.get("id")
            if rid:
                region_color_map[rid] = r.get("color")
    marker_neighbor_states: list = []
    for r in (visible_regions or []):
        if not isinstance(r, dict): continue
        if not r.get("is_multicolor"): continue
        mid = r.get("id")
        n3 = r.get("neighbors_3x3") or {}
        if not n3: continue
        compass = {}
        for direction in ("N","NE","E","SE","S","SW","W","NW"):
            nrid = n3.get(direction)
            if nrid:
                compass[direction] = {
                    "region_id": nrid,
                    "current_color": region_color_map.get(nrid),
                    "clicks": int(board.region_clicks_this_level.get(nrid, 0)),
                }
        if compass:
            marker_neighbor_states.append({
                "marker_id": mid,
                "compass": compass,
            })

    # v587 B12: at the START of this turn, fill compass_changes for the
    # PREVIOUS turn's recent_turn_diffs entry — that entry was waiting
    # for the post-click compass snapshot, which only becomes available
    # now (visible_regions for this turn = post-click frame from last).
    current_compass_snapshot = V57Board._compass_snapshot_from_marker_states(
        marker_neighbor_states
    )
    board.fill_pending_compass_changes(current_compass_snapshot)

    # v587 B13: build LEVEL_BRIDGE_PRIORS from cross-run level_bridges,
    # anonymised (no run-specific R-ids exposed). Cap at 6 entries
    # (most-recent + most-confirmed). Each prior conveys the SHAPE of
    # a successful past transition — not the rule itself.
    level_bridge_priors = _build_level_bridge_priors(
        board.level_bridges.get("bridges", []) or []
    )

    # v587 B14 (ASMW + CPSR): compute stuck severity, expanded turn-diff
    # window, and analogous past segments retrieved from segment_index.
    stuck_sev = board.stuck_severity()
    stuck_expand = stuck_sev // 5  # +2 entries per 5 stuck turns
    expanded_turn_diffs = board.serialised_recent_turn_diffs(expand=stuck_expand)
    analogous_segments: list = []
    if board.stuck_mode and board.segment_index:
        # Build a current-trajectory signature on-the-fly using the same
        # bucket logic as ESC, sourced from the recent_turn_diffs window.
        cur_metrics = _approx_current_metrics(expanded_turn_diffs)
        cur_sig = V57Board.compute_event_signature(cur_metrics)
        # Promote to k=2 once index has ≥10 segments per class.
        succ_n = sum(1 for s in board.segment_index if s.get("did_progress"))
        fail_n = sum(1 for s in board.segment_index if not s.get("did_progress"))
        k = (
            _CPSR_K_PER_CLASS_PROMOTED
            if min(succ_n, fail_n) >= _CPSR_K_PROMOTE_THRESHOLD
            else _CPSR_K_PER_CLASS_DEFAULT
        )
        analogous_segments = _retrieve_analogous_segments(
            current_signature=cur_sig,
            segment_index=board.segment_index,
            k_per_class=k,
        )

    # v589 B17: typed candidate generation. AFTER marker_neighbor_states
    # is built (defined ~30 lines above) and BEFORE spawn_action.
    new_candidates = generate_candidates(
        visible_regions=visible_regions,
        recent_turn_diffs=board.recent_turn_diffs,
        marker_neighbor_states=marker_neighbor_states,
        level_bridges=board.level_bridges.get("bridges", []) or [],
        chain_rule_log=board.chain_rule_log,
        role_history=board.role_history,
        recent_emissions=list(board.recent_candidate_emissions),
        recent_clicks=list(board.recent_clicks),
        turn_index=board.turn_index,
        chain_tokens_len=len(chain_full.get("chain_tokens", []) or []),
        k_per_marker=_CAND_K_PER_MARKER,
        k_global=_CAND_K_GLOBAL,
    )
    board.append_candidate_emissions(new_candidates)
    board.register_candidate_log(new_candidates)
    candidate_tests_for_m1 = board.serialised_candidate_tests_for_m1()

    aresp = await spawn_action(
        summary=board.summary,
        visible_regions=visible_regions,
        active_hypotheses=board.active_hypotheses[:_HYPOTHESIS_POOL_TARGET],
        recent_turns=board.recent_verbose,
        gqb_pair=board.gqb_pair,
        image_b64=image_b64,
        matched_prior_triggers=matched_prior_triggers,
        region_clicks_this_level=dict(board.region_clicks_this_level),
        marker_neighbor_states=marker_neighbor_states,
        recent_turn_diffs=expanded_turn_diffs,
        level_bridge_priors=level_bridge_priors,
        stuck_mode=bool(board.stuck_mode),
        analogous_past_segments=analogous_segments,
        action_state_chain_compact=chain_compact,
        candidate_tests=candidate_tests_for_m1,
    )

    coord = (aresp.get("action") or {}).get("coord") or [32, 32]
    cx, cy = int(coord[0]), int(coord[1])
    cx = max(0, min(63, cx))
    cy = max(0, min(63, cy))

    # v57.2: deterministic snap-to-bbox-center.
    # If LLM emits a coord that lies OUTSIDE every visible_region bbox AND
    # picked a chosen_hypothesis_id, snap coord to the chosen card's
    # region's bbox center. This converts LLM "I want region R22" intent
    # into a coord that actually lands on R22, regardless of LLM coord
    # arithmetic mistakes.
    chosen_id = aresp.get("chosen_hypothesis_id")
    chosen_card = next((c for c in board.active_hypotheses if c.get("id") == chosen_id), None)

    # v586 Phase C: skip dead-region steps in plan-card click_sequence.
    # If the current step's region_id has been marked dead (≥3 _outside_
    # misses), advance sequence_progress past it to the next live step
    # so the click goes somewhere useful rather than repeatedly targeting
    # a phantom-bbox region. If all remaining steps are dead, leave
    # progress at end-of-sequence — verdict logic will resolve the card.
    if chosen_card is not None and chosen_card.get("type") == "plan":
        seq = chosen_card.get("click_sequence") or []
        progress = int(chosen_card.get("sequence_progress", 0))
        while progress < len(seq):
            step_rid = (seq[progress] or {}).get("region_id")
            if step_rid and step_rid in board.dead_regions:
                progress += 1
            else:
                break
        chosen_card["sequence_progress"] = progress

    def _bbox_xyxy(bbox: Any) -> tuple[int, int, int, int] | None:
        """Normalize bbox in either list [x0,y0,x1,y1] or dict form to a tuple."""
        if not bbox:
            return None
        if isinstance(bbox, dict):
            try:
                return (
                    int(bbox.get("min_x", 0)), int(bbox.get("min_y", 0)),
                    int(bbox.get("max_x", 0)), int(bbox.get("max_y", 0)),
                )
            except (TypeError, ValueError):
                return None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            except (TypeError, ValueError):
                return None
        return None

    in_some_region = False
    for r in visible_regions:
        bb = _bbox_xyxy(r.get("bbox"))
        if bb and bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
            in_some_region = True
            break
    snapped = False

    # v57.6: deterministic exploration fallback. After EXPLORATION_TRIGGER
    # turns without any level rise, force-click an unexplored CORNER of a
    # multicolor region. This dodges the failure mode where every plan
    # targets bbox centers and never tests sprite-edge pixels (cycle71's
    # L0 success was a corner click on a multicolor region).
    # cycle78: corner-exploration override hijacked plan steps with
    # coord=None after turn 20. Disabled by default (set to 999); can be
    # re-enabled by setting V57_EXPLORATION_TRIGGER explicitly.
    EXPLORATION_TRIGGER = int(os.environ.get("V57_EXPLORATION_TRIGGER", "999"))
    levels_seen = sum(
        int((t.get("observation") or {}).get("level_delta") or 0)
        for t in board.recent_verbose
    )
    # Approximate "no level rise so far" by checking summary + recent_verbose.
    no_progress = levels_seen == 0 and "level rise" not in (board.summary or "").lower()
    if (
        no_progress
        and board.turn_index >= EXPLORATION_TRIGGER
        and (board.turn_index - EXPLORATION_TRIGGER) % 3 == 0  # every 3 turns post-trigger
    ):
        # Find a multicolor region whose corners haven't been clicked yet.
        clicked_coords = {
            tuple(t.get("coord", []))
            for t in board.recent_verbose
            if t.get("coord")
        }
        for r in visible_regions:
            if not r.get("is_multicolor"):
                continue
            bb = _bbox_xyxy(r.get("bbox"))
            if bb is None:
                continue
            corners = [
                (bb[0], bb[1]),
                (bb[2], bb[1]),
                (bb[0], bb[3]),
                (bb[2], bb[3]),
            ]
            for cor in corners:
                if cor not in clicked_coords:
                    cx, cy = cor
                    snapped = True
                    # We override the LLM's coord but keep its hypothesis id.
                    # This is exploration, not commitment — verdict will
                    # reflect actual observation, not plan-step expectation.
                    break
            if snapped:
                break
    # v57.5: allow plan steps to specify explicit coord (overrides snap).
    # This unblocks "internal pixel targeting" — clicking a specific cell
    # WITHIN a region's bbox (e.g., a corner of a multicolor marker where
    # an unusual-color pixel sits). General puzzle pattern: sprite cells
    # can be click targets, not just region centers.
    if chosen_card is not None and chosen_card.get("type") == "plan":
        seq = chosen_card.get("click_sequence") or []
        progress = int(chosen_card.get("sequence_progress", 0))
        if 0 <= progress < len(seq):
            step = seq[progress] or {}
            step_coord = step.get("coord")
            if (
                isinstance(step_coord, (list, tuple))
                and len(step_coord) == 2
            ):
                try:
                    sx, sy = int(step_coord[0]), int(step_coord[1])
                    if 0 <= sx < 64 and 0 <= sy < 64:
                        cx, cy = sx, sy
                        # Recompute in_some_region with explicit coord.
                        in_some_region = False
                        for r in visible_regions:
                            bb = _bbox_xyxy(r.get("bbox"))
                            if bb and bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
                                in_some_region = True
                                break
                except (TypeError, ValueError):
                    pass

    if not in_some_region and chosen_card is not None:
        target_region_id = chosen_card.get("region_id")
        # Plan card: snap to current sequence step's region, not the card's.
        if chosen_card.get("type") == "plan":
            seq = chosen_card.get("click_sequence") or []
            progress = int(chosen_card.get("sequence_progress", 0))
            if progress < len(seq):
                step_region = (seq[progress] or {}).get("region_id")
                if step_region:
                    target_region_id = step_region
        # Find region in visible_regions
        for r in visible_regions:
            if r.get("id") == target_region_id:
                bb = _bbox_xyxy(r.get("bbox"))
                if bb is not None:
                    cx = (bb[0] + bb[2]) // 2
                    cy = (bb[1] + bb[3]) // 2
                    snapped = True
                break

    # cycle80: anti-oscillation. If the last 2 clicks were the same coord
    # (toggling 9↔8 with no progress), pick a different multicolor-neighbor
    # bbox-center. cycle79's GAME_OVER at action 49 was triggered by a
    # 4-click oscillation on (46,38); preventing the 3rd repeat preserves
    # action budget for new exploration.
    if len(board.recent_verbose) >= 2:
        last2 = [(rv or {}).get("coord") for rv in board.recent_verbose[-2:]]
        if all(lc and lc[0] == cx and lc[1] == cy for lc in last2):
            # Try any multicolor-neighbor whose bbox-center we haven't
            # clicked in the last 4 turns.
            recent_coords = {tuple((rv or {}).get("coord") or [-1, -1])
                              for rv in board.recent_verbose[-4:]}
            for r in visible_regions:
                if not r.get("is_marker_neighbor"):
                    continue
                bb = _bbox_xyxy(r.get("bbox"))
                if bb is None:
                    continue
                ax = (bb[0] + bb[2]) // 2
                ay = (bb[1] + bb[3]) // 2
                if (ax, ay) not in recent_coords:
                    cx, cy = ax, ay
                    snapped = True
                    break

    # 3. Execute (caller-provided).
    obs = await execute_action(cx, cy)

    # 4. Update active hypothesis pool against observation.
    # (chosen_id and chosen_card already resolved above for snap-to-bbox.)
    # v57.3: deterministic dead-region tracking.
    obs_region_id = obs.get("primary_region_id")
    obs_trans = obs.get("dominant_transition")
    if obs_region_id == "_outside_":
        # We don't know which CLAIMED region the click missed; charge it to
        # the chosen card's region_id (or its plan-step's region_id).
        target_rid = None
        if chosen_card is not None:
            if chosen_card.get("type") == "plan":
                seq = chosen_card.get("click_sequence") or []
                progress = int(chosen_card.get("sequence_progress", 0))
                if 0 <= progress < len(seq):
                    target_rid = (seq[progress] or {}).get("region_id")
            target_rid = target_rid or chosen_card.get("region_id")
        if target_rid:
            tally = board.region_click_tally.setdefault(
                target_rid, {"outside": 0, "transitions": 0}
            )
            tally["outside"] += 1
            # cycle78: dead-region auto-block prevented cycle71's revisit
            # pattern (R31 was clicked twice and the 2nd hit was the win).
            # Default threshold raised to effectively-infinity; can be tuned
            # via V57_DEAD_REGION_THRESHOLD if dead-region pruning is needed.
            # v586 Phase C v1: dead-region threshold lowered from 999 to 3.
            # cycle150 confirmed 1-strike was too aggressive (killed valid
            # markers after a single bbox-center miss). 3-strike sweet spot
            # observed in cycle149 (kept markers, pruned phantom-bboxes).
            _DEAD_REGION_THRESHOLD = int(os.environ.get("V57_DEAD_REGION_THRESHOLD", "3"))
            if tally["outside"] >= _DEAD_REGION_THRESHOLD:
                board.dead_regions.add(target_rid)
    elif obs_region_id and obs_trans:
        tally = board.region_click_tally.setdefault(
            obs_region_id, {"outside": 0, "transitions": 0}
        )
        tally["transitions"] += 1
    verdict = "inconclusive"
    if chosen_card is not None:
        chosen_card["tested_count"] = chosen_card.get("tested_count", 0) + 1
        verdict = evaluate_card_against_observation(chosen_card, obs)
        # Plan-card progress bookkeeping.
        if chosen_card.get("type") == "plan":
            chosen_card["sequence_progress"] = int(chosen_card.get("sequence_progress", 0)) + 1
        if verdict == "falsify":
            board.push_falsified(
                chosen_card.get("region_id"),
                (chosen_card.get("expected_signature") or {}).get("dominant_transition"),
            )
            board.remove_hypothesis(chosen_id)
        elif verdict in ("plan_falsify",):
            board.push_falsified(
                f"plan:{chosen_card.get('rule_hypothesis','')[:60]}",
                None,
            )
            board.remove_hypothesis(chosen_id)
        elif verdict == "plan_confirm":
            # Keep the card in active pool but mark for archival; the level
            # rise itself is the canonical signal — Reflexion will summarize.
            board.remove_hypothesis(chosen_id)
        # plan_in_progress / inconclusive / confirm → keep the card

    # 4b. H_RECYCLE: track consecutive inconclusives and evict most-tested
    # card when threshold reached so HYPOTHESIZE re-fires with fresh ideas.
    # Targets the cycle120 plateau pattern: 30+ inconclusives with the same
    # 3-4 cards because falsify never triggered.
    if verdict in ("inconclusive", "plan_in_progress"):
        board.consecutive_inconclusive += 1
    else:
        board.consecutive_inconclusive = 0
    if (board.consecutive_inconclusive >= _INCONCLUSIVE_EVICT_K
            and len(board.active_hypotheses) >= _HYPOTHESIS_POOL_TARGET):
        evict_pool = [c for c in board.active_hypotheses
                      if c.get("tested_count", 0) >= _EVICT_MIN_TESTED]
        if evict_pool:
            most_tested = max(evict_pool, key=lambda c: c.get("tested_count", 0))
            board.push_falsified(
                most_tested.get("region_id") or f"recycle:{most_tested.get('id')}",
                (most_tested.get("expected_signature") or {}).get("dominant_transition"),
            )
            board.remove_hypothesis(most_tested["id"])
            board.consecutive_inconclusive = 0

    # 5. Push verbose turn into rolling window.
    verbose_entry = {
        "turn": board.turn_index,
        "thought": (aresp.get("thought") or "")[:600],
        "chosen_hypothesis_id": chosen_id,
        "coord": [cx, cy],
        "observation": obs,
        "verdict": verdict,
        # v588 B16: include action.expected_observation in the verbose
        # entry so chain_verbose can extract it for prediction_errors.
        "action": {
            "expected_observation": aresp.get("expected_observation") or {},
        },
    }
    board.push_recent(verbose_entry)
    # v588 round-2 D-3: also append a TRIMMED entry into chain_verbose
    # (24-deep) for the chain layer's prediction_errors channel.
    board.append_chain_verbose(verbose_entry)
    # v588 round-2 Q-E + N-2: validate + log chain_rule emissions from
    # the latest M3 call (if any). BUGFIX: use hypothesize_log
    # (defined at function scope as None default) — hresp only exists
    # inside the `if len(active_hypotheses) < target` branch and
    # NameError fires on the no-spawn path.
    if hypothesize_log and isinstance(hypothesize_log.get("chain_rule"), list):
        for cr in hypothesize_log["chain_rule"][:3]:
            board.append_chain_rule_log(cr, turn_emitted=board.turn_index)
    # v588 round-3 reviewer fix — close the loop. After observation,
    # confirm/falsify rules in the chain_rule_log against actual obs
    # so confirmed_count is meaningful and LRU-by-confirmed actually
    # works as designed.
    board.update_chain_rule_log_with_observation(
        observation=obs, turn_now=board.turn_index
    )
    # v589 B17: candidate verdict resolution. Build a richer
    # observation payload (compass_change_count from current
    # turn_diff if available) and resolve in-flight candidates.
    last_diff = (board.recent_turn_diffs[-1]
                 if board.recent_turn_diffs else {})
    cc_count = (
        len(last_diff.get("compass_changes") or [])
        if isinstance(last_diff, dict) else 0
    )
    obs_for_candidates = dict(obs or {})
    obs_for_candidates["compass_change_count"] = cc_count
    new_verdicts = update_candidate_log_with_observation(
        candidate_log=board.candidate_log,
        observation=obs_for_candidates,
        turn_now=board.turn_index,
    )
    # Persist role_history outcomes from new verdicts.
    for v_entry in new_verdicts:
        cid = v_entry.get("candidate_id")
        verdict = v_entry.get("verdict")
        # Find the role for this candidate.
        for c in board.candidate_log:
            if c.get("candidate_id") == cid:
                role = (c.get("suggested_test") or {}).get("role")
                if role and verdict in ("supported", "refuted", "ignored"):
                    board.update_role_history(role, verdict)
                # Round-3 C-9: increment reemit_count when ignored
                # (so M1 gets re-presented up to MAX times).
                if verdict == "ignored":
                    c["reemit_count"] = int(c.get("reemit_count", 0)) + 1
                break
    # Track recent_clicks for next-turn causal_proximity score.
    if obs_region_id and obs_region_id != "_outside_":
        board.recent_clicks.append(obs_region_id)
        if len(board.recent_clicks) > 10:
            board.recent_clicks = board.recent_clicks[-10:]

    # 5b. v585zz: append to L+ event registry on level rise.
    if int(obs.get("level_delta") or 0) >= 1 and obs.get("primary_region_id"):
        board.l_plus_events.append({
            "turn": board.turn_index,
            "region_id": obs.get("primary_region_id"),
            "coord": [cx, cy],
            "dominant_transition": obs.get("dominant_transition"),
            "level_at_event": int(obs.get("level_delta") or 0),
        })

    # B10: per-region click-count tracking. Increment on any click that
    # landed in a real region (not _outside_). Reset on level rise so
    # the counts reflect activity within the current level only — what
    # matters for L+ trigger discovery in the post-reseed layout.
    if obs_region_id and obs_region_id != "_outside_":
        board.region_clicks_this_level[obs_region_id] = (
            board.region_clicks_this_level.get(obs_region_id, 0) + 1
        )

    # v587 B12: append per-turn diff entry. compass_changes is left as
    # None and gets filled at the START of the NEXT turn once the
    # post-click compass snapshot is available. Color transitions /
    # click-region kind / level_delta are filled NOW from `obs`.
    region_color_pre = None
    region_kind_pre = "outside"
    if obs_region_id and obs_region_id != "_outside_":
        for r in (visible_regions or []):
            if isinstance(r, dict) and r.get("id") == obs_region_id:
                region_color_pre = r.get("color")
                region_kind_pre = (
                    "marker_multicolor" if r.get("is_multicolor") else "non_marker"
                )
                break
    color_transitions = []
    dom_t = obs.get("dominant_transition") or {}
    if isinstance(dom_t, dict) and dom_t.get("from") is not None:
        color_transitions.append({
            "region_id": obs_region_id,
            "from": dom_t.get("from"),
            "to": dom_t.get("to"),
        })
    turn_diff_entry = {
        "turn": board.turn_index,
        "click": [cx, cy],
        "click_region_id": obs_region_id,
        "region_kind_pre": region_kind_pre,
        "region_color_pre": region_color_pre,
        "color_transitions": color_transitions,
        "compass_changes": None,  # filled at next turn's start
        "level_delta": int(obs.get("level_delta") or 0),
        "_pre_compass_snapshot": current_compass_snapshot,
    }
    board.append_turn_diff(turn_diff_entry)

    # v587 B13: on level rise, snapshot recent diffs into a level-bridge
    # entry with a Symbolica-derived abstract_signature. Anti-leak:
    # store ONLY structural metrics + compass_change_traj; no rule text.
    # v587 B14: also compute event_signature (ESC) from the bridge
    # metrics; stash it on the board for the M4 promote_to_1A path so
    # the next add_to_1A call clusters by structure not text.
    last_event_signature: str | None = None
    if int(obs.get("level_delta") or 0) >= 1:
        bridge_entry = _compose_level_bridge_entry(
            board=board,
            from_level_delta=int(obs.get("level_delta") or 1),
            recent_diffs_serialised=board.serialised_recent_turn_diffs(),
        )
        if bridge_entry:
            board.append_level_bridge(bridge_entry)
            last_event_signature = V57Board.compute_event_signature(bridge_entry)
        board.region_clicks_this_level = {}
    # v587 B14 (ASMW): update stuck_mode hysteresis after level_delta.
    board.update_stuck_mode(int(obs.get("level_delta") or 0))

    # 5c. v586 B4 v2: defer post-L+ eviction to next turn. The reseed
    # is not visible in the current `visible_regions` (which is the
    # pre-click frame), so eviction here matches against the pre-reseed
    # layout and is mostly a no-op. Set a pending flag and let the next
    # turn's section 0 run the eviction with the post-reseed layout.
    if int(obs.get("level_delta") or 0) >= 1:
        board.pending_post_lp_eviction = True

    # 6. Reflexion every cadence turns OR on level rise.
    reflexion_log = None
    if (
        (board.turn_index + 1) % _REFLEXION_CADENCE == 0
        and len(board.recent_verbose) > 0
    ) or int(obs.get("level_delta") or 0) > 0:
        rresp = await spawn_reflexion(
            previous_summary=board.summary,
            recent_turns=board.recent_verbose,
            active_hypotheses=board.active_hypotheses,
            falsified=board.falsified[-10:],
        )
        reflexion_log = rresp
        new_summary = (rresp or {}).get("summary") or board.summary
        # Soft cap: keep summary ≤500 chars even if LLM exceeds.
        board.summary = new_summary[:500]
        # B9: M4 may emit promote_to_1A — atomic mechanism observations
        # that hold across multiple regions in the recent window. Persist
        # them into cross_run_memory so future runs benefit. Each entry
        # is treated as a distinct mechanism (lower-cased text comparison
        # for dedup); confirmed_runs increments on repeat sightings.
        promotions = (rresp or {}).get("promote_to_1A", []) or []
        if isinstance(promotions, list):
            for entry in promotions[:5]:  # cap per turn to avoid spam
                text = entry if isinstance(entry, str) else (
                    entry.get("text") if isinstance(entry, dict) else None
                )
                if text:
                    # v587 B14 (ESC): pass the L+ event signature when
                    # this Reflexion fired immediately after a level
                    # rise, so cross_run_memory clusters by structural
                    # signature rather than text. Outside of L+ turns
                    # we fall back to text-dedup (signature=None).
                    board.add_to_1A(
                        text, board.turn_index, signature=last_event_signature
                    )

    # 7. Trace.
    board.append_trace(
        {
            "turn": board.turn_index,
            "ts": time.time(),
            "hypothesize_call": hypothesize_log,
            "action": aresp,
            "coord": [cx, cy],
            "snapped_coord": snapped,
            "observation": obs,
            "verdict": verdict,
            "reflexion_call": reflexion_log,
            "summary_after": board.summary,
            "active_hypotheses_after": [c["id"] for c in board.active_hypotheses],
            "visible_region_ids": [r.get("id") for r in (visible_regions or [])][:30],
            # v586: Full compacted visible_regions for splice-fixture support.
            # Each entry = {id, bbox, size, color, is_multicolor, y_band,
            # click_response, is_marker_neighbor, neighbors_3x3?, crop?}.
            "visible_regions": _compact_regions(visible_regions or []),
            "matched_prior_triggers": matched_prior_triggers,
            # v588 visualizer: B14 + B16 runtime payloads exposed for
            # debugging. These were already passed to the LLM but not
            # written to trace.jsonl. Now they are.
            "stuck_mode": bool(board.stuck_mode),
            "stuck_severity": int(board.stuck_severity()),
            "analogous_past_segments": analogous_segments,
            "region_clicks_this_level": dict(board.region_clicks_this_level),
            "marker_neighbor_states": marker_neighbor_states,
            "action_state_chain_compact": chain_compact,
            # v589 B17: candidate-test telemetry.
            "candidate_tests_emitted": new_candidates,
            "candidate_tests_for_m1": candidate_tests_for_m1,
            "candidate_verdicts_this_turn": new_verdicts,
        }
    )

    board.turn_index += 1
    return {
        "verdict": verdict,
        "level_delta": int(obs.get("level_delta") or 0),
        "coord": [cx, cy],
    }


# ---------------------------------------------------------------------------
# Top-level Agent (Swarm-compatible)
# ---------------------------------------------------------------------------


class ArcgenticaV57(Arcgentica):
    """v57 ARC-AGI-3 agent: Action-as-main + Hypothesize sub + Reflexion
    compressor. ARC_NO_GOAL_LEAK strict."""

    MAX_ACTIONS = _MAX_ACTIONS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if os.environ.get("ARC_NO_GOAL_LEAK", "0") != "1":
            raise RuntimeError(
                "v57 requires ARC_NO_GOAL_LEAK=1. Refusing to start."
            )
        ns = os.environ.get("ARC_SIMPLE_NAMESPACE", "").strip()
        if not ns:
            ns = f"v57_{int(time.time())}_{os.getpid()}"
        root = Path("simple_logs")
        self.board = V57Board(
            namespace=ns,
            game_id=self.game_id,
            workdir=root / self.game_id / ns,
        )
        self._prev_level = 0
        logger.info("v57 namespace=%s workdir=%s", ns, self.board.workdir)

    def main(self) -> None:
        """Override Arcgentica.main with our Action-driven loop."""
        asyncio.run(self._async_main())

    async def _async_main(self) -> None:
        # Reuse Arcgentica's submit_action / history infrastructure so
        # SemanticPackets sees Frame objects (with .grid) and an action
        # trail. This is the same wiring agentica_simple uses.
        from agents.structs import GameAction
        from agents.templates.agentica_simple.state import (
            SemanticPackets,
            _grid_to_png_b64,
            _change_pattern_summary,
            _change_bbox,
        )

        submit_action_raw, history, full_history = self._make_submit_action(
            server=None
        )
        # Initialize the game.
        initial_raw = self.take_action(GameAction.RESET)
        if initial_raw:
            self.append_frame(initial_raw)
        initial_frame = submit_action_raw("RESET")

        # SemanticPackets with a NullTrace shim — same pattern as simple.
        class _NullTrace:
            def list(self, kind=None):
                return []
        packets = SemanticPackets(
            history,  # deque[(action_name, Frame)]
            _NullTrace(),
            game_id=self.game_id,
            live_object_supplier=lambda limit=12: [],
        )

        prev_frame = initial_frame
        action_count = 0
        while action_count < self.MAX_ACTIONS:
            current = packets.current(max_regions=30)
            visible_regions = current.get("visible_regions") or []
            current_grid = current.get("current_grid") or [
                row for row in (getattr(prev_frame, "grid", None) or [])
            ]
            png = _grid_to_png_b64(current_grid)
            image_b64 = png

            agent_self = self  # closure
            local_prev_frame = prev_frame

            async def execute(cx: int, cy: int, _prev=local_prev_frame) -> dict:
                try:
                    new_frame = submit_action_raw("ACTION6", x=cx, y=cy)
                except (IndexError, ValueError) as exc:
                    # GAME_OVER / empty frame after end-of-game.
                    logger.info("v57 execute halted: %s", exc)
                    agent_self._latest_frame = None
                    return compose_observation(
                        region_id=None, transition=None, level_delta=0,
                    )
                level = int(getattr(new_frame, "levels_completed", 0) or 0)
                delta = level - agent_self._prev_level
                agent_self._prev_level = level
                # Compute primary_region_id + dominant_transition by
                # diffing prev vs new grid.
                pat = _change_pattern_summary(
                    getattr(_prev, "grid", []) or [],
                    getattr(new_frame, "grid", []) or [],
                )
                bbox = _change_bbox(
                    getattr(_prev, "grid", []) or [],
                    getattr(new_frame, "grid", []) or [],
                )
                # v57.4: precise region attribution.
                # Bug in earlier versions: "first overlapping bbox wins" picked
                # the multicolor marker (large bbox containing many neighbors)
                # whenever a neighbor was clicked. Now: prefer the region
                # whose bbox MOST TIGHTLY contains the click coord (cx, cy);
                # fall back to smallest overlap with the change_bbox.
                primary_region = None
                if bbox is None:
                    primary_region = "_outside_"
                elif visible_regions:
                    # First pass: regions whose bbox CONTAINS the click coord.
                    containing = []
                    for r in visible_regions:
                        rb = r.get("bbox") or {}
                        rx0 = rb.get("min_x", 999); ry0 = rb.get("min_y", 999)
                        rx1 = rb.get("max_x", -1); ry1 = rb.get("max_y", -1)
                        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                            area = max(1, (rx1 - rx0 + 1) * (ry1 - ry0 + 1))
                            containing.append((area, r.get("id")))
                    if containing:
                        # Smallest containing region wins (most specific).
                        containing.sort(key=lambda x: x[0])
                        primary_region = containing[0][1]
                    else:
                        # Fallback: smallest region overlapping change bbox.
                        overlaps = []
                        bx0 = bbox.get("min_x", 99); by0 = bbox.get("min_y", 99)
                        bx1 = bbox.get("max_x", -1); by1 = bbox.get("max_y", -1)
                        for r in visible_regions:
                            rb = r.get("bbox") or {}
                            rx0 = rb.get("min_x", 999); ry0 = rb.get("min_y", 999)
                            rx1 = rb.get("max_x", -1); ry1 = rb.get("max_y", -1)
                            if rx0 <= bx1 and rx1 >= bx0 and ry0 <= by1 and ry1 >= by0:
                                area = max(1, (rx1 - rx0 + 1) * (ry1 - ry0 + 1))
                                overlaps.append((area, r.get("id")))
                        if overlaps:
                            overlaps.sort(key=lambda x: x[0])
                            primary_region = overlaps[0][1]
                # Save updated frame for next turn.
                nonlocal_prev = new_frame
                # We can't `nonlocal prev_frame` from inside async closure
                # cleanly here; just return obs and update outside.
                agent_self._latest_frame = new_frame
                return compose_observation(
                    region_id=primary_region,
                    transition=pat.get("dominant_transition"),
                    level_delta=delta,
                )

            await run_turn(
                board=self.board,
                visible_regions=visible_regions,
                image_b64=image_b64,
                spawn_action=call_action,
                spawn_hypothesize=call_hypothesize,
                spawn_reflexion=call_reflexion,
                execute_action=execute,
            )
            # Refresh prev_frame for next iteration.
            new_latest = getattr(self, "_latest_frame", None)
            if new_latest is None:
                # Game ended (GAME_OVER or WIN); exit loop cleanly.
                logger.info("v57 main loop exiting on game end at action=%d", action_count)
                break
            prev_frame = new_latest
            action_count += 1

        logger.info(
            "v57 main loop ended at action=%d turn=%d level=%d",
            action_count,
            self.board.turn_index,
            self._prev_level,
        )
