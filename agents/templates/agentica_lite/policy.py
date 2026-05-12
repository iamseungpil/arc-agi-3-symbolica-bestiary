"""v601 Role 2: Policy (deterministic). Plan rev C §3.4-3.14.

Computes saturation_status, encodes ArmKey with saturation dimension, runs
UCB1 over visible (predicate × region × saturation_status) arms, applies
per-episode-capped confidence override on near-tie scores, resolves coord,
and computes the verdict against expected_signature.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .predicate_library import PredicateLibrary
from .predicate_posterior import ArmKey, ArmStats, PredicatePosterior

logger = logging.getLogger(__name__)


# Plan §3.6: saturation_status thresholds
def compute_saturation_status(marker: dict | None) -> tuple[int, int, str]:
    """Return (clicked_count, denominator, status)."""
    if not marker:
        return 0, 0, "n/a"
    compass = marker.get("compass") or {}
    n = len(compass)
    if n == 0:
        return 0, 0, "n/a"
    clicked = sum(1 for c in compass.values() if (c or {}).get("clicks", 0) >= 1)
    if clicked == n:
        return clicked, n, "complete"
    if clicked == n - 1:
        return clicked, n, "near_complete"
    return clicked, n, "none"


def compute_constraint_status(state: dict | None, marker_id: str | None = None) -> tuple[int, int, str]:
    """Return (unsatisfied, total, status) for v608 marker constraints.

    v608b: prefer high-`evidence_quality` constraints (multicolor markers).
    If at least one such constraint exists in the filtered set, the count is
    restricted to the high-quality subset. Otherwise the function falls back
    to all constraints so single-color "marker" fallbacks remain accessible.
    """
    if not state:
        return 0, 0, "n/a"
    constraints = state.get("marker_constraints") or []
    rows = [
        c for c in constraints
        if isinstance(c, dict)
        and (marker_id is None or c.get("marker_id") == marker_id)
    ]
    if not rows:
        return 0, 0, "n/a"
    high_rows = [c for c in rows if c.get("evidence_quality") == "high"]
    chosen = high_rows if high_rows else rows
    total = len(chosen)
    if total == 0:
        return 0, 0, "n/a"
    unsat = sum(1 for c in chosen if not bool(c.get("satisfied", True)))
    if unsat == 0:
        return unsat, total, "constraint_satisfied"
    return unsat, total, "constraint_unsatisfied"


def _markers_of(state: Any) -> list[dict]:
    if isinstance(state, dict):
        return state.get("marker_neighbor_states") or []
    return getattr(state, "marker_neighbor_states", None) or []


def _visible_regions(state: Any) -> list[Any]:
    if isinstance(state, dict):
        return state.get("visible_regions") or []
    return getattr(state, "visible_regions", None) or []


def _region_id(r: Any) -> str | None:
    if isinstance(r, dict):
        return r.get("region_id") or r.get("id")
    return getattr(r, "region_id", None) or getattr(r, "id", None)


def select_target_marker(
    state: dict, proposer_output: Any | None,
) -> tuple[dict | None, str]:
    """Plan §3.6 priority list (G19' corrected) + §3.14 marker hint validation (rev C C1).

    Returns (target_marker_dict, reason).
    """
    markers = _markers_of(state)
    primary_markers = [m for m in markers if m.get("is_primary_marker") is not False]  # treat None or True
    # Without is_primary_marker info, we still consider all markers (real traces have it None).
    candidates = primary_markers if primary_markers else list(markers)

    # Priority 0: v608 marker with most unsatisfied constraints. v608b prefers
    # markers whose constraints are high-quality (multicolor); single-color
    # marker fallbacks are only used when no high-quality marker has any
    # unsatisfied constraint.
    constraint_rank_high: list[tuple[int, int, dict]] = []
    constraint_rank_any: list[tuple[int, int, dict]] = []
    constraints_all = state.get("marker_constraints") or []
    high_marker_ids = {
        c.get("marker_id") for c in constraints_all
        if isinstance(c, dict) and c.get("evidence_quality") == "high"
        and not bool(c.get("satisfied", True))
    }
    for m in candidates:
        unsat, total, _ = compute_constraint_status(state, m.get("marker_id"))
        if total > 0 and unsat > 0:
            if m.get("marker_id") in high_marker_ids:
                constraint_rank_high.append((unsat, total, m))
            else:
                constraint_rank_any.append((unsat, total, m))
    if constraint_rank_high:
        constraint_rank_high.sort(key=lambda x: (-x[0], -x[1]))
        return constraint_rank_high[0][2], "constraint_unsatisfied"
    if constraint_rank_any:
        constraint_rank_any.sort(key=lambda x: (-x[0], -x[1]))
        return constraint_rank_any[0][2], "constraint_unsatisfied"

    # Priority 1: proposer-provided marker_id, with rev C C1 validation
    if proposer_output is not None:
        pre_state = getattr(proposer_output, "required_pre_state", None) or {}
        marker_hint = pre_state.get("marker_id") if isinstance(pre_state, dict) else None
        denom_hint = pre_state.get("saturation_denominator") if isinstance(pre_state, dict) else None
        if marker_hint:
            for m in candidates:
                if m.get("marker_id") == marker_hint:
                    if denom_hint is not None and len(m.get("compass") or {}) != int(denom_hint):
                        # Validation failure: demote to priority 2
                        break
                    return m, "proposer_hint"

    # Priority 2: any marker with status == near_complete (argmax saturation_numerator if many)
    near_complete = []
    for m in candidates:
        clicked, _denom, status = compute_saturation_status(m)
        if status == "near_complete":
            near_complete.append((clicked, m))
    if near_complete:
        near_complete.sort(key=lambda x: -x[0])
        return near_complete[0][1], "near_complete"

    # Priority 3: argmax saturation_numerator with at least 1 click
    with_progress = []
    for m in candidates:
        clicked, _denom, _status = compute_saturation_status(m)
        if clicked > 0:
            with_progress.append((clicked, m))
    if with_progress:
        with_progress.sort(key=lambda x: -x[0])
        return with_progress[0][1], "argmax_progress"

    # Priority 4: sentinel
    return None, "n/a_sentinel"


@dataclass
class EpisodeState:
    """Per-episode counters used for confidence override cap (§3.14 rev C C3)."""
    confidence_override_count: int = 0
    policy_decisions: int = 0
    # v608f local-transition-cache repeat-click bookkeeping.
    # `repeats_per_region` counts the EXTRA repeats already spent on each
    # region (0..V608F_MAX_REPEATS_PER_REGION). `repeats_budget_used` is the
    # episode-wide budget consumed. `last_clicked_region_id` is the region
    # id (str) where the previous policy decision landed; the active
    # override gate uses it as the candidate region to force-repeat. It is
    # cleared on level_delta > 0 via `reset_repeat_state`.
    repeats_per_region: dict[str, int] = field(default_factory=dict)
    repeats_budget_used: int = 0
    last_clicked_region_id: str | None = None


def reset_repeat_state(episode: "EpisodeState") -> None:
    """Clear v608f repeat bookkeeping. Call on level_delta > 0 or on a fresh
    episode boundary so the next exploration phase starts clean.
    """
    episode.repeats_per_region.clear()
    episode.repeats_budget_used = 0
    episode.last_clicked_region_id = None


@dataclass
class PolicyDecision:
    arm_key: ArmKey
    coord_xy: tuple[int, int]
    target_marker_id: str | None
    target_marker_status: str
    confidence_override_used: bool = False
    # v608f: True when the policy chose this region as a repeat-click for
    # local transition cache sampling rather than for constraint repair.
    repeat_click_used: bool = False


# v608f tunables. Kept tight per the plan: region x2, episode x30.
V608F_MAX_REPEATS_PER_REGION = 2
V608F_EPISODE_REPEAT_BUDGET = 30
V608F_PROTECTED_FAMILIES = ("constraint_repair",)
V608F_PROTECTED_PIDS = (
    "P13_unsatisfied_marker_constraint",
    "P14_best_constraint_delta",
)
# v608f-fix: explicit predicate id stamped on policy-driven repeat-click
# actions. Standard scoring never picks this; it is set only by
# `_v608f_active_override` below.
V608F_REPEAT_PREDICATE_ID = "P16_transition_cache_repeat"
V608F_REPEAT_STATUS = "transition_cache_collecting"
# v609: graph-solver predicate, never picked by standard scoring.
V609_SEARCH_PREDICATE_ID = "P17_search_step"
# Convenience: all predicates that the standard scoring loop must skip.
V609_RESERVED_PREDICATE_IDS = (V608F_REPEAT_PREDICATE_ID,
                                V609_SEARCH_PREDICATE_ID)


def _override_cap(episode: EpisodeState) -> int:
    """Plan §3.14: cap = min(3, ceil(0.15 * policy_decisions))."""
    if episode.policy_decisions <= 0:
        return 1  # at least one override allowed once decisions begin
    return max(1, min(3, math.ceil(0.15 * episode.policy_decisions)))


def select_arm(
    state: dict,
    posterior: PredicatePosterior,
    library: PredicateLibrary,
    proposer_output: Any | None,
    episode: EpisodeState,
    exploration_boost: dict[ArmKey, float] | None = None,
) -> PolicyDecision | None:
    """Choose arm + coord. Returns None when no visible region is selectable.

    v608f-fix: BEFORE the standard scoring path, the active-override gate
    decides whether to force a repeat-click on the previously clicked
    region. When it fires, the function returns immediately with an arm
    stamped as `P16_transition_cache_repeat` so card evidence attribution
    stays explicit.
    """
    visible_regions = _visible_regions(state)
    if not visible_regions:
        return None
    # v608f-fix: active override path. Fires only when the previous click's
    # region has an unfinished local transition cache (n_samples in {1, 2})
    # AND the region is still visible AND repeat budgets are not exhausted.
    override = _v608f_active_override(state, episode, library, visible_regions)
    if override is not None:
        return override
    target_marker, target_reason = select_target_marker(state, proposer_output)
    target_marker_id = target_marker.get("marker_id") if target_marker else None
    _, _, target_status = compute_saturation_status(target_marker)
    _cu, _ct, constraint_status = compute_constraint_status(state, target_marker_id)
    if constraint_status != "n/a":
        target_status = constraint_status

    # Inject target_marker_id into state copy so P12 can see it
    state_for_predicates = dict(state) if isinstance(state, dict) else {}
    if target_marker_id is not None:
        state_for_predicates["target_marker_id"] = target_marker_id
    # v607 P3: inject turn_count for unknown-predicate grid rotation in resolve_coord.
    if "turn_count" not in state_for_predicates:
        state_for_predicates["turn_count"] = int(episode.policy_decisions)

    preds = library.all_predicates() if library is not None else {}
    if not preds:
        return None
    N = max(1, sum(a.n_emit for a in posterior.arms.values()))
    log_N = math.log(N + 1)

    rids = [_region_id(r) for r in visible_regions]
    rids = [rid for rid in rids if rid is not None]
    if not rids:
        return None

    # Build candidate arm set
    has_target = target_marker_id is not None
    candidates: list[tuple[ArmKey, float]] = []
    for pid, pred in preds.items():
        # v608f-fix + v609: P16 and P17 are reserved for active override
        # and the offline A* solver respectively. Standard scoring loop
        # must never pick either.
        if pid in V609_RESERVED_PREDICATE_IDS:
            continue
        # v601: skip saturation_progress predicate when no target_marker_id
        # (so v600 fixtures without target hint still work).
        if not has_target:
            if pid in ("P12_saturation_progress", "P_saturation_progress"):
                continue
            if getattr(pred, "family", "") == "saturation_progress":
                continue
        pred_rids = set(rids)
        try:
            pred_out = pred.fn(state_for_predicates, episode.policy_decisions)
            pred_rids = {
                _region_id(x)
                for x in pred_out
                if _region_id(x) is not None
            }
        except Exception:  # noqa: BLE001
            pred_rids = set()
        if (
            not pred_rids
            and proposer_output is not None
            and getattr(proposer_output, "candidate_predicate_id", None) == pid
            and getattr(proposer_output, "region_hint", None) in rids
        ):
            # Backward-compatible v601 path: older proposer fixtures supply
            # the intended visible region directly even when the predicate's
            # structural filter has no target-specific output.
            pred_rids = {getattr(proposer_output, "region_hint")}
        if not pred_rids:
            continue
        for rid in [r for r in rids if r in pred_rids]:
            key = ArmKey(pid, rid, target_status)
            arm = posterior.arms.get(key)
            if arm is None or arm.n_emit == 0:
                # Cold start: family priority bumps
                family = getattr(pred, "family", "fallback")
                from .predicate_posterior import _FAMILY_PRIORITY
                score = 1e6 + _FAMILY_PRIORITY.get(family, 0.0)
            else:
                exploit = arm.alpha / (arm.alpha + arm.beta)
                explore = math.sqrt(2.0 * log_N / arm.n_emit)
                score = exploit + explore
            if exploration_boost is not None:
                score += exploration_boost.get(key, 0.0)
            candidates.append((key, score))

    if not candidates:
        return None
    candidates.sort(key=lambda kv: -kv[1])
    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    chosen_key = top[0]
    override_used = False

    # Plan §3.14 confidence override (capped per-episode). When proposer succeeds
    # with confidence >= 0.5 and the top scores are tied within 0.05 (cold-start
    # case where all family-priority-matching arms tie), prefer the arm whose
    # (predicate_id, region_id) match the proposer's exact (candidate_predicate_id,
    # region_hint) tuple. This honors the proposer's specific proposal under §2's
    # architecture without granting it unconditional override (cap per-episode).
    if (
        proposer_output is not None
        and second is not None
        and (top[1] - second[1]) < 0.05
        and getattr(proposer_output, "confidence", 0.0) >= 0.5
    ):
        cap = _override_cap(episode)
        if episode.confidence_override_count < cap:
            region_hint = getattr(proposer_output, "region_hint", None)
            cand_pid = getattr(proposer_output, "candidate_predicate_id", None)
            # Pass 1: exact (predicate, region) match (strongest preference).
            chose = False
            if cand_pid is not None and region_hint is not None:
                for key, _score in candidates:
                    if key.predicate_id == cand_pid and key.region_id == region_hint:
                        chosen_key = key
                        episode.confidence_override_count += 1
                        override_used = True
                        chose = True
                        break
            # Pass 2: region-only match (if exact predicate not present).
            if not chose and region_hint is not None:
                for key, _score in candidates:
                    if key.region_id == region_hint:
                        chosen_key = key
                        episode.confidence_override_count += 1
                        override_used = True
                        break
    if (
        proposer_output is not None
        and not override_used
        and getattr(proposer_output, "confidence", 0.0) >= 0.5
        and getattr(proposer_output, "candidate_predicate_id", None) == chosen_key.predicate_id
        and getattr(proposer_output, "region_hint", None) == chosen_key.region_id
    ):
        # The proposer hint may already be the top-scoring arm after predicate
        # filtering. Count this as an honored confidence override for v601
        # telemetry/backward-compatible tests.
        cap = _override_cap(episode)
        if episode.confidence_override_count < cap:
            episode.confidence_override_count += 1
            override_used = True

    # v608f-fix: the standard scoring path does NOT fire repeat-click. The
    # active override at the top of `select_arm` is the only place that
    # stamps P16. So `repeat_click_used` is always False here.
    repeat_click_used = False

    # Coord resolution via library (carry v600 resolver).
    region_obj = next(
        (r for r in visible_regions if _region_id(r) == chosen_key.region_id),
        None,
    )
    try:
        xy = library.resolve_coord(
            chosen_key.predicate_id,
            region_obj or {},
            state_for_predicates,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("policy.resolve_coord failed: %s", e)
        xy = (0, 0)

    episode.policy_decisions += 1
    return PolicyDecision(
        arm_key=chosen_key,
        coord_xy=xy,
        target_marker_id=target_marker_id,
        target_marker_status=target_status,
        confidence_override_used=override_used,
        repeat_click_used=repeat_click_used,
    )


# ---------------------------------------------------------------------------
# v608f-fix local-transition-cache active override (deterministic).
# ---------------------------------------------------------------------------


def _v608f_active_override(
    state: Any,
    episode: EpisodeState,
    library: Any,
    visible_regions: list[Any],
) -> PolicyDecision | None:
    """Force a repeat-click on the previously clicked region when its local
    transition cache is "almost confirmed".

    Returns a fully formed `PolicyDecision` with `predicate_id =
    P16_transition_cache_repeat` and `target_marker_status =
    transition_cache_collecting` when ALL of the following hold:

      - `episode.last_clicked_region_id` is set;
      - that region is still in `visible_regions` (actionable);
      - the region's `region_transition_cache` entry has `n_samples` 1 or 2
        (some observed transition exists but cycle is not yet confirmed);
      - the per-region repeat cap is not exhausted;
      - the episode-wide repeat budget is not exhausted.

    Returns None otherwise. The caller proceeds with the standard scoring
    path.
    """
    if not isinstance(state, dict):
        return None
    rid = episode.last_clicked_region_id
    if not rid:
        return None
    # Region must still be visible/actionable.
    visible_ids = {_region_id(r) for r in visible_regions}
    visible_ids.discard(None)
    if rid not in visible_ids:
        return None
    cache = (state.get("region_transition_cache")
             or state.get("region_transitions")
             or {})
    if not isinstance(cache, dict):
        return None
    entry = cache.get(rid) or {}
    n_samples = int(entry.get("n_samples", 0) or 0)
    if n_samples not in (1, 2):
        return None
    # Budget guards.
    if episode.repeats_budget_used >= V608F_EPISODE_REPEAT_BUDGET:
        return None
    if (episode.repeats_per_region.get(rid, 0)
            >= V608F_MAX_REPEATS_PER_REGION):
        return None
    # Resolve coord via the library so the click lands inside the region.
    region_obj = next(
        (r for r in visible_regions if _region_id(r) == rid), None,
    )
    try:
        xy = library.resolve_coord(
            V608F_REPEAT_PREDICATE_ID, region_obj or {}, state,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("v608f override resolve_coord failed: %s", e)
        xy = (0, 0)
    arm_key = ArmKey(V608F_REPEAT_PREDICATE_ID, rid, V608F_REPEAT_STATUS)
    # Commit budget bookkeeping.
    episode.repeats_per_region[rid] = (
        episode.repeats_per_region.get(rid, 0) + 1
    )
    episode.repeats_budget_used += 1
    episode.policy_decisions += 1
    return PolicyDecision(
        arm_key=arm_key,
        coord_xy=xy,
        target_marker_id=None,
        target_marker_status=V608F_REPEAT_STATUS,
        confidence_override_used=False,
        repeat_click_used=True,
    )


@dataclass
class Verdict:
    matched: bool
    observed_level_delta: int
    expected_level_delta: int | None
    notes: str = ""


def compute_verdict(observation: dict, expected_signature: dict | None) -> Verdict:
    """Plan §3.7: paired-cf trigger predicate uses level_delta + dt.count.

    Verdict matched := observed.level_delta >= expected.level_delta (when set).
    """
    obs_ld = int((observation or {}).get("level_delta") or 0)
    exp_ld = None
    if expected_signature:
        if "level_delta" in expected_signature:
            try:
                exp_ld = int(expected_signature["level_delta"])
            except (TypeError, ValueError):
                exp_ld = None
    if exp_ld is None:
        return Verdict(matched=obs_ld > 0, observed_level_delta=obs_ld, expected_level_delta=None)
    return Verdict(
        matched=obs_ld >= exp_ld,
        observed_level_delta=obs_ld,
        expected_level_delta=exp_ld,
    )


def record_outcome(
    posterior: PredicatePosterior,
    arm_key: ArmKey,
    verdict: Verdict,
) -> None:
    """Update posterior alpha/beta given a verdict on a chosen arm."""
    arm = posterior.arms.get(arm_key)
    if arm is None:
        arm = ArmStats()
        posterior.arms[arm_key] = arm
    if verdict.matched:
        arm.alpha += 1.0
    else:
        arm.beta += 1.0
    arm.last_level_delta = verdict.observed_level_delta
