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


@dataclass
class PolicyDecision:
    arm_key: ArmKey
    coord_xy: tuple[int, int]
    target_marker_id: str | None
    target_marker_status: str
    confidence_override_used: bool = False


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
    """Choose arm + coord. Returns None when no visible region is selectable."""
    visible_regions = _visible_regions(state)
    if not visible_regions:
        return None
    target_marker, target_reason = select_target_marker(state, proposer_output)
    target_marker_id = target_marker.get("marker_id") if target_marker else None
    _, _, target_status = compute_saturation_status(target_marker)

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
    for pid in preds:
        # v601: skip saturation_progress predicate when no target_marker_id
        # (so v600 fixtures without target hint still work).
        if not has_target:
            if pid in ("P12_saturation_progress", "P_saturation_progress"):
                continue
            if getattr(preds[pid], "family", "") == "saturation_progress":
                continue
        for rid in rids:
            key = ArmKey(pid, rid, target_status)
            arm = posterior.arms.get(key)
            if arm is None or arm.n_emit == 0:
                # Cold start: family priority bumps
                family = getattr(preds[pid], "family", "fallback")
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
