"""B17 plan v589 — Symbolica deterministic typed-candidate generator.

Pure deterministic. No LLM. Emits typed falsifiable candidate tests
that M1 consumes via the candidate_id binding line. Replaces M3
chain_rule emission as the live hypothesis surface.

Design (from codex round 3-4 convergence + plan v589 self-critic round 3):

- Closed role enum (10 roles) — strings are FIXED at code level.
  V-LEAK audit (per-string regex check) is enforced at unit-test time
  AND runtime via _audit_emitted_role.

- Per-marker beam = 2 + global beam = 8 (per-marker cap prevents
  single-marker monopoly while global cap controls prompt bloat).

- Candidate schema:
    {candidate_id, suggested_test:{role, anchor_marker_id,
     abstraction_hint}, expected_observable_signature,
     refutation_signature, score}
  Validator enforces expected ≠ refutation.

- Score formula (round-2 C-2 pinned):
    score = novelty * causal_proximity * delta_correlation_prior
    novelty               = 1 / (1 + n_emitted_same_role_anchor_in_5_turns)
    causal_proximity      = 1.0 if anchor_marker in last_3_clicks else 0.5
    delta_correlation_prior = 0.5 (cold-start) or
                              n_supported / max(1, n_observed)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


# --------------------------------------------------------------------
# Role enum — FIXED, leak-safe.
# --------------------------------------------------------------------

ALLOWED_ROLES: tuple[str, ...] = (
    "unchanged_marker_body_with_changed_adjacent_role",
    "shared_adjacent_region_between_two_markers",
    "same_marker_role_next_to_changed_region",
    "recent_positive_delta_region_role_reused_elsewhere",
    "two_marker_overlap_sector_candidate",
    "not_yet_clicked_neighbor_of_active_marker",
    "recently_clicked_region_revisit_candidate",
    "compass_change_propagation_neighbor",
    # Round-3 force-test gate role.
    "force_test_joint_unrelated_marker_v1",
    # Round-1 unrecognised-pattern fallback (logs warning).
    "unrecognised_pattern_role_v1",
)

# Forbidden vocabulary (mirrors scripts/check_no_leak_prompts.py
# FORBIDDEN_TERMS but kept here for runtime audit independence).
_FORBIDDEN_VOCAB = re.compile(
    r"per_neighbor_target|needs_toggle|marker_progress|joint_neighbors|"
    r"expected_neighbor_colors|target_color|win.?state|win.?condition|"
    r"target.?state|complete the (level|pattern|grid)|"
    r"correct color|right color|should be color|must be color|"
    r"match the indicator|match the pattern|the goal is",
    re.IGNORECASE,
)


def _audit_emitted_role(role: str) -> bool:
    """Per-emission runtime audit. Asserts role ∈ ALLOWED_ROLES AND
    is leak-safe. Returns True if pass; False if fail (caller logs)."""
    if role not in ALLOWED_ROLES:
        return False
    if _FORBIDDEN_VOCAB.search(role):
        return False
    return True


# --------------------------------------------------------------------
# Candidate schema validation.
# --------------------------------------------------------------------


def _validate_candidate_signatures(c: dict) -> bool:
    """Round-3 C-11: enforce expected_observable_signature !=
    refutation_signature per candidate. Otherwise the verdict table
    cannot disambiguate."""
    exp = c.get("expected_observable_signature") or {}
    ref = c.get("refutation_signature") or {}
    if exp == ref:
        return False
    return True


def _candidate_id(turn: int, role: str, anchor: str | None) -> str:
    """Deterministic candidate id. C{turn}:{role-tag}:{6-hex-hash}."""
    role_tag = role.split("_", 2)[0]  # short tag
    payload = f"{turn}|{role}|{anchor or ''}"
    h = hashlib.sha1(payload.encode()).hexdigest()[:6]
    return f"C{turn}:{role_tag}:{h}"


# --------------------------------------------------------------------
# Score formula (plan v589 §4.1, round-2 C-2 pinned).
# --------------------------------------------------------------------


def _novelty(role: str, anchor: str | None,
             recent_emissions: list[dict]) -> float:
    """1 / (1 + n_emitted_same_role_anchor_in_last_5_turns)."""
    n = sum(
        1 for e in recent_emissions[-5 * 8:]   # last 5 turns × beam=8
        if e.get("role") == role
        and e.get("anchor_marker_id") == anchor
    )
    return 1.0 / (1.0 + n)


def _causal_proximity(anchor: str | None,
                      recent_clicks: list[str]) -> float:
    """1.0 if anchor in last 3 clicks else 0.5."""
    if anchor and anchor in recent_clicks[-3:]:
        return 1.0
    return 0.5


def _delta_correlation_prior(role: str,
                             role_history: dict) -> float:
    """Cold-start 0.5; otherwise n_supported / max(1, n_observed)."""
    entry = role_history.get(role)
    if not entry:
        return 0.5
    n_obs = int(entry.get("observed_supported", 0)) + \
            int(entry.get("observed_refuted", 0)) + \
            int(entry.get("observed_ignored", 0))
    if n_obs == 0:
        return 0.5
    return float(entry.get("observed_supported", 0)) / max(1, n_obs)


def _score(role: str, anchor: str | None,
           recent_emissions: list[dict],
           recent_clicks: list[str],
           role_history: dict) -> float:
    return (
        _novelty(role, anchor, recent_emissions)
        * _causal_proximity(anchor, recent_clicks)
        * _delta_correlation_prior(role, role_history)
    )


# --------------------------------------------------------------------
# Per-role candidate proposers.
# --------------------------------------------------------------------


def _propose_per_role(
    *, role: str,
    visible_regions: list[dict],
    recent_turn_diffs: list[dict],
    marker_neighbor_states: list[dict],
) -> list[dict]:
    """Each role has a deterministic generation rule.
    Returns 0+ proposals per (role × anchor) pair.

    proposals = [{role, anchor_marker_id, abstraction_hint,
                  expected_observable_signature, refutation_signature}]
    """
    proposals: list[dict] = []

    # Multicolor markers from visible_regions.
    markers = [r for r in (visible_regions or [])
               if isinstance(r, dict) and r.get("is_multicolor")]
    last_diff = (recent_turn_diffs or [])[-1] if recent_turn_diffs else {}
    last_compass_changes = (last_diff or {}).get("compass_changes") or []
    last_color_transitions = (last_diff or {}).get("color_transitions") or []

    for m in markers[:5]:
        mid = m.get("id")
        if not mid:
            continue

        if role == "unchanged_marker_body_with_changed_adjacent_role":
            # Marker compass changed but marker color did not.
            marker_color_changed = any(
                t.get("region_id") == mid for t in last_color_transitions
            )
            adjacent_changed = any(
                c.get("marker_id") == mid for c in last_compass_changes
            )
            if (not marker_color_changed) and adjacent_changed:
                proposals.append({
                    "role": role, "anchor_marker_id": mid,
                    "abstraction_hint": "test if marker body stays stable while neighbor flips",
                    "expected_observable_signature": {
                        "transition_kind": "neighbor_only",
                        "compass_change_count": 1,
                        "level_delta_min": 0,
                    },
                    "refutation_signature": {
                        "transition_kind": "marker_body_changed",
                        "compass_change_count": 0,
                        "level_delta_max": 0,
                    },
                })

        elif role == "shared_adjacent_region_between_two_markers":
            # Two markers' compass dicts share a region_id.
            for m2 in markers:
                if m2.get("id") == mid:
                    continue
                m1_compass = m.get("neighbors_3x3") or {}
                m2_compass = m2.get("neighbors_3x3") or {}
                shared = set(m1_compass.values()) & set(m2_compass.values())
                shared = {s for s in shared if s and isinstance(s, str)}
                if shared:
                    proposals.append({
                        "role": role, "anchor_marker_id": mid,
                        "abstraction_hint": "test region shared between markers; click affects both",
                        "expected_observable_signature": {
                            "transition_kind": "shared_region_change",
                            "compass_change_count": 2,
                            "level_delta_min": 0,
                        },
                        "refutation_signature": {
                            "transition_kind": "shared_region_change",
                            "compass_change_count": 1,
                            "level_delta_max": 0,
                        },
                    })
                    break  # one per marker is enough

        elif role == "same_marker_role_next_to_changed_region":
            # Marker's adjacent region just changed → re-test same role.
            if any(c.get("marker_id") == mid for c in last_compass_changes):
                proposals.append({
                    "role": role, "anchor_marker_id": mid,
                    "abstraction_hint": "marker neighbor just changed; verify role response",
                    "expected_observable_signature": {
                        "transition_kind": "same_role_repeat",
                        "compass_change_count": 1,
                        "level_delta_min": 0,
                    },
                    "refutation_signature": {
                        "transition_kind": "same_role_no_change",
                        "compass_change_count": 0,
                        "level_delta_max": 0,
                    },
                })

        elif role == "not_yet_clicked_neighbor_of_active_marker":
            n3 = m.get("neighbors_3x3") or {}
            click_response = m.get("click_response") or {}
            clicks = int((click_response.get("clicks") or 0))
            unclicked_neighbors = [
                v for k, v in n3.items()
                if v and v != "_outside_"
            ]
            if unclicked_neighbors and clicks >= 1:
                proposals.append({
                    "role": role, "anchor_marker_id": mid,
                    "abstraction_hint": "click an untested neighbor of an active marker",
                    "expected_observable_signature": {
                        "transition_kind": "new_neighbor_change",
                        "compass_change_count": 1,
                        "level_delta_min": 0,
                    },
                    "refutation_signature": {
                        "transition_kind": "no_change",
                        "compass_change_count": 0,
                        "level_delta_max": 0,
                    },
                })

        elif role == "compass_change_propagation_neighbor":
            # If recent compass changed for this marker, test if the
            # adjacent compass cell propagates further.
            if any(c.get("marker_id") == mid for c in last_compass_changes):
                proposals.append({
                    "role": role, "anchor_marker_id": mid,
                    "abstraction_hint": "test propagation along compass after recent change",
                    "expected_observable_signature": {
                        "transition_kind": "propagation_change",
                        "compass_change_count": 1,
                        "level_delta_min": 0,
                    },
                    "refutation_signature": {
                        "transition_kind": "propagation_blocked",
                        "compass_change_count": 0,
                        "level_delta_max": 0,
                    },
                })

    # Roles that are not per-marker:
    if role == "recent_positive_delta_region_role_reused_elsewhere":
        # Find a recent L+ event region; propose re-test elsewhere.
        for d in (recent_turn_diffs or [])[-10:]:
            if int((d or {}).get("level_delta") or 0) >= 1:
                rid = (d or {}).get("click_region_id")
                if rid and rid != "_outside_":
                    proposals.append({
                        "role": role, "anchor_marker_id": rid,
                        "abstraction_hint": "region role just produced level rise; test similar role elsewhere",
                        "expected_observable_signature": {
                            "transition_kind": "level_rise_signature",
                            "compass_change_count": None,
                            "level_delta_min": 1,
                        },
                        "refutation_signature": {
                            "transition_kind": "no_level_rise",
                            "compass_change_count": None,
                            "level_delta_max": 0,
                        },
                    })
                    break

    if role == "two_marker_overlap_sector_candidate" and len(markers) >= 2:
        # Specifically for joint-test: emit one cross-marker candidate.
        m1 = markers[0]; m2 = markers[1]
        proposals.append({
            "role": role,
            "anchor_marker_id": m1.get("id"),
            "abstraction_hint": "test interaction between two markers' overlapping sector",
            "expected_observable_signature": {
                "transition_kind": "cross_marker_change",
                "compass_change_count": 2,
                "level_delta_min": 0,
            },
            "refutation_signature": {
                "transition_kind": "single_marker_change_only",
                "compass_change_count": 1,
                "level_delta_max": 0,
            },
        })

    if role == "recently_clicked_region_revisit_candidate":
        # Last-clicked region revisit (parity probe) — leak-safe wording.
        last_click_region = (last_diff or {}).get("click_region_id")
        if last_click_region and last_click_region != "_outside_":
            proposals.append({
                "role": role, "anchor_marker_id": last_click_region,
                "abstraction_hint": "revisit recently clicked region to test response symmetry",
                "expected_observable_signature": {
                    "transition_kind": "revisit_inverse",
                    "compass_change_count": 1,
                    "level_delta_min": 0,
                },
                "refutation_signature": {
                    "transition_kind": "revisit_same",
                    "compass_change_count": 0,
                    "level_delta_max": 0,
                },
            })

    return proposals


# --------------------------------------------------------------------
# Top-level entry.
# --------------------------------------------------------------------


def generate_candidates(
    *,
    visible_regions: list[dict],
    recent_turn_diffs: list[dict],
    marker_neighbor_states: list[dict] | None,
    level_bridges: list[dict] | None,
    chain_rule_log: list[dict] | None,
    role_history: dict | None,
    recent_emissions: list[dict] | None,
    recent_clicks: list[str] | None,
    turn_index: int,
    chain_tokens_len: int = 0,
    k_per_marker: int = 2,
    k_global: int = 8,
) -> list[dict]:
    """Returns up to k_global candidates, each leak-safe.

    Per round-3 C-12: force_test_joint_unrelated_marker_v1 only emits
    when chain_tokens_len >= 10 AND ≥2 markers AND no joint candidate
    seen in last 10 turns.
    """
    role_history = role_history or {}
    recent_emissions = recent_emissions or []
    recent_clicks = recent_clicks or []

    # Per-role proposals.
    raw: list[dict] = []
    for role in ALLOWED_ROLES:
        if role == "force_test_joint_unrelated_marker_v1":
            # Cold-start gate.
            if chain_tokens_len < 10:
                continue
            markers = [
                r for r in (visible_regions or [])
                if isinstance(r, dict) and r.get("is_multicolor")
            ]
            if len(markers) < 2:
                continue
            # Last-10-turn joint-emit check.
            recent_joint = sum(
                1 for e in recent_emissions[-10 * 8:]
                if e.get("role") == "two_marker_overlap_sector_candidate"
            )
            if recent_joint == 0:
                # Force-emit one joint candidate.
                m1 = markers[0]; m2 = markers[1]
                raw.append({
                    "role": role,
                    "anchor_marker_id": m1.get("id"),
                    "abstraction_hint": "force probe of two-marker interaction (no recent joint emit)",
                    "expected_observable_signature": {
                        "transition_kind": "cross_marker_change",
                        "compass_change_count": 2,
                        "level_delta_min": 0,
                    },
                    "refutation_signature": {
                        "transition_kind": "single_marker_change_only",
                        "compass_change_count": 1,
                        "level_delta_max": 0,
                    },
                })
            continue
        if role == "unrecognised_pattern_role_v1":
            # Reserved fallback. Generator does not emit this role
            # itself; it is reserved for downstream code that detects
            # an out-of-allowed pattern and wants to log a warning.
            continue
        proposals = _propose_per_role(
            role=role,
            visible_regions=visible_regions,
            recent_turn_diffs=recent_turn_diffs,
            marker_neighbor_states=marker_neighbor_states or [],
        )
        raw.extend(proposals)

    # Score + per-marker cap + global cap.
    scored: list[dict] = []
    for p in raw:
        if not _audit_emitted_role(p["role"]):
            continue
        score = _score(
            p["role"], p.get("anchor_marker_id"),
            recent_emissions, recent_clicks, role_history,
        )
        cid = _candidate_id(turn_index, p["role"], p.get("anchor_marker_id"))
        candidate = {
            "candidate_id": cid,
            "suggested_test": {
                "role": p["role"],
                "anchor_marker_id": p.get("anchor_marker_id"),
                "abstraction_hint": p.get("abstraction_hint"),
            },
            "expected_observable_signature":
                p.get("expected_observable_signature") or {},
            "refutation_signature":
                p.get("refutation_signature") or {},
            "score": round(score, 4),
        }
        if not _validate_candidate_signatures(candidate):
            continue
        scored.append(candidate)

    scored.sort(key=lambda c: -c.get("score", 0))

    # Force-test candidates BYPASS per-marker cap (round-3 C-12: the
    # whole point of force_test_joint_unrelated_marker_v1 is to force
    # joint probing when no recent joint emit). They still respect
    # the global cap.
    forced = [
        c for c in scored
        if c["suggested_test"]["role"] == "force_test_joint_unrelated_marker_v1"
    ]
    non_forced = [
        c for c in scored
        if c["suggested_test"]["role"] != "force_test_joint_unrelated_marker_v1"
    ]
    # Per-marker cap on non-forced.
    per_marker_count: dict[str | None, int] = {}
    after_per_marker: list[dict] = []
    for c in non_forced:
        anchor = c["suggested_test"].get("anchor_marker_id")
        cnt = per_marker_count.get(anchor, 0)
        if cnt < k_per_marker:
            after_per_marker.append(c)
            per_marker_count[anchor] = cnt + 1

    # Forced first (priority slot), then per-marker-capped, global cap.
    return (forced + after_per_marker)[:k_global]


# --------------------------------------------------------------------
# Verdict table (plan v589 §4.2 round-2 C-3 pinned).
# --------------------------------------------------------------------


def update_candidate_log_with_observation(
    *, candidate_log: list[dict],
    observation: dict,
    turn_now: int,
) -> list[dict]:
    """Per round-2 C-3 verdict table:
       expected match | refutation match | turns_since | verdict
       ---------------+------------------+-------------+--------
       true           | false            | ≤3          | supported
       false          | true             | ≤3          | refuted
       both           | both             | any         | inconclusive
       neither        | neither          | >5          | ignored

       Returns list of new verdict entries (may be empty)."""
    out: list[dict] = []
    if not isinstance(observation, dict):
        return out
    obs_transition = observation.get("dominant_transition") or {}
    obs_compass_count = int(observation.get("compass_change_count") or 0)
    obs_ld = int(observation.get("level_delta") or 0)

    for c in candidate_log:
        if c.get("verdict") in ("supported", "refuted", "ignored"):
            continue
        emitted = int(c.get("emitted_at_turn", 0))
        turns_since = turn_now - emitted
        exp = c.get("expected_observable_signature") or {}
        ref = c.get("refutation_signature") or {}

        def _match(sig: dict) -> bool:
            ok = True
            if sig.get("compass_change_count") is not None:
                ok = ok and (obs_compass_count == sig["compass_change_count"])
            if sig.get("level_delta_min") is not None:
                ok = ok and (obs_ld >= sig["level_delta_min"])
            if sig.get("level_delta_max") is not None:
                ok = ok and (obs_ld <= sig["level_delta_max"])
            return ok

        exp_match = _match(exp)
        ref_match = _match(ref)
        verdict = None
        if exp_match and not ref_match and turns_since <= 3:
            verdict = "supported"
        elif ref_match and not exp_match and turns_since <= 3:
            verdict = "refuted"
        elif exp_match and ref_match:
            verdict = "inconclusive"
        elif (not exp_match) and (not ref_match) and turns_since > 5:
            verdict = "ignored"
        if verdict:
            c["verdict"] = verdict
            c["resolved_at_turn"] = turn_now
            out.append({
                "candidate_id": c.get("candidate_id"),
                "verdict": verdict,
                "resolved_at_turn": turn_now,
            })
    return out
