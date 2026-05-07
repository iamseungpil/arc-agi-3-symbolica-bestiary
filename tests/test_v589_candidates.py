"""B17 plan v589 — candidate_generator tests.

14 tests covering: role enumeration / leak audit / score formula /
schema validator / verdict table / backward compat.
"""
import json
import re
import pytest

from tools.candidate_generator import (
    ALLOWED_ROLES,
    _FORBIDDEN_VOCAB,
    _audit_emitted_role,
    _candidate_id,
    _causal_proximity,
    _delta_correlation_prior,
    _novelty,
    _validate_candidate_signatures,
    generate_candidates,
    update_candidate_log_with_observation,
)


def _marker(rid="R12", color=12, neighbors=None):
    return {
        "id": rid, "color": color, "is_multicolor": True,
        "size": 36, "bbox": {"min_x": 0, "min_y": 0, "max_x": 8, "max_y": 8},
        "neighbors_3x3": neighbors or {"N": "R20", "E": "R21", "S": "R22", "W": "R23"},
        "click_response": {"clicks": 1, "responses": 1},
    }


def _diff(turn, region_id="R20", level_delta=0, compass=None, transitions=None):
    return {
        "turn": turn,
        "click_region_id": region_id,
        "level_delta": level_delta,
        "compass_changes": compass or [],
        "color_transitions": transitions or [],
    }


# --- T-CAND-1..3: generator produces candidates ---


def test_cand_1_basic_marker_with_change():
    out = generate_candidates(
        visible_regions=[_marker(), _marker(rid="R13")],
        recent_turn_diffs=[
            _diff(5, region_id="R20", level_delta=0,
                  compass=[{"marker_id": "R12", "direction": "N", "from": 9, "to": 12}]),
        ],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=6,
    )
    assert isinstance(out, list)
    assert 1 <= len(out) <= 8


def test_cand_2_no_markers_returns_empty_or_minimal():
    out = generate_candidates(
        visible_regions=[],
        recent_turn_diffs=[],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=0,
    )
    assert isinstance(out, list)
    # No markers + no diffs → no proposals.
    assert len(out) == 0


def test_cand_3_per_marker_cap_enforced():
    out = generate_candidates(
        visible_regions=[_marker(rid="R12"), _marker(rid="R13"),
                          _marker(rid="R14"), _marker(rid="R15")],
        recent_turn_diffs=[
            _diff(5, region_id="R20", level_delta=0,
                  compass=[{"marker_id": "R12", "direction": "N", "from": 9, "to": 12}]),
        ],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=6,
        k_per_marker=2, k_global=20,
    )
    # No marker should have more than 2 candidates anchored to it.
    from collections import Counter
    anchor_counts = Counter(c["suggested_test"]["anchor_marker_id"] for c in out)
    for a, n in anchor_counts.items():
        assert n <= 2, f"per-marker cap violated for {a}: {n}"


# --- T-CAND-4..5: leak audit ---


def test_cand_4_all_emitted_roles_in_allowed_set():
    out = generate_candidates(
        visible_regions=[_marker(), _marker(rid="R13")],
        recent_turn_diffs=[
            _diff(5, region_id="R20", level_delta=1,
                  compass=[{"marker_id": "R12", "direction": "N", "from": 9, "to": 12}]),
        ],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=6,
    )
    for c in out:
        role = c["suggested_test"]["role"]
        assert role in ALLOWED_ROLES, f"unrecognised role: {role}"


def test_cand_5_no_forbidden_vocab_anywhere():
    out = generate_candidates(
        visible_regions=[_marker(), _marker(rid="R13")],
        recent_turn_diffs=[
            _diff(5, region_id="R20", level_delta=1,
                  compass=[{"marker_id": "R12", "direction": "N", "from": 9, "to": 12}]),
        ],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=6,
    )
    serialised = json.dumps(out)
    hits = _FORBIDDEN_VOCAB.findall(serialised)
    assert hits == [], f"leak vocab found: {hits}"


# --- T-CAND-6..7: schema + score ---


def test_cand_6_unique_candidate_ids_per_turn():
    out = generate_candidates(
        visible_regions=[_marker(), _marker(rid="R13")],
        recent_turn_diffs=[
            _diff(5, region_id="R20", level_delta=1,
                  compass=[{"marker_id": "R12", "direction": "N", "from": 9, "to": 12}]),
        ],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=6,
    )
    ids = [c["candidate_id"] for c in out]
    assert len(ids) == len(set(ids))


def test_cand_7_score_descending_order():
    out = generate_candidates(
        visible_regions=[_marker(), _marker(rid="R13")],
        recent_turn_diffs=[
            _diff(5, region_id="R20", level_delta=1,
                  compass=[{"marker_id": "R12", "direction": "N", "from": 9, "to": 12}]),
        ],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=6,
    )
    scores = [c["score"] for c in out]
    assert scores == sorted(scores, reverse=True), \
        f"scores not in descending order: {scores}"


# --- T-CAND-8..9: signature validator ---


def test_cand_8_validate_signatures_rejects_equal():
    same_sig = {"transition_kind": "x", "compass_change_count": 1}
    c = {
        "candidate_id": "C1:test:abc",
        "suggested_test": {"role": ALLOWED_ROLES[0], "anchor_marker_id": "R1"},
        "expected_observable_signature": same_sig,
        "refutation_signature": dict(same_sig),
        "score": 1.0,
    }
    assert not _validate_candidate_signatures(c)


def test_cand_9_validate_signatures_accepts_distinct():
    c = {
        "candidate_id": "C1:test:abc",
        "suggested_test": {"role": ALLOWED_ROLES[0], "anchor_marker_id": "R1"},
        "expected_observable_signature": {"compass_change_count": 1, "level_delta_min": 0},
        "refutation_signature": {"compass_change_count": 0, "level_delta_max": 0},
        "score": 1.0,
    }
    assert _validate_candidate_signatures(c)


# --- T-CAND-10..11: verdict table ---


def test_cand_10_verdict_supported():
    c = {
        "candidate_id": "C1:test:abc",
        "expected_observable_signature": {"compass_change_count": 1, "level_delta_min": 0},
        "refutation_signature": {"compass_change_count": 0, "level_delta_max": 0},
        "emitted_at_turn": 10,
    }
    obs = {"compass_change_count": 1, "level_delta": 0}
    out = update_candidate_log_with_observation(
        candidate_log=[c], observation=obs, turn_now=11,
    )
    # expected matches (cc=1, ld>=0); refutation does NOT match (cc=0 fails).
    assert len(out) == 1
    assert out[0]["verdict"] == "supported"


def test_cand_11_verdict_refuted():
    c = {
        "candidate_id": "C1:test:abc",
        "expected_observable_signature": {"compass_change_count": 1, "level_delta_min": 0},
        "refutation_signature": {"compass_change_count": 0, "level_delta_max": 0},
        "emitted_at_turn": 10,
    }
    obs = {"compass_change_count": 0, "level_delta": 0}
    out = update_candidate_log_with_observation(
        candidate_log=[c], observation=obs, turn_now=11,
    )
    assert out[0]["verdict"] == "refuted"


def test_cand_12_verdict_ignored_after_5_turns():
    c = {
        "candidate_id": "C1:test:abc",
        "expected_observable_signature": {"compass_change_count": 5, "level_delta_min": 1},
        "refutation_signature": {"compass_change_count": 9, "level_delta_max": -1},
        "emitted_at_turn": 10,
    }
    obs = {"compass_change_count": 0, "level_delta": 0}  # neither match
    out = update_candidate_log_with_observation(
        candidate_log=[c], observation=obs, turn_now=17,  # 7 turns since
    )
    assert out[0]["verdict"] == "ignored"


# --- T-CAND-13: force_test joint role gating ---


def test_cand_13_force_test_gate_chain_threshold():
    """force_test_joint_unrelated_marker_v1 emits ONLY when chain≥10."""
    args = dict(
        visible_regions=[_marker(rid="R12"), _marker(rid="R13")],
        recent_turn_diffs=[],
        marker_neighbor_states=[],
        level_bridges=[], chain_rule_log=[],
        role_history={}, recent_emissions=[], recent_clicks=[],
        turn_index=20,
    )
    out_short = generate_candidates(**args, chain_tokens_len=5)
    out_long = generate_candidates(**args, chain_tokens_len=15)
    forced_short = [c for c in out_short
                    if c["suggested_test"]["role"] == "force_test_joint_unrelated_marker_v1"]
    forced_long = [c for c in out_long
                   if c["suggested_test"]["role"] == "force_test_joint_unrelated_marker_v1"]
    assert len(forced_short) == 0
    assert len(forced_long) == 1


# --- T-CAND-14: per-emission audit + V-LEAK on every role ---


def test_cand_14_per_role_string_passes_v_leak():
    for role in ALLOWED_ROLES:
        assert _audit_emitted_role(role), f"role failed audit: {role}"
        assert not _FORBIDDEN_VOCAB.search(role), f"role contains forbidden vocab: {role}"
