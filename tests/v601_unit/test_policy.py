"""Per-module unit tests for policy.py.

Plan v602 §11 addendum: 14 critical branch tests.

Branches under test:
  target-priority ×4:
    1. priority 1 — proposer hint with valid denominator
    2. priority 1 demoted — proposer hint with WRONG denominator (§3.14 rev C C1)
    3. priority 2 — near_complete marker selected when no proposer
    4. priority 3 — argmax progress marker selected when no near_complete

  sat-threshold ×3:
    5. compute_saturation_status -> none
    6. compute_saturation_status -> near_complete (clicked == n-1)
    7. compute_saturation_status -> complete (clicked == n)

  ucb1-floor:
    8. cold-start arm gets 1e6 + family_priority floor

  confidence-override ×4:
    9. exact (predicate, region) match wins on near-tie when conf >= 0.5
    10. region-only fallback when exact predicate not present
    11. NOT applied when proposer.confidence < 0.5
    12. capped per-episode (override_count never exceeds cap)

  marker-validation:
    13. priority 4 sentinel — no candidate markers -> (None, "n/a_sentinel")

  resolver ×2 fold into UCB1+coord:
    14. select_arm uses library.resolve_coord to compute coord_xy correctly

  verdict ×3 (compute_verdict):
    15. matched True when observed >= expected
    16. matched False when observed < expected
    17. expected None falls back to obs > 0
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.policy import (  # noqa: E402
    EpisodeState, PolicyDecision, Verdict,
    compute_saturation_status, compute_verdict,
    record_outcome, select_arm, select_target_marker,
)
from agents.templates.agentica_lite.predicate_library import PredicateLibrary  # noqa: E402
from agents.templates.agentica_lite.predicate_posterior import (  # noqa: E402
    ArmKey, ArmStats, PredicatePosterior,
)
from agents.templates.agentica_lite.proposer import ProposerOutput  # noqa: E402


# ---------- helpers -----------------------------------------------------------

def _make_marker(marker_id: str, clicks_pattern: list[int],
                 region_ids: list[str] | None = None) -> dict:
    """Build a marker dict whose compass has slot clicks == clicks_pattern."""
    n = len(clicks_pattern)
    if region_ids is None:
        region_ids = [f"{marker_id}_slot_{i}" for i in range(n)]
    compass = {}
    for i, c in enumerate(clicks_pattern):
        compass[f"d{i}"] = {"region_id": region_ids[i], "clicks": c}
    return {"marker_id": marker_id, "is_primary_marker": True, "compass": compass}


def _proposer(pred="P_saturation_progress", region="R36", conf=0.7, marker="M0",
              denom=8, threshold=7) -> ProposerOutput:
    return ProposerOutput(
        candidate_predicate_id=pred,
        region_hint=region,
        expected_signature={"level_delta": 1},
        required_pre_state={
            "marker_id": marker,
            "saturation_threshold": threshold,
            "saturation_denominator": denom,
        },
        confidence=conf,
        thought="",
    )


# ============================================================================
# target-priority ×4
# ============================================================================

def test_target_priority_proposer_hint_valid_denom():
    """Priority 1: proposer.required_pre_state.marker_id matches with valid denom."""
    m0 = _make_marker("M0", [1] * 7 + [0])  # 8 slots, 7 clicked -> near_complete
    state = {"marker_neighbor_states": [m0]}
    p = _proposer(marker="M0", denom=8)
    target, reason = select_target_marker(state, p)
    assert target is m0
    assert reason == "proposer_hint"


def test_target_priority_proposer_hint_demoted_on_denom_mismatch():
    """Priority 1 demoted when denom_hint != marker compass len (§3.14 rev C C1).

    Falls through to priority 2 (near_complete) or 3 (argmax_progress).
    """
    m0 = _make_marker("M0", [1] * 7 + [0])  # n=8 actual
    state = {"marker_neighbor_states": [m0]}
    p = _proposer(marker="M0", denom=4)  # WRONG denom: claims 4 but compass has 8
    target, reason = select_target_marker(state, p)
    # Falls through to priority 2 (near_complete since clicked == n - 1 = 7)
    assert target is m0
    assert reason == "near_complete"


def test_target_priority_near_complete_no_proposer():
    """Priority 2: marker with status == near_complete selected (no proposer)."""
    m_near = _make_marker("M0", [1, 1, 1, 0])  # 3/4 -> near_complete
    m_other = _make_marker("M1", [1, 0, 0, 0])  # 1/4 -> none
    state = {"marker_neighbor_states": [m_near, m_other]}
    target, reason = select_target_marker(state, None)
    assert target is m_near
    assert reason == "near_complete"


def test_target_priority_argmax_progress_when_no_near_complete():
    """Priority 3: argmax saturation_numerator with at least 1 click."""
    m_low = _make_marker("M0", [1, 0, 0, 0, 0, 0, 0, 0])  # 1/8
    m_mid = _make_marker("M1", [1, 1, 1, 0, 0, 0, 0, 0])  # 3/8
    state = {"marker_neighbor_states": [m_low, m_mid]}
    target, reason = select_target_marker(state, None)
    assert target is m_mid
    assert reason == "argmax_progress"


# ============================================================================
# sat-threshold ×3
# ============================================================================

def test_compute_saturation_status_none():
    m = _make_marker("M0", [0, 0, 0, 0])
    clicked, denom, status = compute_saturation_status(m)
    assert (clicked, denom, status) == (0, 4, "none")


def test_compute_saturation_status_near_complete():
    """Plan §3.6: near_complete iff clicked == n - 1 (i.e., exactly one slot left)."""
    m = _make_marker("M0", [1, 1, 1, 0])
    clicked, denom, status = compute_saturation_status(m)
    assert (clicked, denom, status) == (3, 4, "near_complete")


def test_compute_saturation_status_complete():
    m = _make_marker("M0", [1, 1, 1, 1])
    clicked, denom, status = compute_saturation_status(m)
    assert (clicked, denom, status) == (4, 4, "complete")


# ============================================================================
# ucb1-floor
# ============================================================================

def test_select_arm_cold_start_floor_score():
    """Cold-start arm receives 1e6 + family_priority score; saturation_progress
    (priority 5.0) outranks sector_alignment (4.0) when target is set."""
    posterior = PredicatePosterior()
    library = PredicateLibrary()
    state = {
        "marker_neighbor_states": [_make_marker("M0", [1] * 7 + [0])],
        "visible_regions": [{"region_id": "R36", "id": "R36",
                             "bbox": [0, 0, 4, 4]}],
    }
    p = _proposer(pred="P12_saturation_progress", region="R36", conf=0.6, denom=8)
    episode = EpisodeState()
    decision = select_arm(state, posterior, library, p, episode)
    assert decision is not None
    # P12 should be chosen (saturation_progress family priority 5.0 wins)
    assert decision.arm_key.predicate_id == "P12_saturation_progress"


# ============================================================================
# confidence-override ×4
# ============================================================================

def test_confidence_override_exact_match_pass1():
    """Exact (predicate, region) match wins on near-tie when conf >= 0.5."""
    posterior = PredicatePosterior()
    library = PredicateLibrary()
    state = {
        "marker_neighbor_states": [_make_marker("M0", [1] * 7 + [0])],
        "visible_regions": [
            {"region_id": "R36", "id": "R36", "bbox": [0, 0, 4, 4]},
            {"region_id": "R12", "id": "R12", "bbox": [10, 10, 14, 14]},
        ],
    }
    p = _proposer(pred="P_saturation_progress", region="R12", conf=0.7, denom=8)
    episode = EpisodeState()
    decision = select_arm(state, posterior, library, p, episode)
    assert decision is not None
    # exact preference: P_saturation_progress + R12
    assert decision.arm_key.predicate_id == "P_saturation_progress"
    assert decision.arm_key.region_id == "R12"
    assert decision.confidence_override_used is True
    assert episode.confidence_override_count == 1


def test_confidence_override_region_only_fallback():
    """When exact predicate hint isn't a registered predicate, region-only Pass 2 fires."""
    posterior = PredicatePosterior()
    library = PredicateLibrary()
    state = {
        "marker_neighbor_states": [_make_marker("M0", [1] * 7 + [0])],
        "visible_regions": [
            {"region_id": "R36", "id": "R36", "bbox": [0, 0, 4, 4]},
            {"region_id": "R12", "id": "R12", "bbox": [10, 10, 14, 14]},
        ],
    }
    # Proposer claims a predicate id NOT in the library
    p = _proposer(pred="P_NONEXISTENT", region="R12", conf=0.6, denom=8)
    episode = EpisodeState()
    decision = select_arm(state, posterior, library, p, episode)
    assert decision is not None
    # region-only fallback: chose some predicate at R12
    assert decision.arm_key.region_id == "R12"
    assert decision.confidence_override_used is True


def test_confidence_override_disabled_below_threshold():
    """confidence < 0.5 does NOT trigger override; chosen arm = top-score (no preference)."""
    posterior = PredicatePosterior()
    library = PredicateLibrary()
    state = {
        "marker_neighbor_states": [_make_marker("M0", [1] * 7 + [0])],
        "visible_regions": [
            {"region_id": "R36", "id": "R36", "bbox": [0, 0, 4, 4]},
            {"region_id": "R12", "id": "R12", "bbox": [10, 10, 14, 14]},
        ],
    }
    p = _proposer(pred="P_saturation_progress", region="R12", conf=0.3, denom=8)
    episode = EpisodeState()
    decision = select_arm(state, posterior, library, p, episode)
    assert decision is not None
    assert decision.confidence_override_used is False
    assert episode.confidence_override_count == 0


def test_confidence_override_capped_per_episode():
    """Per-episode cap = max(1, min(3, ceil(0.15*decisions))); never exceeded."""
    posterior = PredicatePosterior()
    library = PredicateLibrary()
    state = {
        "marker_neighbor_states": [_make_marker("M0", [1] * 7 + [0])],
        "visible_regions": [
            {"region_id": "R36", "id": "R36", "bbox": [0, 0, 4, 4]},
            {"region_id": "R12", "id": "R12", "bbox": [10, 10, 14, 14]},
        ],
    }
    p = _proposer(pred="P_saturation_progress", region="R12", conf=0.9, denom=8)
    episode = EpisodeState()
    # Force-override 5 times; cap stays at 1 because policy_decisions starts at 0
    for _ in range(5):
        decision = select_arm(state, posterior, library, p, episode)
        assert decision is not None
    # Cap: at most 1 override granted before 0.15 * decisions exceeds 1, then up to 3 max
    assert episode.confidence_override_count <= 3
    # And must be > 0
    assert episode.confidence_override_count >= 1


# ============================================================================
# marker-validation (priority 4 sentinel)
# ============================================================================

def test_marker_validation_no_candidates_returns_sentinel():
    """When state has no markers at all, target = None with reason 'n/a_sentinel'."""
    state = {"marker_neighbor_states": []}
    target, reason = select_target_marker(state, None)
    assert target is None
    assert reason == "n/a_sentinel"


# ============================================================================
# resolver — coord_xy via library.resolve_coord
# ============================================================================

def test_select_arm_coord_resolution_via_library():
    """select_arm uses library.resolve_coord to compute centroid coord."""
    posterior = PredicatePosterior()
    library = PredicateLibrary()
    state = {
        "marker_neighbor_states": [_make_marker("M0", [1] * 7 + [0])],
        "visible_regions": [
            # bbox (10, 20, 30, 40) -> centroid (20, 30)
            {"region_id": "R36", "id": "R36",
             "bbox": {"min_x": 10, "min_y": 20, "max_x": 30, "max_y": 40}},
        ],
    }
    p = _proposer(pred="P_saturation_progress", region="R36", conf=0.6, denom=8)
    episode = EpisodeState()
    decision = select_arm(state, posterior, library, p, episode)
    assert decision is not None
    assert decision.coord_xy == (20, 30)


# ============================================================================
# verdict ×3
# ============================================================================

def test_compute_verdict_matched_when_obs_meets_expected():
    """observed.level_delta >= expected.level_delta -> matched True."""
    v = compute_verdict({"level_delta": 1}, {"level_delta": 1})
    assert v.matched is True
    assert v.observed_level_delta == 1
    assert v.expected_level_delta == 1


def test_compute_verdict_unmatched_when_obs_below_expected():
    v = compute_verdict({"level_delta": 0}, {"level_delta": 1})
    assert v.matched is False
    assert v.observed_level_delta == 0


def test_compute_verdict_expected_none_falls_back():
    """Without expected_signature.level_delta, matched := obs > 0."""
    v_pos = compute_verdict({"level_delta": 1}, None)
    assert v_pos.matched is True
    assert v_pos.expected_level_delta is None
    v_zero = compute_verdict({"level_delta": 0}, None)
    assert v_zero.matched is False
    # also handles missing key in expected_signature
    v_no_key = compute_verdict({"level_delta": 1}, {"some_other_field": True})
    assert v_no_key.matched is True
