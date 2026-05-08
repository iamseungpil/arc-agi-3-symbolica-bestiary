"""B20 v592 — TIER balance runtime forcing tests.

Tests _v592_stuck_rotation_should_force, _v592_score_gate_should_force,
_v592_min_ratio_should_force in isolation. Mock board.recent_verbose +
candidate_tests as plain dicts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import the helpers from agent.py — they live as module-level functions.
from agents.templates.agentica_v57.agent import (
    _v592_stuck_rotation_should_force,
    _v592_score_gate_should_force,
    _v592_min_ratio_should_force,
)


def _make_board(recent_chid_tiers, recent_lds=None):
    """Build a mock board with recent_verbose + sensible defaults."""
    if recent_lds is None:
        recent_lds = [0] * len(recent_chid_tiers)
    rv = []
    for tier, ld in zip(recent_chid_tiers, recent_lds):
        rv.append({
            "chid_tier": tier,
            "observation": {"level_delta": ld},
        })
    return SimpleNamespace(recent_verbose=rv)


def _make_cand(pid, score=0.5, verdict=None):
    return {"predicate_id": pid, "score": score, "verdict": verdict}


# ---------------- T-V592-1..5: stuck rotation ----------------


def test_stuck_rotation_returns_none_with_fewer_than_3_turns():
    board = _make_board(["B", "B"])
    assert _v592_stuck_rotation_should_force(board, [_make_cand("P01", 0.6)]) is None


def test_stuck_rotation_returns_none_when_not_all_tier_b():
    board = _make_board(["B", "A", "B"])
    assert _v592_stuck_rotation_should_force(board, [_make_cand("P01", 0.6)]) is None


def test_stuck_rotation_returns_none_when_some_ld_nonzero():
    board = _make_board(["B", "B", "B"], [0, 1, 0])
    assert _v592_stuck_rotation_should_force(board, [_make_cand("P01", 0.6)]) is None


def test_stuck_rotation_returns_top_score_when_triggered():
    board = _make_board(["B", "B", "B"], [0, 0, 0])
    cands = [_make_cand("P01", 0.4), _make_cand("P05", 0.7), _make_cand("P03", 0.6)]
    forced = _v592_stuck_rotation_should_force(board, cands)
    assert forced is not None
    assert forced["predicate_id"] == "P05"


def test_stuck_rotation_returns_none_when_candidates_empty():
    board = _make_board(["B", "B", "B"], [0, 0, 0])
    assert _v592_stuck_rotation_should_force(board, []) is None


# ---------------- T-V592-6..8: score gate ----------------


def test_score_gate_fires_only_on_unresolved_high_score():
    cands = [
        _make_cand("P01", 0.5, verdict=None),
        _make_cand("P05", 0.9, verdict=None),
        _make_cand("P03", 0.85, verdict="supported"),  # resolved → skip
    ]
    forced = _v592_score_gate_should_force(cands)
    assert forced is not None
    assert forced["predicate_id"] == "P05"


def test_score_gate_returns_none_when_all_below_threshold():
    cands = [_make_cand("P01", 0.7), _make_cand("P05", 0.84)]
    assert _v592_score_gate_should_force(cands) is None


def test_score_gate_returns_none_when_all_resolved():
    cands = [_make_cand("P01", 0.95, verdict="refuted")]
    assert _v592_score_gate_should_force(cands) is None


# ---------------- T-V592-9..11: min ratio ----------------


def test_min_ratio_returns_none_when_window_too_small():
    board = _make_board(["B"] * 10)
    assert _v592_min_ratio_should_force(board, [_make_cand("P01", 0.6)]) is None


def test_min_ratio_returns_none_when_ratio_above_floor():
    # 5 TIER-A out of 20 = 25%, well above 0.15 floor.
    board = _make_board(["A"] * 5 + ["B"] * 15)
    assert _v592_min_ratio_should_force(board, [_make_cand("P01", 0.6)]) is None


def test_min_ratio_forces_when_ratio_below_floor():
    # 1 TIER-A out of 20 = 5%, below 0.15 floor.
    board = _make_board(["A"] * 1 + ["B"] * 19)
    forced = _v592_min_ratio_should_force(board, [_make_cand("P01", 0.6)])
    assert forced is not None
    assert forced["predicate_id"] == "P01"


def test_min_ratio_skips_low_confidence_candidates():
    # Below floor but only low-score candidates → skip.
    board = _make_board(["B"] * 20)
    cands = [_make_cand("P01", 0.4), _make_cand("P05", 0.45)]
    assert _v592_min_ratio_should_force(board, cands) is None


def test_min_ratio_respects_cap():
    # Below floor + 3 prior forces in last 50 → cap reached, no more forces.
    rv = [{"chid_tier": "B", "observation": {"level_delta": 0}} for _ in range(20)]
    # Add 3 prior forces in the surrounding 50-window
    for i in range(3):
        rv.append({
            "chid_tier": "A",
            "observation": {"level_delta": 0},
            "v592_min_ratio_forced": True,
        })
    board = SimpleNamespace(recent_verbose=rv[-50:] if len(rv) > 50 else rv)
    cands = [_make_cand("P01", 0.6)]
    # The window of last 20 is all TIER-B, but cap exhausted.
    # We construct a 23-turn rv; last 20 includes the 3 forced + 17 B
    # → tier_a_ratio = 3/20 = 0.15 (right at floor) — actually floor not breached.
    # To test cap: ratio < floor AND cap reached.
    # Build 20 B + 3 forced before; last_20 = 17 B + 3 forced → 0.15.
    # Edit: rv = 20 B + 3 forced, last_50 = all 23 → 3 forces in window.
    # Last 20 of last 23 = 17 B + 3 forced → 3/20 = 0.15 — at floor, no force.
    # Make ratio strictly below: 18 B + 1 forced, total 19 → not 20 yet.
    # Simpler test: 20 B + 3 forced earlier, but last 20 is recent → all 20 are B.
    # Use distinct turns: last 20 turns = 20 B; cap window (last 50) includes those 20 + 3 prior forced + filler.
    pre_forced = [{"chid_tier": "A", "observation": {"level_delta": 0},
                   "v592_min_ratio_forced": True} for _ in range(3)]
    last20 = [{"chid_tier": "B", "observation": {"level_delta": 0}} for _ in range(20)]
    rv2 = pre_forced + last20  # 23 turns
    board2 = SimpleNamespace(recent_verbose=rv2)
    # Last 50 contains all 23 (3 forced visible). Last 20 = all B (ratio=0.0).
    assert _v592_min_ratio_should_force(board2, cands) is None  # cap exhausted


# ---------------- T-V592-14: priority order ----------------


def test_priority_stuck_then_score_then_min_ratio():
    """When stuck condition + high-score predicate + low ratio all hold,
    stuck-rotation should fire first (highest priority)."""
    board = _make_board(["B"] * 20, [0] * 20)
    cands = [_make_cand("P09", 0.95, verdict=None),
             _make_cand("P01", 0.6, verdict=None)]
    stuck = _v592_stuck_rotation_should_force(board, cands)
    score = _v592_score_gate_should_force(cands)
    min_r = _v592_min_ratio_should_force(board, cands)
    # All three should fire
    assert stuck is not None
    assert score is not None
    assert min_r is not None
    # Priority "stuck or score or min_ratio" — Python "or" picks first truthy.
    chosen = stuck or score or min_r
    assert chosen["predicate_id"] == stuck["predicate_id"]
