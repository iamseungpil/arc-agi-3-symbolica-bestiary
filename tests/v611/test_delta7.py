"""v611 Step 2a — Δ7 multi-role schema fixture suite.

Plan v611 rev D §Δ7, codex round 10 ACCEPT.

These tests pin the SCHEMA CONTRACT for the 3 roles. They do NOT
enforce input-isolation (separate context, no frame for verifier) —
that is agent.py's runtime responsibility, audited at smoke time.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.v611_schemas import (  # noqa: E402
    validate_m1_proposer_output,
    validate_m2e_executor_output,
    validate_m2v_verifier_output,
)


# ─────────────────────────────────────────────────────────────────
# Δ7a — M1 Proposer (NL intent only, NO coords)
# ─────────────────────────────────────────────────────────────────


def _m1_proposer_clean() -> dict:
    return {
        "nl_strategy": ("Looking at the grid, I see a marker tile in the "
                         "lower-right region with an unsatisfied neighbor. "
                         "I propose clicking a corner area of that marker "
                         "to test the repair."),
        "suggested_click_region": ("bottom-right tile area near the marker, "
                                     "approximately a quarter of the way in"),
        "expected_signature": {"frame_changed": True, "unsat_delta": -1},
        "rollback_trigger": ("if no frame change after this click, the "
                              "proposed region is wrong"),
    }


def test_delta7a_clean_proposer_passes():
    res = validate_m1_proposer_output(_m1_proposer_clean())
    assert res.ok, f"clean proposer rejected: {res.violations}"


def test_delta7a_rejects_click_xy_hint_in_proposer():
    """CRITICAL: M1 Proposer must NOT emit coords."""
    bad = _m1_proposer_clean()
    bad["click_xy_hint"] = [44, 44]
    res = validate_m1_proposer_output(bad)
    assert not res.ok
    assert any("click_xy_hint forbidden" in v for v in res.violations)


def test_delta7a_rejects_raw_xy_keys():
    bad = _m1_proposer_clean()
    bad["x"] = 44
    bad["y"] = 44
    res = validate_m1_proposer_output(bad)
    assert not res.ok
    assert any("x/y coords forbidden" in v for v in res.violations)


def test_delta7a_rejects_missing_suggested_region():
    bad = _m1_proposer_clean()
    del bad["suggested_click_region"]
    res = validate_m1_proposer_output(bad)
    assert not res.ok
    assert any("suggested_click_region" in v for v in res.violations)


def test_delta7a_rejects_short_nl_strategy():
    bad = _m1_proposer_clean()
    bad["nl_strategy"] = "click"
    res = validate_m1_proposer_output(bad)
    assert not res.ok
    assert any("nl_strategy" in v for v in res.violations)


def test_delta7a_rejects_missing_rollback_trigger():
    bad = _m1_proposer_clean()
    del bad["rollback_trigger"]
    res = validate_m1_proposer_output(bad)
    assert not res.ok
    assert any("rollback_trigger" in v for v in res.violations)


# ─────────────────────────────────────────────────────────────────
# Δ7b — M2v Verifier (separate context, only verdict + reason)
# ─────────────────────────────────────────────────────────────────


def _m2v_verifier_clean() -> dict:
    return {
        "verdict": "approve",
        "reason_nl": ("the proposed bottom-right tile region matches an "
                       "untested marker with high information value"),
    }


def test_delta7b_clean_verifier_passes():
    res = validate_m2v_verifier_output(_m2v_verifier_clean())
    assert res.ok, f"clean verifier rejected: {res.violations}"


def test_delta7b_three_verdicts_accepted():
    for v in ("approve", "reject_replan", "reject_anchor"):
        out = _m2v_verifier_clean()
        out["verdict"] = v
        res = validate_m2v_verifier_output(out)
        assert res.ok, f"verdict={v} rejected: {res.violations}"


def test_delta7b_rejects_unknown_verdict():
    bad = _m2v_verifier_clean()
    bad["verdict"] = "maybe"
    res = validate_m2v_verifier_output(bad)
    assert not res.ok
    assert any("verdict must be" in v for v in res.violations)


def test_delta7b_rejects_short_reason():
    bad = _m2v_verifier_clean()
    bad["reason_nl"] = "ok"
    res = validate_m2v_verifier_output(bad)
    assert not res.ok
    assert any("reason_nl" in v for v in res.violations)


# ─────────────────────────────────────────────────────────────────
# Δ7c — M2e Executor (PNG visual → xy coords + grounding text)
# ─────────────────────────────────────────────────────────────────


def _m2e_executor_clean() -> dict:
    return {
        "click_xy_hint": [44, 44],
        "grounding_text": ("the marker tile I see in the PNG occupies "
                            "approximately pixels (40-48, 40-48); I aim "
                            "for its center at (44, 44)"),
    }


def test_delta7c_clean_executor_passes():
    res = validate_m2e_executor_output(_m2e_executor_clean())
    assert res.ok, f"clean executor rejected: {res.violations}"


def test_delta7c_rejects_missing_click_xy():
    bad = _m2e_executor_clean()
    del bad["click_xy_hint"]
    res = validate_m2e_executor_output(bad)
    assert not res.ok
    assert any("click_xy_hint" in v for v in res.violations)


def test_delta7c_rejects_out_of_bounds():
    bad = _m2e_executor_clean()
    bad["click_xy_hint"] = [100, 200]
    res = validate_m2e_executor_output(bad)
    assert not res.ok
    assert any("out of bounds" in v for v in res.violations)


def test_delta7c_rejects_missing_grounding_text():
    """grounding_text is the load-bearing 'I saw this in the PNG' field —
    without it, M2e collapses to a coord-emitter and Δ7c is invalid."""
    bad = _m2e_executor_clean()
    del bad["grounding_text"]
    res = validate_m2e_executor_output(bad)
    assert not res.ok
    assert any("grounding_text" in v for v in res.violations)


def test_delta7c_rejects_short_grounding():
    bad = _m2e_executor_clean()
    bad["grounding_text"] = "click here"
    res = validate_m2e_executor_output(bad)
    assert not res.ok
    assert any("grounding_text" in v for v in res.violations)


# ─────────────────────────────────────────────────────────────────
# Integration: full 3-role turn handoff
# ─────────────────────────────────────────────────────────────────


def test_delta7_integration_full_handoff_per_turn():
    """A single v611+Δ7 turn flows: M1 Proposer → M2v Verifier → M2e
    Executor. All three outputs must individually pass."""
    m1 = _m1_proposer_clean()
    m2v = _m2v_verifier_clean()
    m2e = _m2e_executor_clean()
    assert validate_m1_proposer_output(m1).ok
    assert validate_m2v_verifier_output(m2v).ok
    assert validate_m2e_executor_output(m2e).ok


def test_delta7_role_boundary_violation_blocked():
    """If M1 Proposer accidentally emits xy AND M2e Executor accidentally
    emits NL strategy, BOTH should fail their respective validators."""
    m1_bad = _m1_proposer_clean()
    m1_bad["click_xy_hint"] = [44, 44]
    m2e_bad = _m2e_executor_clean()
    del m2e_bad["grounding_text"]
    assert not validate_m1_proposer_output(m1_bad).ok
    assert not validate_m2e_executor_output(m2e_bad).ok


def test_delta7_reject_anchor_path_valid():
    """When verifier returns reject_anchor, the orchestrator (agent.py)
    must spawn a fresh M1 next turn. This test pins the verdict shape
    that triggers the fresh-spawn path."""
    out = _m2v_verifier_clean()
    out["verdict"] = "reject_anchor"
    out["reason_nl"] = ("prior strategy has been anchored on a failing "
                          "region for 3+ turns; spawn fresh proposer")
    res = validate_m2v_verifier_output(out)
    assert res.ok
    assert out["verdict"] == "reject_anchor"  # orchestrator key
