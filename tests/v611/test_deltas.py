"""v611 Step 1 — Δ1, Δ3, Δ5 schema fixture suite.

Plan v611 rev B §Δ deltas, FROZEN.

These fixtures verify the SCHEMA CONTRACT for each module output.
They do NOT make LLM calls; they only check structural validity of
emitted JSON. Live agent.py must apply these validators as gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.v611_schemas import (  # noqa: E402
    m1_nl_grounded_by_strategy,
    validate_m1_output,
    validate_m3_skill_output,
    validate_m4_output,
)


# ─────────────────────────────────────────────────────────────────
# Δ1 — M1 NL strategy first + grounded click_xy_hint
# ─────────────────────────────────────────────────────────────────


def _m1_clean_sample() -> dict:
    return {
        "nl_strategy": ("Looking at the grid, I see a marker tile near "
                         "the bottom-right corner with an unsatisfied "
                         "neighbor on its east side. I will click near "
                         "that east neighbor to test the repair."),
        "predicate_id": "P_NL_grounded",
        "click_xy_hint": [44, 44],
        "expected_signature": {"frame_changed": True, "unsat_delta": -1},
        "rollback_trigger": ("if frame unchanged after this click, "
                              "the strategy does not apply here"),
    }


def test_delta1_clean_m1_passes():
    res = validate_m1_output(_m1_clean_sample())
    assert res.ok, f"clean M1 rejected: {res.violations}"


def test_delta1_missing_nl_strategy_rejected():
    bad = _m1_clean_sample()
    del bad["nl_strategy"]
    res = validate_m1_output(bad)
    assert not res.ok
    assert any("nl_strategy" in v for v in res.violations)


def test_delta1_short_nl_strategy_rejected():
    bad = _m1_clean_sample()
    bad["nl_strategy"] = "too short"
    res = validate_m1_output(bad)
    assert not res.ok
    assert any("too short" in v for v in res.violations)


def test_delta1_missing_click_xy_hint_rejected():
    bad = _m1_clean_sample()
    del bad["click_xy_hint"]
    res = validate_m1_output(bad)
    assert not res.ok
    assert any("click_xy_hint" in v for v in res.violations)


def test_delta1_out_of_bounds_xy_rejected():
    bad = _m1_clean_sample()
    bad["click_xy_hint"] = [64, 100]
    res = validate_m1_output(bad)
    assert not res.ok
    assert any("out of bounds" in v for v in res.violations)


def test_delta1_old_predicate_id_rejected():
    """Δ1: predicate_id must start with P_NL. Old P_C4_probe_edge
    style is REJECTED (v608d era)."""
    bad = _m1_clean_sample()
    bad["predicate_id"] = "P_C4_probe_edge"
    res = validate_m1_output(bad)
    assert not res.ok
    assert any("P_NL" in v for v in res.violations)


def test_delta1_missing_rollback_trigger_rejected():
    bad = _m1_clean_sample()
    bad["rollback_trigger"] = ""
    res = validate_m1_output(bad)
    assert not res.ok


def test_delta1_grounding_by_spatial_words():
    """Grounding rule: nl_strategy mentions spatial/visual word."""
    out = _m1_clean_sample()
    assert m1_nl_grounded_by_strategy(out), \
        "clean sample has 'corner', 'neighbor', 'east' — should ground"


def test_delta1_grounding_fails_on_generic_strategy():
    """If nl_strategy is purely abstract ('I will optimize') without
    spatial or visual reference, grounding fails."""
    out = _m1_clean_sample()
    out["nl_strategy"] = ("I will execute a click action and observe "
                          "the resulting feedback to update my belief "
                          "state about the world.")
    # No spatial/visual word -> grounding False
    assert not m1_nl_grounded_by_strategy(out)


# ─────────────────────────────────────────────────────────────────
# Δ3 — M3 NL-only skill (no executable, no coords)
# ─────────────────────────────────────────────────────────────────


def _m3_clean_skill() -> dict:
    return {
        "skill_id": "S-NL-abc123",
        "nl_description": ("Clicking near a corner tile of the same "
                            "color as a marker's neighbor tends to "
                            "trigger a satisfaction change at that "
                            "marker."),
        "abstract_precondition": ("the marker has at least one "
                                    "unsatisfied neighbor on a "
                                    "visible side"),
        "expected_observed_effect": ("neighbor satisfaction count "
                                       "decreases by one in the "
                                       "subsequent frame"),
    }


def test_delta3_clean_skill_passes():
    res = validate_m3_skill_output(_m3_clean_skill())
    assert res.ok, f"clean M3 rejected: {res.violations}"


def test_delta3_rejects_executable_python_in_nl():
    """Δ3 REJECTS executable code in NL field (Round-1 retraction)."""
    bad = _m3_clean_skill()
    bad["nl_description"] = ("def click_corner(state): return (44, 44) "
                              "if state.marker else None")
    res = validate_m3_skill_output(bad)
    assert not res.ok
    assert any("code tokens" in v for v in res.violations)


def test_delta3_rejects_raw_coord_in_nl():
    """Δ3 + memory hygiene: coords like '38' or '44' rejected."""
    bad = _m3_clean_skill()
    bad["nl_description"] = "Click at position 44 in the grid to win."
    res = validate_m3_skill_output(bad)
    assert not res.ok
    assert any("44" in v for v in res.violations)


def test_delta3_rejects_wrong_skill_id_prefix():
    bad = _m3_clean_skill()
    bad["skill_id"] = "M-001"  # old v608 style
    res = validate_m3_skill_output(bad)
    assert not res.ok
    assert any("S-NL-" in v for v in res.violations)


def test_delta3_rejects_missing_fields():
    bad = _m3_clean_skill()
    del bad["abstract_precondition"]
    res = validate_m3_skill_output(bad)
    assert not res.ok
    assert any("abstract_precondition" in v for v in res.violations)


def test_delta3_rejects_too_short_description():
    bad = _m3_clean_skill()
    bad["nl_description"] = "do it"
    res = validate_m3_skill_output(bad)
    assert not res.ok
    assert any("too short" in v for v in res.violations)


# ─────────────────────────────────────────────────────────────────
# Δ5 — M4 3-step self-verification
# ─────────────────────────────────────────────────────────────────


def _m4_clean_output() -> dict:
    return {
        "paragraph": ("The click produced the predicted frame change "
                       "but the unsat delta was zero rather than minus "
                       "one as predicted. The strategy partially held."),
        "verify": {
            "predicted_vs_observed": ("predicted frame_changed=True, "
                                        "observed True; predicted "
                                        "unsat_delta=-1, observed 0"),
            "strategy_validity": ("strategy is partially valid: click "
                                    "does affect frame but does not "
                                    "satisfy marker constraints"),
            "skillmd_update": {
                "add": [],
                "promote": [],
                "falsify": ["S-NL-old-precondition-needs-revision"],
            },
        },
        "verdict": "neutral",
        "next_directive": ("explore a different marker neighbor with "
                            "a stronger hypothesis about color match"),
    }


def test_delta5_clean_m4_passes():
    res = validate_m4_output(_m4_clean_output())
    assert res.ok, f"clean M4 rejected: {res.violations}"


def test_delta5_missing_verify_dict_rejected():
    bad = _m4_clean_output()
    del bad["verify"]
    res = validate_m4_output(bad)
    assert not res.ok
    assert any("verify" in v for v in res.violations)


def test_delta5_missing_predicted_vs_observed_rejected():
    bad = _m4_clean_output()
    del bad["verify"]["predicted_vs_observed"]
    res = validate_m4_output(bad)
    assert not res.ok
    assert any("predicted_vs_observed" in v for v in res.violations)


def test_delta5_missing_strategy_validity_rejected():
    bad = _m4_clean_output()
    del bad["verify"]["strategy_validity"]
    res = validate_m4_output(bad)
    assert not res.ok
    assert any("strategy_validity" in v for v in res.violations)


def test_delta5_missing_skillmd_update_rejected():
    bad = _m4_clean_output()
    del bad["verify"]["skillmd_update"]
    res = validate_m4_output(bad)
    assert not res.ok
    assert any("skillmd_update" in v for v in res.violations)


def test_delta5_skillmd_update_missing_promote_rejected():
    bad = _m4_clean_output()
    del bad["verify"]["skillmd_update"]["promote"]
    res = validate_m4_output(bad)
    assert not res.ok
    assert any("promote" in v for v in res.violations)


def test_delta5_invalid_verdict_rejected():
    bad = _m4_clean_output()
    bad["verdict"] = "maybe"
    res = validate_m4_output(bad)
    assert not res.ok
    assert any("verdict" in v for v in res.violations)


def test_delta5_three_verdict_values_accepted():
    for v in ("support", "refute", "neutral"):
        out = _m4_clean_output()
        out["verdict"] = v
        res = validate_m4_output(out)
        assert res.ok, f"verdict={v} rejected: {res.violations}"


# ─────────────────────────────────────────────────────────────────
# Cross-delta integration: a single turn's M1 → M4 chain
# ─────────────────────────────────────────────────────────────────


def test_integration_m1_m3_m4_all_valid_for_one_turn():
    """All three module outputs in a single hypothetical v611 turn
    must pass their respective validators."""
    m1 = _m1_clean_sample()
    m3 = _m3_clean_skill()
    m4 = _m4_clean_output()
    assert validate_m1_output(m1).ok
    assert validate_m3_skill_output(m3).ok
    assert validate_m4_output(m4).ok


def test_integration_v608d_legacy_m1_fails_under_delta1():
    """A v608d-style M1 output (no nl_strategy, just region_hint thought)
    must FAIL under v611 Δ1 validator. This proves the contract
    actually rejects the old shape."""
    v608d_legacy = {
        "candidate_predicate_id": "P_C4_probe_edge",
        "region_hint": "C5",
        "expected_signature": {"level_delta": 1},
        "thought": ("Marker C4 slot E has neighbor C5 unsatisfied. "
                     "Region-anchored repair on C4."),
        "confidence": 0.41,
    }
    res = validate_m1_output(v608d_legacy)
    assert not res.ok
    # Specifically: missing nl_strategy + old predicate prefix +
    # missing click_xy_hint + missing rollback_trigger.
    assert any("nl_strategy" in v for v in res.violations)
    assert any("click_xy_hint" in v for v in res.violations)
