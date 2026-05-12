"""Round-15 alignment tests — structured state_summary + grounding
attributes + M2e SUBSTITUTE prefix detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.v611_orchestrator import (  # noqa: E402
    StateSummary,
)
from agents.templates.agentica_lite.v611_schemas import (  # noqa: E402
    m2e_is_substitute,
    validate_m2e_executor_output,
    validate_m2e_grounding_attributes,
)


# ─────────────────────────────────────────────────────────────────
# StateSummary structured rendering
# ─────────────────────────────────────────────────────────────────


def test_state_summary_renders_all_fields():
    s = StateSummary(
        state_now="8 markers, 5 satisfied; primary bottom-right",
        last_strategies=[
            {"text": "click bottom-right marker east neighbor",
             "verdict": "approve", "frame_changed": False},
            {"text": "click bottom-right marker east neighbor",
             "verdict": "approve", "frame_changed": False},
        ],
        repeat_axis_count=2,
        null_effect_streak=2,
    )
    text = s.render()
    assert "state_now:" in text
    assert "last_5_strategies:" in text
    assert "repeat_axis_count: 2" in text
    assert "null_effect_streak: 2" in text
    assert "frame_changed=False" in text


def test_state_summary_empty_render_safe():
    s = StateSummary()
    text = s.render()
    assert "state_now:" in text
    assert "repeat_axis_count: 0" in text
    assert "null_effect_streak: 0" in text


def test_state_summary_caps_strategies_at_5():
    s = StateSummary(last_strategies=[
        {"text": f"s{i}", "verdict": "approve", "frame_changed": False}
        for i in range(10)
    ])
    text = s.render()
    # Only first 5 appear with numbered prefixes
    assert "1. \"s0\"" in text
    assert "5. \"s4\"" in text
    assert "6. \"s5\"" not in text


# ─────────────────────────────────────────────────────────────────
# M2e grounding_text two-attribute validation
# ─────────────────────────────────────────────────────────────────


def test_grounding_valid_with_color_and_pixel_range():
    text = ("the bottom-right marker tile, color 8, occupies "
            "pixels 40-48 by 36-44; I click its center")
    assert validate_m2e_grounding_attributes(text)


def test_grounding_valid_with_color_name_and_shape():
    text = ("a red 4x4 cluster in the lower-right; centroid at "
            "approximately (44, 44)")
    assert validate_m2e_grounding_attributes(text)


def test_grounding_invalid_generic_blurb():
    text = "I see a region near (44, 44)"
    assert not validate_m2e_grounding_attributes(text)


def test_grounding_invalid_color_only():
    text = "blue thing somewhere"
    assert not validate_m2e_grounding_attributes(text)


def test_grounding_invalid_shape_only():
    text = "a cluster of cells, bbox unclear"
    # 'cluster' AND 'cells' are shape words but no color → fail
    assert not validate_m2e_grounding_attributes(text)


def test_executor_validator_rejects_generic_grounding():
    out = {
        "click_xy_hint": [44, 44],
        "grounding_text": "I see a region near 44,44 and click it",
    }
    res = validate_m2e_executor_output(out)
    assert not res.ok
    assert any("two visual attributes" in v for v in res.violations)


def test_executor_validator_accepts_specific_grounding():
    out = {
        "click_xy_hint": [44, 44],
        "grounding_text": ("the bottom-right marker, color 8, occupies "
                            "pixels 40-48 by 36-44; clicking centroid"),
    }
    res = validate_m2e_executor_output(out)
    assert res.ok, f"specific grounding rejected: {res.violations}"


# ─────────────────────────────────────────────────────────────────
# SUBSTITUTE prefix detection
# ─────────────────────────────────────────────────────────────────


def test_substitute_prefix_detected():
    text = ("SUBSTITUTE: proposer asked for top-right marker but only "
            "bottom-right visible, color 8 at pixels 50-56")
    assert m2e_is_substitute(text)


def test_substitute_prefix_not_detected_in_normal_grounding():
    text = "the bottom-right marker, color 8, occupies pixels 40-48"
    assert not m2e_is_substitute(text)


def test_substitute_prefix_with_leading_whitespace():
    text = "   SUBSTITUTE: missing region, using closest match in red"
    assert m2e_is_substitute(text)


def test_orchestrator_renders_statesummary_for_verifier(tmp_path, monkeypatch):
    """Round 16: when state_text_summary is a StateSummary instance,
    the verifier must receive the structured RENDERED multi-line text,
    not the dataclass itself."""
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    from agents.templates.agentica_lite.v611_orchestrator import (
        AnchorCounter, StateSummary, run_v611_turn,
    )

    received: list[dict] = []

    def m1(state_text, skill_md_summary, anchor_summary=None,
           rejection_reason=None):
        return {
            "nl_strategy": ("Looking at the grid, I propose clicking the "
                             "bottom-right marker to test the repair."),
            "suggested_click_region": "bottom-right marker area",
            "expected_signature": {"frame_changed": True, "unsat_delta": -1},
            "rollback_trigger": "no frame change after click",
        }

    def m2v(proposer_out, state_text_summary):
        received.append({"state_text_summary": state_text_summary})
        return {"verdict": "approve", "reason_nl": "ok approval"}

    def m2e(approved_out, png_bytes):
        return {
            "click_xy_hint": [44, 44],
            "grounding_text": ("color 8 marker at pixels 40-48 by 40-48, "
                                "clicking center"),
        }

    summary = StateSummary(
        state_now="8 markers, 5 satisfied",
        last_strategies=[
            {"text": "click bottom-right", "verdict": "approve",
             "frame_changed": False},
        ],
        repeat_axis_count=1,
        null_effect_streak=1,
    )
    res = run_v611_turn(
        turn_id=0, state_text="full", state_text_summary=summary,
        png_bytes=b"png", skill_md_summary="skill",
        anchor=AnchorCounter(), m1_proposer=m1, m2v_verifier=m2v,
        m2e_executor=m2e,
    )
    assert res.success
    # verifier received the RENDERED string, not the StateSummary instance
    passed = received[0]["state_text_summary"]
    assert isinstance(passed, str)
    assert "state_now: 8 markers, 5 satisfied" in passed
    assert "repeat_axis_count: 1" in passed
    assert "null_effect_streak: 1" in passed


def test_orchestrator_logs_substitute_drift_when_m2e_uses_prefix(
    tmp_path, monkeypatch
):
    """Round 16: when M2e emits SUBSTITUTE-prefixed grounding_text,
    the orchestrator must log 'substitute_drift' event."""
    log_path = tmp_path / "t.jsonl"
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(log_path))
    from agents.templates.agentica_lite.v611_orchestrator import (
        AnchorCounter, run_v611_turn,
    )
    from agents.templates.agentica_lite.v611_telemetry import read_telemetry

    def m1(state_text, skill_md_summary, anchor_summary=None,
           rejection_reason=None):
        return {
            "nl_strategy": ("Looking at the grid, propose clicking the "
                             "imaginary top-right area to test a marker."),
            "suggested_click_region": "imagined top-right marker",
            "expected_signature": {"frame_changed": True, "unsat_delta": -1},
            "rollback_trigger": "no frame change",
        }

    def m2v(proposer_out, state_text_summary):
        return {"verdict": "approve", "reason_nl": "approving for test"}

    def m2e(approved_out, png_bytes):
        return {
            "click_xy_hint": [50, 50],
            "grounding_text": ("SUBSTITUTE: top-right not visible, using "
                                "bottom-right color 8 marker at pixels "
                                "48-56 by 48-56"),
        }

    res = run_v611_turn(
        turn_id=7, state_text="full", state_text_summary="ok",
        png_bytes=b"png", skill_md_summary="skill",
        anchor=AnchorCounter(), m1_proposer=m1, m2v_verifier=m2v,
        m2e_executor=m2e,
    )
    assert res.success
    events = read_telemetry(log_path)
    substitute_events = [e for e in events if e.event == "substitute_drift"]
    assert len(substitute_events) == 1
    assert substitute_events[0].turn_id == 7


def test_substitute_grounding_still_must_validate_attributes():
    """SUBSTITUTE: prefix doesn't bypass the two-attribute rule."""
    out_generic_sub = {
        "click_xy_hint": [50, 50],
        "grounding_text": "SUBSTITUTE: missing thing, click here",
    }
    res = validate_m2e_executor_output(out_generic_sub)
    assert not res.ok  # still missing color + shape concreteness

    out_concrete_sub = {
        "click_xy_hint": [50, 50],
        "grounding_text": ("SUBSTITUTE: missing top-right, using "
                            "bottom-right marker color 8 at pixels "
                            "48-56 by 48-56"),
    }
    res = validate_m2e_executor_output(out_concrete_sub)
    assert res.ok
    assert m2e_is_substitute(out_concrete_sub["grounding_text"])
