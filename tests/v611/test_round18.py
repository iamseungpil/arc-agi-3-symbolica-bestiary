"""Round 18 enforcement tests — length clamping + per-role telemetry counts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.v611_orchestrator import (  # noqa: E402
    MAX_SKILL_MD_SUMMARY_CHARS,
    MAX_STATE_NOW_CHARS,
    MAX_STATE_TEXT_CHARS,
    AnchorCounter,
    StateSummary,
    _clamp,
    run_v611_turn,
)
from agents.templates.agentica_lite.v611_telemetry import (  # noqa: E402
    read_telemetry,
)


def test_clamp_short_text_unchanged():
    assert _clamp("hello", 100) == "hello"


def test_clamp_truncates_with_ellipsis():
    long = "a" * 500
    out = _clamp(long, 100)
    assert len(out) == 100
    assert out.endswith("...")


def test_clamp_handles_none_or_empty():
    assert _clamp("", 10) == ""
    assert _clamp(None, 10) == ""


def test_state_now_clamped_in_render():
    long_state = "x" * 1000
    s = StateSummary(state_now=long_state)
    rendered = s.render()
    # state_now line should be clamped to MAX_STATE_NOW_CHARS chars
    state_line = [ln for ln in rendered.splitlines()
                   if ln.startswith("state_now:")][0]
    assert len(state_line) <= len("state_now: ") + MAX_STATE_NOW_CHARS
    assert state_line.endswith("...")


def test_state_text_clamped_when_passed_to_run_v611_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    received_m1: list[dict] = []

    def m1(state_text, skill_md_summary, anchor_summary=None,
           rejection_reason=None):
        received_m1.append({"state_text": state_text,
                              "skill_md_summary": skill_md_summary})
        return {
            "nl_strategy": ("a strategy of clicking the bottom-right marker "
                             "tile to test for repair patterns in the grid"),
            "suggested_click_region": "bottom-right marker area",
            "expected_signature": {"frame_changed": True, "unsat_delta": -1},
            "rollback_trigger": "no frame change",
        }

    def m2v(proposer_out, state_text_summary):
        return {"verdict": "approve", "reason_nl": "test approval here"}

    def m2e(approved_out, png_bytes):
        return {
            "click_xy_hint": [44, 44],
            "grounding_text": ("color 8 marker at pixels 40-48 by 40-48, "
                                "clicking center"),
        }

    huge_state_text = "S" * 10000  # 10k chars
    huge_skill_md = "K" * 5000      # 5k chars
    run_v611_turn(
        turn_id=0, state_text=huge_state_text,
        state_text_summary="ok summary",
        png_bytes=b"png", skill_md_summary=huge_skill_md,
        anchor=AnchorCounter(), m1_proposer=m1, m2v_verifier=m2v,
        m2e_executor=m2e,
    )
    assert len(received_m1) == 1
    # state_text clamped to MAX_STATE_TEXT_CHARS
    assert len(received_m1[0]["state_text"]) <= MAX_STATE_TEXT_CHARS
    # skill_md_summary clamped to MAX_SKILL_MD_SUMMARY_CHARS
    assert (len(received_m1[0]["skill_md_summary"])
            <= MAX_SKILL_MD_SUMMARY_CHARS)


def test_per_role_telemetry_counts_5_turns(tmp_path, monkeypatch):
    """Round 18: per-role telemetry count invariants for 5 turns —
    we should see >=5 m1 role_returned events, >=5 m2v role_returned,
    and >=5 m2e role_returned (assuming all approve)."""
    log_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(log_path))
    anchor = AnchorCounter()

    def m1(state_text, skill_md_summary, anchor_summary=None,
           rejection_reason=None):
        return {
            "nl_strategy": ("propose clicking the bottom-right marker "
                             "tile to test the repair mechanism"),
            "suggested_click_region": "bottom-right tile area",
            "expected_signature": {"frame_changed": True, "unsat_delta": -1},
            "rollback_trigger": "no frame change after click",
        }

    def m2v(proposer_out, state_text_summary):
        return {"verdict": "approve", "reason_nl": "ok mocked approval"}

    def m2e(approved_out, png_bytes):
        return {
            "click_xy_hint": [44, 44],
            "grounding_text": ("color 8 marker tile at pixels 40-48 by "
                                "40-48 clicking center"),
        }

    for t in range(5):
        run_v611_turn(
            turn_id=t, state_text="s", state_text_summary="ok",
            png_bytes=b"png", skill_md_summary="skill",
            anchor=anchor, m1_proposer=m1, m2v_verifier=m2v,
            m2e_executor=m2e,
        )

    events = read_telemetry(log_path)

    m1_returned = [e for e in events
                    if e.role == "m1" and e.event == "role_returned"]
    m2v_returned = [e for e in events
                     if e.role == "m2v" and e.event == "role_returned"]
    m2e_returned = [e for e in events
                     if e.role == "m2e" and e.event == "role_returned"]

    # Each role called at least once per turn (no retries in clean path)
    assert len(m1_returned) >= 5, \
        f"expected >=5 m1 calls, got {len(m1_returned)}"
    assert len(m2v_returned) >= 5, \
        f"expected >=5 m2v calls, got {len(m2v_returned)}"
    assert len(m2e_returned) >= 5, \
        f"expected >=5 m2e calls (all approved), got {len(m2e_returned)}"


def test_per_role_telemetry_drops_m2e_when_rejected(tmp_path, monkeypatch):
    """If verifier rejects, M2e should NOT be called → m2e role_returned
    count should be less than m2v count."""
    log_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(log_path))
    anchor = AnchorCounter()

    def m1(state_text, skill_md_summary, anchor_summary=None,
           rejection_reason=None):
        return {
            "nl_strategy": ("propose clicking the upper-right area to "
                             "test if anchoring fails as expected here"),
            "suggested_click_region": "upper-right area",
            "expected_signature": {"frame_changed": True, "unsat_delta": -1},
            "rollback_trigger": "no frame change",
        }

    def m2v(proposer_out, state_text_summary):
        return {"verdict": "reject_anchor",
                "reason_nl": "anchored failure pattern detected"}

    def m2e(approved_out, png_bytes):
        return {
            "click_xy_hint": [44, 44],
            "grounding_text": ("color 8 at pixels 40-48 by 40-48 "
                                "clicking center"),
        }

    for t in range(3):
        run_v611_turn(
            turn_id=t, state_text="s", state_text_summary="ok",
            png_bytes=b"png", skill_md_summary="skill",
            anchor=anchor, m1_proposer=m1, m2v_verifier=m2v,
            m2e_executor=m2e,
        )

    events = read_telemetry(log_path)
    m2e_returned = [e for e in events
                     if e.role == "m2e" and e.event == "role_returned"]
    # On reject_anchor, m2e is never called
    assert len(m2e_returned) == 0, \
        f"m2e called on reject path: {len(m2e_returned)} events"
    # anchor rejects should be logged
    anchor_evts = [e for e in events
                    if e.role == "anchor" and e.event == "reject_anchor"]
    assert len(anchor_evts) == 3
